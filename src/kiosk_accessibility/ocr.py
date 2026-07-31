from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .models import OCRItem


PRICE_ONLY = re.compile(r"^[\d\s,.\-원₩]+$")
HAS_LETTER = re.compile(r"[A-Za-z가-힣]")
SINGLE_HANGUL = re.compile(r"^[가-힣]$")
NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
TRAILING_PRICE = re.compile(
    r"\s+(?:₩\s*)?\d[\d\s,.\-]*(?:원|₩)\s*$",
    re.UNICODE,
)


@dataclass(frozen=True)
class OCRPass:
    image: np.ndarray
    scale_x: float
    scale_y: float
    psm: int
    name: str


def find_tesseract(project_root: Path) -> Path | None:
    candidates = [
        project_root / ".runtime" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    command = shutil.which("tesseract")
    if command:
        candidates.insert(0, Path(command))
    return next((path for path in candidates if path.exists()), None)


class TesseractOCR:
    def __init__(self, project_root: Path, minimum_confidence: float = 32.0) -> None:
        self.project_root = project_root
        self.minimum_confidence = minimum_confidence
        self.executable = find_tesseract(project_root)
        if self.executable is not None:
            pytesseract.pytesseract.tesseract_cmd = str(self.executable)
        local_tessdata = project_root / ".runtime" / "tessdata"
        self.tessdata = self._prepare_tessdata(local_tessdata)
        self.language = self._language()
        self.last_diagnostics: dict = {}

    @property
    def available(self) -> bool:
        return self.executable is not None

    @property
    def name(self) -> str:
        if not self.available:
            return "OCR 엔진 없음"
        return f"Tesseract 다중 경로 ({self.language})"

    def _language(self) -> str:
        if self.tessdata and (self.tessdata / "kor.traineddata").exists():
            return "kor+eng" if (self.tessdata / "eng.traineddata").exists() else "kor"
        return "eng"

    @staticmethod
    def _prepare_tessdata(source: Path) -> Path | None:
        if not source.exists():
            return None
        try:
            str(source).encode("ascii")
            return source
        except UnicodeEncodeError:
            # Windows Tesseract 5.4 may print a locale-encoded error when its
            # --tessdata-dir contains Hangul. Mirror only the small language
            # models into an ASCII-only temporary cache.
            cache = Path(tempfile.gettempdir()) / "gbl_kiosk_tessdata"
            cache.mkdir(parents=True, exist_ok=True)
            for model in source.glob("*.traineddata"):
                destination = cache / model.name
                if (
                    not destination.exists()
                    or destination.stat().st_size != model.stat().st_size
                ):
                    shutil.copy2(model, destination)
            return cache

    @staticmethod
    def _preprocess_passes(image: np.ndarray) -> tuple[list[OCRPass], OCRPass]:
        height, width = image.shape[:2]
        sparse_scale = 1.65
        sparse_size = (
            max(1, int(width * sparse_scale)),
            max(1, int(height * sparse_scale)),
        )
        colour = cv2.resize(image, sparse_size, interpolation=cv2.INTER_CUBIC)

        block_scale = 1.0
        block_size = (
            max(1, int(width * block_scale)),
            max(1, int(height * block_scale)),
        )
        block_colour = cv2.resize(
            image, block_size, interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8)).apply(gray)
        soft = cv2.GaussianBlur(clahe, (0, 0), 1.0)
        sharpened = cv2.addWeighted(clahe, 1.55, soft, -0.55, 0)
        enhanced = cv2.resize(
            sharpened, sparse_size, interpolation=cv2.INTER_CUBIC
        )
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        primary = [
            OCRPass(
                colour,
                sparse_scale,
                sparse_scale,
                11,
                "확대 컬러·희소 텍스트",
            ),
            OCRPass(
                block_colour,
                block_scale,
                block_scale,
                6,
                "확대 컬러·텍스트 블록",
            ),
        ]
        fallback = OCRPass(
            enhanced_bgr,
            sparse_scale,
            sparse_scale,
            11,
            "명암 보정·희소 텍스트",
        )
        return primary, fallback

    def _config(self, psm: int) -> str:
        config_parts = [
            "--oem",
            "1",
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ]
        if self.tessdata:
            config_parts.extend(["--tessdata-dir", str(self.tessdata)])
        return " ".join(config_parts)

    @staticmethod
    def _clean_display_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip(" \t\r\n|:;·•")
        text = re.sub(r"^[^\w가-힣]+|[^\w가-힣]+$", "", text, flags=re.UNICODE)
        text = TRAILING_PRICE.sub("", text).strip()
        return text

    @staticmethod
    def _join_words(indices: list[int], data: dict, spec: OCRPass) -> str:
        parts: list[str] = []
        previous_index: int | None = None
        for index in indices:
            word = unicodedata.normalize("NFC", str(data["text"][index]).strip())
            if not word:
                continue
            separator = " "
            if previous_index is not None:
                previous_word = unicodedata.normalize(
                    "NFC", str(data["text"][previous_index]).strip()
                )
                previous_right = (
                    int(data["left"][previous_index])
                    + int(data["width"][previous_index])
                ) / spec.scale_x
                current_left = int(data["left"][index]) / spec.scale_x
                gap = current_left - previous_right
                typical_height = max(
                    int(data["height"][previous_index]) / spec.scale_y,
                    int(data["height"][index]) / spec.scale_y,
                    6,
                )
                if (
                    SINGLE_HANGUL.fullmatch(previous_word)
                    and SINGLE_HANGUL.fullmatch(word)
                    and gap <= typical_height * 0.65
                ):
                    separator = ""
            if not parts:
                parts.append(word)
            else:
                parts.append(f"{separator}{word}")
            previous_index = index
        return "".join(parts)

    def _recognize_pass(
        self,
        spec: OCRPass,
        original_shape: tuple[int, ...],
    ) -> tuple[list[OCRItem], int]:
        rgb = cv2.cvtColor(spec.image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb,
            lang=self.language,
            config=self._config(spec.psm),
            output_type=Output.DICT,
            timeout=10,
        )
        lines: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        accepted_indices: set[int] = set()
        for index, raw_text in enumerate(data["text"]):
            text = unicodedata.normalize("NFC", str(raw_text).strip())
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1
            if text and confidence >= max(15.0, self.minimum_confidence - 12.0):
                key = (
                    int(data["block_num"][index]),
                    int(data["par_num"][index]),
                    int(data["line_num"][index]),
                )
                lines[key].append(index)
                accepted_indices.add(index)

        results: list[OCRItem] = []
        for line_indices in lines.values():
            ordered = sorted(line_indices, key=lambda index: int(data["left"][index]))
            segments: list[list[int]] = []
            for index in ordered:
                if not segments:
                    segments.append([index])
                    continue
                previous = segments[-1][-1]
                previous_right = (
                    int(data["left"][previous]) + int(data["width"][previous])
                ) / spec.scale_x
                current_left = int(data["left"][index]) / spec.scale_x
                gap = current_left - previous_right
                typical_height = max(
                    int(data["height"][previous]) / spec.scale_y,
                    int(data["height"][index]) / spec.scale_y,
                    8,
                )
                if gap > max(36, typical_height * 2.0):
                    segments.append([index])
                else:
                    segments[-1].append(index)

            for indices in segments:
                text = self._clean_display_text(
                    self._join_words(indices, data, spec)
                )
                normalized = NON_WORD.sub("", text)
                if (
                    not text
                    or len(text) > 45
                    or PRICE_ONLY.fullmatch(text)
                    or not HAS_LETTER.search(text)
                    or len(normalized) < 2
                ):
                    continue
                raw_left = min(int(data["left"][index]) for index in indices)
                raw_top = min(int(data["top"][index]) for index in indices)
                raw_right = max(
                    int(data["left"][index]) + int(data["width"][index])
                    for index in indices
                )
                raw_bottom = max(
                    int(data["top"][index]) + int(data["height"][index])
                    for index in indices
                )
                left = int(np.floor(raw_left / spec.scale_x))
                top = int(np.floor(raw_top / spec.scale_y))
                right = int(np.ceil(raw_right / spec.scale_x))
                bottom = int(np.ceil(raw_bottom / spec.scale_y))
                original_height, original_width = original_shape[:2]
                left = min(original_width - 1, max(0, left))
                top = min(original_height - 1, max(0, top))
                right = min(original_width, max(left + 1, right))
                bottom = min(original_height, max(top + 1, bottom))
                weights = [
                    max(1, len(NON_WORD.sub("", str(data["text"][index]))))
                    for index in indices
                ]
                confidence = sum(
                    float(data["conf"][index]) * weight
                    for index, weight in zip(indices, weights)
                ) / sum(weights)
                width, height = right - left, bottom - top
                alphanumeric_ratio = len(normalized) / max(1, len(text))
                ascii_letters = re.fullmatch(r"[A-Za-z\s]+", text)
                has_hangul = re.search(r"[가-힣]", text) is not None
                if (
                    confidence < self.minimum_confidence
                    or width < 8
                    or height < 6
                    or alphanumeric_ratio < 0.48
                    or (not has_hangul and len(normalized) < 3)
                    or (not has_hangul and confidence < 48)
                    or (
                        ascii_letters is not None
                        and len(normalized) <= 3
                        and not normalized.isupper()
                        and confidence < 65
                    )
                ):
                    continue
                # Discard only a wide top banner. Small category tabs near the
                # top remain valid touch targets.
                if (
                    top < original_height * 0.12
                    and width > original_width * 0.28
                ):
                    continue
                results.append(
                    OCRItem(
                        text=text,
                        x=left,
                        y=top,
                        width=width,
                        height=height,
                        confidence=confidence,
                    )
                )
        nonempty_count = sum(bool(str(text).strip()) for text in data["text"])
        return (
            sorted(results, key=lambda item: (item.y, item.x)),
            max(0, nonempty_count - len(accepted_indices)),
        )

    @staticmethod
    def _normalized_text(text: str) -> str:
        return NON_WORD.sub("", unicodedata.normalize("NFC", text)).casefold()

    @staticmethod
    def _overlap_over_smaller(first: OCRItem, second: OCRItem) -> float:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        smaller = min(first.width * first.height, second.width * second.height)
        return intersection / max(1, smaller)

    @classmethod
    def _deduplicate(cls, candidates: list[OCRItem]) -> list[OCRItem]:
        selected: list[OCRItem] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.confidence
                + min(10, len(cls._normalized_text(item.text))) * 1.4
            ),
            reverse=True,
        ):
            candidate_text = cls._normalized_text(candidate.text)
            duplicate_index = None
            for index, existing in enumerate(selected):
                overlap = cls._overlap_over_smaller(candidate, existing)
                if overlap < 0.48:
                    continue
                existing_text = cls._normalized_text(existing.text)
                similarity = SequenceMatcher(
                    None, candidate_text, existing_text
                ).ratio()
                if (
                    candidate_text == existing_text
                    or similarity >= 0.72
                    or (
                        min(len(candidate_text), len(existing_text)) >= 3
                        and (
                            candidate_text in existing_text
                            or existing_text in candidate_text
                        )
                    )
                ):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                selected.append(candidate)
            else:
                existing = selected[duplicate_index]
                existing_text = cls._normalized_text(existing.text)
                candidate_is_more_complete = (
                    len(candidate_text) > len(existing_text)
                    and existing_text in candidate_text
                    and candidate.confidence >= existing.confidence - 12
                )
                if candidate_is_more_complete:
                    selected[duplicate_index] = candidate
        return sorted(selected, key=lambda item: (item.y, item.x))

    @staticmethod
    def _quality_diagnostics(image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(gray, (300, 500), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(reduced, cv2.CV_64F).var())
        overexposed = float(np.mean(reduced >= 245))
        underexposed = float(np.mean(reduced <= 20))
        contrast = float(reduced.std())
        hints: list[str] = []
        if sharpness < 240:
            hints.append("흐림 가능성: 카메라를 고정하거나 조금 가까이 이동")
        if overexposed > 0.46:
            hints.append("반사·과다노출 가능성: 카메라 각도를 조금 변경")
        if underexposed > 0.28:
            hints.append("화면이 어두움: 화면 밝기 또는 주변 조명 확인")
        if contrast < 27:
            hints.append("낮은 대비: 화면을 더 크게 비추기")
        return {
            "sharpness": sharpness,
            "overexposed_ratio": overexposed,
            "underexposed_ratio": underexposed,
            "contrast": contrast,
            "hints": hints,
        }

    def recognize(self, image) -> list[OCRItem]:
        if not self.available:
            raise RuntimeError(
                "Tesseract OCR 실행 파일을 찾지 못했습니다. 내장 데모는 예상 좌표로 실행할 수 있습니다."
            )
        primary, fallback = self._preprocess_passes(image)
        candidates: list[OCRItem] = []
        pass_names: list[str] = []
        excluded = 0
        errors: list[str] = []
        for spec in primary:
            try:
                items, rejected = self._recognize_pass(spec, image.shape)
            except RuntimeError as error:
                errors.append(f"{spec.name}: {error}")
                continue
            candidates.extend(items)
            excluded += rejected
            pass_names.append(spec.name)

        merged = self._deduplicate(candidates)
        diagnostics = self._quality_diagnostics(image)
        median_confidence = (
            float(np.median([item.confidence for item in merged]))
            if merged
            else 0.0
        )
        if len(merged) < 3 or median_confidence < 46 or diagnostics["hints"]:
            try:
                items, rejected = self._recognize_pass(fallback, image.shape)
            except RuntimeError as error:
                errors.append(f"{fallback.name}: {error}")
            else:
                candidates.extend(items)
                excluded += rejected
                pass_names.append(fallback.name)
                merged = self._deduplicate(candidates)

        diagnostics.update(
            {
                "passes": pass_names,
                "candidate_count": len(candidates),
                "final_count": len(merged),
                "excluded_count": excluded,
                "median_confidence": (
                    float(np.median([item.confidence for item in merged]))
                    if merged
                    else 0.0
                ),
                "errors": errors,
            }
        )
        self.last_diagnostics = diagnostics
        if not pass_names and errors:
            raise RuntimeError("OCR 처리 시간이 초과되었습니다. 화면을 고정하고 다시 시도하세요.")
        return merged


class StaticOCR:
    def __init__(self, items: list[OCRItem]) -> None:
        self.items = items
        self.name = "내장 데모 좌표"
        self.available = True
        self.last_diagnostics = {
            "passes": ["내장 데모 좌표"],
            "candidate_count": len(items),
            "final_count": len(items),
            "excluded_count": 0,
            "median_confidence": 99.0,
            "hints": [],
        }

    def recognize(self, image) -> list[OCRItem]:
        return [
            OCRItem(
                text=item.text,
                x=item.x,
                y=item.y,
                width=item.width,
                height=item.height,
                confidence=item.confidence,
            )
            for item in self.items
        ]
