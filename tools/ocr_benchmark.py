from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
versioned_vendor = ROOT / f".vendor-py{sys.version_info.major}{sys.version_info.minor}"
for candidate in (versioned_vendor, ROOT / ".vendor"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kiosk_accessibility.demo import create_mock_kiosk  # noqa: E402
from kiosk_accessibility.ocr import TesseractOCR  # noqa: E402
from kiosk_accessibility.vision import detect_screen, warp_screen  # noqa: E402


def normalized(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def match_count(expected: list[str], found: list[str]) -> int:
    normalized_found = [normalized(text) for text in found]
    count = 0
    for target in expected:
        wanted = normalized(target)
        if any(
            wanted == candidate
            or SequenceMatcher(None, wanted, candidate).ratio() >= 0.72
            for candidate in normalized_found
        ):
            count += 1
    return count


def variants(image: np.ndarray) -> dict[str, np.ndarray]:
    height, width = image.shape[:2]
    small = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    low_resolution = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    low_contrast = cv2.convertScaleAbs(image, alpha=0.55, beta=82)
    blur = cv2.GaussianBlur(image, (7, 7), 1.6)
    glare = image.astype(np.float32)
    x_gradient = np.linspace(0, 1, width, dtype=np.float32)
    highlight = np.clip(1 - np.abs(x_gradient - 0.72) / 0.20, 0, 1)
    glare += highlight[None, :, None] * 105
    glare = np.clip(glare, 0, 255).astype(np.uint8)
    return {
        "clean": image,
        "low_resolution": low_resolution,
        "low_contrast": low_contrast,
        "blur": blur,
        "glare": glare,
    }


def main() -> None:
    frame, expected_items = create_mock_kiosk()
    detection = detect_screen(frame)
    if detection is None:
        raise RuntimeError("Mock kiosk screen was not detected")
    screen = warp_screen(frame, detection.corners)
    engine = TesseractOCR(ROOT)
    expected = [item.text for item in expected_items]
    total = 0
    for name, image in variants(screen).items():
        items = engine.recognize(image)
        texts = [item.text for item in items]
        matched = match_count(expected, texts)
        total += matched
        labelled = [f"{item.text}({item.confidence:.0f})" for item in items]
        print(
            f"{name:>14}: {matched}/{len(expected)} {labelled} "
            f"sharp={engine.last_diagnostics.get('sharpness', 0):.0f} "
            f"over={engine.last_diagnostics.get('overexposed_ratio', 0):.0%} "
            f"hints={engine.last_diagnostics.get('hints', [])}"
        )
    print(f"TOTAL_MATCHES={total}/{len(expected) * len(variants(screen))}")


if __name__ == "__main__":
    main()
