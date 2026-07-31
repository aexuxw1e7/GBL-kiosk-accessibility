from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .analyzer import KioskAnalyzer
from .demo import create_mock_kiosk
from .gui import launch
from .ocr import StaticOCR, TesseractOCR


def _write_image(path: Path, image) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"이미지를 인코딩할 수 없습니다: {path}")
    path.write_bytes(encoded.tobytes())


def run_demo(project_root: Path, output: Path, force_static: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame, expected = create_mock_kiosk()
    ocr = TesseractOCR(project_root)
    analyzer = KioskAnalyzer(ocr)
    provider = StaticOCR(expected) if force_static or not ocr.available else None
    try:
        result = analyzer.analyze(frame, provider)
    except Exception:
        result = analyzer.analyze(frame, StaticOCR(expected))
    _write_image(output / "mock_kiosk_input.png", frame)
    _write_image(output / "screen_detected.png", result.source_annotated)
    _write_image(output / "grid_analysis.png", result.warped_annotated)
    (output / "analysis.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"완료: {output}")
    print(f"OCR: {result.ocr_engine}")
    for item in result.items:
        print(f"- {item.guidance} (신뢰도 {item.confidence:.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="키오스크 접근성 프로토타입")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="GUI 없이 내장 모의 키오스크를 분석합니다.",
    )
    parser.add_argument(
        "--static-demo",
        action="store_true",
        help="OCR 대신 내장 정답 좌표로 영상처리·격자 변환을 검증합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_output"),
        help="데모 결과 저장 폴더",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.demo or args.static_demo:
        run_demo(project_root, args.output.resolve(), force_static=args.static_demo)
    else:
        launch(project_root)

