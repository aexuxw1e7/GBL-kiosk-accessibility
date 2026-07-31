from __future__ import annotations

import cv2
import numpy as np

from .grid import assign_grid
from .models import AnalysisResult, OCRItem
from .vision import (
    alignment_message,
    detect_screen,
    draw_detection,
    warp_screen,
)


class KioskAnalyzer:
    def __init__(
        self,
        ocr,
        output_size: tuple[int, int] = (600, 1000),
        columns: int = 6,
        rows: int = 10,
    ) -> None:
        self.ocr = ocr
        self.output_size = output_size
        self.columns = columns
        self.rows = rows

    def analyze(
        self,
        frame: np.ndarray,
        ocr_override=None,
        detection_override=None,
    ) -> AnalysisResult:
        detection = detection_override or detect_screen(frame)
        if detection is None:
            raise ValueError(
                "키오스크 화면의 네 모서리를 찾지 못했습니다. 화면 전체가 보이도록 다시 촬영하세요."
            )
        message = alignment_message(frame.shape, detection)
        warped = warp_screen(frame, detection.corners, self.output_size)
        provider = ocr_override or self.ocr
        items = provider.recognize(warped)
        assign_grid(items, self.output_size[0], self.output_size[1], self.columns, self.rows)
        source_annotated = draw_detection(frame, detection, message)
        warped_annotated = annotate_warped(
            warped, items, columns=self.columns, rows=self.rows
        )
        return AnalysisResult(
            source=frame,
            source_annotated=source_annotated,
            warped=warped,
            warped_annotated=warped_annotated,
            detection=detection,
            items=items,
            alignment_message=message,
            ocr_engine=provider.name,
            ocr_diagnostics=getattr(provider, "last_diagnostics", {}).copy(),
        )


def annotate_warped(
    image: np.ndarray,
    items: list[OCRItem],
    columns: int = 6,
    rows: int = 10,
    selected: OCRItem | None = None,
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    overlay = annotated.copy()
    for column in range(1, columns):
        x = int(width * column / columns)
        cv2.line(overlay, (x, 0), (x, height), (70, 165, 255), 2)
    for row in range(1, rows):
        y = int(height * row / rows)
        cv2.line(overlay, (0, y), (width, y), (70, 165, 255), 2)
    annotated = cv2.addWeighted(overlay, 0.42, annotated, 0.58, 0)
    for item in items:
        is_selected = item is selected
        if is_selected:
            color = (0, 215, 255)
        elif item.confidence < 55:
            color = (20, 145, 255)
        else:
            color = (50, 210, 120)
        thickness = 5 if is_selected else 3
        cv2.rectangle(
            annotated,
            (item.x, item.y),
            (item.x + item.width, item.y + item.height),
            color,
            thickness,
        )
        cv2.putText(
            annotated,
            f"{item.grid} {item.confidence:.0f}%",
            (item.x, max(20, item.y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
