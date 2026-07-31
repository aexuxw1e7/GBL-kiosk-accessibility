from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class OCRItem:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    grid: str = ""
    relative: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def guidance(self) -> str:
        suffix = f" {self.relative}" if self.relative else ""
        return f"{self.text}, {self.grid}{suffix}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"center": list(self.center), "guidance": self.guidance}


@dataclass
class ScreenDetection:
    corners: np.ndarray
    center: tuple[float, float]
    area_ratio: float
    score: float


@dataclass
class AnalysisResult:
    source: np.ndarray
    source_annotated: np.ndarray
    warped: np.ndarray
    warped_annotated: np.ndarray
    detection: ScreenDetection
    items: list[OCRItem] = field(default_factory=list)
    alignment_message: str = ""
    ocr_engine: str = ""
    ocr_diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "alignment_message": self.alignment_message,
            "area_ratio": self.detection.area_ratio,
            "ocr_engine": self.ocr_engine,
            "ocr_diagnostics": self.ocr_diagnostics,
            "items": [item.to_dict() for item in self.items],
        }
