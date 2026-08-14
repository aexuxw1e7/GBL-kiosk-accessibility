from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .grid import vibration_interval_ms
from .models import OCRItem
from .vision import order_corners


@dataclass(frozen=True)
class TargetTracking:
    pointer_screen: tuple[float, float]
    target_frame_center: tuple[float, float]
    target_frame_polygon: np.ndarray
    normalized_distance: float
    interval_ms: int
    inside: bool
    direction: str


class ApproachRateTracker:
    """Estimate d(distance)/dt; positive values mean approaching the target."""

    def __init__(self, smoothing: float = 0.35) -> None:
        self.smoothing = smoothing
        self.previous_distance: float | None = None
        self.previous_time: float | None = None
        self.rate = 0.0

    def reset(self) -> None:
        self.previous_distance = None
        self.previous_time = None
        self.rate = 0.0

    def update(self, distance: float, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else timestamp
        if self.previous_distance is None or self.previous_time is None:
            self.previous_distance = distance
            self.previous_time = now
            return 0.0
        elapsed = now - self.previous_time
        if elapsed <= 0:
            return self.rate
        raw_rate = (self.previous_distance - distance) / elapsed
        self.rate = self.smoothing * raw_rate + (1.0 - self.smoothing) * self.rate
        self.previous_distance = distance
        self.previous_time = now
        return self.rate


def _destination(output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2)


def frame_point_to_screen(
    point: tuple[float, float],
    screen_corners: np.ndarray,
    output_size: tuple[int, int] = (600, 1000),
) -> tuple[float, float]:
    matrix = cv2.getPerspectiveTransform(
        order_corners(screen_corners), _destination(output_size)
    )
    mapped = transform_points(np.array([point], dtype=np.float32), matrix)[0]
    return float(mapped[0]), float(mapped[1])


def _direction(pointer: tuple[float, float], target: OCRItem) -> str:
    px, py = pointer
    horizontal = ""
    vertical = ""
    if px < target.x:
        horizontal = "오른쪽"
    elif px > target.x + target.width:
        horizontal = "왼쪽"
    if py < target.y:
        vertical = "아래"
    elif py > target.y + target.height:
        vertical = "위"
    if not horizontal and not vertical:
        return "목표 일치"
    return "·".join(part for part in (horizontal, vertical) if part) + "로 이동"


def track_camera_center(
    frame_shape: tuple[int, ...],
    screen_corners: np.ndarray,
    target: OCRItem,
    output_size: tuple[int, int] = (600, 1000),
) -> TargetTracking:
    frame_height, frame_width = frame_shape[:2]
    source = order_corners(screen_corners)
    destination = _destination(output_size)
    frame_to_screen = cv2.getPerspectiveTransform(source, destination)
    screen_to_frame = cv2.getPerspectiveTransform(destination, source)

    frame_center = np.array([[frame_width / 2, frame_height / 2]], dtype=np.float32)
    pointer = transform_points(frame_center, frame_to_screen)[0]
    pointer_screen = (float(pointer[0]), float(pointer[1]))

    target_polygon_screen = np.array(
        [
            [target.x, target.y],
            [target.x + target.width, target.y],
            [target.x + target.width, target.y + target.height],
            [target.x, target.y + target.height],
        ],
        dtype=np.float32,
    )
    target_polygon_frame = transform_points(target_polygon_screen, screen_to_frame)
    target_center_screen = np.array([target.center], dtype=np.float32)
    target_center_frame = transform_points(target_center_screen, screen_to_frame)[0]
    interval, inside, normalized = vibration_interval_ms(
        pointer_screen, target, output_size
    )
    return TargetTracking(
        pointer_screen=pointer_screen,
        target_frame_center=(
            float(target_center_frame[0]),
            float(target_center_frame[1]),
        ),
        target_frame_polygon=target_polygon_frame,
        normalized_distance=normalized,
        interval_ms=interval,
        inside=inside,
        direction=_direction(pointer_screen, target),
    )


def draw_source_tracking(
    frame: np.ndarray,
    tracking: TargetTracking,
) -> np.ndarray:
    annotated = frame.copy()
    color = (45, 220, 90) if tracking.inside else (0, 205, 255)
    polygon = np.round(tracking.target_frame_polygon).astype(np.int32)
    cv2.polylines(
        annotated,
        [polygon.reshape((-1, 1, 2))],
        True,
        color,
        5,
        cv2.LINE_AA,
    )
    height, width = annotated.shape[:2]
    camera_center = (width // 2, height // 2)
    cv2.drawMarker(
        annotated,
        camera_center,
        color,
        cv2.MARKER_CROSS,
        42,
        5,
        cv2.LINE_AA,
    )
    label = (
        "TARGET LOCK"
        if tracking.inside
        else f"TARGET {tracking.normalized_distance:.0%} / {tracking.interval_ms}ms"
    )
    cv2.rectangle(annotated, (18, height - 64), (440, height - 16), (18, 28, 34), -1)
    cv2.putText(
        annotated,
        label,
        (32, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_warped_pointer(
    image: np.ndarray,
    tracking: TargetTracking,
    target: OCRItem,
) -> np.ndarray:
    return draw_pointer_to_target(
        image,
        tracking.pointer_screen,
        target,
        tracking.inside,
    )


def draw_pointer_to_target(
    image: np.ndarray,
    pointer: tuple[float, float],
    target: OCRItem,
    inside: bool,
) -> np.ndarray:
    annotated = image.copy()
    color = (45, 220, 90) if inside else (255, 120, 30)
    px, py = (int(round(value)) for value in pointer)
    cv2.drawMarker(
        annotated,
        (px, py),
        color,
        cv2.MARKER_CROSS,
        40,
        5,
        cv2.LINE_AA,
    )
    return annotated


def draw_exploration_paths(
    image: np.ndarray,
    routes: Mapping[
        str,
        Sequence[Sequence[tuple[float, float]]],
    ],
) -> np.ndarray:
    """Draw mission routes in the corrected 600×1000 screen coordinates."""
    annotated = image.copy()
    colors = {
        "camera": (224, 108, 38),
        "cursor": (28, 146, 245),
        "click": (28, 146, 245),
    }
    for source, segments in routes.items():
        color = colors.get(source, (190, 120, 70))
        for segment in segments:
            points = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
            if not len(points):
                continue
            rounded = np.round(points).astype(np.int32)
            if len(rounded) >= 2:
                cv2.polylines(
                    annotated,
                    [rounded.reshape((-1, 1, 2))],
                    False,
                    color,
                    3,
                    cv2.LINE_AA,
                )
            stride = max(1, len(rounded) // 80)
            for x, y in rounded[::stride]:
                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    4,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
    return annotated
