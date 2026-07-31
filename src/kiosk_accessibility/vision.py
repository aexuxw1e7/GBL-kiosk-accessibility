from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .models import ScreenDetection


MIN_OCR_FRAME_QUALITY = 5.4


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(differences)]  # top-right
    ordered[3] = points[np.argmax(differences)]  # bottom-left
    return ordered


def _candidate_score(contour: np.ndarray, frame_area: float) -> tuple[float, np.ndarray] | None:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approximation) != 4 or not cv2.isContourConvex(approximation):
        return None
    area = abs(cv2.contourArea(approximation))
    if area / frame_area < 0.12:
        return None
    points = order_corners(approximation.reshape(4, 2))
    width_top = np.linalg.norm(points[1] - points[0])
    width_bottom = np.linalg.norm(points[2] - points[3])
    height_left = np.linalg.norm(points[3] - points[0])
    height_right = np.linalg.norm(points[2] - points[1])
    if min(width_top, width_bottom, height_left, height_right) < 80:
        return None
    rectangularity = area / max(
        1.0, ((width_top + width_bottom) / 2) * ((height_left + height_right) / 2)
    )
    score = (area / frame_area) * max(0.0, min(1.0, rectangularity))
    return score, points


def detect_screen(frame: np.ndarray) -> ScreenDetection | None:
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        candidate = _candidate_score(contour, frame_area)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    score, corners = max(candidates, key=lambda value: value[0])
    center_point = corners.mean(axis=0)
    area_ratio = abs(cv2.contourArea(corners)) / frame_area
    return ScreenDetection(
        corners=corners,
        center=(float(center_point[0]), float(center_point[1])),
        area_ratio=float(area_ratio),
        score=float(score),
    )


def full_frame_detection(
    frame: np.ndarray, margin_ratio: float = 0.01
) -> ScreenDetection:
    """Treat almost the entire frame as a kiosk screen for borderless displays."""
    height, width = frame.shape[:2]
    margin_x = max(0, int(width * margin_ratio))
    margin_y = max(0, int(height * margin_ratio))
    corners = np.array(
        [
            [margin_x, margin_y],
            [width - 1 - margin_x, margin_y],
            [width - 1 - margin_x, height - 1 - margin_y],
            [margin_x, height - 1 - margin_y],
        ],
        dtype=np.float32,
    )
    area_ratio = ((width - 2 * margin_x) * (height - 2 * margin_y)) / max(
        1.0, float(width * height)
    )
    return ScreenDetection(
        corners=corners,
        center=(width / 2, height / 2),
        area_ratio=float(area_ratio),
        score=float(area_ratio),
    )


def warp_screen(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = (600, 1000),
) -> np.ndarray:
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_corners(corners), destination)
    return cv2.warpPerspective(frame, matrix, (width, height))


def alignment_message(
    frame_shape: tuple[int, ...],
    detection: ScreenDetection,
    center_tolerance: float = 0.06,
) -> str:
    height, width = frame_shape[:2]
    dx = (detection.center[0] - width / 2) / width
    dy = (detection.center[1] - height / 2) / height
    instructions: list[str] = []
    if dx < -center_tolerance:
        instructions.append("왼쪽으로 이동")
    elif dx > center_tolerance:
        instructions.append("오른쪽으로 이동")
    if dy < -center_tolerance:
        instructions.append("위로 이동")
    elif dy > center_tolerance:
        instructions.append("아래로 이동")
    if detection.area_ratio < 0.34:
        instructions.append("조금 가까이")
    elif detection.area_ratio > 0.82:
        instructions.append("조금 멀리")
    return ", ".join(instructions) if instructions else "화면 정렬 완료"


def screen_change_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 1.0
    old = cv2.resize(previous, (120, 200))
    new = cv2.resize(current, (120, 200))
    old_gray = cv2.GaussianBlur(
        cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), (3, 3), 0
    ).astype(np.float32)
    new_gray = cv2.GaussianBlur(
        cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), (3, 3), 0
    ).astype(np.float32)

    old_std = float(old_gray.std())
    new_std = float(new_gray.std())
    if old_std < 5.0 or new_std < 5.0:
        return float(np.mean(np.abs(old_gray - new_gray)) / 255.0)

    old_normalized = np.clip(
        (old_gray - float(old_gray.mean())) * (42.0 / old_std) + 127.0,
        0,
        255,
    )
    new_normalized = np.clip(
        (new_gray - float(new_gray.mean())) * (42.0 / new_std) + 127.0,
        0,
        255,
    )
    best_score = 1.0
    max_shift = 3
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            old_y = slice(max(0, dy), min(200, 200 + dy))
            new_y = slice(max(0, -dy), min(200, 200 - dy))
            old_x = slice(max(0, dx), min(120, 120 + dx))
            new_x = slice(max(0, -dx), min(120, 120 - dx))
            difference = np.mean(
                np.abs(
                    old_normalized[old_y, old_x]
                    - new_normalized[new_y, new_x]
                )
            )
            best_score = min(best_score, float(difference / 255.0))
    return best_score


def ocr_frame_quality(image: np.ndarray) -> float:
    """Rank stable frames for OCR using detail, contrast, and clipping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (300, 500), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(reduced, cv2.CV_64F).var())
    contrast = float(reduced.std())
    clipped = float(np.mean((reduced <= 12) | (reduced >= 250)))
    median = cv2.medianBlur(reduced, 3)
    impulse_noise = float(
        np.mean(
            cv2.absdiff(reduced, median)
        )
    )
    detail_score = float(np.log1p(max(0.0, sharpness)))
    contrast_factor = min(1.0, max(0.22, contrast / 48.0))
    clipping_factor = max(0.18, 1.0 - clipped * 1.35)
    noise_factor = max(0.35, 1.0 - impulse_noise / 36.0)
    return detail_score * contrast_factor * clipping_factor * noise_factor


def screen_is_fully_visible(
    detection: ScreenDetection,
    frame_shape: tuple[int, ...],
    margin: int = 6,
) -> bool:
    height, width = frame_shape[:2]
    corners = order_corners(detection.corners)
    return bool(
        np.all(corners[:, 0] >= margin)
        and np.all(corners[:, 0] <= width - 1 - margin)
        and np.all(corners[:, 1] >= margin)
        and np.all(corners[:, 1] <= height - 1 - margin)
    )


def validate_screen_corners(
    corners: np.ndarray,
    frame_shape: tuple[int, ...],
    previous: ScreenDetection | None = None,
) -> bool:
    height, width = frame_shape[:2]
    points = order_corners(corners)
    contour = points.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        return False
    frame_area = float(height * width)
    area_ratio = abs(cv2.contourArea(points)) / max(1.0, frame_area)
    if not 0.10 <= area_ratio <= 0.96:
        return False
    sides = [
        np.linalg.norm(points[(index + 1) % 4] - points[index])
        for index in range(4)
    ]
    if min(sides) < max(45.0, np.hypot(width, height) * 0.035):
        return False
    margin_x = width * 0.08
    margin_y = height * 0.08
    if (
        np.any(points[:, 0] < -margin_x)
        or np.any(points[:, 0] > width - 1 + margin_x)
        or np.any(points[:, 1] < -margin_y)
        or np.any(points[:, 1] > height - 1 + margin_y)
    ):
        return False
    if previous is not None:
        previous_points = order_corners(previous.corners)
        previous_area = abs(cv2.contourArea(previous_points)) / max(
            1.0, frame_area
        )
        area_change = area_ratio / max(previous_area, 1e-6)
        if not 0.58 <= area_change <= 1.72:
            return False
        diagonal = float(np.hypot(width, height))
        movement = float(
            np.mean(np.linalg.norm(points - previous_points, axis=1))
            / diagonal
        )
        if movement > 0.09:
            return False
    return True


def recover_screen_detection(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_detection: ScreenDetection,
) -> ScreenDetection | None:
    """Recover a briefly missed screen using tracked interior features."""
    if (
        previous_frame is None
        or current_frame is None
        or previous_frame.shape[:2] != current_frame.shape[:2]
    ):
        return None
    previous_gray = (
        previous_frame
        if previous_frame.ndim == 2
        else cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    )
    current_gray = (
        current_frame
        if current_frame.ndim == 2
        else cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    )
    mask = np.zeros(previous_gray.shape, dtype=np.uint8)
    polygon = np.round(order_corners(previous_detection.corners)).astype(
        np.int32
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=mask,
        maxCorners=120,
        qualityLevel=0.012,
        minDistance=7,
        blockSize=7,
    )
    if previous_points is None or len(previous_points) < 16:
        return None

    optical_flow_settings = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **optical_flow_settings,
    )
    if current_points is None or forward_status is None:
        return None
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points,
        None,
        **optical_flow_settings,
    )
    if backward_points is None or backward_status is None:
        return None

    previous_flat = previous_points.reshape(-1, 2)
    current_flat = current_points.reshape(-1, 2)
    backward_flat = backward_points.reshape(-1, 2)
    forward_backward_error = np.linalg.norm(
        previous_flat - backward_flat, axis=1
    )
    good = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (forward_backward_error < 2.2)
        & (forward_error.reshape(-1) < 45.0)
    )
    if int(np.count_nonzero(good)) < 14:
        return None

    homography, inlier_mask = cv2.findHomography(
        previous_flat[good],
        current_flat[good],
        cv2.RANSAC,
        3.0,
    )
    if homography is None or inlier_mask is None:
        return None
    inlier_ratio = float(np.mean(inlier_mask.reshape(-1).astype(bool)))
    if inlier_ratio < 0.62:
        return None

    recovered = cv2.perspectiveTransform(
        order_corners(previous_detection.corners).reshape(-1, 1, 2),
        homography,
    ).reshape(4, 2)
    if not validate_screen_corners(
        recovered, current_frame.shape, previous_detection
    ):
        return None
    ordered = order_corners(recovered)
    center = ordered.mean(axis=0)
    area_ratio = abs(cv2.contourArea(ordered)) / float(
        current_frame.shape[0] * current_frame.shape[1]
    )
    return ScreenDetection(
        corners=ordered,
        center=(float(center[0]), float(center[1])),
        area_ratio=float(area_ratio),
        score=float(previous_detection.score * inlier_ratio * 0.92),
    )


class ScreenDetectionTracker:
    """Bridge a few missed contour detections during handheld movement."""

    def __init__(self, max_recoveries: int = 4) -> None:
        self.max_recoveries = max_recoveries
        self._previous_frame: np.ndarray | None = None
        self._detection: ScreenDetection | None = None
        self._recoveries = 0

    def reset(self) -> None:
        self._previous_frame = None
        self._detection = None
        self._recoveries = 0

    def update(
        self,
        frame: np.ndarray,
        detection: ScreenDetection | None,
    ) -> tuple[ScreenDetection | None, str, float]:
        if detection is not None:
            self._previous_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )
            self._detection = detection
            self._recoveries = 0
            return detection, "raw", 1.0
        if (
            self._previous_frame is None
            or self._detection is None
            or self._recoveries >= self.max_recoveries
        ):
            self._recoveries += 1
            if self._recoveries > self.max_recoveries:
                self._previous_frame = None
                self._detection = None
            return None, "missing", 0.0

        self._recoveries += 1
        recovered = recover_screen_detection(
            self._previous_frame,
            frame,
            self._detection,
        )
        if recovered is None:
            return None, "missing", 0.0
        self._previous_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        self._detection = recovered
        confidence = max(0.58, 0.90 - 0.08 * (self._recoveries - 1))
        return recovered, "tracked", confidence


def select_best_ocr_candidate(
    candidates,
    now: float,
    max_age: float = 2.0,
):
    """Choose a recent sharp frame while keeping its matching detection."""
    recent = [
        candidate
        for candidate in candidates
        if now - candidate.get("timestamp", now) <= max_age
        and candidate.get("quality", 0.0) >= MIN_OCR_FRAME_QUALITY
        and candidate.get("tracking_confidence", 1.0) >= 0.58
    ]
    if not recent:
        return None

    def score(candidate) -> float:
        age = max(0.0, now - candidate.get("timestamp", now))
        freshness = max(0.82, 1.0 - 0.10 * age / max(max_age, 0.01))
        source_weight = 1.0 if candidate.get("source") == "raw" else 0.86
        confidence = candidate.get("tracking_confidence", 1.0)
        detection_score = min(
            1.0,
            max(0.55, candidate["detection"].score / 0.55),
        )
        return (
            candidate["quality"]
            * freshness
            * source_weight
            * confidence
            * detection_score
        )

    return max(recent, key=score)


class StabilityTracker:
    def __init__(self, history: int = 8, movement_threshold: float = 0.006) -> None:
        self.history = history
        self.movement_threshold = movement_threshold
        self._corners: deque[np.ndarray] = deque(maxlen=history)
        self._stable = False
        self._unstable_windows = 0

    def reset(self) -> None:
        self._corners.clear()
        self._stable = False
        self._unstable_windows = 0

    def update(self, corners: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[bool, float]:
        self._corners.append(np.asarray(corners, dtype=np.float32))
        if len(self._corners) < self.history:
            return False, 1.0
        diagonal = float(np.hypot(frame_shape[1], frame_shape[0]))
        movements = [
            np.mean(np.linalg.norm(b - a, axis=1)) / diagonal
            for a, b in zip(self._corners, list(self._corners)[1:])
        ]
        median_movement = float(np.median(movements))
        high_movement = float(np.percentile(movements, 90))
        can_enter = (
            median_movement <= self.movement_threshold
            and high_movement <= self.movement_threshold * 2.0
        )
        should_exit = (
            median_movement > self.movement_threshold * 1.45
            or high_movement > self.movement_threshold * 2.8
        )
        if self._stable:
            self._unstable_windows = (
                self._unstable_windows + 1 if should_exit else 0
            )
            if self._unstable_windows >= 3:
                self._stable = False
                self._unstable_windows = 0
        elif can_enter:
            self._stable = True
            self._unstable_windows = 0
        return self._stable, median_movement


def draw_detection(frame: np.ndarray, detection: ScreenDetection, message: str) -> np.ndarray:
    annotated = frame.copy()
    corners = detection.corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [corners], True, (46, 220, 160), 5, cv2.LINE_AA)
    cx, cy = map(int, detection.center)
    cv2.drawMarker(annotated, (cx, cy), (0, 220, 255), cv2.MARKER_CROSS, 34, 4)
    frame_center = (annotated.shape[1] // 2, annotated.shape[0] // 2)
    cv2.drawMarker(annotated, frame_center, (255, 150, 40), cv2.MARKER_CROSS, 34, 3)
    cv2.putText(
        annotated,
        f"screen {detection.area_ratio:.0%}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (46, 220, 160),
        2,
        cv2.LINE_AA,
    )
    return annotated
