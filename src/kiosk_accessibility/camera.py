from __future__ import annotations

import locale
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str

    @property
    def is_camo(self) -> bool:
        return "camo" in self.name.casefold()

    @property
    def display_name(self) -> str:
        return "Camo/iPhone" if self.is_camo else self.name


@dataclass
class CameraSession:
    capture: cv2.VideoCapture
    index: int
    backend: str
    width: int
    height: int
    quality_score: float
    camera_kind: str
    device_name: str = ""
    initial_frame: np.ndarray | None = None

    @property
    def is_camo(self) -> bool:
        return "camo" in self.device_name.casefold()

    @property
    def description(self) -> str:
        device_prefix = ""
        if self.device_name:
            device = "Camo/iPhone" if self.is_camo else self.device_name
            device_prefix = f"{device} · "
        return (
            f"{device_prefix}카메라 {self.index} · {self.backend} · "
            f"{self.camera_kind} · "
            f"{self.width}×{self.height}"
        )


@dataclass
class CameraAttempt:
    index: int
    backend: str
    opened: bool
    frame_read: bool
    quality_score: float = 0.0
    saturation: float = 0.0
    brightness: float = 0.0
    device_name: str = ""


@dataclass
class CameraOpenResult:
    session: CameraSession | None
    attempts: list[CameraAttempt]

    @property
    def error_message(self) -> str:
        attempted = ", ".join(
            (
                f"{attempt.device_name or '카메라'} {attempt.index} "
                f"({attempt.backend})"
            )
            for attempt in self.attempts
        )
        return (
            f"사용 가능한 카메라 프레임을 찾지 못했습니다. 확인한 조합: {attempted}. "
            "다른 앱이 카메라를 사용 중인지와 Windows 카메라 개인정보 설정을 확인하세요."
        )


def _parse_pnputil_camera_names(output: str) -> list[str]:
    names: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.casefold().replace(" ", "")
        if "devicedescription" not in normalized_key and "장치설명" not in normalized_key:
            continue
        name = value.strip()
        if name:
            names.append(name)
    return names


def enumerate_camera_devices(maximum_indices: int = 8) -> list[CameraDevice]:
    """Return Windows camera names in the order used by this PC.

    OpenCV's Windows backend normally exposes the same order as the connected
    camera class. If enumeration is unavailable, callers retain the existing
    numeric-index fallback.
    """

    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            [
                "pnputil",
                "/enum-devices",
                "/connected",
                "/class",
                "Camera",
            ],
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            timeout=6,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    names = _parse_pnputil_camera_names(result.stdout)
    return [
        CameraDevice(index=index, name=name)
        for index, name in enumerate(names[:maximum_indices])
    ]


def available_backends() -> list[tuple[int, str]]:
    if sys.platform == "win32":
        # opencv-python Windows wheels consistently include DirectShow, while
        # Media Foundation may be absent. CAP_ANY is retained as a fallback.
        return [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "자동")]
    return [(cv2.CAP_ANY, "자동")]


def camera_indices(preferred_index: int | None, maximum: int = 4) -> list[int]:
    indices = list(range(maximum))
    if preferred_index is None:
        return indices
    return [preferred_index] + [index for index in indices if index != preferred_index]


def _frame_quality(frame: np.ndarray) -> tuple[float, float, float, str]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    brightness = float(np.mean(hsv[:, :, 2]))
    channel_spread = float(np.mean(np.max(frame, axis=2) - np.min(frame, axis=2)))
    dark_ratio = float(np.mean(hsv[:, :, 2] < 25))
    exposure_score = max(0.0, 127.0 - abs(brightness - 127.0))
    score = (
        saturation
        + 4.0 * channel_spread
        + 0.2 * exposure_score
        - 120.0 * dark_ratio
    )
    camera_kind = (
        "일반 RGB 추정"
        if saturation >= 5.0 or channel_spread >= 5.0
        else "IR/흑백 추정"
    )
    return score, saturation, brightness, camera_kind


def frame_is_blank(frame: np.ndarray | None) -> bool:
    """Detect a virtual-camera placeholder without rejecting a dark scene."""

    if frame is None or not frame.size:
        return True
    height, width = frame.shape[:2]
    sample = frame[
        :: max(1, height // 90),
        :: max(1, width // 160),
    ]
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return (
        float(np.percentile(gray, 99.0)) <= 5.0
        and float(np.std(gray)) <= 2.0
    )


def _try_capture(
    index: int,
    backend_id: int,
    backend_name: str,
    warmup_reads: int,
    device_name: str = "",
) -> tuple[cv2.VideoCapture, np.ndarray | None, CameraAttempt]:
    capture = cv2.VideoCapture(index, backend_id)
    opened = capture.isOpened()
    valid_frame = None
    is_camo = "camo" in device_name.casefold()
    if opened:
        # Camo's DirectShow driver already negotiates 1280x720 correctly.
        # Setting width/height after opening switches this installation to a
        # YUY2 path that reports successful reads containing only zero pixels.
        # CAP_PROP_BUFFERSIZE is unsupported by the same virtual driver.
        if not is_camo:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        read_limit = max(warmup_reads, 30) if is_camo else warmup_reads
        deadline = time.monotonic() + 3.0 if is_camo else None
        usable_streak = 0
        for _ in range(read_limit):
            frame_read, frame = capture.read()
            if frame_read and frame is not None and frame.size:
                if not is_camo or not frame_is_blank(frame):
                    valid_frame = frame
                    usable_streak += 1
                    if is_camo and usable_streak >= 2:
                        break
                else:
                    usable_streak = 0
            else:
                time.sleep(0.03)
            if is_camo:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
    if valid_frame is None:
        attempt = CameraAttempt(
            index=index,
            backend=backend_name,
            opened=opened,
            frame_read=False,
            device_name=device_name,
        )
    else:
        score, saturation, brightness, _ = _frame_quality(valid_frame)
        attempt = CameraAttempt(
            index=index,
            backend=backend_name,
            opened=opened,
            frame_read=True,
            quality_score=score,
            saturation=saturation,
            brightness=brightness,
            device_name=device_name,
        )
    return capture, valid_frame, attempt


def _session_from_frame(
    capture: cv2.VideoCapture,
    frame: np.ndarray,
    attempt: CameraAttempt,
) -> CameraSession:
    height, width = frame.shape[:2]
    _, _, _, camera_kind = _frame_quality(frame)
    if "camo" in attempt.device_name.casefold():
        camera_kind = "아이폰 영상"
    return CameraSession(
        capture=capture,
        index=attempt.index,
        backend=attempt.backend,
        width=width,
        height=height,
        quality_score=attempt.quality_score,
        camera_kind=camera_kind,
        device_name=attempt.device_name,
        initial_frame=frame.copy(),
    )


def open_camera(
    preferred_index: int | None = None,
    maximum_indices: int = 8,
    warmup_reads: int = 3,
    known_devices: list[CameraDevice] | None = None,
) -> CameraOpenResult:
    attempts: list[CameraAttempt] = []
    backends = available_backends()
    devices = (
        enumerate_camera_devices(maximum_indices)
        if known_devices is None
        else list(known_devices)
    )
    device_by_index = {device.index: device for device in devices}
    scan_indices = (
        [device.index for device in devices]
        if devices
        else list(range(min(maximum_indices, 4)))
    )

    # A user's explicit camera choice wins, even if it is an IR camera.
    if preferred_index is not None:
        device_name = device_by_index.get(
            preferred_index, CameraDevice(preferred_index, "")
        ).name
        for backend_id, backend_name in backends:
            capture, frame, attempt = _try_capture(
                preferred_index,
                backend_id,
                backend_name,
                warmup_reads,
                device_name,
            )
            attempts.append(attempt)
            if frame is not None:
                return CameraOpenResult(
                    session=_session_from_frame(capture, frame, attempt),
                    attempts=attempts,
                )
            capture.release()

    # Camo is a virtual camera. Its first frames can be dark or nearly
    # monochrome while the iPhone stream warms up, so its device name takes
    # priority over the colour heuristic used for ordinary laptop cameras.
    camo_indices = [
        device.index
        for device in devices
        if device.is_camo and device.index != preferred_index
    ]
    for index in camo_indices:
        device_name = device_by_index[index].name
        for backend_id, backend_name in backends:
            capture, frame, attempt = _try_capture(
                index,
                backend_id,
                backend_name,
                max(warmup_reads, 6),
                device_name,
            )
            attempts.append(attempt)
            if frame is not None:
                return CameraOpenResult(
                    session=_session_from_frame(capture, frame, attempt),
                    attempts=attempts,
                )
            capture.release()

    # Automatic mode evaluates every working DirectShow camera and selects the
    # first useful colour result after the named Camo candidate.
    fallback_candidates: list[tuple[int, str, CameraAttempt]] = []
    for backend_id, backend_name in backends:
        for index in scan_indices:
            if index == preferred_index or index in camo_indices:
                continue
            device_name = device_by_index.get(
                index, CameraDevice(index, "")
            ).name
            capture, frame, attempt = _try_capture(
                index,
                backend_id,
                backend_name,
                warmup_reads,
                device_name,
            )
            attempts.append(attempt)
            if frame is not None:
                _, _, _, camera_kind = _frame_quality(frame)
                if camera_kind == "일반 RGB 추정":
                    return CameraOpenResult(
                        session=_session_from_frame(capture, frame, attempt),
                        attempts=attempts,
                    )
                fallback_candidates.append((backend_id, backend_name, attempt))
            capture.release()
    for backend_id, backend_name, selected in sorted(
        fallback_candidates,
        key=lambda candidate: candidate[2].quality_score,
        reverse=True,
    ):
        capture, frame, reopened = _try_capture(
            selected.index,
            backend_id,
            backend_name,
            warmup_reads,
            selected.device_name,
        )
        attempts.append(reopened)
        if frame is not None:
            return CameraOpenResult(
                session=_session_from_frame(capture, frame, reopened),
                attempts=attempts,
            )
        capture.release()
    return CameraOpenResult(session=None, attempts=attempts)
