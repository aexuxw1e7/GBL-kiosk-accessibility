from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
versioned_vendor = ROOT / f".vendor-py{sys.version_info.major}{sys.version_info.minor}"
for candidate in (versioned_vendor, ROOT / ".vendor"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Auto")]
    for backend, name in backends:
        for index in range(6):
            capture = cv2.VideoCapture(index, backend)
            opened = capture.isOpened()
            read_ok = False
            shape = None
            frame = None
            if opened:
                for _ in range(24):
                    read_ok, frame = capture.read()
                    if read_ok and frame is not None and frame.size:
                        shape = frame.shape
                    time.sleep(0.03)
            if frame is not None and frame.size:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturation = float(np.mean(hsv[:, :, 1]))
                brightness = float(np.mean(hsv[:, :, 2]))
                channel_spread = float(
                    np.mean(np.max(frame, axis=2) - np.min(frame, axis=2))
                )
                dark_ratio = float(np.mean(hsv[:, :, 2] < 25))
                metrics = (
                    f"sat={saturation:.1f} bright={brightness:.1f} "
                    f"spread={channel_spread:.1f} dark={dark_ratio:.1%}"
                )
            else:
                metrics = "no-frame"
            print(
                f"{name:10} index={index} opened={opened} "
                f"read={read_ok} shape={shape} {metrics}"
            )
            capture.release()


if __name__ == "__main__":
    main()
