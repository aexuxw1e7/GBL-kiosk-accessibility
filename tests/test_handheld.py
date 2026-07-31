import unittest
from collections import deque
from types import MethodType

import cv2
import numpy as np

from kiosk_accessibility.demo import create_mock_kiosk
from kiosk_accessibility.gui import PrototypeApp
from kiosk_accessibility.vision import (
    StabilityTracker,
    detect_screen,
    warp_screen,
)


class _CandidateState:
    pass


class HandheldCameraTests(unittest.TestCase):
    @staticmethod
    def _run_sequence(
        translation_sigma: float,
        rotation_sigma: float,
        seed: int,
    ) -> tuple[list[bool], _CandidateState]:
        base, _ = create_mock_kiosk()
        height, width = base.shape[:2]
        random = np.random.default_rng(seed)
        state = _CandidateState()
        state.frame_candidates = deque(maxlen=12)
        state.last_candidate_at = 0.0
        state.candidate_change_votes = 0
        state.screen_epoch = 0
        state._clear_frame_candidates = MethodType(
            PrototypeApp._clear_frame_candidates,
            state,
        )
        stability = StabilityTracker(
            history=6,
            movement_threshold=0.018,
        )
        ready = []

        for index in range(16):
            matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                random.normal(0, rotation_sigma),
                1.0,
            )
            matrix[:, 2] += (
                random.normal(0, translation_sigma),
                random.normal(0, translation_sigma),
            )
            frame = cv2.warpAffine(
                base,
                matrix,
                (width, height),
                borderValue=(32, 38, 44),
            )
            detection = detect_screen(frame)
            if detection is None:
                continue
            stable, movement = stability.update(
                detection.corners,
                frame.shape,
            )
            warped = warp_screen(frame, detection.corners)
            now = index * 0.14
            PrototypeApp._record_frame_candidate(
                state,
                frame,
                detection,
                warped,
                now,
                movement,
                stable,
                "raw",
                1.0,
                False,
            )
            ready.append(
                PrototypeApp._camera_candidates_ready(
                    state,
                    now,
                    stable,
                    movement,
                )
            )
        return ready, state

    def test_moderate_handheld_shake_still_becomes_ocr_ready(self):
        ready, state = self._run_sequence(
            translation_sigma=15,
            rotation_sigma=1.5,
            seed=3,
        )
        self.assertTrue(any(ready))
        self.assertGreaterEqual(len(state.frame_candidates), 5)

    def test_strong_shake_waits_for_a_clearer_moment(self):
        ready, state = self._run_sequence(
            translation_sigma=25,
            rotation_sigma=2.5,
            seed=4,
        )
        self.assertFalse(any(ready))
        self.assertLess(len(state.frame_candidates), 5)


if __name__ == "__main__":
    unittest.main()
