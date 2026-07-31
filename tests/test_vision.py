import unittest

import cv2
import numpy as np

from kiosk_accessibility.demo import create_mock_kiosk
from kiosk_accessibility.vision import (
    MIN_OCR_FRAME_QUALITY,
    ScreenDetectionTracker,
    detect_screen,
    full_frame_detection,
    ocr_frame_quality,
    order_corners,
    recover_screen_detection,
    screen_is_fully_visible,
    screen_change_score,
    select_best_ocr_candidate,
    warp_screen,
)


class VisionTests(unittest.TestCase):
    def test_corner_order(self):
        points = np.array([[90, 100], [10, 10], [100, 12], [12, 90]], dtype=np.float32)
        ordered = order_corners(points)
        np.testing.assert_array_equal(ordered[0], [10, 10])
        np.testing.assert_array_equal(ordered[2], [90, 100])

    def test_detects_and_warps_mock_screen(self):
        frame, _ = create_mock_kiosk()
        detection = detect_screen(frame)
        self.assertIsNotNone(detection)
        self.assertGreater(detection.area_ratio, 0.4)
        warped = warp_screen(frame, detection.corners)
        self.assertEqual(warped.shape[:2], (1000, 600))

    def test_change_score(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertLess(screen_change_score(black, black), 0.001)
        self.assertGreater(screen_change_score(black, white), 0.99)

    def test_full_frame_detection_uses_nearly_entire_image(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = full_frame_detection(frame)
        self.assertAlmostEqual(detection.center[0], 320)
        self.assertAlmostEqual(detection.center[1], 240)
        self.assertGreater(detection.area_ratio, 0.9)

    def test_ocr_frame_quality_prefers_sharp_text_edges(self):
        sharp = np.zeros((500, 300, 3), dtype=np.uint8)
        for x in range(20, 280, 20):
            sharp[:, x : x + 4] = 255
        blurred = cv2.GaussianBlur(sharp, (21, 21), 5)
        self.assertGreater(ocr_frame_quality(sharp), ocr_frame_quality(blurred))

    def test_motion_blurred_screen_is_not_an_ocr_candidate(self):
        frame, _ = create_mock_kiosk()
        length = 7
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        blurred = cv2.filter2D(frame, -1, kernel)
        detection = detect_screen(blurred)
        self.assertIsNotNone(detection)
        quality = ocr_frame_quality(
            warp_screen(blurred, detection.corners)
        )
        self.assertLess(quality, MIN_OCR_FRAME_QUALITY)

    def test_clipped_screen_is_not_fully_visible(self):
        frame, _ = create_mock_kiosk()
        matrix = np.float32([[1, 0, 0], [0, 1, -180]])
        clipped = cv2.warpAffine(
            frame,
            matrix,
            (frame.shape[1], frame.shape[0]),
        )
        detection = detect_screen(clipped)
        self.assertIsNotNone(detection)
        self.assertFalse(
            screen_is_fully_visible(detection, clipped.shape)
        )

    def test_recovers_screen_after_small_handheld_motion(self):
        frame, _ = create_mock_kiosk()
        detection = detect_screen(frame)
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            1.2,
            1.0,
        )
        matrix[:, 2] += (10, 7)
        moved = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            borderValue=(32, 38, 44),
        )
        expected = cv2.transform(
            detection.corners.reshape(-1, 1, 2),
            matrix,
        ).reshape(4, 2)

        recovered = recover_screen_detection(
            frame,
            moved,
            detection,
        )

        self.assertIsNotNone(recovered)
        mean_error = float(
            np.mean(
                np.linalg.norm(recovered.corners - expected, axis=1)
            )
        )
        self.assertLess(mean_error, 4.0)

    def test_detection_tracker_expires_after_repeated_misses(self):
        frame, _ = create_mock_kiosk()
        detection = detect_screen(frame)
        tracker = ScreenDetectionTracker(max_recoveries=2)
        tracked, source, _ = tracker.update(frame, detection)
        self.assertIsNotNone(tracked)
        self.assertEqual(source, "raw")

        blank = np.zeros_like(frame)
        for _ in range(3):
            tracked, source, _ = tracker.update(blank, None)
        self.assertIsNone(tracked)
        self.assertEqual(source, "missing")

        moved = np.roll(frame, 8, axis=1)
        tracked, source, _ = tracker.update(moved, None)
        self.assertIsNone(tracked)
        self.assertEqual(source, "missing")

    def test_candidate_selector_ignores_old_and_low_confidence_frames(self):
        frame, _ = create_mock_kiosk()
        detection = detect_screen(frame)
        now = 20.0
        old = {
            "frame": frame,
            "warp": frame,
            "detection": detection,
            "quality": 20.0,
            "timestamp": 10.0,
            "source": "raw",
            "tracking_confidence": 1.0,
        }
        recent_raw = {
            "frame": frame.copy(),
            "warp": frame,
            "detection": detection,
            "quality": 6.0,
            "timestamp": 19.8,
            "source": "raw",
            "tracking_confidence": 1.0,
        }
        uncertain_tracked = {
            "frame": frame.copy(),
            "warp": frame,
            "detection": detection,
            "quality": 9.0,
            "timestamp": 19.9,
            "source": "tracked",
            "tracking_confidence": 0.5,
        }

        selected = select_best_ocr_candidate(
            [old, recent_raw, uncertain_tracked],
            now,
            max_age=2.0,
        )

        self.assertIs(selected, recent_raw)

    def test_change_score_tolerates_small_residual_warp_shift(self):
        frame, _ = create_mock_kiosk()
        detection = detect_screen(frame)
        warped = warp_screen(frame, detection.corners)
        shifted = np.roll(warped, 2, axis=1)
        self.assertLess(screen_change_score(warped, shifted), 0.01)


if __name__ == "__main__":
    unittest.main()
