import unittest
from pathlib import Path

import numpy as np

from kiosk_accessibility.models import OCRItem
from kiosk_accessibility.tracking import (
    ApproachRateTracker,
    frame_point_to_screen,
    track_camera_center,
)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.corners = np.array(
            [[0, 0], [599, 0], [599, 999], [0, 999]], dtype=np.float32
        )

    def test_camera_center_maps_to_normalized_screen_center(self):
        target = OCRItem("가운데", 270, 470, 60, 60, 99)
        tracking = track_camera_center(
            (1000, 600, 3), self.corners, target, (600, 1000)
        )
        self.assertAlmostEqual(tracking.pointer_screen[0], 300, delta=1)
        self.assertAlmostEqual(tracking.pointer_screen[1], 500, delta=1)
        self.assertTrue(tracking.inside)
        self.assertEqual(tracking.interval_ms, 80)
        self.assertEqual(tracking.direction, "목표 일치")

    def test_tracking_gives_physical_move_direction(self):
        target = OCRItem("오른쪽 아래", 450, 760, 80, 60, 99)
        tracking = track_camera_center(
            (1000, 600, 3), self.corners, target, (600, 1000)
        )
        self.assertFalse(tracking.inside)
        self.assertEqual(tracking.direction, "오른쪽·아래로 이동")

    def test_clicked_source_point_maps_to_corrected_coordinates(self):
        mapped = frame_point_to_screen((150, 250), self.corners, (600, 1000))
        self.assertAlmostEqual(mapped[0], 150, delta=1)
        self.assertAlmostEqual(mapped[1], 250, delta=1)

    def test_approach_rate_is_positive_when_distance_decreases(self):
        tracker = ApproachRateTracker(smoothing=1.0)
        tracker.update(0.5, timestamp=1.0)
        rate = tracker.update(0.3, timestamp=2.0)
        self.assertAlmostEqual(rate, 0.2)

    def test_source_contains_no_arrow_overlay(self):
        source_root = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "kiosk_accessibility"
        )
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in source_root.glob("*.py")
        )
        self.assertNotIn("arrowedLine", source)


if __name__ == "__main__":
    unittest.main()
