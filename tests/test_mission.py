from __future__ import annotations

import unittest

from kiosk_accessibility.mission import (
    MissionPhase,
    MissionSession,
    clamp_point,
    distance_to_rect,
    path_efficiency,
    path_length,
    point_in_rect,
    same_target_identity,
)


class MissionMathTests(unittest.TestCase):
    def test_path_length_uses_each_segment(self) -> None:
        self.assertEqual(path_length([]), 0.0)
        self.assertEqual(path_length([(0, 0)]), 0.0)
        self.assertAlmostEqual(
            path_length([(0, 0), (3, 4), (6, 8)]),
            10.0,
        )

    def test_distance_and_hit_test_use_target_rectangle(self) -> None:
        rect = (3, 4, 10, 10)
        self.assertAlmostEqual(distance_to_rect((0, 0), rect), 5.0)
        self.assertEqual(distance_to_rect((5, 6), rect), 0.0)
        self.assertTrue(point_in_rect((3, 4), rect))
        self.assertFalse(point_in_rect((2.9, 4), rect))

    def test_efficiency_rewards_direct_route(self) -> None:
        rect = (10, 0, 2, 2)
        direct = [(0, 1), (10, 1)]
        detour = [(0, 1), (0, 11), (10, 1)]
        self.assertAlmostEqual(path_efficiency(direct, rect), 1.0)
        self.assertLess(path_efficiency(detour, rect), 0.5)

    def test_points_are_clamped_to_corrected_screen(self) -> None:
        self.assertEqual(clamp_point((-20, 1100), (600, 1000)), (0.0, 999.0))

    def test_target_identity_allows_ocr_jitter_but_not_duplicate_menu(self) -> None:
        original = (100, 100, 40, 30)
        self.assertTrue(
            same_target_identity(
                "불고기 버거",
                original,
                "불고기버거",
                (112, 106, 42, 29),
            )
        )
        self.assertFalse(
            same_target_identity(
                "불고기버거",
                original,
                "불고기버거",
                (150, 100, 40, 30),
            )
        )
        self.assertFalse(
            same_target_identity(
                "불고기버거",
                original,
                "불고기버거",
                (300, 500, 40, 30),
            )
        )
        self.assertFalse(
            same_target_identity(
                "불고기버거",
                original,
                "치즈버거",
                original,
            )
        )


class MissionSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = MissionSession(
            "불고기버거",
            (100, 100, 40, 30),
            (600, 1000),
            started_at=10.0,
            min_step_px=4.0,
            min_sample_interval=0.0,
        )

    def test_wrong_click_then_target_click_completes_once(self) -> None:
        first = self.session.register_click((10, 10), 11.0)
        second = self.session.register_click((110, 110), 12.5)
        frozen = self.session.register_click((20, 20), 20.0)

        self.assertTrue(first.wrong_click)
        self.assertTrue(second.completed)
        self.assertFalse(frozen.completed)
        self.assertEqual(self.session.phase, MissionPhase.SUCCEEDED)
        result = self.session.snapshot(100.0)
        self.assertEqual(result.wrong_clicks, 1)
        self.assertEqual(result.elapsed_seconds, 2.5)

    def test_camera_entry_completes_mission(self) -> None:
        self.session.observe((20, 20), 10.5, "camera")
        update = self.session.observe((101, 101), 11.0, "camera")
        self.assertTrue(update.completed)
        self.assertEqual(
            self.session.snapshot(12.0).completion_source,
            "camera",
        )

    def test_small_jitter_is_ignored_but_target_entry_is_kept(self) -> None:
        self.session.observe((96, 110), 10.1, "cursor", complete_on_inside=False)
        jitter = self.session.observe(
            (98, 110), 10.2, "cursor", complete_on_inside=False
        )
        target = self.session.observe((100, 110), 10.3, "cursor")
        self.assertFalse(jitter.recorded)
        self.assertTrue(target.recorded)
        self.assertTrue(target.completed)
        points = self.session.snapshot(10.3).route_segments[0]
        self.assertEqual(points[-1], (100.0, 110.0))

    def test_route_break_does_not_add_missing_interval_distance(self) -> None:
        self.session.observe((0, 0), 10.0, "camera", complete_on_inside=False)
        self.session.observe((10, 0), 10.1, "camera", complete_on_inside=False)
        self.session.break_route("camera")
        self.session.observe((500, 900), 11.0, "camera", complete_on_inside=False)
        result = self.session.snapshot(12.0)
        self.assertAlmostEqual(result.path_length_px, 10.0)
        self.assertEqual(len(result.route_segments), 2)

    def test_camera_and_cursor_routes_do_not_create_a_jump(self) -> None:
        self.session.observe((0, 0), 10.0, "camera", complete_on_inside=False)
        self.session.observe((10, 0), 10.1, "camera", complete_on_inside=False)
        self.session.observe((500, 900), 10.2, "cursor", complete_on_inside=False)
        self.session.observe((510, 900), 10.3, "cursor", complete_on_inside=False)
        routes = self.session.route_segments()
        self.assertEqual(path_length(list(routes["camera"][0])), 10.0)
        self.assertEqual(path_length(list(routes["cursor"][0])), 10.0)

    def test_target_can_move_during_ocr_refresh(self) -> None:
        self.session.observe((20, 20), 10.2, "camera")
        self.session.update_target((200, 200, 40, 30))
        update = self.session.observe((210, 210), 11.0, "camera")
        self.assertTrue(update.completed)
        self.assertAlmostEqual(self.session.snapshot(20.0).elapsed_seconds, 1.0)

    def test_completed_result_is_not_changed_by_later_ocr_refresh(self) -> None:
        self.session.observe((20, 20), 10.2, "camera")
        self.session.observe((110, 110), 11.0, "camera")
        before = self.session.snapshot(11.0)
        self.session.update_target((400, 700, 40, 30))
        after = self.session.snapshot(30.0)
        self.assertEqual(after.elapsed_seconds, before.elapsed_seconds)
        self.assertEqual(after.direct_distance_px, before.direct_distance_px)
        self.assertEqual(after.efficiency, before.efficiency)


if __name__ == "__main__":
    unittest.main()
