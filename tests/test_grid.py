import unittest

from kiosk_accessibility.grid import (
    assign_grid,
    find_matching_item_index,
    find_query_match,
    find_same_text_item_index,
    grid_numeric_coordinate,
    screen_position_label,
    target_direction_label,
    vibration_interval_ms,
)
from kiosk_accessibility.models import OCRItem


class GridTests(unittest.TestCase):
    def test_assigns_expected_grid(self):
        items = [OCRItem("불고기버거", 90, 190, 20, 20, 99)]
        assign_grid(items, 600, 1000)
        self.assertEqual(items[0].grid, "B3")

    def test_duplicate_items_receive_relative_positions(self):
        items = [
            OCRItem("왼쪽 메뉴", 101, 201, 20, 20, 99),
            OCRItem("오른쪽 메뉴", 151, 201, 20, 20, 99),
        ]
        assign_grid(items, 600, 1000)
        self.assertEqual(items[0].grid, items[1].grid)
        self.assertEqual([item.relative for item in items], ["왼쪽", "오른쪽"])

    def test_haptic_is_strong_inside_target(self):
        target = OCRItem("목표", 100, 100, 80, 50, 99)
        interval, inside, _ = vibration_interval_ms((120, 120), target, (600, 1000))
        self.assertTrue(inside)
        self.assertEqual(interval, 80)

    def test_selection_is_preserved_by_text_after_position_shift(self):
        previous = OCRItem("새우 버거", 300, 200, 80, 30, 99, grid="D3")
        candidates = [
            OCRItem("불고기버거", 90, 200, 90, 30, 99, grid="B3"),
            OCRItem("새우버거", 312, 208, 80, 30, 95, grid="D3"),
        ]
        self.assertEqual(find_matching_item_index(previous, candidates), 1)

    def test_selection_falls_back_to_nearest_item(self):
        previous = OCRItem("이전 항목", 300, 500, 80, 30, 99, grid="D6")
        candidates = [
            OCRItem("위쪽", 300, 100, 80, 30, 99, grid="D2"),
            OCRItem("가까운 새 항목", 305, 510, 80, 30, 99, grid="D6"),
        ]
        self.assertEqual(find_matching_item_index(previous, candidates), 1)

    def test_query_matches_ocr_text_with_spacing_difference(self):
        candidates = [
            OCRItem("새우 버거", 300, 200, 80, 30, 99),
            OCRItem("불고기버거", 90, 200, 90, 30, 99),
        ]
        match = find_query_match("불고기 버거", candidates)
        self.assertIsNotNone(match)
        self.assertEqual(match[0], 1)
        self.assertGreater(match[1], 0.9)

    def test_same_text_does_not_silently_switch_to_another_menu(self):
        previous = OCRItem("치킨버거", 300, 500, 80, 30, 99)
        candidates = [OCRItem("새우버거", 305, 510, 80, 30, 99)]
        self.assertIsNone(find_same_text_item_index(previous, candidates))

    def test_screen_position_label_is_natural_korean(self):
        upper_left = OCRItem("목표", 20, 30, 40, 30, 99)
        center = OCRItem("목표", 280, 480, 40, 30, 99)
        self.assertEqual(
            screen_position_label(upper_left, (600, 1000)), "왼쪽 위"
        )
        self.assertEqual(screen_position_label(center, (600, 1000)), "가운데")

    def test_target_direction_is_relative_to_clicked_point(self):
        target = OCRItem("목표", 400, 700, 80, 50, 99)
        self.assertEqual(target_direction_label((100, 100), target), "오른쪽 아래")
        self.assertEqual(target_direction_label((420, 720), target), "현재 위치")

    def test_grid_label_converts_to_numeric_coordinate(self):
        item = OCRItem("목표", 100, 100, 80, 50, 99, grid="D4")
        self.assertEqual(grid_numeric_coordinate(item), (4, 4))


if __name__ == "__main__":
    unittest.main()
