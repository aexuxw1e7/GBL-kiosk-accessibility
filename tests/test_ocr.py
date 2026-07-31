import unittest
from unittest.mock import patch

import numpy as np

from kiosk_accessibility.models import OCRItem
from kiosk_accessibility.ocr import OCRPass, TesseractOCR


class OCRTests(unittest.TestCase):
    def test_scaled_pass_restores_original_coordinates(self):
        engine = object.__new__(TesseractOCR)
        engine.minimum_confidence = 32.0
        engine.language = "kor+eng"
        engine.tessdata = None
        data = {
            "text": ["", "불고기버거"],
            "conf": ["-1", "90"],
            "block_num": [0, 1],
            "par_num": [0, 1],
            "line_num": [0, 1],
            "left": [0, 200],
            "top": [0, 400],
            "width": [0, 100],
            "height": [0, 40],
        }
        image = np.zeros((2000, 1200, 3), dtype=np.uint8)
        spec = OCRPass(image, 2.0, 2.0, 11, "test")
        with patch(
            "kiosk_accessibility.ocr.pytesseract.image_to_data",
            return_value=data,
        ):
            items, _ = engine._recognize_pass(spec, (1000, 600, 3))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].bbox, (100, 200, 50, 20))

    def test_deduplicate_prefers_complete_text_at_same_position(self):
        candidates = [
            OCRItem("즈버거", 100, 200, 70, 30, 96),
            OCRItem("치즈버거", 95, 198, 85, 34, 88),
        ]
        merged = TesseractOCR._deduplicate(candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "치즈버거")

    def test_same_text_at_different_positions_is_not_removed(self):
        candidates = [
            OCRItem("콜라", 50, 200, 50, 25, 90),
            OCRItem("콜라", 400, 700, 50, 25, 92),
        ]
        merged = TesseractOCR._deduplicate(candidates)
        self.assertEqual(len(merged), 2)

    def test_single_hangul_syllables_join_when_gaps_are_small(self):
        data = {
            "text": ["불", "고", "기"],
            "left": [0, 24, 48],
            "width": [20, 20, 20],
            "height": [30, 30, 30],
        }
        spec = OCRPass(np.zeros((100, 100, 3), dtype=np.uint8), 1, 1, 11, "test")
        self.assertEqual(
            TesseractOCR._join_words([0, 1, 2], data, spec),
            "불고기",
        )


if __name__ == "__main__":
    unittest.main()
