import unittest
from unittest.mock import patch

import numpy as np

from kiosk_accessibility import camera


class FakeCapture:
    def __init__(self, index, backend, readable, frame=None):
        self.index = index
        self.backend = backend
        self.readable = readable
        self.frame = (
            frame
            if frame is not None
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        if self.readable:
            return True, self.frame.copy()
        return False, None

    def release(self):
        self.released = True


class CameraTests(unittest.TestCase):
    def test_falls_back_to_next_index_after_failed_frame_reads(self):
        created = []

        def factory(index, backend):
            capture = FakeCapture(
                index,
                backend,
                readable=backend == camera.cv2.CAP_DSHOW and index == 1,
            )
            created.append(capture)
            return capture

        with patch.object(camera.cv2, "VideoCapture", side_effect=factory):
            result = camera.open_camera(
                preferred_index=0,
                maximum_indices=2,
                warmup_reads=1,
                known_devices=[],
            )

        self.assertIsNotNone(result.session)
        self.assertEqual(result.session.index, 1)
        self.assertEqual(result.session.backend, "DirectShow")
        self.assertFalse(result.session.capture.released)
        self.assertTrue(
            all(
                capture.released
                for capture in created
                if capture is not result.session.capture
            )
        )
        result.session.capture.release()

    def test_automatic_mode_prefers_colour_camera_over_ir_camera(self):
        grayscale = np.full((480, 640, 3), 45, dtype=np.uint8)
        colour = np.empty((480, 640, 3), dtype=np.uint8)
        colour[:, :, 0] = 25
        colour[:, :, 1] = 120
        colour[:, :, 2] = 220

        def factory(index, backend):
            frame = grayscale if index == 0 else colour
            return FakeCapture(index, backend, readable=index in (0, 1), frame=frame)

        with patch.object(camera.cv2, "VideoCapture", side_effect=factory):
            result = camera.open_camera(
                maximum_indices=2,
                warmup_reads=1,
                known_devices=[],
            )

        self.assertIsNotNone(result.session)
        self.assertEqual(result.session.index, 1)
        self.assertEqual(result.session.camera_kind, "일반 RGB 추정")
        result.session.capture.release()

    def test_reports_all_attempts_when_no_camera_yields_a_frame(self):
        with patch.object(
            camera.cv2,
            "VideoCapture",
            side_effect=lambda index, backend: FakeCapture(index, backend, False),
        ):
            result = camera.open_camera(
                maximum_indices=2,
                warmup_reads=1,
                known_devices=[],
            )

        self.assertIsNone(result.session)
        self.assertEqual(len(result.attempts), 4)
        self.assertIn("카메라 0 (DirectShow)", result.error_message)
        self.assertIn("카메라 1 (자동)", result.error_message)

    def test_automatic_mode_prioritizes_named_camo_even_when_frame_is_gray(self):
        grayscale = np.full((720, 1280, 3), 80, dtype=np.uint8)
        colour = np.empty((480, 640, 3), dtype=np.uint8)
        colour[:, :, 0] = 25
        colour[:, :, 1] = 120
        colour[:, :, 2] = 220
        devices = [
            camera.CameraDevice(index=0, name="LGE Camera"),
            camera.CameraDevice(index=1, name="Camo"),
        ]

        def factory(index, backend):
            frame = grayscale if index == 1 else colour
            return FakeCapture(
                index,
                backend,
                readable=backend == camera.cv2.CAP_DSHOW,
                frame=frame,
            )

        with patch.object(camera.cv2, "VideoCapture", side_effect=factory):
            result = camera.open_camera(
                maximum_indices=2,
                warmup_reads=1,
                known_devices=devices,
            )

        self.assertIsNotNone(result.session)
        self.assertEqual(result.session.index, 1)
        self.assertTrue(result.session.is_camo)
        self.assertEqual(result.session.camera_kind, "아이폰 영상")
        self.assertIn("Camo/iPhone", result.session.description)
        result.session.capture.release()

    def test_parses_camera_names_from_pnputil_output(self):
        output = """
Microsoft PnP Utility

Instance ID: ROOT\\CAMERA\\0000
Device Description: Camo
Class Name: Camera

Instance ID: USB\\CAMERA\\0001
Device Description: LGE Camera
Class Name: Camera
"""
        self.assertEqual(
            camera._parse_pnputil_camera_names(output),
            ["Camo", "LGE Camera"],
        )

    def test_camo_keeps_driver_negotiated_resolution_and_buffering(self):
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        created = []

        def factory(index, backend):
            capture = FakeCapture(index, backend, readable=True, frame=frame)
            created.append(capture)
            return capture

        devices = [camera.CameraDevice(index=0, name="Camo")]
        with patch.object(camera.cv2, "VideoCapture", side_effect=factory):
            result = camera.open_camera(
                maximum_indices=1,
                warmup_reads=1,
                known_devices=devices,
            )

        self.assertIsNotNone(result.session)
        selected = result.session.capture
        configured_properties = {prop for prop, _ in selected.set_calls}
        self.assertNotIn(camera.cv2.CAP_PROP_FRAME_WIDTH, configured_properties)
        self.assertNotIn(camera.cv2.CAP_PROP_FRAME_HEIGHT, configured_properties)
        self.assertNotIn(camera.cv2.CAP_PROP_BUFFERSIZE, configured_properties)
        selected.release()

    def test_blank_frame_detection_preserves_dark_but_real_image(self):
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        dark_scene = black.copy()
        dark_scene[30:90, 50:110] = 20

        self.assertTrue(camera.frame_is_blank(black))
        self.assertFalse(camera.frame_is_blank(dark_scene))


if __name__ == "__main__":
    unittest.main()
