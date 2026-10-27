import unittest
from unittest.mock import patch

import cv2

from app.engine.camera.modes import CameraMode, preferred_mode
from app.engine.camera.manager import CameraManager
from tests.test_camera import _FakeCapture


class CameraModeTests(unittest.TestCase):
    def test_max_resolution_and_rate_without_sacrificing_preview_rate(self):
        modes = [CameraMode(3840, 2160, 15, "MJPG"),
                 CameraMode(2592, 1944, 30, "MJPG"),
                 CameraMode(1920, 1080, 60, "MJPG"),
                 CameraMode(2592, 1944, 2, "YUY2")]
        self.assertEqual(preferred_mode(modes, 30), modes[1])
        self.assertEqual(preferred_mode(modes, 60), modes[2])
        self.assertEqual(preferred_mode(modes, 120), modes[2])
        self.assertIsNone(preferred_mode([], 30))

    def test_preview_budget_selects_supported_mode_without_live_benchmark(self):
        modes = [CameraMode(2592, 1944, 30, "MJPG"),
                 CameraMode(1920, 1080, 30, "MJPG"),
                 CameraMode(1280, 960, 30, "MJPG")]
        self.assertEqual(preferred_mode(modes, 60, max_pixels=1920*1080), modes[1])
        self.assertEqual(preferred_mode(modes[:1], 30, max_pixels=1920*1080), modes[0])

    def test_public_capture_applies_supported_mode_and_reads_back(self):
        class RenegotiatingCapture(_FakeCapture):
            def set(self, prop, value):
                result = super().set(prop, value)
                if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS):
                    self.props[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
                return result

        capture = RenegotiatingCapture()
        mode = CameraMode(2592, 1944, 30, "MJPG")
        with patch("app.engine.camera.modes.supported_modes", return_value=[mode]) as enumerate_modes, patch.object(
                CameraManager, "_open_device", return_value=(capture, "DirectShow")):
            camera = CameraManager(index=0, auto_mode=True, fps=30)
            try:
                self.assertTrue(camera.start())
                self.assertEqual(camera.actual_resolution, (2592, 1944))
                self.assertEqual(camera.actual_fps, 30)
                self.assertIsNotNone(camera.read())
                enumerate_modes.assert_called_once_with(0)
                self.assertEqual(capture.get(cv2.CAP_PROP_FOURCC), cv2.VideoWriter_fourcc(*"MJPG"))
            finally:
                camera.stop()

    def test_auto_mode_never_reopens_or_changes_running_stream(self):
        class StreamingCapture(_FakeCapture):
            streaming = False

            def set(self, prop, value):
                if self.streaming:
                    raise RuntimeError("cannot change a running stream")
                return super().set(prop, value)

            def read(self):
                self.streaming = True
                return super().read()

        capture = StreamingCapture()
        modes = [CameraMode(2592, 1944, 30, "MJPG"), CameraMode(1920, 1080, 30, "MJPG")]
        with patch("app.engine.camera.modes.supported_modes", return_value=modes) as enumerate_modes, patch.object(
                CameraManager, "_open_device", return_value=(capture, "DirectShow")) as opener:
            camera = CameraManager(index=2, auto_mode=True, fps=60)
            try:
                self.assertTrue(camera.start())
                enumerate_modes.assert_called_once_with(2)
                opener.assert_called_once_with(2)
                self.assertEqual(camera.actual_resolution, (1920, 1080))
                self.assertIsNotNone(camera.read())
                self.assertFalse(capture.released)
            finally:
                camera.stop()

    def test_connected_device_without_frames_is_not_reported_open(self):
        class EmptyCapture(_FakeCapture):
            def read(self):
                return False, None

        capture = EmptyCapture()
        with patch.object(CameraManager, "_open_device", return_value=(capture, "DirectShow")):
            camera = CameraManager(index=0)
            self.assertFalse(camera.start())
            self.assertFalse(camera.is_opened)
            self.assertIn("没有收到画面", camera.last_error)
            self.assertTrue(capture.released)
            self.assertIsNone(camera.registry.owner_of(0))

    def test_enumeration_failure_keeps_manual_settings(self):
        capture = _FakeCapture()
        with patch("app.engine.camera.modes.supported_modes", side_effect=RuntimeError("driver")), patch.object(
                CameraManager, "_open_device", return_value=(capture, "DirectShow")):
            camera = CameraManager(index=1, auto_mode=True, resolution=(640, 480), fps=25)
            try:
                self.assertTrue(camera.start())
                self.assertEqual(camera.actual_resolution, (640, 480))
                self.assertEqual(camera.actual_fps, 25)
                self.assertIn("手动", camera.mode_note)
            finally:
                camera.stop()
