import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.camera.manager import CameraManager


class _FakeCapture:
    def __init__(self):
        self.counter = 0
        self.released = False
        self.props = {}

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 30)

    def read(self):
        time.sleep(0.002)
        self.counter += 1
        return True, np.full((8, 12, 3), self.counter % 255, dtype=np.uint8)

    def release(self):
        self.released = True


class CameraManagerTests(unittest.TestCase):
    def test_capture_thread_keeps_latest_frame_for_multiple_consumers(self):
        capture = _FakeCapture()
        with patch.object(CameraManager, "_open_device", return_value=(capture, "fake")):
            camera = CameraManager(index=1, resolution=(12, 8), fps=30)
            self.assertTrue(camera.start())
            deadline = time.monotonic() + 0.5
            first = None
            while first is None and time.monotonic() < deadline:
                first = camera.read()
                time.sleep(0.005)
            self.assertIsNotNone(first)
            time.sleep(0.02)
            second = camera.read()
            self.assertIsNotNone(second)
            self.assertGreater(int(second[0, 0, 0]), int(first[0, 0, 0]))
            second[0, 0, 0] = 0
            self.assertNotEqual(int(camera.read()[0, 0, 0]), 0)
            camera.stop()
        self.assertTrue(capture.released)

    def test_clarity_assist_switches_profiles_without_reopening_camera(self):
        capture = _FakeCapture()
        settings = {
            "enabled": True,
            "preview_exposure": -6,
            "preview_gain": 0,
            "motion_exposure": -7,
            "motion_gain": 80,
            "check_frames": 1,
            # 本测试只验证配置切换；禁用随假画面触发的自适应增益。
            "trigger_checks": 100,
        }
        with patch.object(CameraManager, "_open_device", return_value=(capture, "fake")):
            camera = CameraManager(
                index=1, resolution=(12, 8), fps=30,
                clarity_config=settings,
            )
            self.assertTrue(camera.start())
            self.assertEqual(capture.props.get(cv2.CAP_PROP_EXPOSURE), -6)
            camera.set_clarity_assist(True)
            deadline = time.monotonic() + 0.5
            while not camera.clarity_status()["enabled"] and time.monotonic() < deadline:
                time.sleep(0.005)
            status = camera.clarity_status()
            self.assertTrue(status["enabled"])
            self.assertTrue(status["software_enabled"])
            self.assertEqual(status["exposure"], -7)
            self.assertEqual(status["gain"], 80)
            camera.set_clarity_assist(False)
            deadline = time.monotonic() + 0.5
            while camera.clarity_status()["enabled"] and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertFalse(camera.clarity_status()["enabled"])
            camera.stop()


if __name__ == "__main__":
    unittest.main()
