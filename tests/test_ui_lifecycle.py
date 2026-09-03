"""主界面非硬件资源生命周期测试。"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.ui.app import YoloCamApp


class _StopStub:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def stop(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


class _TrackerStub:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls += 1


class _CameraPanelStub:
    def __init__(self):
        self.status = ""

    def set_clarity_status(self, text):
        self.status = text


class _LogStub:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


class CameraLifecycleTests(unittest.TestCase):
    def test_agent_export_uses_selected_path_and_current_conversation(self):
        app = YoloCamApp.__new__(YoloCamApp)
        app.root = object()
        app.agent_panel = Mock()
        entries = (object(), object())
        app.agent_panel.conversation_entries.return_value = entries
        app.log = _LogStub()
        selected = r"D:\实验记录\助手对话.md"

        with (
            patch(
                "tkinter.filedialog.asksaveasfilename",
                return_value=selected,
            ) as choose_path,
            patch(
                "src.ui.app.export_conversation",
                return_value=Path(selected),
            ) as export,
            patch("src.ui.app.messagebox.showinfo") as show_info,
        ):
            app._on_agent_export_chat()

        choose_path.assert_called_once()
        export.assert_called_once_with(selected, entries)
        show_info.assert_called_once()
        self.assertIn("对话已导出", app.log.lines[-1])

    def test_image_review_question_detection_is_specific(self):
        self.assertTrue(YoloCamApp._is_image_review_question(
            "请分析当前条纹图"))
        self.assertTrue(YoloCamApp._is_image_review_question(
            "帮我看图并指导调整"))
        self.assertFalse(YoloCamApp._is_image_review_question(
            "下一步应该做什么"))

    def test_close_camera_resets_all_camera_consumers_and_tracking(self):
        app = object.__new__(YoloCamApp)
        app.cam = _StopStub()
        camera = app.cam
        app.recorder = _StopStub()
        app.camera_running = True
        app.camera_plugin = _CameraPanelStub()
        app.log = _LogStub()
        app._center_tracker = _TrackerStub()
        app._center_line_x = 10.0
        app._center_line_box = (1, 2, 3, 4)
        app._zero_box_x = 20.0
        app._zero_box_confidence = 0.9
        app._center_yolo_misses = 3
        calls = []
        app._stop_preview = lambda: calls.append("preview")
        app._stop_predict = lambda: calls.append("predict")
        app._reset_box_stability = lambda: calls.append("stability")
        statuses = []
        app._set_status = statuses.append

        app._close_interferometer_camera("测试关闭")

        self.assertEqual(calls, ["preview", "predict", "stability"])
        self.assertEqual(camera.calls, 1)
        self.assertEqual(app.recorder.calls, 1)
        self.assertIsNone(app.cam)
        self.assertFalse(app.camera_running)
        self.assertIsNone(app._center_line_x)
        self.assertIsNone(app._center_line_box)
        self.assertIsNone(app._zero_box_x)
        self.assertEqual(app._zero_box_confidence, 0.0)
        self.assertEqual(app._center_yolo_misses, 0)
        self.assertEqual(app._center_tracker.calls, 1)
        self.assertEqual(statuses, ["测试关闭"])
        self.assertEqual(app.camera_plugin.status, "摄像头未连接")
        self.assertIn("测试关闭", app.log.lines[0])

    def test_close_camera_continues_cleanup_when_devices_raise(self):
        app = object.__new__(YoloCamApp)
        app.cam = _StopStub(RuntimeError("camera busy"))
        app.recorder = _StopStub(RuntimeError("writer busy"))
        app.camera_running = True
        app.camera_plugin = _CameraPanelStub()
        app.log = _LogStub()
        app._center_tracker = _TrackerStub()
        app._center_line_x = 10.0
        app._center_line_box = (1, 2, 3, 4)
        app._zero_box_x = 20.0
        app._zero_box_confidence = 0.9
        app._center_yolo_misses = 3
        app._stop_preview = lambda: None
        app._stop_predict = lambda: None
        app._reset_box_stability = lambda: None
        app._set_status = lambda _text: None

        app._close_interferometer_camera()

        self.assertIsNone(app.cam)
        self.assertFalse(app.camera_running)
        self.assertIsNone(app._center_line_x)
        self.assertEqual(app._center_tracker.calls, 1)
        self.assertTrue(any("录像停止失败" in line for line in app.log.lines))
        self.assertTrue(any("摄像头释放失败" in line for line in app.log.lines))


if __name__ == "__main__":
    unittest.main()
