"""公共 UI 数值解析与显示组件测试。"""
from __future__ import annotations

import gc
import math
import unittest

from src.ui.values import bounded_float, bounded_int
from src.ui.widgets.log_panel import LogPanel


class BoundedValueTests(unittest.TestCase):
    def test_float_rejects_non_finite_and_invalid_values(self):
        self.assertEqual(bounded_float("nan", default=2.0), 2.0)
        self.assertEqual(bounded_float(math.inf, default=3.0), 3.0)
        self.assertEqual(bounded_float("bad", default=4.0), 4.0)

    def test_float_and_int_are_clamped(self):
        self.assertEqual(
            bounded_float("12.5", default=1.0, minimum=0.0, maximum=10.0),
            10.0,
        )
        self.assertEqual(
            bounded_int("-3", default=1, minimum=0, maximum=10),
            0,
        )

    def test_log_level_classification_is_deterministic(self):
        self.assertEqual(LogPanel.classify_level("[错误] 无法连接"), "error")
        self.assertEqual(LogPanel.classify_level("[警告] 读数过期"), "warning")
        self.assertEqual(LogPanel.classify_level("摄像头已打开"), "info")


try:
    import tkinter as tk
except Exception:  # pragma: no cover - Python without Tk
    tk = None


class LogPanelDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if tk is None:
            raise unittest.SkipTest("Tkinter 不可用")
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"图形环境不可用: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "root", None) is not None:
            cls.root.destroy()
            cls.root = None
            gc.collect()

    def test_clear_removes_visible_and_structured_history(self):
        panel = LogPanel(self.root)
        try:
            panel.write("普通消息")
            panel.write("[警告] 测试警告")
            self.assertEqual(len(panel.recent_entries()), 2)
            self.assertIn("警告 1", panel.summary_var.get())

            panel.clear()

            self.assertEqual(panel.recent_entries(), [])
            self.assertEqual(panel.summary_var.get(), "0 条 · 错误 0 · 警告 0")
            self.assertEqual(str(panel._text["state"]), "disabled")
        finally:
            panel.destroy()


if __name__ == "__main__":
    unittest.main()
