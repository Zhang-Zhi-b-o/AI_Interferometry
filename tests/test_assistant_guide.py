"""实验助手互动引导：选项标记解析与实时诊断规则的单元测试。"""
import unittest

from src.agent.tools import diagnose_context, parse_options


class ParseOptionsTests(unittest.TestCase):
    def test_no_marker_returns_text_unchanged(self):
        text, options = parse_options("请你完成第一步")
        self.assertEqual(text, "请你完成第一步")
        self.assertEqual(options, [])

    def test_marker_at_end(self):
        text, options = parse_options(
            "请确认你看到的现象\n【选项】看到条纹了；还没看到")
        self.assertEqual(text, "请确认你看到的现象")
        self.assertEqual(options, ["看到条纹了", "还没看到"])

    def test_marker_strips_only_marker_line(self):
        text, options = parse_options(
            "第一行说明\n【选项】甲、乙，丙；丁\n标记行之后的内容")
        self.assertEqual(text, "第一行说明\n\n标记行之后的内容")
        self.assertEqual(options, ["甲", "乙", "丙", "丁"])

    def test_empty_options(self):
        text, options = parse_options("正文【选项】")
        self.assertEqual(text, "正文")
        self.assertEqual(options, [])

    def test_english_and_cn_separators(self):
        _, options = parse_options("【选项】A;B、C，D")
        self.assertEqual(options, ["A", "B", "C", "D"])


class DiagnoseContextTests(unittest.TestCase):
    def _ctx(self, **overrides):
        base = {
            "vision": {
                "model_loaded": True, "prediction_running": True,
                "fringe_present": False, "center_offset_px": None,
            },
            "motor": {"auto_enabled": False, "auto_control_state": "idle"},
            "micrometer": {"connected": True, "reading_age_seconds": 0.5},
            "experiment_progress": {
                "stage": "模型预测与条纹分析", "next_action": "开始预测"},
        }
        base.update(overrides)
        return base

    def test_empty_context(self):
        self.assertIn("等待实时状态", diagnose_context({}))

    def test_stale_micrometer(self):
        context = self._ctx()
        context["micrometer"]["reading_age_seconds"] = 8.2
        self.assertIn("微分表读数已过期", diagnose_context(context))

    def test_centered(self):
        context = self._ctx()
        context["motor"]["auto_control_state"] = "centered"
        self.assertIn("中心条纹已稳定", diagnose_context(context))

    def test_auto_enabled(self):
        context = self._ctx()
        context["motor"]["auto_enabled"] = True
        self.assertIn("自动寻中运行中", diagnose_context(context))

    def test_center_offset(self):
        context = self._ctx()
        context["vision"]["center_offset_px"] = 120.0
        self.assertIn("偏离画面中心", diagnose_context(context))

    def test_fringe_present(self):
        context = self._ctx()
        context["vision"]["fringe_present"] = True
        self.assertIn("已识别到干涉条纹", diagnose_context(context))

    def test_model_not_loaded(self):
        context = self._ctx()
        context["vision"]["model_loaded"] = False
        self.assertIn("模型或预测未就绪", diagnose_context(context))

    def test_fallback_stage(self):
        context = self._ctx()
        insight = diagnose_context(context)
        self.assertIn("当前阶段", insight)
        self.assertIn("模型预测与条纹分析", insight)


if __name__ == "__main__":
    unittest.main()
