import unittest

try:
    import tkinter as tk
except Exception:  # pragma: no cover - Python without Tk
    tk = None

from src.ui.widgets.agent_plugin import AgentPluginPanel


class AgentPanelDashboardTests(unittest.TestCase):
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

    def setUp(self):
        self.panel = AgentPluginPanel(self.root)

    def tearDown(self):
        self.panel.destroy()

    @staticmethod
    def context(*, stage="adaptive", ready=True, auto_enabled=False):
        return {
            "camera": {
                "interferometer_running": True,
                "micrometer_running": True,
            },
            "vision": {
                "detections": {"zero_order": 0.9},
                "model_loaded": True,
                "center_offset_px": 2.5,
                "guidance_execution_stage": stage,
                "fringe_guidance": {
                    "execution_stage": stage,
                    "measurement_ready": ready,
                    "quality_score": 0.91 if ready else 0.43,
                    "phase": "measurement_ready" if ready else "adjusting",
                    "summary": "条纹清晰稳定" if ready else "仍需改善",
                    "recommendations": ["等待稳定"],
                    "metrics": {
                        "angle_deg": 1.2,
                        "spacing_px": 40.3,
                        "spacing_cv_percent": 2.1,
                        "movement": "stable",
                    },
                    "actions": [{"code": "apply_angle_correction", "label": "校正角度"}],
                },
                "adaptive_response": {
                    "confidence": 0.6,
                    "response_samples": 12,
                    "learned_settle_seconds": 0.35,
                },
            },
            "motor": {
                "connected": True,
                "auto_enabled": auto_enabled,
                "auto_control_state": "centering" if auto_enabled else "idle",
            },
            "micrometer": {},
            "measurement": {},
            "experiment_progress": {
                "step_number": 5,
                "progress_percent": 95,
                "stage": "条纹调节",
                "next_action": "改善画面",
                "completion_criterion": "质量门通过",
            },
        }

    def test_dashboard_renders_quality_geometry_and_adaptive_state(self):
        self.panel.set_experiment_context(self.context())

        self.assertIn("质量门通过", self.panel.fringe_quality_var.get())
        self.assertIn("40.30px", self.panel.fringe_metrics_var.get())
        self.assertIn("响应样本 12", self.panel.adaptive_var.get())
        self.assertEqual(self.panel.guidance_stage, "adaptive")
        self.assertEqual(str(self.panel.auto_center_button["state"]), "normal")

    def test_read_only_stage_disables_all_execution_shortcuts(self):
        self.panel.set_experiment_context(
            self.context(stage="advisory", ready=False))

        self.assertEqual(str(self.panel.apply_guidance_button["state"]), "disabled")
        self.assertEqual(str(self.panel.auto_center_button["state"]), "disabled")

    def test_running_closed_loop_always_keeps_stop_available(self):
        self.panel.set_experiment_context(
            self.context(stage="advisory", ready=False, auto_enabled=True))

        self.assertEqual(str(self.panel.auto_center_button["state"]), "normal")
        self.assertEqual(self.panel.auto_center_button["text"], "停止闭环")


if __name__ == "__main__":
    unittest.main()
