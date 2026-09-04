import gc
import unittest

try:
    import tkinter as tk
    from tkinter import font as tkfont
except Exception:  # pragma: no cover - Python without Tk
    tk = None
    tkfont = None

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
            cls.root = None
            gc.collect()

    def setUp(self):
        self.panel = AgentPluginPanel(self.root)

    def tearDown(self):
        self.panel.destroy()
        # Drop Tk variables while this test is still on the Tk/main thread.
        # Otherwise cyclic collection may finalize them later on a camera
        # worker, where Tk raises "main thread is not in main loop".
        del self.panel

    @staticmethod
    def context(*, stage="adaptive", ready=True, auto_enabled=False,
                laser_active=False):
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
                "laser_alignment_active": laser_active,
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
                    "laser_vertical_alignment": {
                        "stage": "straighten", "ready": False,
                        "observation": "条纹顺时针倾斜 8°",
                        "action": "从动镜背面逆时针微调上方旋钮（位于动镜背面左上侧）约 1/16 圈。",
                        "expected_change": "倾角绝对值减小。",
                        "stop_condition": "倾角不超过 3°。",
                    },
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

    def test_font_controls_scale_the_entire_assistant_and_default_is_larger(self):
        def size(widget):
            return int(tkfont.Font(
                root=self.root, font=widget.cget("font")).actual("size"))

        action_before = size(self.panel.laser_action_label)
        advanced_before = size(self.panel.advanced_status_button)
        input_before = size(self.panel.input)
        self.assertEqual(self.panel.font_size, 11)

        self.panel.change_font_size(2)

        self.assertEqual(size(self.panel.laser_action_label), action_before + 2)
        self.assertEqual(size(self.panel.advanced_status_button), advanced_before + 2)
        self.assertEqual(size(self.panel.input), input_before + 2)
        self.panel.reset_font_size()
        self.assertEqual(self.panel.font_size, 11)

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

    def test_proactive_blocking_decision_is_prominent_and_tracks_budget(self):
        self.panel.set_proactive_guidance({
            "priority": "blocking",
            "diagnosis": "微分表读数已经过期",
            "action": "等待新读数",
            "evidence": ["读数已 8.0 秒未刷新"],
            "issues": [{"code": "STALE_MICROMETER"}],
        }, llm_calls=2)

        self.assertIn("读数已经过期", self.panel.proactive_var.get())
        self.assertIn("等待新读数", self.panel.proactive_var.get())
        self.assertIn("2 次", self.panel.proactive_budget_var.get())
        self.assertEqual(self.panel.proactive_label["fg"], "#c53030")

    def test_intent_and_response_mode_emit_structured_values(self):
        intents = []
        modes = []
        self.panel.on_set_intent = intents.append
        self.panel.on_set_response_mode = modes.append
        self.panel.intent_var.set("调节并测量条纹间距")
        self.panel.response_mode_var.set("教学")

        self.panel._on_intent_selected()
        self.panel._on_response_mode_selected()

        self.assertEqual(intents, ["fringe_spacing"])
        self.assertEqual(modes, ["teaching"])

    def test_image_review_button_follows_busy_state(self):
        self.panel.set_busy(True)
        self.assertEqual(str(self.panel.image_review_button["state"]), "disabled")
        self.panel.set_busy(False)
        self.assertEqual(str(self.panel.image_review_button["state"]), "normal")

    def test_laser_alignment_button_activates_live_knob_instruction(self):
        changes = []
        self.panel.on_toggle_laser_alignment = changes.append
        self.panel.laser_alignment_button.invoke()
        self.assertEqual(changes, [True])
        self.assertIn("结束", self.panel.laser_alignment_button["text"])

        self.panel.set_experiment_context(
            self.context(laser_active=True, ready=False))
        instruction = self.panel.laser_alignment_var.get()
        self.assertIn("上方旋钮", instruction)
        self.assertIn("左上侧", instruction)
        self.assertIn("逆时针", instruction)
        self.assertIn("停下复测", instruction)

    def test_focus_card_renders_single_step_and_automatic_comparison(self):
        self.panel.set_laser_workflow({
            "step_number": 3, "total_steps": 6, "step_title": "调直条纹",
            "state": "action_required", "diagnosis": "条纹倾斜 8.2°",
            "action": "上方旋钮（位于动镜背面左上侧）逆时针约 1/16 圈",
            "expected_change": "倾角绝对值减小", "stop_condition": "松手复测",
            "metrics": {"angle_deg": 8.2, "spacing_px": 42,
                        "bright_fringe_count": 7, "spacing_valid": True},
            "target": {"min_bright_fringes": 4, "max_bright_fringes": 10},
            "comparison": {"outcome": "improved", "summary": "本次调节有效",
                           "recommendation": "可以同方向再微调一次"},
        })
        self.assertIn("第 3/6 步", self.panel.laser_step_var.get())
        self.assertIn("上方旋钮", self.panel.laser_action_var.get())
        self.assertIn("明纹 7条·合适", self.panel.laser_metrics_var.get())
        self.assertIn("本次调节有效", self.panel.laser_comparison_var.get())

    def test_auto_ai_toggle_activates_laser_mode(self):
        laser_changes = []
        ai_changes = []
        self.panel.on_toggle_laser_alignment = laser_changes.append
        self.panel.on_toggle_laser_ai_guidance = ai_changes.append
        self.panel.laser_ai_button.invoke()
        self.assertEqual(laser_changes, [True])
        self.assertEqual(ai_changes, [True])

    def test_conversation_snapshot_contains_displayed_messages_and_options(self):
        self.panel.append("你", "我看到了彩色条纹")
        self.panel.append("助手", "请判断条纹是否稳定。\n【选项】稳定；仍在漂移")

        entries = self.panel.conversation_entries()

        self.assertEqual(entries[-2].role, "你")
        self.assertEqual(entries[-2].text, "我看到了彩色条纹")
        self.assertEqual(entries[-1].text, "请判断条纹是否稳定。")
        self.assertEqual(entries[-1].options, ("稳定", "仍在漂移"))
        self.assertRegex(entries[-1].timestamp, r"^\d{4}-\d{2}-\d{2} ")

    def test_export_button_invokes_ui_callback(self):
        calls = []
        self.panel.on_export_chat = lambda: calls.append("export")

        self.panel.export_chat_button.invoke()

        self.assertEqual(calls, ["export"])


if __name__ == "__main__":
    unittest.main()
