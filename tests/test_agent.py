import tempfile
import time
import unittest
from unittest.mock import Mock
from pathlib import Path

from src.agent.knowledge import KnowledgeBase
from src.agent.service import AgentService, SYSTEM_PROMPT
from src.agent.session import AgentSession
from src.agent.provider import ProviderCancelled
from src.agent.tools import build_suggestion, suggest_next
from src.ui.runtime_context import build_runtime_context


class KnowledgeBaseTests(unittest.TestCase):
    def test_retrieval_prefers_matching_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text(
                "---\ntitle: 条纹故障\ntags: [条纹, 故障]\n---\n## 无条纹\n检查返回光斑是否重合。",
                encoding="utf-8")
            (root / "b.md").write_text(
                "---\ntitle: 不确定度\n---\n重复测量用于A类评定。", encoding="utf-8")
            kb = KnowledgeBase(root)
            result = kb.search("看不到条纹怎么排查", top_k=1)
            self.assertEqual(result[0].title, "条纹故障")

    def test_no_match_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("完全不相关的内容", encoding="utf-8")
            self.assertEqual(KnowledgeBase(root).search("激光条纹"), [])


class AgentServiceTests(unittest.TestCase):
    def test_system_prompt_is_immersive_and_safe(self):
        self.assertIn("迈克尔逊干涉实验教学搭档", SYSTEM_PROMPT)
        self.assertIn("实验预习", SYSTEM_PROMPT)
        self.assertIn("实验过程指导", SYSTEM_PROMPT)
        self.assertIn("误差计算", SYSTEM_PROMPT)
        self.assertIn("# 迈克尔逊干涉实验报告", SYSTEM_PROMPT)
        self.assertIn("## 8. 实验结论", SYSTEM_PROMPT)
        self.assertIn("[待补充：具体内容]", SYSTEM_PROMPT)
        self.assertIn("不要主动讨论软件功能边界", SYSTEM_PROMPT)
        self.assertIn("现场判断", SYSTEM_PROMPT)
        self.assertIn("experiment_progress", SYSTEM_PROMPT)
        self.assertIn("固定实验流程（不得跳步）", SYSTEM_PROMPT)
        self.assertIn("打开激光光源，调出非定域干涉条纹", SYSTEM_PROMPT)
        self.assertIn("沿同一方向寻找条纹", SYSTEM_PROMPT)
        self.assertIn("固定七步总流程", SYSTEM_PROMPT)
        self.assertIn("progress_percent", SYSTEM_PROMPT)
        self.assertIn("绝不声称自己已经启动", SYSTEM_PROMPT)
        self.assertIn("不输出资料来源编号", SYSTEM_PROMPT)

    def test_offline_mode_returns_sources_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text(
                "---\ntitle: 实验原理\nurl: https://example.test\n---\n迈克尔逊干涉由两束相干光叠加形成。",
                encoding="utf-8")
            service = AgentService(lambda: {"camera": {"running": True}}, root)
            service.provider.api_key = ""
            response = service.ask("迈克尔逊干涉原理是什么？")
            self.assertFalse(response.online)
            self.assertEqual(len(response.sources), 1)
            self.assertIn("迈克尔逊干涉", response.answer)
            self.assertIn("本地实验指导", response.answer)
            self.assertNotIn("[来源", response.answer)

    def test_offline_guidance_uses_live_progress_without_knowledge_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = {"experiment_progress": {
                "stage": "视觉准备", "progress_percent": 30,
                "next_action": "启动模型预测",
                "completion_criterion": "画面出现识别结果",
            }}
            service = AgentService(lambda: context, Path(tmp))
            service.provider.api_key = ""
            response = service.ask("下一步做什么")
            self.assertIn("进度 30%", response.answer)
            self.assertIn("启动模型预测", response.answer)
            self.assertIn("画面出现识别结果", response.answer)

    def test_empty_question_is_rejected(self):
        service = AgentService(knowledge_root=Path("missing"))
        self.assertEqual(service.ask("  ").answer, "请输入问题。")

    def test_online_model_is_used_even_without_retrieval_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(knowledge_root=Path(tmp))
            service.provider.api_key = "sk-test"
            service.autonomous_enabled = False
            service.provider.chat = Mock(return_value="在线回答")
            response = service.ask("你好")
            self.assertTrue(response.online)
            self.assertEqual(response.answer, "在线回答")
            service.provider.chat.assert_called_once()

    def test_connection_reports_model(self):
        service = AgentService(knowledge_root=Path("missing"))
        service.provider.api_key = "sk-test"
        service.provider.chat = Mock(return_value="连接成功")
        response = service.test_connection()
        self.assertTrue(response.online)
        self.assertIn(service.provider.model, response.answer)

    def test_report_request_uses_larger_output_budget(self):
        service = AgentService(knowledge_root=Path("missing"))
        service.provider.api_key = "sk-test"
        service.autonomous_enabled = False
        service.provider.chat = Mock(return_value="报告")
        service.ask("请生成实验报告")
        self.assertGreaterEqual(
            service.provider.chat.call_args.kwargs["max_tokens"], 3000)

    def test_normal_questions_have_full_experiment_output_budget(self):
        service = AgentService(knowledge_root=Path("missing"))
        self.assertEqual(service.provider.max_tokens, 6000)
        self.assertEqual(service.provider.model, "deepseek-v4-pro")

    def test_short_history_is_bounded_and_sent_to_model(self):
        service = AgentService(knowledge_root=Path("missing"))
        service.provider.api_key = "sk-test"
        service.autonomous_enabled = False
        service.provider.chat = Mock(return_value="回答")
        for index in range(15):
            service.ask(f"问题{index}")
        messages = service.provider.chat.call_args.args[0]
        history_questions = [item["content"] for item in messages
                             if item["role"] == "user" and item["content"].startswith("问题")]
        self.assertLessEqual(len(history_questions), 13)
        self.assertNotIn("问题0", history_questions)

    def test_agent_session_can_cancel_without_waiting_for_thread_shutdown(self):
        class BlockingService:
            def ask(self, _question, _include_status, context_override, cancel_event):
                self.context = context_override
                while not cancel_event.wait(0.005):
                    pass
                raise ProviderCancelled("请求已取消")

        session = AgentSession(BlockingService())
        self.assertTrue(session.ask("问题", True, {"camera": {"running": True}}))
        self.assertTrue(session.cancel())
        deadline = time.monotonic() + 1
        result = None
        while result is None and time.monotonic() < deadline:
            result = session.poll()
            time.sleep(0.005)
        self.assertTrue(result.cancelled)


class RuntimeProgressTests(unittest.TestCase):
    @staticmethod
    def context(**overrides):
        values = {
            "camera_running": False, "fps": 0.0,
            "model_loaded": False, "prediction_running": False,
            "detections": {}, "center_x_px": None,
            "fringe_motion": {}, "motor_connected": False,
            "motor_mode": "manual", "auto_enabled": False,
            "auto_state": "未启动", "auto_control_state": "idle",
            "micrometer_connected": False, "micrometer_reading_mm": None,
            "micrometer_reading_at": None, "scale_factor": 1.0,
            "record_count": 0,
        }
        values.update(overrides)
        return build_runtime_context(**values)

    def test_progress_starts_with_camera_guidance(self):
        progress = self.context()["experiment_progress"]
        self.assertEqual(progress["step_number"], 1)
        self.assertIn("白光光源", progress["next_action"])

    def test_progress_follows_the_five_required_steps(self):
        progress = self.context(
            camera_running=True, fps=30,
        )["experiment_progress"]
        self.assertEqual(progress["step_number"], 2)
        self.assertIn("微分表摄像头", progress["next_action"])

        progress = self.context(
            camera_running=True, fps=30, micrometer_connected=True,
            motor_connected=True,
        )["experiment_progress"]
        self.assertEqual(progress["step_number"], 3)
        self.assertIn("ROI", progress["next_action"])

        progress = self.context(
            camera_running=True, fps=30, micrometer_connected=True,
            motor_connected=True, preview_adjusted=True,
            roi_xywh=(10, 20, 300, 400),
        )["experiment_progress"]
        self.assertEqual(progress["step_number"], 4)
        self.assertIn("加载模型", progress["next_action"])

        progress = self.context(
            camera_running=True, fps=30, micrometer_connected=True,
            motor_connected=True, preview_adjusted=True,
            roi_xywh=(10, 20, 300, 400), model_loaded=True,
            prediction_running=True, auto_analysis_enabled=True,
        )["experiment_progress"]
        self.assertEqual(progress["step_number"], 5)
        self.assertEqual(progress["next_action"], "开始自动寻中")

    def test_runtime_context_exposes_all_state_fields(self):
        context = self.context(
            camera_running=True, fps=30, micrometer_connected=True,
            motor_connected=True,
            fringe_band_overlay=[{"width": 1.0}],
            fringe_count_overlay={"fringe_width": 12.5, "fringe_count": 8},
            fringe_realtime_active=True,
            texture_analysis={"blur": 0.1},
            auto_direction_mapping="forward",
            live_measurement={"reading_mm": 1.234},
            live_measurement_active=True,
            calibration_rows=[{"opd_um": 0.0, "r": 255, "g": 255, "b": 255}],
        )
        self.assertTrue(context["vision"]["fringe_realtime_active"])
        self.assertEqual(
            context["vision"]["fringe_count_overlay"]["fringe_width"], 12.5)
        self.assertEqual(context["vision"]["texture_analysis"]["blur"], 0.1)
        self.assertEqual(context["motor"]["auto_direction_mapping"], "forward")
        self.assertTrue(context["measurement"]["live_measurement_active"])
        self.assertEqual(len(context["measurement"]["calibration"]), 1)


class SuggestionTests(unittest.TestCase):
    def test_suggest_next_without_context_waits_for_state(self):
        suggestion = suggest_next({})
        self.assertEqual(suggestion["next_action"], "请先打开设备并连接仪器")
        self.assertEqual(suggestion["suggestions"], [])

    def test_build_suggestion_reads_extra_state(self):
        context = {
            "experiment_progress": {
                "stage": "自动寻中完成", "progress_percent": 100,
                "next_action": "核对读数并记录数据",
                "completion_criterion": "读数已核对",
            },
            "vision": {
                "model_loaded": True, "prediction_running": True,
                "fringe_present": True, "center_offset_px": 3.0,
                "fringe_count_overlay": {"fringe_width": 12.5, "fringe_count": 8},
            },
            "motor": {"auto_control_state": "centered", "auto_enabled": False},
            "micrometer": {"connected": True, "reading_age_seconds": 1.0},
            "measurement": {
                "record_count": 3,
                "calibration": [{"opd_um": 0.0, "r": 255, "g": 255, "b": 255}],
                "live_measurement": {"reading_mm": 1.2345},
                "live_measurement_active": True,
                "thickness": {"records": [{"thickness_mm": 1.0}]},
                "experiment_assistant": {"session": {"rounds": [{"thickness_mm": 1.0}]}},
            },
        }
        suggestion = suggest_next(context)
        self.assertEqual(suggestion["next_action"], "核对读数并记录数据")
        text = "\n".join(suggestion["suggestions"])
        self.assertIn("不确定度", text)       # record_count
        self.assertIn("颜色→OPD 标定", text)  # calibration
        self.assertIn("实时测量", text)       # live_measurement_active
        self.assertIn("厚度", text)           # thickness records
        self.assertIn("居中", text)           # centered
        self.assertIn("实时间隔", text)       # fringe_count_overlay

    def test_build_suggestion_renders_lines(self):
        text = build_suggestion({
            "experiment_progress": {
                "next_action": "加载模型", "completion_criterion": "模型已加载"},
        })
        self.assertIn("当前状态", text)
        self.assertIn("下一步任务：加载模型", text)
        self.assertIn("完成判据：模型已加载", text)

    def test_system_prompt_allows_advice_beyond_fixed_flow(self):
        self.assertIn("程序已生成的确定性建议", SYSTEM_PROMPT)
        self.assertIn("其他建议", SYSTEM_PROMPT)
        self.assertIn("颜色→OPD标定", SYSTEM_PROMPT)


class SuggestCallTests(unittest.TestCase):
    def test_suggest_uses_small_output_budget(self):
        service = AgentService(knowledge_root=Path("missing"))
        service.provider.api_key = "sk-test"
        service.provider.chat = Mock(return_value="建议内容")
        text = service.suggest({"experiment_progress": {"stage": "准备"}})
        self.assertEqual(text, "建议内容")
        service.provider.chat.assert_called_once()
        max_tokens = service.provider.chat.call_args.kwargs["max_tokens"]
        self.assertLessEqual(max_tokens, 500)

    def test_compact_suggestion_context_is_concise(self):
        context = build_runtime_context(
            camera_running=True, fps=30, model_loaded=True,
            prediction_running=True, detections={}, center_x_px=None,
            fringe_motion={}, motor_connected=True, motor_mode="manual",
            auto_enabled=False, auto_state="未启动", auto_control_state="idle",
            micrometer_connected=True, micrometer_reading_mm=1.234,
            micrometer_reading_at=None, scale_factor=1.0, record_count=2,
            recent_logs=[{"msg": "x" * 200} for _ in range(100)],
            measurement_records=[{"a": "b"} for _ in range(50)],
        )
        import json
        full = json.dumps(context, ensure_ascii=False)
        compact = AgentService._compact_suggestion_context(context)
        self.assertLess(len(compact), len(full))
        self.assertIn("阶段", compact)
        self.assertIn("标定点", compact)


if __name__ == "__main__":
    unittest.main()
