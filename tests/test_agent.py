import tempfile
import time
import unittest
from unittest.mock import Mock
from pathlib import Path

from src.agent.knowledge import KnowledgeBase
from src.agent.service import AgentService, SYSTEM_PROMPT
from src.agent.session import AgentSession
from src.agent.provider import ProviderCancelled


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
        self.assertIn("实验台协作伙伴", SYSTEM_PROMPT)
        self.assertIn("现场判断", SYSTEM_PROMPT)
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
            self.assertIn("camera", response.answer)
            self.assertIn("本地知识库", response.answer)
            self.assertNotIn("[来源", response.answer)

    def test_empty_question_is_rejected(self):
        service = AgentService(knowledge_root=Path("missing"))
        self.assertEqual(service.ask("  ").answer, "请输入问题。")

    def test_online_model_is_used_even_without_retrieval_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(knowledge_root=Path(tmp))
            service.provider.api_key = "sk-test"
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

    def test_short_history_is_bounded_and_sent_to_model(self):
        service = AgentService(knowledge_root=Path("missing"))
        service.provider.api_key = "sk-test"
        service.provider.chat = Mock(return_value="回答")
        for index in range(6):
            service.ask(f"问题{index}")
        messages = service.provider.chat.call_args.args[0]
        history_questions = [item["content"] for item in messages
                             if item["role"] == "user" and item["content"].startswith("问题")]
        self.assertLessEqual(len(history_questions), 5)
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


if __name__ == "__main__":
    unittest.main()
