import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

from src.agent.knowledge import KnowledgeBase
from src.agent.service import AgentService


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


if __name__ == "__main__":
    unittest.main()
