from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

from src.agent.conversation_export import (
    ConversationEntry,
    export_conversation,
    render_conversation,
)


class ConversationExportTests(unittest.TestCase):
    def setUp(self):
        self.entries = (
            ConversationEntry("你", "当前条纹怎么样？", "2026-09-03 10:00:00"),
            ConversationEntry(
                "助手",
                "条纹基本稳定。",
                "2026-09-03 10:00:02",
                ("继续观察", "重新调整"),
            ),
        )
        self.exported_at = datetime(
            2026, 9, 3, 10, 1, tzinfo=timezone.utc)

    def test_markdown_preserves_roles_text_times_and_options(self):
        text = render_conversation(
            self.entries, exported_at=self.exported_at)

        self.assertIn("消息数量：2", text)
        self.assertIn("## 你 · 2026-09-03 10:00:00", text)
        self.assertIn("当前条纹怎么样？", text)
        self.assertIn("- 继续观察", text)

    def test_text_export_uses_utf8_and_supports_chinese_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "实验对话.txt"
            result = export_conversation(path, self.entries)

            self.assertEqual(result, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("[2026-09-03 10:00:00] 你", text)
            self.assertIn("可选回复：继续观察 / 重新调整", text)

    def test_unknown_render_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持"):
            render_conversation(self.entries, format_name="pdf")


if __name__ == "__main__":
    unittest.main()
