"""DeepSeekProvider function-calling 流式解析的单元测试。

不发起真实网络请求：直接 mock ``urllib.request.urlopen`` 返回假 SSE 流，
验证 tool_calls 按 index 归并、name 与 arguments 片段拼接、JSON 解析与降级。
"""
import json
import unittest
from unittest.mock import patch

from src.agent.provider import ChatResult, DeepSeekProvider, ProviderError


def _sse_lines(*choices):
    """把一串 choices 字典转成 ``data: …`` 字节行，末尾补 [DONE]。"""
    lines = []
    for choice in choices:
        payload = json.dumps({"choices": [choice]}, ensure_ascii=False)
        lines.append(b"data: " + payload.encode("utf-8") + b"\n")
    lines.append(b"data: [DONE]\n")
    return lines


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ProviderToolCallTests(unittest.TestCase):
    def _provider(self):
        return DeepSeekProvider(
            "https://api.deepseek.com/v1", "sk-test", "deepseek-v4-pro")

    def test_chat_with_tools_accumulates_fragments_by_index(self):
        provider = self._provider()
        stream = _sse_lines(
            {"delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "measurement_start", "arguments": ""}}]},
             "finish_reason": None},
            {"delta": {"tool_calls": [{
                "index": 0, "function": {"name": "", "arguments": "{\"target_"}}]},
             "finish_reason": None},
            {"delta": {"tool_calls": [{
                "index": 0, "function": {"name": "", "arguments": "mm\":1.5}"}}]},
             "finish_reason": None},
            {"delta": {"tool_calls": [{
                "index": 1, "id": "call_2", "type": "function",
                "function": {"name": "get_context", "arguments": "{}"}}]},
             "finish_reason": None},
            {"delta": {}, "finish_reason": "tool_calls"},
        )
        tools = [{"type": "function", "function": {
            "name": "x", "description": "d", "parameters": {"type": "object"}}}]
        with patch("urllib.request.urlopen", return_value=_FakeResponse(stream)):
            result = provider.chat_with_tools(
                [{"role": "user", "content": "hi"}], tools)
        self.assertIsInstance(result, ChatResult)
        self.assertTrue(result.wants_tool)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0]["id"], "call_1")
        self.assertEqual(result.tool_calls[0]["name"], "measurement_start")
        self.assertEqual(result.tool_calls[0]["arguments"], {"target_mm": 1.5})
        self.assertEqual(result.tool_calls[1]["name"], "get_context")
        self.assertEqual(result.tool_calls[1]["arguments"], {})

    def test_chat_returns_concatenated_text(self):
        provider = self._provider()
        stream = _sse_lines(
            {"delta": {"content": "你好"}, "finish_reason": None},
            {"delta": {"content": "世界"}, "finish_reason": None},
            {"delta": {}, "finish_reason": "stop"},
        )
        with patch("urllib.request.urlopen", return_value=_FakeResponse(stream)):
            text = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "你好世界")

    def test_chat_with_tools_raises_on_empty_response(self):
        provider = self._provider()
        stream = _sse_lines({"delta": {}, "finish_reason": "stop"})
        with patch("urllib.request.urlopen", return_value=_FakeResponse(stream)):
            with self.assertRaises(ProviderError):
                provider.chat_with_tools(
                    [{"role": "user", "content": "hi"}], [])

    def test_finalise_tool_call_falls_back_on_bad_json(self):
        result = DeepSeekProvider._finalise_tool_call(
            {"id": "call_x", "name": "tool", "arguments": "not-json"})
        self.assertEqual(result["name"], "tool")
        self.assertEqual(result["arguments"], {"_raw": "not-json"})


if __name__ == "__main__":
    unittest.main()
