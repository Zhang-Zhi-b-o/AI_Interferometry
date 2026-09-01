"""智能体执行循环（AgentLoop）的单元测试：计划 → 工具 → 观察 → 报告。"""
import threading
import unittest

from src.agent.loop import AgentLoop
from src.agent.provider import ChatResult
from src.agent.toolkit import Tool, ToolRegistry, ToolRisk


class FakeProvider:
    """按顺序弹出预置的 ChatResult，并记录每次调用。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.models: list[str | None] = []

    def chat_with_tools(self, messages, tools, cancel_event=None, **kwargs):
        self.calls.append(list(messages))
        self.models.append(kwargs.get("model"))
        return self.responses.pop(0)


def _make_registry():
    calls = {"read": 0, "move": 0}

    def read(_args):
        calls["read"] += 1
        return {"read": calls["read"]}

    def move(_args):
        calls["move"] += 1
        return {"move": calls["move"]}

    registry = ToolRegistry([
        Tool("read_tool", "只读工具", {"type": "object"}, ToolRisk.READ, read),
        Tool("move_tool", "运动工具", {"type": "object"}, ToolRisk.MOTION, move),
    ])
    return registry, calls


def _task():
    return [{"role": "user", "content": "请执行任务"}]


class AgentLoopTests(unittest.TestCase):
    def test_read_tool_auto_executes_then_returns_final_answer(self):
        registry, calls = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "read_tool", "arguments": {}}]),
            ChatResult(content="已完成", tool_calls=[]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5)
        result = loop.run(_task())
        self.assertEqual(result.final_answer, "已完成")
        self.assertEqual(result.tool_calls_made, 1)
        self.assertEqual(calls["read"], 1)
        self.assertEqual([s.kind for s in result.steps], ["tool", "final"])

    def test_selected_model_is_forwarded_to_each_tool_round(self):
        registry, _calls = _make_registry()
        provider = FakeProvider([ChatResult(content="完成", tool_calls=[])])
        loop = AgentLoop(provider, registry)

        loop.run(_task(), model="deepseek-v4-flash")

        self.assertEqual(provider.models, ["deepseek-v4-flash"])

    def test_motion_tool_runs_when_confirmed(self):
        registry, calls = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "move_tool", "arguments": {}}]),
            ChatResult(content="已移动", tool_calls=[]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5)
        result = loop.run(_task(), confirm_handler=lambda name, args: True)
        self.assertEqual(calls["move"], 1)
        self.assertTrue(result.steps[0].ok)

    def test_motion_tool_rejected_then_replans(self):
        registry, calls = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "move_tool", "arguments": {}}]),
            ChatResult(content="", tool_calls=[
                {"id": "2", "name": "read_tool", "arguments": {}}]),
            ChatResult(content="改用只读方案", tool_calls=[]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5)
        result = loop.run(_task(), confirm_handler=lambda name, args: False)
        self.assertEqual(calls["move"], 0)   # 被拒绝，未执行
        self.assertEqual(calls["read"], 1)   # 重新规划到只读工具
        self.assertEqual(result.final_answer, "改用只读方案")
        self.assertFalse(result.steps[0].ok)  # 拒绝产生的工具步骤 ok=False
        self.assertIn("拒绝", result.steps[0].result)

    def test_step_limit_terminates_loop(self):
        registry, _ = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": str(i), "name": "read_tool", "arguments": {}}])
            for i in range(20)
        ])
        loop = AgentLoop(provider, registry, max_steps=3)
        result = loop.run(_task())
        self.assertTrue(result.error)
        self.assertIn("最大步骤数", result.error)
        self.assertEqual(result.tool_calls_made, 3)

    def test_cancel_event_stops_loop_before_tools(self):
        registry, calls = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "read_tool", "arguments": {}}]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5)
        cancel = threading.Event()
        cancel.set()
        result = loop.run(_task(), cancel_event=cancel)
        self.assertTrue(result.cancelled)
        self.assertEqual(calls["read"], 0)

    def test_dry_run_does_not_execute_tools(self):
        registry, calls = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "read_tool", "arguments": {}}]),
            ChatResult(content="规划完成", tool_calls=[]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5, dry_run=True)
        result = loop.run(_task())
        self.assertEqual(calls["read"], 0)
        self.assertTrue(result.steps[0].ok)  # 占位结果视为成功
        self.assertEqual(result.final_answer, "规划完成")

    def test_on_step_callback_fires(self):
        registry, _ = _make_registry()
        provider = FakeProvider([
            ChatResult(content="", tool_calls=[
                {"id": "1", "name": "read_tool", "arguments": {}}]),
            ChatResult(content="好了", tool_calls=[]),
        ])
        loop = AgentLoop(provider, registry, max_steps=5)
        seen = []
        loop.run(_task(), on_step=lambda step: seen.append(step.kind))
        self.assertEqual(seen, ["tool", "final"])


if __name__ == "__main__":
    unittest.main()
