"""工具注册表 / 风险分级 / 确认流程 / 执行器的单元测试。"""
import unittest

from src.agent.toolkit import (
    Confirmation,
    Tool,
    ToolExecutor,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolRisk,
)


def _tool(name="read_tool", risk=ToolRisk.READ, fn=lambda args: {"ok": True}):
    return Tool(
        name=name,
        description="测试工具",
        parameters={"type": "object", "properties": {}, "required": []},
        risk=risk,
        fn=fn,
    )


class ToolRiskTests(unittest.TestCase):
    def test_risk_values(self):
        self.assertEqual(ToolRisk.READ.value, "read")
        self.assertEqual(ToolRisk.MOTION.value, "motion")
        self.assertEqual(ToolRisk.STOP.value, "stop")


class ToolSchemaTests(unittest.TestCase):
    def test_to_openai_schema(self):
        tool = Tool("fringe_width", "测条纹", {"type": "object"}, ToolRisk.READ, lambda a: {})
        schema = tool.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "fringe_width")
        self.assertEqual(schema["function"]["description"], "测条纹")
        self.assertEqual(schema["function"]["parameters"], {"type": "object"})


class ToolResultTests(unittest.TestCase):
    def test_ok_serializes_dict(self):
        self.assertEqual(ToolResult(True, {"a": 1}).text, '{"a": 1}')

    def test_error_text(self):
        self.assertEqual(ToolResult(False, error="拒绝").text, "错误：拒绝")


class ToolRegistryTests(unittest.TestCase):
    def test_register_get_names(self):
        registry = ToolRegistry()
        registry.register(_tool("a"))
        self.assertEqual(registry.names(), ["a"])
        self.assertEqual(registry.get("a").name, "a")
        self.assertIsNone(registry.get("missing"))

    def test_to_openai_schemas_and_risk_of(self):
        registry = ToolRegistry([_tool("a", ToolRisk.READ), _tool("b", ToolRisk.MOTION)])
        self.assertEqual(len(registry.to_openai_schemas()), 2)
        self.assertEqual(registry.risk_of("a"), ToolRisk.READ)
        self.assertEqual(registry.risk_of("b"), ToolRisk.MOTION)
        self.assertEqual(registry.risk_of("missing"), ToolRisk.READ)

    def test_describe_includes_risk(self):
        registry = ToolRegistry([_tool("a", ToolRisk.MOTION)])
        rows = registry.describe()
        self.assertEqual(rows[0]["name"], "a")
        self.assertEqual(rows[0]["risk"], "motion")


class ToolPolicyTests(unittest.TestCase):
    def test_read_never_needs_confirmation(self):
        policy = ToolPolicy(confirm_motion=True)
        self.assertFalse(policy.needs_confirmation(_tool(risk=ToolRisk.READ)))

    def test_stop_never_needs_confirmation(self):
        policy = ToolPolicy(confirm_motion=True)
        self.assertFalse(policy.needs_confirmation(_tool(risk=ToolRisk.STOP)))

    def test_motion_confirmation_respects_flag(self):
        self.assertTrue(
            ToolPolicy(confirm_motion=True).needs_confirmation(
                _tool(risk=ToolRisk.MOTION)))
        self.assertFalse(
            ToolPolicy(confirm_motion=False).needs_confirmation(
                _tool(risk=ToolRisk.MOTION)))


class ConfirmationTests(unittest.TestCase):
    def test_approve_sets_event(self):
        confirmation = Confirmation(tool_name="auto_center_start", arguments={})
        self.assertFalse(confirmation.event.is_set())
        confirmation.approve()
        self.assertTrue(confirmation.approved)
        self.assertTrue(confirmation.event.is_set())

    def test_reject_sets_event(self):
        confirmation = Confirmation(tool_name="auto_center_start", arguments={})
        confirmation.reject()
        self.assertTrue(confirmation.rejected)
        self.assertTrue(confirmation.event.is_set())


class ToolExecutorTests(unittest.TestCase):
    def test_read_tool_auto_executes(self):
        registry = ToolRegistry([_tool("read_tool", ToolRisk.READ)])
        executor = ToolExecutor(registry)
        result = executor.execute("read_tool", {})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"ok": True})

    def test_motion_tool_runs_when_confirmed(self):
        registry = ToolRegistry([_tool("move", ToolRisk.MOTION)])
        executor = ToolExecutor(
            registry, confirm_handler=lambda name, args: True)
        result = executor.execute("move", {})
        self.assertTrue(result.ok)

    def test_motion_tool_rejected_returns_error(self):
        registry = ToolRegistry([_tool("move", ToolRisk.MOTION)])
        executor = ToolExecutor(
            registry, confirm_handler=lambda name, args: False)
        result = executor.execute("move", {})
        self.assertFalse(result.ok)
        self.assertIn("拒绝", result.error)

    def test_motion_tool_without_confirm_channel(self):
        registry = ToolRegistry([_tool("move", ToolRisk.MOTION)])
        executor = ToolExecutor(registry)  # 无 confirm_handler
        result = executor.execute("move", {})
        self.assertFalse(result.ok)
        self.assertIn("确认通道", result.error)

    def test_stop_tool_always_runs_without_confirmation(self):
        registry = ToolRegistry([_tool("stop", ToolRisk.STOP)])
        executor = ToolExecutor(registry)  # 无 confirm_handler
        result = executor.execute("stop", {})
        self.assertTrue(result.ok)

    def test_unknown_tool(self):
        executor = ToolExecutor(ToolRegistry())
        result = executor.execute("nope", {})
        self.assertFalse(result.ok)
        self.assertIn("未知工具", result.error)

    def test_tool_exception_is_captured(self):
        def boom(_args):
            raise ValueError("坏了")
        registry = ToolRegistry([_tool("boom", ToolRisk.READ, boom)])
        executor = ToolExecutor(registry)
        result = executor.execute("boom", {})
        self.assertFalse(result.ok)
        self.assertIn("坏了", result.error)


if __name__ == "__main__":
    unittest.main()
