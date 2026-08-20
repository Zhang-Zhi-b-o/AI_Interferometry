"""智能体工具注册表与执行策略（借鉴 OpenClaw 的 skills/tool registry）。

每个工具 = ``name + description + JSON Schema + 实现``，模型通过 DeepSeek
function calling 自动发现并决定调用。工具带风险分级，运动类工具需要人工确认：

- ``READ``   只读（读表 / 查状态 / 测条纹 / 算厚度 / 误差），自动执行；
- ``MOTION`` 使电机转动的操作，需 UI 弹确认框，用户点「确认」才执行；
- ``STOP``   急停 / 停止操作，始终放行。

本模块不依赖 Tkinter 或任何硬件，只做注册、分派与策略判断，便于单元测试和
CLI 复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import json
import threading


class ToolRisk(str, Enum):
    READ = "read"
    MOTION = "motion"
    STOP = "stop"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema（properties / required / type）
    risk: ToolRisk
    fn: Callable[[dict], Any]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""

    @property
    def text(self) -> str:
        if self.ok:
            if isinstance(self.data, (dict, list, tuple, str, int, float, bool)):
                return json.dumps(self.data, ensure_ascii=False, default=str)
            return str(self.data)
        return f"错误：{self.error}"


class ToolRegistry:
    """按名称登记工具，提供 OpenAI function-calling 所需的 schema 列表。"""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai_schemas(self) -> list[dict]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def risk_of(self, name: str) -> ToolRisk:
        tool = self._tools.get(name)
        return tool.risk if tool is not None else ToolRisk.READ

    def describe(self) -> list[dict]:
        return [
            {"name": tool.name, "risk": tool.risk.value,
             "description": tool.description}
            for tool in self._tools.values()
        ]


class ToolPolicy:
    """判断一个工具是否需要人工确认。"""

    def __init__(self, confirm_motion: bool = True):
        self.confirm_motion = bool(confirm_motion)

    def needs_confirmation(self, tool: Tool) -> bool:
        if tool.risk is ToolRisk.STOP:
            return False
        if tool.risk is ToolRisk.MOTION:
            return self.confirm_motion
        return False


@dataclass
class Confirmation:
    """一个待确认的运动操作；UI 主线程 approve/reject 后事件被置位。"""

    tool_name: str
    arguments: dict
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    rejected: bool = False

    def approve(self) -> None:
        self.approved = True
        self.event.set()

    def reject(self) -> None:
        self.rejected = True
        self.event.set()

    @property
    def summary(self) -> str:
        return f"{self.tool_name}({json.dumps(self.arguments, ensure_ascii=False)})"


class ToolExecutor:
    """按策略执行一次工具调用，运动工具经 ``confirm_handler`` 门控。

    ``confirm_handler`` 形如 ``(tool_name, arguments) -> bool``，返回 True 表示
    用户同意执行；由 UI 注入，可阻塞等待用户点击。返回 ``ToolResult``。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
        confirm_handler: Callable[[str, dict], bool] | None = None,
    ):
        self.registry = registry
        self.policy = policy or ToolPolicy()
        self.confirm_handler = confirm_handler

    def execute(self, name: str, arguments: dict | None) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(False, error=f"未知工具：{name}")
        if self.policy.needs_confirmation(tool):
            if self.confirm_handler is None:
                return ToolResult(
                    False, error=f"工具 {name} 需要人工确认，但当前没有确认通道")
            if not self.confirm_handler(name, arguments or {}):
                return ToolResult(False, error="用户拒绝了该运动操作")
        try:
            data = tool.fn(arguments or {})
            return ToolResult(True, data=data)
        except Exception as exc:  # 工具异常不应中断智能体循环
            return ToolResult(False, error=str(exc))
