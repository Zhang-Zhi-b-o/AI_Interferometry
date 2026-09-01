"""智能体执行循环：Receive task → Plan → Call tools → Observe → Report。

借鉴 OpenClaw 的核心循环，但用 DeepSeek 的 OpenAI 兼容 function calling 实现：

1. 把「用户任务 + 实时上下文 + 工具 schema」发给模型；
2. 模型返回 ``tool_calls`` → 逐个经 ``ToolExecutor`` 执行（运动工具阻塞等待
   UI 确认，拒绝则把错误回灌给模型，促其重新规划）；
3. 工具结果作为 ``role:tool`` 消息回填，循环再决策；
4. 模型返回纯文本且无 tool_calls → 视为最终回答，返回。

循环本身运行在 ``AgentSession`` 的后台线程上，不阻塞 Tk 主线程；运动确认在
确认框处阻塞等待，也不影响主线程。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import json

from src.agent.provider import ProviderCancelled, ProviderError
from src.agent.toolkit import ToolExecutor, ToolPolicy, ToolRegistry, ToolResult


@dataclass
class AgentStep:
    kind: str          # "tool" | "final" | "error"
    title: str
    detail: str = ""
    ok: bool = True
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    result: str = ""


@dataclass
class AgentLoopResult:
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    cancelled: bool = False
    error: str = ""

    @property
    def tool_calls_made(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")


class AgentLoop:
    """用 function calling 驱动工具执行直至模型给出最终回答。"""

    def __init__(
        self,
        provider,
        registry: ToolRegistry,
        *,
        executor: ToolExecutor | None = None,
        max_steps: int = 12,
        max_tool_result_chars: int = 4000,
        dry_run: bool = False,
        confirm_motion: bool = True,
    ):
        self.provider = provider
        self.registry = registry
        self.executor = executor or ToolExecutor(
            registry, policy=ToolPolicy(confirm_motion=confirm_motion))
        self.max_steps = max(1, int(max_steps))
        self.max_tool_result_chars = max(200, int(max_tool_result_chars))
        self.dry_run = bool(dry_run)

    def run(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        thinking: bool = False,
        cancel_event=None,
        confirm_handler: Callable[[str, dict], bool] | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentLoopResult:
        self.executor.confirm_handler = confirm_handler
        steps: list[AgentStep] = []
        tools = self.registry.to_openai_schemas()
        working = list(messages)

        def record(step: AgentStep) -> None:
            steps.append(step)
            if on_step is not None:
                try:
                    on_step(step)
                except Exception:
                    pass  # 回调异常不打断循环

        for _ in range(self.max_steps):
            if cancel_event is not None and cancel_event.is_set():
                return AgentLoopResult("", steps, cancelled=True)
            try:
                result = self.provider.chat_with_tools(
                    working, tools, cancel_event=cancel_event, model=model,
                    thinking=thinking)
            except ProviderCancelled:
                return AgentLoopResult("", steps, cancelled=True)
            except ProviderError as exc:
                record(AgentStep("error", "模型调用失败", str(exc), ok=False))
                return AgentLoopResult("", steps, error=str(exc))

            if not result.wants_tool:
                answer = result.content.strip()
                if not answer:
                    answer = "（模型未返回最终回答）"
                record(AgentStep("final", "最终回答", answer))
                working.append({"role": "assistant", "content": answer})
                return AgentLoopResult(answer, steps)

            # 记录 assistant 的 tool_calls，再逐个执行并回填 role:tool 结果。
            assistant_msg = self._assistant_tool_message(result.content, result.tool_calls)
            working.append(assistant_msg)
            for tool_call in result.tool_calls:
                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments") or {}
                if self.dry_run:
                    tool_result = ToolResult(
                        True, data={"dry_run": True,
                                    "note": "仅规划模式：未实际执行该工具"})
                else:
                    tool_result = self.executor.execute(name, arguments)
                record(AgentStep(
                    kind="tool",
                    title=name,
                    detail=json.dumps(arguments, ensure_ascii=False, default=str),
                    ok=tool_result.ok,
                    tool_name=name,
                    arguments=arguments,
                    result=self._clip(tool_result.text),
                ))
                working.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": self._clip(tool_result.text),
                })

        record(AgentStep(
            "error", "达到最大步骤数",
            f"在 {self.max_steps} 步内未收敛到最终回答", ok=False))
        return AgentLoopResult("", steps, error=f"达到最大步骤数（{self.max_steps}）")

    @staticmethod
    def _assistant_tool_message(content: str, tool_calls: list[dict]) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        message["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(
                        tc.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ]
        return message

    def _clip(self, text: str) -> str:
        text = str(text)
        if len(text) <= self.max_tool_result_chars:
            return text
        return text[:self.max_tool_result_chars] + "…[工具结果已截断]"
