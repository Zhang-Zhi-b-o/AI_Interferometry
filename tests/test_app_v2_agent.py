"""助手协议与确认边界；完全离线，不使用真实串口和模型额度。"""
import threading
import time
from types import SimpleNamespace as NS

import pytest

from app.agent.guidance import guidance
from app.agent.runtime import AgentRuntime
from app.agent.tools import ToolRegistry
from app.core.config import AppConfig
from app.core.state import AppState
from app.services.compute import ComputeService
from src.agent.provider import ChatResult


def context():
    calls = []
    motion = NS(connected=True, stop=lambda: calls.append("motor-stop"))
    search = NS(running=False, start=lambda: (calls.append("start") or (True, "started")),
                stop=lambda: calls.append("search-stop"))
    return NS(
        state=AppState(), config=AppConfig(), roi=None, history=[],
        camera=NS(is_open=False), detect=NS(is_loaded=False),
        reading=NS(connected=False, value_mm=None, stable=False),
        motion=motion, search=search,
        target=NS(running=False, stop=lambda: calls.append("target-stop")),
        compute=ComputeService(), frame=lambda: None, calls=calls)


class Provider:
    available = True

    def __init__(self, results):
        self.results = iter(results)
        self.messages = []

    def chat_with_tools(self, messages, tools, **kwargs):
        self.messages.append(list(messages))
        return next(self.results)


def call(name, arguments=None):
    return ChatResult(tool_calls=[{"id": "call1", "name": name, "arguments": arguments or {}}])


def test_registry_and_numeric_tool_cycle():
    ctx = context()
    provider = Provider([call("calculate_thickness", {"d1_mm": 1, "d2_mm": 2}),
                         ChatResult(content="厚度已计算")])
    runtime = AgentRuntime(ctx, provider=provider)
    assert len(runtime.registry.schemas()) == 9
    assert runtime.ask("计算厚度") == "厚度已计算"
    output = provider.messages[1][-1]["content"]
    assert "thickness_mm" in output
    assert len(runtime.history) == 2
    assert ctx.calls == []


@pytest.mark.parametrize("approved", [False, True])
def test_motion_requires_confirmation(approved):
    ctx = context()
    ctx.state.update(center_x=100)
    confirmations = []
    runtime = AgentRuntime(
        ctx, provider=Provider([call("start_search"), ChatResult(content="已处理")]),
        confirm_motion=lambda *args: confirmations.append(args[0].name) or approved)
    runtime.ask("开始寻中")
    assert confirmations == ["start_search"]
    assert ("start" in ctx.calls) is approved
    assert "error" in ToolRegistry(ctx).execute("start_search", {})


def test_stop_never_requests_confirmation():
    ctx = context()
    runtime = AgentRuntime(
        ctx, provider=Provider([call("emergency_stop"), ChatResult(content="已停止")]),
        confirm_motion=lambda *args: pytest.fail("停车不应请求确认"))
    runtime.ask("停止")
    assert ctx.calls == ["search-stop", "target-stop", "motor-stop"]


def test_invalid_tool_arguments_never_move():
    ctx = context()
    registry = ToolRegistry(ctx)
    for name, args in [("start_search", {"unexpected": True}),
                       ("calculate_thickness", {"d1_mm": float("nan"), "d2_mm": 1}),
                       ("evaluate_uncertainty", {"values_mm": []}),
                       ("unknown", {})]:
        with pytest.raises(ValueError):
            registry.execute(name, args, confirmed=True)
    assert ctx.calls == []


def test_cancel_blocked_provider_discards_late_motion():
    ctx = context()
    entered, release = threading.Event(), threading.Event()

    class Blocking:
        available = True

        def chat_with_tools(self, *args, **kwargs):
            entered.set()
            release.wait(3)
            return call("start_search")

    runtime = AgentRuntime(ctx, provider=Blocking())
    answers = []
    thread = threading.Thread(target=lambda: answers.append(runtime.ask("开始")))
    thread.start()
    assert entered.wait(2)
    started = time.monotonic()
    runtime.cancel()
    thread.join(0.5)
    assert not thread.is_alive()
    assert time.monotonic() - started < 1
    release.set()
    assert "已取消" in answers[0]
    assert ctx.calls == []


def test_cancel_before_worker_starts():
    ctx = context()
    provider = Provider([call("start_search")])
    runtime = AgentRuntime(ctx, provider=provider)
    turn = threading.Event()
    turn.set()
    assert "已取消" in runtime.ask("开始", cancel_event=turn)
    assert provider.messages == []
    assert ctx.calls == []


def test_cancel_during_confirmation_prevents_motion():
    ctx = context()
    ctx.state.update(center_x=100)
    runtime = None

    def confirm(*args):
        runtime.cancel()
        return True

    runtime = AgentRuntime(ctx, provider=Provider([call("start_search")]),
                           confirm_motion=confirm)
    assert "已取消" in runtime.ask("开始")
    assert ctx.calls == []


def test_late_provider_keeps_original_cancel_token():
    ctx = context()
    entered, release = threading.Event(), threading.Event()
    tokens = []

    class Blocking:
        available = True

        def chat_with_tools(self, *args, cancel_event, **kwargs):
            tokens.append(cancel_event)
            entered.set()
            release.wait(3)
            return ChatResult(content="迟到结果")

    runtime = AgentRuntime(ctx, provider=Blocking())
    thread = threading.Thread(target=lambda: runtime.ask("第一轮"))
    thread.start()
    assert entered.wait(2)
    runtime.cancel()
    thread.join(0.5)
    assert not thread.is_alive()
    runtime.provider = Provider([ChatResult(content="第二轮")])
    assert runtime.ask("继续") == "第二轮"
    assert tokens[0].is_set()
    release.set()


def test_offline_and_four_guidance_stages():
    ctx = context()
    runtime = AgentRuntime(ctx, provider=NS(available=False))
    assert "[离线模式]" in runtime.ask("下一步")
    payload = {"state": {}}
    assert guidance(payload).startswith("观察")
    payload["state"]["frame_size"] = [640, 480]
    assert guidance(payload).startswith("定位")
    payload["state"]["center_x"] = 320
    assert guidance(payload).startswith("读数")
    payload["state"]["reading_mm"] = 1
    assert guidance(payload).startswith("计算")
    payload["reading_age_seconds"] = 2
    assert guidance(payload).startswith("读数")


def test_streaming_deltas_come_before_final_answer():
    ctx = context()
    events = []

    class Streaming:
        available = True

        def chat_with_tools(self, messages, tools, *, on_delta=None, **kwargs):
            for piece in ("先测", "条纹，", "再读数。"):
                on_delta(piece)
            return ChatResult(content="先测条纹，再读数。")

    runtime = AgentRuntime(ctx, provider=Streaming(),
                           on_event=lambda kind, value: events.append((kind, value)))
    assert runtime.ask("下一步") == "先测条纹，再读数。"
    deltas = "".join(value for kind, value in events if kind == "delta")
    assert deltas == "先测条纹，再读数。"
    assert events[-1][0] == "text"
