"""低 token 的实验助手主动响应调度器。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from src.agent.experiment_guidance import (
    ExperimentIntent,
    GuidanceDecision,
    build_guidance_decision,
)


@dataclass(frozen=True)
class ProactiveUpdate:
    decision: GuidanceDecision
    changed: bool
    urgent: bool
    llm_reason: str
    request_key: tuple[Any, ...] | None


class ProactiveCoordinator:
    """根据语义变化决定本地更新及是否值得调用一次模型。"""

    def __init__(
        self,
        *,
        min_llm_interval: float = 60.0,
        repeat_suppression: float = 300.0,
        max_calls_per_window: int = 3,
        window_seconds: float = 600.0,
        max_calls_per_session: int = 12,
        stalled_stage_seconds: float = 120.0,
    ) -> None:
        self.min_llm_interval = max(1.0, float(min_llm_interval))
        self.repeat_suppression = max(1.0, float(repeat_suppression))
        self.max_calls_per_window = max(1, int(max_calls_per_window))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_calls_per_session = max(1, int(max_calls_per_session))
        self.stalled_stage_seconds = max(5.0, float(stalled_stage_seconds))
        self._semantic_key: tuple[Any, ...] | None = None
        self._intent_key: tuple[str, str] | None = None
        self._stage = ""
        self._stage_since = 0.0
        self._last_llm_at = float("-inf")
        self._llm_by_key: dict[tuple[Any, ...], float] = {}
        self._llm_calls: deque[float] = deque()
        self._session_calls = 0
        self._stalled_reported_stage = ""

    def observe(self, context: dict[str, Any], *, now: float) -> ProactiveUpdate:
        decision = build_guidance_decision(context)
        key = decision.semantic_key
        changed = key != self._semantic_key
        previous_stage = self._stage
        if decision.stage != self._stage:
            self._stage = decision.stage
            self._stage_since = now
            self._stalled_reported_stage = ""
        elif not self._stage_since:
            self._stage_since = now
        self._semantic_key = key

        intent = ExperimentIntent.from_mapping(context.get("experiment_intent"))
        intent_key = (intent.kind, intent.objective)
        intent_changed = self._intent_key is not None and intent_key != self._intent_key
        self._intent_key = intent_key
        urgent = any(issue.severity == "blocking" for issue in decision.issues)
        llm_reason = ""
        request_key = None
        if intent.response_mode != "quiet":
            if intent_changed:
                llm_reason = "实验目的发生变化，需要据此调整现场指导"
            elif intent.response_mode == "teaching" and previous_stage and changed:
                llm_reason = "实验阶段发生变化，需要结合实验目的解释下一步"
            elif len(decision.issues) >= 2 and changed:
                llm_reason = "多个现场问题同时出现，需要确定处理优先级"
            elif (now - self._stage_since >= self.stalled_stage_seconds
                  and self._stalled_reported_stage != self._stage):
                llm_reason = "当前实验阶段长时间没有进展"
                self._stalled_reported_stage = self._stage
            if llm_reason:
                request_key = (llm_reason,) + key
        return ProactiveUpdate(decision, changed, urgent, llm_reason, request_key)

    def reserve_llm(self, request_key: tuple[Any, ...] | None, *, now: float) -> bool:
        """原子式占用一次后台模型预算；返回 False 表示应使用本地建议。"""
        if request_key is None or self._session_calls >= self.max_calls_per_session:
            return False
        while self._llm_calls and now - self._llm_calls[0] > self.window_seconds:
            self._llm_calls.popleft()
        if len(self._llm_calls) >= self.max_calls_per_window:
            return False
        if now - self._last_llm_at < self.min_llm_interval:
            return False
        last_same = self._llm_by_key.get(request_key, float("-inf"))
        if now - last_same < self.repeat_suppression:
            return False
        self._last_llm_at = now
        self._llm_by_key[request_key] = now
        self._llm_calls.append(now)
        self._session_calls += 1
        return True

    @property
    def session_calls(self) -> int:
        return self._session_calls
