"""可取消、不会阻止进程退出的实验助手会话。"""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading

from src.agent.service import AgentResponse, AgentService


@dataclass(frozen=True)
class AgentSessionResult:
    response: AgentResponse | None = None
    error: Exception | None = None
    cancelled: bool = False


class AgentSession:
    def __init__(self, service: AgentService):
        self.service = service
        self._cancel_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._results: queue.SimpleQueue = queue.SimpleQueue()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ask(self, question: str, include_status: bool, context: dict) -> bool:
        return self._start(
            lambda cancel: self.service.ask(
                question, include_status, context_override=context,
                cancel_event=cancel))

    def test_connection(self) -> bool:
        return self._start(lambda cancel: self.service.test_connection(cancel_event=cancel))

    def cancel(self) -> bool:
        if not self.busy or self._cancel_event is None:
            return False
        self._cancel_event.set()
        return True

    def poll(self) -> AgentSessionResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        self.cancel()

    def _start(self, operation) -> bool:
        if self.busy:
            return False
        self._cancel_event = threading.Event()

        def run():
            try:
                response = operation(self._cancel_event)
                self._results.put(AgentSessionResult(response=response))
            except Exception as exc:
                cancelled = self._cancel_event.is_set()
                self._results.put(AgentSessionResult(error=exc, cancelled=cancelled))

        self._thread = threading.Thread(target=run, name="agent-session", daemon=True)
        self._thread.start()
        return True
