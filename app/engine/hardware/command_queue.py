"""串行执行硬件命令的后台队列。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import itertools
import queue
import threading
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    name: str
    value: Any = None
    error: Exception | None = None


class SerialCommandQueue:
    """在单个守护线程中顺序执行串口操作，并在 UI 线程领取结果。"""

    def __init__(self, name: str = "motor-serial"):
        self._tasks: queue.PriorityQueue = queue.PriorityQueue()
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._sequence = itertools.count()
        self._accepting = True
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, name: str, operation: Callable[[], Any], *,
               priority: int = 10, coalesce: bool = False) -> bool:
        with self._lock:
            if not self._accepting or (coalesce and name in self._pending):
                return False
            self._pending.add(name)
        self._tasks.put((priority, next(self._sequence), name, operation))
        return True

    def drain(self) -> list[CommandResult]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def shutdown(
        self,
        safety_action: Callable[[], Any] | None = None,
        *,
        timeout: float = 3.0,
    ) -> CommandResult | None:
        """停止接收任务，优先执行安全动作，并等待工作线程退出。

        返回安全动作结果；超时或重复关闭时返回 ``None``。调用者可据此在
        销毁 UI 前确认停车命令是否真正写入串口。
        """
        with self._lock:
            if not self._accepting:
                self._thread.join(timeout=max(0.0, float(timeout)))
                return None
            self._accepting = False
        safety_done = threading.Event()
        safety_result: list[CommandResult] = []
        if safety_action is not None:
            def run_safety() -> Any:
                try:
                    return safety_action()
                finally:
                    safety_done.set()

            self._tasks.put((-100, next(self._sequence), "shutdown", run_safety))
        # 安全动作之后立即退出，丢弃关闭前尚未执行的普通启动/调速命令。
        quit_priority = -99 if safety_action is not None else -100
        self._tasks.put((quit_priority, next(self._sequence), "__quit__", lambda: None))
        wait_timeout = max(0.0, float(timeout))
        if safety_action is not None:
            safety_done.wait(wait_timeout)
        self._thread.join(timeout=wait_timeout)
        for result in self.drain():
            if result.name == "shutdown":
                safety_result.append(result)
            else:
                self._results.put(result)
        return safety_result[-1] if safety_result else None

    def _run(self) -> None:
        while True:
            _, _, name, operation = self._tasks.get()
            if name == "__quit__":
                return
            try:
                result = CommandResult(name, operation())
            except Exception as exc:  # 由 UI 统一展示硬件异常
                result = CommandResult(name, error=exc)
            finally:
                with self._lock:
                    self._pending.discard(name)
            self._results.put(result)
