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

    def shutdown(self, safety_action: Callable[[], Any] | None = None) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
        if safety_action is not None:
            self._tasks.put((-100, next(self._sequence), "shutdown", safety_action))
        self._tasks.put((1000, next(self._sequence), "__quit__", lambda: None))

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
