"""后台任务执行器：所有阻塞调用都丢到这里，界面线程只做渲染。

用法::

    tasks.submit(lambda: do_slow_thing(), on_done=..., on_error=...)

回调在**工作线程**里执行，界面层负责用 Qt 信号把它们marshal回主线程
（``app/ui/bridge.py`` 已封装好，页面里直接用 ``ctx.run_async``）。
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class Background:
    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._inference_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def submit(
        self,
        work: Callable[[], object],
        *,
        on_done: Callable[[object], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        name: str = "task",
        serial: bool = False,
    ) -> threading.Thread | Future:
        def runner() -> None:
            if self._stopping.is_set():
                return
            try:
                result = work()
            except BaseException as exc:  # noqa: BLE001 - 后台异常必须交回界面
                if on_error is not None and not self._stopping.is_set():
                    try:
                        on_error(exc)
                    except Exception:
                        traceback.print_exc()
                return
            if on_done is not None and not self._stopping.is_set():
                try:
                    on_done(result)
                except Exception:
                    traceback.print_exc()

        if serial:
            return self._inference_worker.submit(runner)
        thread = threading.Thread(target=runner, name=f"bg-{name}", daemon=True)
        self._threads.append(thread)
        if len(self._threads) > 64:
            self._threads = [t for t in self._threads if t.is_alive()]
        thread.start()
        return thread

    def shutdown(self) -> None:
        self._stopping.set()
        self._inference_worker.shutdown(wait=False, cancel_futures=True)
