"""页面与服务共享的上下文对象。

每个页面拿到的都是它：``ctx.camera`` 开相机、``ctx.state`` 读当前值、
``ctx.run_async`` 跑耗时任务、``ctx.log`` 写状态栏。页面之间**不直接互相调用**。
"""
from __future__ import annotations

from typing import Any, Callable
import copy
import threading

import numpy as np

from app.core.config import AppConfig
from app.core.state import AppState
from app.core.tasks import Background
from app.core.timeline import Timeline
from app.core.config import ROOT
from app.ui.bridge import Bridge


class AppContext:
    def __init__(
        self,
        *,
        config: AppConfig,
        state: AppState,
        bridge: Bridge,
        tasks: Background,
        frame_provider: Callable[[], np.ndarray | None],
        navigate: Callable[[str], None],
        theme_toggle: Callable[[], None],
        stop_motion: Callable[[], None],
    ) -> None:
        # core
        self.config = config
        self.state = state
        self.bridge = bridge
        self.tasks = tasks
        # 服务（由 MainWindow 装配后注入）
        self.camera: Any = None
        self.detect: Any = None
        self.fringe: Any = None
        self.reading: Any = None
        self.motion: Any = None
        self.search: Any = None
        self.target: Any = None
        self.compute: Any = None
        self.record: Any = None
        self.thickness: Any = None
        self.calibration: Any = None
        # 界面回调
        self._frame_provider = frame_provider
        self._navigate = navigate
        self._theme_toggle = theme_toggle
        self.stop_motion = stop_motion
        self._pending: dict[str, tuple[Any, Any]] = {}
        self._ticket = 0
        # 跨页面共享的小数据（读数历史、当前 ROI、叠加层开关）
        self.history: list[dict] = []
        self.measurements: list[dict] = []
        self.latest_measurement: dict | None = None
        self.last_map: dict | None = None
        self._data_lock = threading.Lock()
        self.timeline = Timeline(ROOT / "exports" / "sessions")
        from app.services.performance import PerformanceLog

        self.performance = PerformanceLog(self.timeline.path.with_name(
            f"performance-{self.timeline.session_id}.jsonl"))
        self._last_transitions = {}
        self.state.changed.subscribe(self._track_transitions)
        self.roi: tuple[int, int, int, int] | None = None
        self.overlay_flags: dict[str, bool] = {"center": True, "boxes": True}
        self.bridge.task_ok.connect(self._on_task_ok)
        self.bridge.task_err.connect(self._on_task_err)

    # ------------------------------------------------------------------
    def set_overlay_flag(self, name: str, value: bool) -> None:
        self.overlay_flags[str(name)] = bool(value)

    def overlay_enabled(self, name: str) -> bool:
        return bool(self.overlay_flags.get(name, True))

    # ------------------------------------------------------------------
    def frame(self) -> np.ndarray | None:
        return self._frame_provider()

    def frame_size(self) -> tuple[int, int]:
        frame = self._frame_provider()
        if frame is None:
            return (0, 0)
        return frame.shape[1], frame.shape[0]

    def navigate(self, key: str) -> None:
        self._navigate(key)

    def toggle_theme(self) -> None:
        self._theme_toggle()

    def log(self, text: str) -> None:
        self.bridge.message.emit(str(text))

    def _track_transitions(self, state):
        for key in ("camera_open", "motor_connected", "search_running", "target_running"):
            value = getattr(state, key)
            old = self._last_transitions.get(key, False)
            self._last_transitions[key] = value
            if old != value:
                self.timeline.add(key, result={"value": value})

    def save_measurement(self, d1: float, d2: float, value: float) -> dict:
        import time

        result = {"d1": d1, "d2": d2, "value_mm": value,
                  "refractive_index": self.compute.refractive_index,
                  "at": time.time(), "data_source": "demo" if getattr(self, "demo_mode", False) else "live"}
        event = self.timeline.add("calculation", parameters={"d1_mm": d1, "d2_mm": d2,
                                  "refractive_index": self.compute.refractive_index}, result=result)
        result["event_id"] = event["id"]
        with self._data_lock:
            self.latest_measurement = result
        return copy.deepcopy(result)

    def add_measurement(self) -> dict | None:
        with self._data_lock:
            if self.latest_measurement is None:
                return None
            item = copy.deepcopy(self.latest_measurement)
            self.measurements.append(item)
        self.timeline.add("round_recorded", result=item)
        return item

    def clear_measurements(self):
        with self._data_lock:
            self.measurements.clear()
        self.timeline.add("rounds_cleared")

    def report_snapshot(self) -> dict:
        from app.agent.context import snapshot
        from app import __version__

        with self._data_lock:
            measurements = copy.deepcopy(self.measurements)
            latest = copy.deepcopy(self.latest_measurement)
        return {**snapshot(self), "measurements": measurements or ([latest] if latest else []),
                "software_version": __version__,
                "session_id": self.timeline.session_id,
                "timeline_path": str(self.timeline.path),
                "timeline_error": self.timeline.persistence_error,
                "performance": self.performance.snapshot(),
                "uncertainty_parameters": {
                    "micrometer_accuracy_mm": self.compute.micrometer_accuracy_mm,
                    "refractive_index_tolerance": self.compute.refractive_index_tolerance,
                    "confidence": self.compute.confidence}}

    # ------------------------------------------------------------------
    def run_async(self, work: Callable[[], Any], *, on_done: Callable[[Any], None],
                  on_error: Callable[[BaseException], None] | None = None,
                  name: str = "task", serial: bool = False) -> None:
        self._ticket += 1
        ticket = f"t{self._ticket}"
        self._pending[ticket] = (on_done, on_error)
        self.tasks.submit(
            work,
            on_done=lambda result: self.bridge.task_ok.emit(ticket, result),
            on_error=lambda exc: self.bridge.task_err.emit(ticket, str(exc)),
            name=name,
            serial=serial,
        )

    def _on_task_ok(self, ticket: str, result: Any) -> None:
        callbacks = self._pending.pop(ticket, None)
        if callbacks is None:
            return
        on_done, _ = callbacks
        if on_done is not None:
            on_done(result)

    def _on_task_err(self, ticket: str, message: str) -> None:
        callbacks = self._pending.pop(ticket, None)
        if callbacks is None:
            self.log(f"后台任务失败：{message}")
            return
        _, on_error = callbacks
        if on_error is not None:
            on_error(RuntimeError(message))
        else:
            self.log(f"后台任务失败：{message}")
