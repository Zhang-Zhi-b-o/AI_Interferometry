"""目标读数闭环，由 search 模块保留兼容导出。"""
from __future__ import annotations

import math
import threading
import time


class TargetReadingService:
    """把微分表转到指定读数（简化版：只用读数误差决定方向与档位）。"""

    def __init__(self, *, reading, motion, busy=lambda: False, on_update=None, on_finish=None) -> None:
        self._lock = motion.control_lock
        self._busy = busy
        self._reading = reading
        self._motion = motion
        self._on_update = on_update
        self._on_finish = on_finish
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.running = False

    @property
    def active(self):
        return self.running or bool(self._thread and self._thread.is_alive())

    def start(self, **kwargs):
        with self._lock:
            if self.active or self._busy():
                return False, "已有运动闭环运行或正在停止，请稍候"
            return self._start(**kwargs)

    def _start(self, *, target_mm: float, tolerance_mm: float = 0.001,
              gear: int = 10, tick_ms: int = 250, max_seconds: float = 120.0,
              invert: bool = False) -> tuple[bool, str]:
        if self.running:
            return False, "目标读数已在运行"
        if not getattr(self._motion, "connected", False):
            return False, "请先连接电机"
        if not getattr(self._reading, "connected", False):
            return False, "请先启动微分表读数"
        values = (target_mm, tolerance_mm, tick_ms, max_seconds)
        if any(not math.isfinite(float(v)) for v in values) or min(tolerance_mm, tick_ms, max_seconds) <= 0:
            return False, "目标读数及控制参数必须是有限有效数值"
        if not 1 <= int(gear) <= 10:
            return False, "电机档位范围为 1–10"
        value, at = self._reading.latest()
        if value is None or not math.isfinite(float(value)) or not 0 <= time.time() - at <= 1.5:
            return False, "请等待新的微分表读数（有效期 1.5 秒）"
        self._stop_event.clear()
        self._stop_reason = ""
        self.running = True
        self._thread = threading.Thread(
            target=self._loop,
            kwargs=dict(target_mm=float(target_mm), tolerance_mm=float(tolerance_mm),
                        gear=int(gear), tick_ms=int(tick_ms),
                        max_seconds=float(max_seconds), invert=bool(invert)),
            name="target-reading", daemon=True,
        )
        self._thread.start()
        return True, f"正在转向 {target_mm:.3f} mm"

    def stop(self, reason: str = "手动停止") -> None:
        with self._lock:
            if not self.active:
                return
            self._stop_reason = reason
            self._stop_event.set()
            self._motion.stop()
            self.running = False
        self._emit(reason=reason, running=False)

    def _loop(self, *, target_mm, tolerance_mm, gear, tick_ms, max_seconds, invert) -> None:
        started = time.monotonic()
        reason = "已到达"
        last_direction = None
        try:
            while not self._stop_event.is_set():
                if time.monotonic() - started > max_seconds:
                    reason = "超时停止"
                    break
                if not self._motion.connected or not self._reading.connected:
                    reason = "电机或读数设备失联"
                    break
                value, at = self._reading.latest()
                if (value is None or not math.isfinite(float(value))
                        or not 0 <= time.time() - at <= 1.5):
                    reason = "微分表读数无效或超过 1.5 秒未更新"
                    break
                err = target_mm - value
                if abs(err) <= tolerance_mm:
                    reason = "已到达"
                    break
                with self._lock:
                    if self._stop_event.is_set():
                        break
                    self._motion.set_speed(gear)
                    want_forward = (err > 0) != invert
                    if want_forward:
                        self._motion.forward()
                        last_direction = "forward"
                    else:
                        self._motion.reverse()
                        last_direction = "reverse"
                self._emit(value=value, error=err, direction=last_direction)
                self._stop_event.wait(min(tick_ms / 1000.0, 0.25))
        finally:
            with self._lock:
                self._motion.stop()
                self.running = False
                reason = self._stop_reason or reason
            self._emit(reason=reason, running=False)
            if self._on_finish is not None:
                try:
                    self._on_finish(reason)
                except Exception:
                    pass

    def _emit(self, **payload) -> None:
        if self._on_update is not None:
            try:
                self._on_update(payload)
            except Exception:
                pass
