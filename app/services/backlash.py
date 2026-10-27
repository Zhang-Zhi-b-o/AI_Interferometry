"""旧版五阶段回程差实验的无 GUI 适配。所有位置均为微分表读数 mm。"""
from __future__ import annotations

import math
import threading
import time


class BacklashService:
    def __init__(self, *, reading, motion, center_provider, busy=lambda: False,
                 on_update=None, on_finish=None, ready=lambda: ""):
        self._reading, self._motion = reading, motion
        self._center_provider, self._busy = center_provider, busy
        self._on_update, self._on_finish = on_update, on_finish
        self._ready = ready
        self._lock = motion.control_lock
        self._thread = None
        self._event = threading.Event()
        self.running = False
        self.phase = "idle"
        self.forward_mm = self.backward_mm = None
        self._reason = ""

    @property
    def active(self):
        return self.running or bool(self._thread and self._thread.is_alive())

    def start(self, *, start_mm, end_mm, gear=10, tolerance_mm=.001,
              endpoint_tolerance_mm=.01, center_tolerance_px=8, tick_ms=250,
              max_seconds=60, reading_timeout=1.5):
        with self._lock:
            if self.active or self._busy():
                return False, "已有运动闭环运行或正在停止"
            values = (start_mm, end_mm, gear, tolerance_mm, endpoint_tolerance_mm,
                      center_tolerance_px, tick_ms, max_seconds, reading_timeout)
            if any(not math.isfinite(float(v)) for v in values):
                return False, "回程差参数必须为有限数值"
            if end_mm - start_mm < .001 or not 1 <= gear <= 10 or min(values[3:]) <= 0:
                return False, "终点须比起点至少大 0.001 mm，档位为 1–10，容差及时间须为正数"
            if not self._motion.connected or not self._reading.connected:
                return False, "请先连接电机并启动微分表读数"
            problem = self._ready()
            if problem:
                return False, problem
            current = self._fresh(reading_timeout)
            if current is None:
                return False, "请等待新的可信微分表读数"
            self._event = threading.Event()
            self._reason = ""
            self.forward_mm = self.backward_mm = None
            self._center_after = time.monotonic()
            self._last_center_at = None
            self.phase, self.running = "move_to_start", True
            self._thread = threading.Thread(target=self._loop, kwargs=dict(
                start=float(start_mm), end=float(end_mm), gear=int(gear),
                tolerance=max(float(tolerance_mm), float(endpoint_tolerance_mm)),
                center_tolerance=float(center_tolerance_px), interval=float(tick_ms)/1000,
                max_seconds=float(max_seconds), timeout=float(reading_timeout),
                approach="forward" if current < start_mm else "reverse"),
                name="backlash", daemon=True)
            self._thread.start()
            return True, "回程差测量已启动"

    def _fresh(self, timeout):
        value, at = getattr(self._reading, "latest_stable", self._reading.latest)()
        age = time.time() - at
        if value is None or not math.isfinite(float(value)) or not 0 <= age <= timeout:
            return None
        # OCR timestamps use wall time; center timestamps use monotonic time.
        self._reading_at_mono = time.monotonic() - age
        return float(value)

    @staticmethod
    def endpoint(current, target, tolerance, direction):
        return current >= target-tolerance if direction == "forward" else current <= target+tolerance

    def _aligned(self, tolerance, timeout):
        sample = self._center_provider()
        if not sample or len(sample) != 3:
            return False
        center, width, at = sample
        aligned = (center is not None and width and 0 <= time.monotonic()-at <= 1.5
                   and at >= self._center_after
                   and self._reading_at_mono >= self._center_after
                   and (self._last_center_at is None or at > self._last_center_at)
                   and abs(at-self._reading_at_mono) <= min(timeout, .5)
                   and abs(center-width/2) <= tolerance)
        if aligned:
            self._last_center_at = at
        return aligned

    def stop(self, reason="手动停止"):
        with self._lock:
            if not self.active:
                return
            self._reason = reason
            self._event.set()
            self._motion.stop()
            self.running = False

    def _emit(self, reason=""):
        if self._on_update:
            self._on_update(dict(running=self.running, phase=self.phase, reason=reason,
                forward_mm=self.forward_mm, backward_mm=self.backward_mm,
                backlash_mm=(abs(self.forward_mm-self.backward_mm)
                    if self.forward_mm is not None and self.backward_mm is not None else None)))

    def _loop(self, *, start, end, gear, tolerance, center_tolerance, interval,
              max_seconds, timeout, approach):
        event, started, lost_at, direction = self._event, time.monotonic(), None, None
        reason = "已完成"
        try:
            while not event.is_set():
                now = time.monotonic()
                if now-started > max_seconds:
                    reason = "达到最大运行时间"
                    break
                if not self._motion.connected or not self._reading.connected:
                    reason = "电机或读数设备失联"
                    break
                current = self._fresh(timeout)
                with self._lock:
                    if event.is_set():
                        break
                    if current is None:
                        self._motion.stop()
                        direction = None
                        lost_at = now if lost_at is None else lost_at
                        if now-lost_at > timeout*3:
                            reason = "可信微分表读数持续丢失"
                            break
                        self._emit("等待可信读数，电机已停车")
                    else:
                        lost_at = None
                        if self.phase == "move_to_start" and self.endpoint(current,start,tolerance,approach):
                            self.phase = "forward"
                        elif self.phase == "forward":
                            if self._aligned(center_tolerance, timeout):
                                self.forward_mm = current
                                self.phase = "to_end"
                            elif self.endpoint(current,end,tolerance,"forward"):
                                self.phase = "backward"
                        elif self.phase == "to_end" and self.endpoint(current,end,tolerance,"forward"):
                            self.phase = "backward"
                        elif self.phase == "backward":
                            if self._aligned(center_tolerance, timeout):
                                self.backward_mm = current
                                self.phase = "to_start"
                            elif self.endpoint(current,start,tolerance,"reverse"):
                                break
                        elif self.phase == "to_start" and self.endpoint(current,start,tolerance,"reverse"):
                            break
                        wanted = approach if self.phase == "move_to_start" else (
                            "forward" if self.phase in {"forward","to_end"} else "reverse")
                        if direction != wanted:
                            self._motion.set_speed(gear)
                            getattr(self._motion, wanted)()
                            direction = wanted
                            self._center_after = time.monotonic()
                        self._emit()
                event.wait(interval)
        except Exception as exc:
            reason = "回程差测量失败：" + str(exc)
        finally:
            with self._lock:
                event.set()
                self._motion.stop()
                self.running = False
                reason = self._reason or reason
                self.phase = "done"
            if reason == "已完成" and (self.forward_mm is None or self.backward_mm is None):
                reason = "已完成行程，但未获得双向中心对齐读数"
            self._emit(reason)
            if self._on_finish:
                self._on_finish(reason)
