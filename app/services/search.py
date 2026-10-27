"""旧 CenterControlStateMachine 的无 GUI 执行适配：观察、决策、串行下发。"""
from __future__ import annotations

from dataclasses import asdict
import math
import threading
import time

from app.engine.control.center_control import CenterControlStateMachine
from app.services.target_reading import TargetReadingService  # noqa: F401


class SearchService:
    def __init__(self, *, frame_provider, motion, observe, ready=lambda: "",
                 busy=lambda: False, settings=None, on_update=None, on_finish=None):
        self._frame_provider = frame_provider
        self._motion = motion
        self._observe = observe
        self._ready = ready
        self._busy = busy
        self._settings = dict(settings or {})
        self._on_update = on_update
        self._on_finish = on_finish
        self._thread = None
        self._lock = motion.control_lock
        self._stop_event = threading.Event()
        self._reason = ""
        self.running = False
        self.forward_positive = True

    @property
    def active(self):
        return self.running or bool(self._thread and self._thread.is_alive())

    def start(self, *, tolerance_px=12.0, stable_frames=5, slow_zone_px=120.0,
              fast_gear=9, slow_gear=10, tick_ms=200, max_seconds=180.0,
              direction_mode="single_direction", recognition_mode="continuous",
              search_direction="forward", auto_learn_direction=False, **options):
        with self._lock:
            if self.active or self._busy():
                return False, "已有运动闭环运行或正在停止，请稍候"
            if not self._motion.connected:
                return False, "请先连接电机"
            problem = self._ready()
            if problem:
                return False, problem
            params = dict(self._settings)
            params.update(tolerance_px=tolerance_px, stable_frames=stable_frames,
                          slow_zone_px=slow_zone_px, fast_gear=fast_gear,
                          slow_gear=slow_gear, direction_mode=direction_mode,
                          recognition_mode=recognition_mode, search_direction=search_direction,
                          auto_learn_direction=auto_learn_direction,
                          invert_direction=not self.forward_positive)
            params.update(options)
            if (direction_mode not in {"single_direction", "bidirectional"}
                    or recognition_mode not in {"continuous", "stop_and_detect"}
                    or search_direction not in {"forward", "reverse"}):
                return False, "搜索方向或识别方式无效"
            numbers = [tolerance_px, stable_frames, slow_zone_px, fast_gear,
                       slow_gear, tick_ms, max_seconds]
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in numbers):
                return False, "寻中参数必须是有限正数"
            if not (1 <= fast_gear <= 10 and 1 <= slow_gear <= 10):
                return False, "电机档位范围为 1–10"
            self._stop_event = threading.Event()
            self._reason = ""
            self._observation_at = time.monotonic()
            if direction_mode == "single_direction" and not self.forward_positive:
                # 旧 invert 同时翻转搜索方向；v2 的起始方向表示实际电机指令。
                params["search_direction"] = "reverse" if search_direction == "forward" else "forward"
            self.running = True
            self._thread = threading.Thread(
                target=self._loop, args=(params, float(tick_ms) / 1000, float(max_seconds)),
                name="search", daemon=True)
            self._thread.start()
            return True, "自动寻中已启动"

    def stop(self, reason="手动停止"):
        with self._lock:
            if not self.active:
                return
            self._reason = reason
            self._stop_event.set()
            self._motion.stop()
            self.running = False
        self._emit(reason=reason, running=False)

    def _loop(self, params, interval, max_seconds):
        machine = CenterControlStateMachine()
        deadline = time.monotonic() + max_seconds
        event = self._stop_event
        reason = "自动寻中结束"
        machine.start(time.monotonic())

        def watchdog():
            while not event.wait(0.1):
                problem = self._ready()
                if time.monotonic() - self._observation_at > 1.5:
                    problem = "参与识别的画面已过期"
                if time.monotonic() >= deadline or problem or not self._motion.connected:
                    self.stop(problem or ("串口失联" if not self._motion.connected else "达到最大运行时间"))
                    return

        watcher = threading.Thread(target=watchdog, name="search-watchdog", daemon=True)
        watcher.start()
        try:
            if hasattr(self._observe, "reset"):
                self._observe.reset()
            while not event.is_set():
                sample = self._frame_provider()
                if isinstance(sample, tuple):
                    frame, self._observation_at = sample
                else:
                    frame, self._observation_at = sample, time.monotonic()
                if frame is None:
                    reason = "没有可用画面"
                    break
                observation = self._observe(frame)
                with self._lock:
                    # 图像推理期间可能已急停；迟到结果不得重新启动电机。
                    if event.is_set():
                        break
                    problem = self._ready()
                    if time.monotonic() - self._observation_at > 1.5:
                        problem = "参与识别的画面已过期"
                    if problem or time.monotonic() >= deadline:
                        reason = problem or "达到最大运行时间"
                        break
                    decision = machine.update(
                        **observation, frame_width=frame.shape[1],
                        connected=self._motion.connected, params=params,
                        safety={"max_run_seconds": max_seconds,
                                "max_missing_frames": self._settings.get("max_missing_frames", 90)},
                        now=time.monotonic())
                    for command, value in decision.commands:
                        if command == "set_speed":
                            self._motion.set_speed(value)
                        elif command == "start_forward":
                            self._motion.forward()
                        elif command == "start_reverse":
                            self._motion.reverse()
                        elif command == "stop":
                            self._motion.stop()
                    payload = asdict(decision)
                    payload.pop("commands")
                    payload.update(center=observation.get("center_x"),
                                   confidence=observation.get("confidence", 0.0),
                                   running=machine.enabled)
                    self._emit(**payload)
                    if not machine.enabled:
                        reason = decision.stopped_reason or decision.message
                        break
                if event.wait(interval):
                    break
        except Exception as exc:
            reason = "自动寻中失败：" + str(exc)
        finally:
            with self._lock:
                event.set()
                self._motion.stop()
                self.running = False
                reason = self._reason or reason
            self._emit(reason=reason, running=False)
            if self._on_finish:
                self._on_finish(reason)

    def _emit(self, **payload):
        if self._on_update:
            self._on_update(payload)
