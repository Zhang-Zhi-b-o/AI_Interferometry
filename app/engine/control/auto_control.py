"""可独立测试的电机自动寻零状态机。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoControlDecision:
    """一次视觉更新产生的硬件命令和界面事件。"""

    commands: tuple[tuple[str, int | None], ...] = ()
    stopped_reason: str = ""
    status: str = ""
    log: str = ""


class AutoControlStateMachine:
    """只计算状态转换，不直接访问串口或 Tkinter。"""

    def __init__(self):
        self.enabled = False
        self.mode = "manual"
        self.stage = "idle"
        self.phase = "idle"
        self.started_at = 0.0
        self.phase_started_at = 0.0
        self.phase_ms = 1000
        self.black_frames = 0
        self.missing_frames = 0
        self.best_black_conf = 0.0

    def start(self, mode: str, now: float) -> None:
        self.enabled = True
        self.mode = mode
        self.stage = "idle"
        self.phase = "idle"
        self.started_at = now
        self.phase_started_at = now
        self.phase_ms = 1000
        self.black_frames = 0
        self.missing_frames = 0
        self.best_black_conf = 0.0

    def stop(self, reason: str) -> AutoControlDecision:
        was_enabled = self.enabled
        self.enabled = False
        self.stage = "idle"
        self.phase = "idle"
        commands = (("stop", None),) if was_enabled else ()
        return AutoControlDecision(commands, reason if was_enabled else "")

    def update(
        self,
        *,
        color_conf: float,
        black_conf: float,
        connected: bool,
        params: dict,
        safety: dict,
        now: float,
    ) -> AutoControlDecision:
        if not self.enabled:
            return AutoControlDecision()
        if not connected:
            return self.stop("串口失联")

        max_run_s = max(0.1, float(safety.get("max_run_seconds", 60)))
        confirm_frames = max(1, int(safety.get("black_confirm_frames", 3)))
        max_missing = max(1, int(safety.get("max_missing_frames", 30)))
        if now - self.started_at > max_run_s:
            return self.stop("达到最大运行时间")

        if color_conf <= 0 and black_conf <= 0:
            self.missing_frames += 1
            if self.missing_frames >= max_missing:
                return self.stop("连续未检测到条纹")
        else:
            self.missing_frames = 0

        if self.mode == "step":
            return self._update_step(black_conf, params, confirm_frames, now)
        if self.mode == "continuous":
            return self._update_continuous(
                color_conf, black_conf, params, confirm_frames)
        return self.stop("自动模式无效")

    def _update_step(self, black_conf: float, params: dict,
                     confirm_frames: int, now: float) -> AutoControlDecision:
        if self.phase == "idle":
            self.phase = "move"
            self.phase_started_at = now
            self.phase_ms = (params["first_ms"] if self.best_black_conf == 0
                             else params["cycle_ms"])
            return AutoControlDecision((("set_speed", params["speed"]),
                                        ("start", None)))
        if self.phase == "move" and now - self.phase_started_at > self.phase_ms / 1000.0:
            self.phase = "pause"
            self.phase_started_at = now
            return AutoControlDecision((("stop", None),))
        if self.phase == "pause" and now - self.phase_started_at > params["pause_ms"] / 1000.0:
            self.best_black_conf = max(self.best_black_conf, black_conf)
            self.black_frames = self.black_frames + 1 if black_conf > params["black_threshold"] else 0
            if self.black_frames >= confirm_frames:
                self.phase = "locked"
                return AutoControlDecision(
                    (("stop", None),), status="自动控制: 已锁定", log="[AUTO] 步进锁定")
            self.phase = "idle"
        return AutoControlDecision()

    def _update_continuous(self, color_conf: float, black_conf: float,
                           params: dict, confirm_frames: int) -> AutoControlDecision:
        threshold = params["black_threshold"]
        if self.stage == "idle":
            self.stage = "searching"
            return AutoControlDecision((("set_speed", params["search_speed"]),
                                        ("start", None)))
        if self.stage == "searching" and color_conf > 0.3:
            self.stage = "color"
            return AutoControlDecision((("set_speed", params["color_speed"]),))
        if self.stage == "color":
            if black_conf > threshold:
                self.black_frames += 1
                if self.black_frames >= confirm_frames:
                    self.stage = "black"
                    return AutoControlDecision(
                        (("set_speed", params["black_speed"]),),
                        log="[AUTO] 连续检测到黑条")
            else:
                self.black_frames = 0
                if color_conf <= 0.3:
                    self.stage = "searching"
                    return AutoControlDecision((("set_speed", params["search_speed"]),))
        elif self.stage == "black":
            if black_conf > threshold:
                self.black_frames += 1
                if self.black_frames >= confirm_frames * 2:
                    self.stage = "locked"
                    return AutoControlDecision(
                        (("stop", None),), status="自动控制: 已锁定", log="[AUTO] 黑条锁定")
            else:
                self.black_frames = 0
                if color_conf > 0.3:
                    self.stage = "color"
                    return AutoControlDecision((("set_speed", params["color_speed"]),))
                self.stage = "searching"
                return AutoControlDecision((("set_speed", params["search_speed"]),))
        return AutoControlDecision()
