"""白光干涉实验流程识别与自动推进状态机。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentObservation:
    """一次流程更新所需的只读实验状态。"""

    camera_running: bool = False
    model_loaded: bool = False
    prediction_running: bool = False
    motor_connected: bool = False
    micrometer_connected: bool = False
    micrometer_reading_mm: float | None = None
    center_x_px: float | None = None
    center_confidence: float = 0.0
    frame_width_px: float | None = None


@dataclass(frozen=True)
class ExperimentWorkflowDecision:
    """当前阶段、下一步和需要由 UI 执行的确定性动作。"""

    stage: str
    title: str
    next_action: str
    progress: int
    actions: tuple[str, ...] = ()
    warning: str = ""


STAGE_TITLES = {
    "manual_alignment": "人工调整仪器",
    "waiting_white_light": "等待放置白光光源",
    "initializing": "设备初始化",
    "ready": "等待启动自动实验",
    "searching": "自动寻找白光条纹",
    "centering": "将中心条纹移至画面中央",
    "confirming_center": "确认中心条纹",
    "waiting_micrometer": "等待微分表读数稳定",
    "center_found": "中心条纹定位完成",
    "application_ready": "等待白光干涉应用",
    "error": "实验暂停",
}


class ExperimentWorkflowStateMachine:
    """识别实验进度，并在自动模式下产生搜索与停车动作。"""

    def __init__(self):
        self.instrument_adjusted = False
        self.white_light_placed = False
        self.auto_enabled = False
        self.stage = "manual_alignment"
        self.started_at: float | None = None
        self.search_started = False
        self.center_recorded = False
        self.reference_reading_mm: float | None = None
        self.center_reading_mm: float | None = None
        self.center_x_px: float | None = None
        self._last_center_x: float | None = None
        self._stable_center_frames = 0
        self._center_waiting_for_meter = False
        self.warning = ""

    def confirm_instrument_adjusted(self, confirmed: bool = True) -> None:
        self.instrument_adjusted = bool(confirmed)
        if not confirmed:
            self.reset_automatic_state()

    def confirm_white_light_placed(self, confirmed: bool = True) -> None:
        self.white_light_placed = bool(confirmed)
        if not confirmed:
            self.reset_automatic_state()

    def set_auto_enabled(self, enabled: bool, now: float | None = None) -> None:
        self.auto_enabled = bool(enabled)
        if enabled:
            self.started_at = now
            self.warning = ""
        else:
            self.reset_automatic_state(keep_result=True)

    def reset(self) -> None:
        self.instrument_adjusted = False
        self.white_light_placed = False
        self.auto_enabled = False
        self.reference_reading_mm = None
        self.center_reading_mm = None
        self.center_x_px = None
        self.center_recorded = False
        self.reset_automatic_state()

    def reset_automatic_state(self, keep_result: bool = False) -> None:
        self.started_at = None
        self.search_started = False
        self._last_center_x = None
        self._stable_center_frames = 0
        self._center_waiting_for_meter = False
        self.warning = ""
        if not keep_result:
            self.center_recorded = False
            self.center_reading_mm = None
            self.center_x_px = None

    def snapshot(self) -> dict:
        displacement = None
        if self.reference_reading_mm is not None and self.center_reading_mm is not None:
            displacement = self.center_reading_mm - self.reference_reading_mm
        decision = self._describe_without_observation()
        return {
            "stage": decision.stage,
            "stage_title": decision.title,
            "next_action": decision.next_action,
            "progress_percent": decision.progress,
            "auto_enabled": self.auto_enabled,
            "instrument_adjusted": self.instrument_adjusted,
            "white_light_placed": self.white_light_placed,
            "reference_reading_mm": self.reference_reading_mm,
            "center_reading_mm": self.center_reading_mm,
            "reading_change_mm": displacement,
            "center_x_px": self.center_x_px,
            "warning": self.warning,
        }

    def update(
        self,
        observation: ExperimentObservation,
        now: float,
        *,
        max_seconds: float = 600.0,
        stable_frames: int = 5,
        center_min_confidence: float = 0.18,
        center_max_jitter_px: float = 12.0,
        center_target_tolerance_px: float = 15.0,
    ) -> ExperimentWorkflowDecision:
        actions: list[str] = []

        if not self.instrument_adjusted:
            return self._decision(
                "manual_alignment", "请先用红光调整仪器，完成后点击确认。", 10)
        if not self.white_light_placed:
            return self._decision(
                "waiting_white_light", "请放置白光光源，完成后点击确认。", 25)

        missing = []
        if not observation.camera_running:
            missing.append("相机")
        if not observation.model_loaded:
            missing.append("YOLO 模型")
        if not observation.prediction_running:
            missing.append("实时预测")
        if not observation.motor_connected:
            missing.append("电机")
        if missing:
            if self.search_started:
                actions.append("stop_search")
                self.search_started = False
            return self._decision(
                "initializing", "等待程序连接：" + "、".join(missing), 40,
                tuple(actions))

        if (self.reference_reading_mm is None
                and observation.micrometer_reading_mm is not None):
            self.reference_reading_mm = observation.micrometer_reading_mm

        if self.center_recorded:
            return self._decision(
                "application_ready", "中心位置已保存，等待选择白光干涉应用。", 100)

        if not self.auto_enabled:
            return self._decision(
                "ready", "打开“自动进行实验”开关，程序将开始寻找中心条纹。", 55)

        if self.started_at is None:
            self.started_at = now
        if now - self.started_at > max(1.0, float(max_seconds)):
            self.warning = "自动实验超过最大运行时间"
            if self.search_started:
                actions.append("stop_search")
                self.search_started = False
            return self._decision(
                "error", "检查光路和白光位置，确认后重新启动自动实验。", 55,
                tuple(actions), self.warning)

        if self._center_waiting_for_meter:
            if (observation.micrometer_connected
                    and observation.micrometer_reading_mm is None):
                return self._decision(
                    "waiting_micrometer",
                    "电机已停车，正在等待视觉微分表连续读数稳定。",
                    92, tuple(actions))
            self.center_recorded = True
            self.center_reading_mm = observation.micrometer_reading_mm
            self._center_waiting_for_meter = False
            actions.append("record_center")
            return self._decision(
                "center_found", "中心条纹和微分表读数已保存。", 95,
                tuple(actions))

        if not self.search_started:
            self.search_started = True
            actions.append("start_search")

        valid_center = (
            observation.center_x_px is not None
            and observation.center_confidence >= float(center_min_confidence)
        )
        if not valid_center:
            self._stable_center_frames = 0
            self._last_center_x = None
            return self._decision(
                "searching", "程序正在驱动电机寻找白光中心条纹。", 70,
                tuple(actions))

        center_x = float(observation.center_x_px)
        if observation.frame_width_px is not None and observation.frame_width_px > 0:
            target_x = float(observation.frame_width_px) / 2.0
            target_error = center_x - target_x
            if abs(target_error) > max(1.0, float(center_target_tolerance_px)):
                self._stable_center_frames = 0
                self._last_center_x = center_x
                return self._decision(
                    "centering",
                    f"中心条纹偏离画面中央 {target_error:+.1f} px，程序正在自动调整。",
                    80, tuple(actions))

        if (self._last_center_x is None
                or abs(center_x - self._last_center_x) <= float(center_max_jitter_px)):
            self._stable_center_frames += 1
        else:
            self._stable_center_frames = 1
        self._last_center_x = center_x

        if self._stable_center_frames < max(1, int(stable_frames)):
            return self._decision(
                "confirming_center",
                f"正在确认中心稳定性（{self._stable_center_frames}/{max(1, int(stable_frames))}）。",
                85, tuple(actions))

        self.center_x_px = center_x
        self.search_started = False
        actions.append("stop_search")
        if (observation.micrometer_connected
                and observation.micrometer_reading_mm is None):
            self._center_waiting_for_meter = True
            return self._decision(
                "waiting_micrometer",
                "中心条纹已稳定并停车，正在等待视觉微分表读数稳定。",
                92, tuple(actions))
        self.center_recorded = True
        self.center_reading_mm = observation.micrometer_reading_mm
        actions.append("record_center")
        return self._decision(
            "center_found", "中心条纹已确认，正在停车并保存实验数据。", 95,
            tuple(actions))

    def _describe_without_observation(self) -> ExperimentWorkflowDecision:
        if not self.instrument_adjusted:
            return self._decision(
                "manual_alignment", "请先用红光调整仪器，完成后点击确认。", 10)
        if not self.white_light_placed:
            return self._decision(
                "waiting_white_light", "请放置白光光源，完成后点击确认。", 25)
        if self.center_recorded:
            return self._decision(
                "application_ready", "中心位置已保存，等待选择白光干涉应用。", 100)
        if not self.auto_enabled:
            return self._decision(
                "ready", "打开“自动进行实验”开关。", 55)
        return self._decision(self.stage, "程序正在根据实时状态推进实验。", 70)

    def _decision(
        self,
        stage: str,
        next_action: str,
        progress: int,
        actions: tuple[str, ...] = (),
        warning: str = "",
    ) -> ExperimentWorkflowDecision:
        self.stage = stage
        return ExperimentWorkflowDecision(
            stage=stage,
            title=STAGE_TITLES[stage],
            next_action=next_action,
            progress=max(0, min(100, int(progress))),
            actions=actions,
            warning=warning,
        )
