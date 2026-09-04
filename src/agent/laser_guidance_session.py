"""激光竖直条纹调节的只读步骤门控、自动比较与事件记录。"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from src.vision.fringe_adjustment import compare_fringe_adjustment


@dataclass(frozen=True)
class LaserGuidanceConfig:
    max_tilt_deg: float = 3.0
    min_bright_fringes: int = 4
    max_bright_fringes: int = 10
    consecutive_passes: int = 3
    settle_seconds: float = 1.0


class LaserGuidanceSession:
    """消费实时快照并给出唯一下一步；从不执行电机或旋钮动作。"""

    STEP_TITLES = (
        "等待有效画面", "等待连续条纹", "调直条纹",
        "调节条纹粗细", "等待条纹稳定", "调节完成",
    )

    def __init__(self, config: LaserGuidanceConfig | None = None):
        self.config = config or LaserGuidanceConfig()
        self._pass_count = 0
        self._stable_baseline: dict[str, Any] | None = None
        self._movement_seen = False
        self._settle_started_at: float | None = None
        self._last_comparison: dict[str, Any] | None = None
        self._last_signature: tuple | None = None
        self._events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.__init__(self.config)

    def events(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._events]

    def observe(
        self, context: dict[str, Any], *, now: float | None = None,
    ) -> dict[str, Any]:
        monotonic_now = time.monotonic() if now is None else float(now)
        camera = context.get("camera") or {}
        vision = context.get("vision") or {}
        guidance = vision.get("fringe_guidance") or {}
        alignment = guidance.get("laser_vertical_alignment") or {}
        metrics = dict(guidance.get("metrics") or {})
        metrics["quality_score"] = guidance.get("quality_score")
        movement = str(
            metrics.get("movement") or vision.get("fringe_movement") or "unknown")
        self._update_comparison(metrics, movement, monotonic_now)

        if (not camera.get("interferometer_running")
                or not vision.get("prediction_running")
                or not vision.get("roi_defined")):
            result = self._result(
                0, "blocked", "当前画面还不能支持旋钮判断",
                "打开干涉相机、启动预测，并在条纹区域框选 ROI。",
                "画面稳定且 ROI 内出现可分析内容。",
                "证据有效前不要转动任何动镜旋钮", metrics, {})
        elif not bool(vision.get("fringe_present")):
            result = self._result(
                1, "observing", "尚未检测到连续明暗条纹",
                "先调整光路，使两束返回光斑重合并让连续条纹进入 ROI。",
                "ROI 内出现连续、可辨认的明暗条纹。",
                "未出现连续条纹时不要按旋钮方向盲调", metrics, {})
        else:
            angle = self._number(metrics.get("angle_deg"))
            count = int(alignment.get("bright_fringe_count") or 0)
            spacing_valid = bool(alignment.get("spacing_valid", False))
            if angle is None:
                result = self._result(
                    1, "observing", "已看到条纹，正在积累可靠角度证据",
                    "保持装置不动，等待角度分析稳定。",
                    "界面给出可靠的有符号倾角。",
                    "没有可靠倾角时不要转动旋钮", metrics, {})
            elif abs(angle) > self.config.max_tilt_deg:
                self._pass_count = 0
                result = self._from_alignment(
                    2, "action_required", alignment, metrics)
            elif not spacing_valid or not (
                    self.config.min_bright_fringes <= count
                    <= self.config.max_bright_fringes):
                self._pass_count = 0
                result = self._from_alignment(
                    3, "action_required", alignment, metrics)
            elif movement != "stable":
                self._pass_count = 0
                result = self._result(
                    4, "evaluating", "检测到条纹仍在移动",
                    "立即松手，保持两个旋钮不动，等待系统自动比较。",
                    "条纹恢复稳定后显示本次调节是否有效。",
                    "稳定前不要继续转动", metrics, {})
            else:
                self._pass_count += 1
                required = self.config.consecutive_passes
                if self._pass_count < required:
                    result = self._result(
                        4, "observing",
                        f"指标已达标，正在连续复核（{self._pass_count}/{required}）",
                        "保持上方旋钮和下方旋钮都不动。",
                        "倾角、粗细和运动状态连续多帧通过。",
                        "任一指标退出范围就返回相应调节步骤", metrics, {})
                else:
                    result = self._result(
                        5, "passed", "激光条纹已经竖直、粗细合适且稳定",
                        "停止调节并保存当前检查点，随后准备切换白光。",
                        "检查点包含当前画面、ROI 和条纹指标。",
                        "换白光前不要再改变两个旋钮", metrics, {})
        self._record_state_change(result)
        return result

    def _from_alignment(
        self, step: int, state: str, alignment: dict, metrics: dict,
    ) -> dict[str, Any]:
        return self._result(
            step, state,
            str(alignment.get("observation") or "等待可靠条纹判断"),
            str(alignment.get("action") or "保持装置不动并等待分析"),
            str(alignment.get("expected_change") or "等待下一帧比较"),
            str(alignment.get("stop_condition") or "每次只调一个小步"),
            metrics, alignment)

    def _result(
        self, step: int, state: str, diagnosis: str, action: str,
        expected: str, stop: str, metrics: dict, alignment: dict,
    ) -> dict[str, Any]:
        count = int(alignment.get("bright_fringe_count") or 0)
        return {
            "step_number": step + 1,
            "total_steps": len(self.STEP_TITLES),
            "step_title": self.STEP_TITLES[step],
            "state": state,
            "diagnosis": diagnosis,
            "action": action,
            "expected_change": expected,
            "stop_condition": stop,
            "knob": alignment.get("knob"),
            "direction": alignment.get("direction"),
            "ready": state == "passed",
            "comparison": self._last_comparison,
            "metrics": {
                "angle_deg": self._number(metrics.get("angle_deg")),
                "spacing_px": self._number(metrics.get("spacing_px")),
                "spacing_cv_percent": self._number(
                    metrics.get("spacing_cv_percent")),
                "curvature": self._number(metrics.get("curvature")),
                "sharpness": self._number(metrics.get("sharpness")),
                "quality_score": self._number(metrics.get("quality_score")),
                "movement": metrics.get("movement"),
                "bright_fringe_count": count,
                "spacing_valid": bool(alignment.get("spacing_valid", False)),
            },
            "target": {
                "max_tilt_deg": self.config.max_tilt_deg,
                "min_bright_fringes": self.config.min_bright_fringes,
                "max_bright_fringes": self.config.max_bright_fringes,
                "consecutive_passes": self.config.consecutive_passes,
            },
            "read_only": True,
        }

    def _update_comparison(
        self, metrics: dict[str, Any], movement: str, now: float,
    ) -> None:
        if movement == "stable":
            if self._movement_seen:
                if self._settle_started_at is None:
                    self._settle_started_at = now
                if now - self._settle_started_at >= self.config.settle_seconds:
                    if self._stable_baseline:
                        comparison = compare_fringe_adjustment(
                            self._stable_baseline, metrics)
                        if comparison != self._last_comparison:
                            self._last_comparison = comparison
                            self._events.append({
                                "at": time.time(),
                                "event": "adjustment_comparison",
                                "comparison": dict(comparison),
                            })
                    self._stable_baseline = dict(metrics)
                    self._movement_seen = False
                    self._settle_started_at = None
            elif self._stable_baseline is None:
                self._stable_baseline = dict(metrics)
        elif movement in {"left", "right", "moving"}:
            self._movement_seen = True
            self._settle_started_at = None

    def _record_state_change(self, result: dict[str, Any]) -> None:
        signature = (
            result.get("step_number"), result.get("state"),
            result.get("knob"), result.get("direction"))
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self._events.append({
            "at": time.time(), "event": "state_changed",
            "step_number": result.get("step_number"),
            "step_title": result.get("step_title"),
            "state": result.get("state"),
        })

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            return round(float(value), 4) if value is not None else None
        except (TypeError, ValueError):
            return None
