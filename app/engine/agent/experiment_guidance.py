"""实验意图、现场审计与统一指导决策。

本模块只读取运行时快照，不执行设备动作，也不重新计算厚度结果。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


INTENT_LABELS = {
    "white_light_centering": "寻找白光中心条纹",
    "fringe_observation": "观察并调出清晰干涉条纹",
    "fringe_spacing": "调节并测量条纹间距",
    "glass_thickness": "完成玻璃片厚度实验",
}

INTENT_READY_ACTIONS = {
    "white_light_centering": "核对中心位置和新鲜读数，并保存中心条纹画面",
    "fringe_observation": "保存当前原始画面并记录条纹形态、光源和调节条件",
    "fringe_spacing": "记录当前条纹间距及对应原始画面，随后完成重复测量",
    "glass_thickness": "按现有厚度实验流程核对读数并记录下一轮数据",
}


@dataclass(frozen=True)
class ExperimentIntent:
    kind: str = "white_light_centering"
    objective: str = "寻找稳定、清晰并居中的白光零级条纹"
    required_repeats: int = 5
    response_mode: str = "standard"
    confirmed: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "ExperimentIntent":
        raw = value or {}
        kind = str(raw.get("kind") or cls.kind)
        if kind not in INTENT_LABELS:
            kind = cls.kind
        mode = str(raw.get("response_mode") or "standard")
        if mode not in {"quiet", "standard", "teaching"}:
            mode = "standard"
        try:
            repeats = max(1, min(100, int(raw.get("required_repeats", 5))))
        except (TypeError, ValueError):
            repeats = 5
        objective = str(raw.get("objective") or INTENT_LABELS[kind]).strip()
        return cls(
            kind=kind,
            objective=objective or INTENT_LABELS[kind],
            required_repeats=repeats,
            response_mode=mode,
            confirmed=bool(raw.get("confirmed", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentIssue:
    code: str
    severity: str
    message: str
    evidence: str
    recovery: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GuidanceDecision:
    objective: str
    stage: str
    priority: str
    diagnosis: str
    evidence: tuple[str, ...]
    action: str
    expected_change: str
    completion_criterion: str
    issues: tuple[ExperimentIssue, ...]
    can_record: bool

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.objective,
            self.stage,
            self.priority,
            tuple(issue.code for issue in self.issues),
            self.action,
            self.can_record,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["semantic_key"] = list(self.semantic_key)
        return result


def audit_experiment(context: dict[str, Any]) -> tuple[ExperimentIssue, ...]:
    """按安全和数据有效性优先级审查现场快照。"""
    camera = context.get("camera", {}) or {}
    vision = context.get("vision", {}) or {}
    motor = context.get("motor", {}) or {}
    micrometer = context.get("micrometer", {}) or {}
    guidance = vision.get("fringe_guidance") or {}
    issues: list[ExperimentIssue] = []

    if motor.get("auto_enabled") and not motor.get("connected"):
        issues.append(ExperimentIssue(
            "MOTOR_DISCONNECTED", "blocking", "自动运动期间电机连接已丢失",
            "自动控制已启用，但电机状态为未连接", "立即停车并检查串口连接"))

    age = micrometer.get("reading_age_seconds")
    age_value = _number(age)
    if micrometer.get("connected") and age_value is not None and age_value > 5.0:
        issues.append(ExperimentIssue(
            "STALE_MICROMETER", "blocking", "微分表读数已经过期",
            f"读数已 {age_value:.1f} 秒未刷新", "等待新的稳定读数后再记录"))

    fps = _number(camera.get("fps")) or 0.0
    if camera.get("interferometer_running") and fps < 5.0:
        issues.append(ExperimentIssue(
            "LOW_CAMERA_RATE", "warning", "干涉画面刷新率过低",
            f"当前 FPS 为 {fps:.1f}",
            "检查相机连接、分辨率和计算负载"))

    if vision.get("prediction_running") and not vision.get("fringe_present"):
        issues.append(ExperimentIssue(
            "FRINGE_NOT_FOUND", "warning", "当前没有可靠识别到干涉条纹",
            "预测正在运行，但条纹存在判据未通过",
            "检查光路、曝光和 ROI，先让连续条纹进入画面"))

    severity_map = {"high": "blocking", "medium": "warning", "low": "notice"}
    for index, item in enumerate((guidance.get("issues") or [])[:3]):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        issues.append(ExperimentIssue(
            _fringe_issue_code(text, index),
            severity_map.get(str(item.get("severity")), "warning"),
            text,
            str(guidance.get("summary") or "来自实时条纹诊断"),
            str((guidance.get("recommendations") or ["按条纹诊断恢复画面质量"])[0]),
        ))

    intent = ExperimentIntent.from_mapping(context.get("experiment_intent"))
    measurement = context.get("measurement", {}) or {}
    if intent.kind == "glass_thickness":
        assistant = measurement.get("experiment_assistant", {}) or {}
        count = len(((assistant.get("session") or {}).get("rounds") or []))
    else:
        count = _integer(measurement.get("record_count")) or 0
    if 0 < count < intent.required_repeats:
        issues.append(ExperimentIssue(
            "INSUFFICIENT_REPEATS", "notice", "有效重复测量次数尚不足",
            f"当前 {count} 轮，计划 {intent.required_repeats} 轮",
            f"保持条件一致并继续完成至少 {intent.required_repeats} 轮测量"))

    order = {"blocking": 0, "warning": 1, "notice": 2}
    return tuple(sorted(issues, key=lambda issue: order.get(issue.severity, 9)))


def build_guidance_decision(context: dict[str, Any]) -> GuidanceDecision:
    """把实验目的、进度、视觉证据和数据审查合成为单一下一步。"""
    intent = ExperimentIntent.from_mapping(context.get("experiment_intent"))
    progress = context.get("experiment_progress", {}) or {}
    vision = context.get("vision", {}) or {}
    guidance = vision.get("fringe_guidance") or {}
    metrics = guidance.get("metrics") or {}
    issues = audit_experiment(context)
    stage = str(progress.get("stage") or "等待实时状态")
    criterion = str(progress.get("completion_criterion") or "现场状态满足当前步骤要求")

    if issues:
        primary = issues[0]
        diagnosis = primary.message
        action = primary.recovery
        expected = "该问题解除后重新评估条纹质量和实验进度"
        priority = primary.severity
    elif guidance.get("measurement_ready"):
        diagnosis = "条纹清晰、稳定且质量门已经通过"
        action = INTENT_READY_ACTIONS[intent.kind]
        expected = "形成一条带时间戳、读数和画面状态的有效记录"
        priority = "ready"
    else:
        diagnosis = str(guidance.get("summary") or stage)
        recommendations = guidance.get("recommendations") or []
        action = str(recommendations[0] if recommendations else (
            progress.get("next_action") or "等待实时状态更新"))
        expected = "相关指标改善并达到当前阶段完成判据"
        priority = "normal"

    evidence = []
    angle = _number(metrics.get("angle_deg"))
    spacing = _number(metrics.get("spacing_px"))
    spacing_cv = _number(metrics.get("spacing_cv_percent"))
    if angle is not None:
        evidence.append(f"条纹倾角 {angle:+.1f}°")
    if spacing is not None:
        evidence.append(f"法向间距 {spacing:.2f} px")
    if spacing_cv is not None:
        evidence.append(f"间距波动 {spacing_cv:.1f}%")
    if metrics.get("movement"):
        evidence.append(f"运动状态 {metrics['movement']}")
    evidence.extend(issue.evidence for issue in issues[:2])

    can_record = bool(
        guidance.get("measurement_ready")
        and not any(issue.severity == "blocking" for issue in issues))
    return GuidanceDecision(
        objective=intent.objective,
        stage=stage,
        priority=priority,
        diagnosis=diagnosis,
        evidence=tuple(evidence[:4]),
        action=action,
        expected_change=expected,
        completion_criterion=criterion,
        issues=issues,
        can_record=can_record,
    )


def render_guidance_decision(decision: GuidanceDecision) -> str:
    """渲染适合实时卡片的短文本，不消耗模型 token。"""
    evidence = "；".join(decision.evidence[:2]) or "等待新的现场证据"
    return (
        f"判断：{decision.diagnosis}\n"
        f"依据：{evidence}\n"
        f"操作：{decision.action}\n"
        f"完成：{decision.completion_criterion}"
    )


def _fringe_issue_code(text: str, index: int) -> str:
    categories = (
        (("模糊", "清晰"), "BLUR"),
        (("曝光", "亮度"), "EXPOSURE"),
        (("移动", "运动"), "MOTION"),
        (("倾斜", "倾角"), "TILT"),
        (("弯曲", "曲率"), "CURVATURE"),
        (("间距", "条纹数"), "SPACING"),
        (("中心", "偏离"), "CENTER"),
        (("识别", "置信"), "RECOGNITION"),
    )
    for keywords, code in categories:
        if any(keyword in text for keyword in keywords):
            return f"FRINGE_{code}"
    return f"FRINGE_OTHER_{index}"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
