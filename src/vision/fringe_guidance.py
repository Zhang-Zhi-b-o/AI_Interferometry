"""实时条纹诊断与只读操作指导。

本模块只根据视觉状态生成诊断和建议，不产生、发送或执行任何电机命令。
重型二维几何分析可放在线程池中低频运行；逐帧指导生成保持为轻量纯函数。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.vision.fringe_angle import estimate_fringe_angle_2d
from src.vision.fringe_width import (
    measure_center_fringe_width_2d,
    measure_fringe_spacing_2d,
)


def analyse_guidance_geometry(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """低频分析角度、曲率和法向间距，供实时指导器复用。"""
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    image = bgr
    roi_rect = None
    if roi is not None:
        height, width = bgr.shape[:2]
        x, y, w, h = (int(value) for value in roi)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        image = bgr[y:y + h, x:x + w]
        roi_rect = [x, y, w, h]

    analysis_2d = measure_center_fringe_width_2d(image)
    angle = estimate_fringe_angle_2d(image, analysis_2d=analysis_2d)
    spacing = measure_fringe_spacing_2d(
        image, angle_deg=angle.get("tilt_deg"), analysis_2d=analysis_2d)
    return {"roi": roi_rect, "angle": angle, "spacing": spacing}


def build_fringe_guidance(
    *,
    recognition: dict[str, Any] | None,
    motion: dict[str, Any] | None,
    texture: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    clarity: dict[str, Any] | None = None,
    center_x: float | None = None,
    frame_width: int | None = None,
    motor_connected: bool = False,
    auto_enabled: bool = False,
    current_correction_deg: float = 0.0,
    motion_enhancement_enabled: bool = False,
) -> dict[str, Any]:
    """把实时视觉证据整理为可解释、无执行副作用的操作指导。"""
    recognition = recognition or {}
    motion = motion or {}
    texture = texture or {}
    geometry = geometry or {}
    clarity = clarity or {}
    angle = geometry.get("angle") or {}
    spacing = geometry.get("spacing") or {}

    has_fringe = bool(
        recognition.get("has_fringe", motion.get("has_fringe", False)))
    recognition_confidence = _unit(
        recognition.get("confidence", motion.get("recognition_confidence", 0.0)))
    sharpness = _unit(texture.get("sharpness", 0.0))
    blurred = bool(
        recognition.get("blurred", motion.get("blurred", False))
        or texture.get("blurred", False))
    held = bool(recognition.get("held", motion.get("held", False)))
    movement = str(motion.get("movement") or "unknown")
    velocity = _finite_float(
        recognition.get("velocity_px_s", motion.get("velocity_px_s", 0.0)), 0.0)
    brightness = clarity.get("brightness")

    issues: list[dict[str, str]] = []
    recommendations: list[str] = []
    actions: list[dict[str, Any]] = []

    if not has_fringe:
        issues.append(_issue("high", "未可靠识别到干涉条纹"))
        recommendations.extend([
            "检查光路、观察屏和相机视场，确认条纹已进入画面。",
            "框选有效条纹区域；若画面过暗或过曝，先调整曝光与照明。",
        ])
        if motor_connected and not auto_enabled:
            actions.append(_action(
                "start_auto_search", "启动闭环搜索",
                "按当前安全范围启动自动旋转，搜索条纹并在发现中心后寻中。",
                risk="motor", requires_confirmation=True))
    else:
        if held or recognition_confidence < 0.35:
            issues.append(_issue("high", "条纹识别不稳定，当前结果依赖弱证据或历史保持"))
            recommendations.append("保持装置不动并等待连续清晰帧，再依据结果调节。")
        elif recognition_confidence < 0.60:
            issues.append(_issue("medium", "条纹识别置信度偏低"))
            recommendations.append("缩小 ROI 到清晰条纹区，并检查照明和对焦。")

        if blurred or sharpness < 0.30:
            issues.append(_issue("high", "画面存在运动模糊或清晰度不足"))
            recommendations.append("降低电机速度或停车等待稳定，并缩短曝光时间。")
            if auto_enabled:
                actions.append(_action(
                    "stop_auto_center", "停车恢复清晰画面",
                    "停止当前自动寻中，等待装置稳定后重新分析。",
                    risk="motor", requires_confirmation=True))
            if not motion_enhancement_enabled:
                actions.append(_action(
                    "enable_motion_enhancement", "开启运动清晰度增强",
                    "启用已配置的短曝光、增益和软件增强参数。",
                    risk="camera", requires_confirmation=True))
        elif sharpness < 0.50:
            issues.append(_issue("medium", "条纹清晰度一般"))
            recommendations.append("微调对焦或曝光，确认条纹边缘清晰后再记录。")

        if brightness is not None and _finite_float(brightness, 128.0) >= 230.0:
            issues.append(_issue("medium", "画面平均亮度过高，可能存在过曝"))
            recommendations.append("降低曝光或光源强度，避免亮纹峰值饱和。")
        elif brightness is not None and _finite_float(brightness, 128.0) <= 25.0:
            issues.append(_issue("medium", "画面平均亮度过低"))
            recommendations.append("增加照明或适当提高曝光，避免只提高增益放大噪声。")

        if movement not in {"stable", "unknown"} or abs(velocity) > 8.0:
            issues.append(_issue("medium", f"条纹仍在移动（{velocity:+.1f} px/s）"))
            recommendations.append("等待条纹稳定；正式测量时优先采用转动—停车—采集。")

        tilt = angle.get("tilt_deg")
        angle_confidence = _unit(angle.get("confidence", 0.0))
        curvature = angle.get("curvature")
        if tilt is not None and angle_confidence >= 0.35:
            tilt_value = float(tilt)
            if abs(tilt_value) > 3.0:
                direction = "顺时针" if tilt_value > 0 else "逆时针"
                issues.append(_issue(
                    "medium", f"条纹相对竖直方向{direction}倾斜 {abs(tilt_value):.1f}°"))
                recommendations.append(
                    f"可将画面校正 {float(angle.get('correction_deg') or -tilt_value):+.1f}°；"
                    "若调整光路，每次只微调一个旋钮并观察变化。")
                correction_delta = float(
                    angle.get("correction_deg")
                    if angle.get("correction_deg") is not None else -tilt_value)
                if angle_confidence >= 0.55 and abs(correction_delta) <= 30.0:
                    actions.append(_action(
                        "apply_angle_correction", "应用画面角度校正",
                        f"将当前画面校正量调整 {correction_delta:+.1f}°。",
                        risk="display", requires_confirmation=True,
                        params={
                            "delta_deg": round(correction_delta, 3),
                            "target_deg": round(
                                float(current_correction_deg) + correction_delta, 3),
                            "confidence": round(angle_confidence, 3),
                        }))
        if curvature is not None and float(curvature) > 0.035:
            issues.append(_issue("medium", "条纹弯曲较明显，整体旋转不能完全校正"))
            recommendations.append("检查镜面平行度和光路稳定性，采用单变量小步调节。")

        spacing_px = spacing.get("spacing_px")
        if spacing_px is None:
            issues.append(_issue("medium", "有效条纹不足，暂时无法给出稳健间距"))
            recommendations.append("调整 ROI，使其中包含至少四条连续、清晰的同类条纹。")
        else:
            if not bool(spacing.get("quality_valid", False)):
                reasons = []
                if not spacing.get("min_fringes_ok", False):
                    reasons.append("条纹数不足")
                if not spacing.get("cv_ok", False):
                    reasons.append("间距波动较大")
                if not spacing.get("rejection_ok", False):
                    reasons.append("异常间隔较多")
                detail = "、".join(reasons) or "几何质量不足"
                issues.append(_issue("medium", f"间距结果仅供参考：{detail}"))
                recommendations.append("重新框选均匀清晰的条纹区域，排除反光和黑色边缘。")

        if center_x is not None and frame_width:
            offset = float(center_x) - float(frame_width) / 2.0
            tolerance = max(15.0, float(frame_width) * 0.03)
            if abs(offset) > tolerance:
                side = "右" if offset > 0 else "左"
                issues.append(_issue("low", f"中心条纹位于画面中心{side}侧 {abs(offset):.1f}px"))
                recommendations.append("如需寻中，请按已标定的电机方向小步调节并观察中心误差变化。")
                if motor_connected and not auto_enabled:
                    actions.append(_action(
                        "start_auto_center", "启动闭环自动寻中",
                        "使用现有方向学习、模糊降速和安全范围控制移至画面中心。",
                        risk="motor", requires_confirmation=True))

    spacing_confidence = _unit(spacing.get("confidence", 0.0))
    angle_confidence = _unit(angle.get("confidence", 0.0))
    stable_score = 1.0 if movement == "stable" and abs(velocity) <= 8.0 else 0.35
    if not has_fringe:
        quality_score = 0.0
    else:
        quality_score = float(np.clip(
            0.30 * recognition_confidence
            + 0.22 * sharpness
            + 0.20 * spacing_confidence
            + 0.10 * angle_confidence
            + 0.18 * stable_score,
            0.0, 1.0))
        if held:
            quality_score *= 0.65

    high_count = sum(item["severity"] == "high" for item in issues)
    medium_count = sum(item["severity"] == "medium" for item in issues)
    measurement_ready = bool(
        has_fringe and high_count == 0 and medium_count == 0 and not held
        and movement == "stable" and abs(velocity) <= 8.0
        and bool(spacing.get("quality_valid", False))
        and quality_score >= 0.68)
    if measurement_ready:
        phase = "measurement_ready"
        summary = "条纹清晰且稳定，间距结果通过质量检查，可进行记录。"
        recommendations = ["保持装置和环境稳定，保存原始帧后记录测量数据。"]
    elif not has_fringe:
        phase = "searching"
        summary = "尚未获得可靠条纹，当前不适合测量。"
    elif high_count:
        phase = "quality_recovery"
        summary = "已看到条纹，但画面质量或识别稳定性不足。"
    elif medium_count:
        phase = "adjusting"
        summary = "条纹可分析，仍需根据提示改善后再测量。"
    else:
        phase = "observing"
        summary = "条纹已识别，正在等待更多稳定证据。"

    recommendations = _unique(recommendations)[:3]
    grade = "可靠" if quality_score >= 0.80 else (
        "可参考" if quality_score >= 0.50 else "不足")
    return {
        "read_only": True,
        "analysis_read_only": True,
        "phase": phase,
        "summary": summary,
        "quality_score": round(quality_score, 3),
        "quality_grade": grade,
        "measurement_ready": measurement_ready,
        "issues": issues,
        "recommendations": recommendations,
        "actions": _unique_actions(actions),
        "metrics": {
            "recognition_confidence": round(recognition_confidence, 3),
            "sharpness": round(sharpness, 3),
            "movement": movement,
            "velocity_px_s": round(velocity, 2),
            "angle_deg": angle.get("tilt_deg"),
            "angle_confidence": round(angle_confidence, 3),
            "curvature": angle.get("curvature"),
            "spacing_px": spacing.get("spacing_px"),
            "spacing_cv_percent": spacing.get("cv_percent"),
            "spacing_confidence": round(spacing_confidence, 3),
            "num_fringes": spacing.get("num_fringes", 0),
            "num_valid_intervals": spacing.get("num_valid_intervals", 0),
        },
    }


def _issue(severity: str, text: str) -> dict[str, str]:
    return {"severity": severity, "text": text}


def _action(
    code: str,
    label: str,
    description: str,
    *,
    risk: str,
    requires_confirmation: bool,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "description": description,
        "risk": risk,
        "requires_confirmation": bool(requires_confirmation),
        "params": params or {},
    }


def _unit(value: Any) -> float:
    return float(np.clip(_finite_float(value, 0.0), 0.0, 1.0))


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _unique_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for action in actions:
        code = str(action.get("code") or "")
        if code and code not in seen:
            seen.add(code)
            result.append(action)
    return result[:3]
