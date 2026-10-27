"""比较一次人工调节前后的条纹指标，仅提供只读反馈。"""
from __future__ import annotations

import math
from typing import Any


def compare_fringe_adjustment(
    before: dict[str, Any], after: dict[str, Any],
) -> dict[str, Any]:
    """比较倾角、曲率、间距波动和质量分数，避免猜测旋钮方向。"""
    if not before or not after:
        return {
            "outcome": "insufficient",
            "summary": "缺少调整前或调整后的条纹指标，无法比较效果。",
            "recommendation": "先记录调整前状态，再只微调一个旋钮并重新比较。",
            "changes": {},
        }
    movement = str(after.get("movement") or "unknown")
    if movement not in {"stable", "unknown"}:
        return {
            "outcome": "insufficient",
            "summary": "条纹仍在移动，调整前后指标暂时不可直接比较。",
            "recommendation": "停止调节并等待条纹稳定，系统将自动完成比较。",
            "changes": {},
        }

    changes: dict[str, float] = {}
    improvements = 0
    regressions = 0
    definitions = (
        ("angle_deg", True, 0.5),
        ("curvature", True, 0.003),
        ("spacing_cv_percent", True, 1.0),
        ("quality_score", False, 0.03),
    )
    for name, lower_is_better, threshold in definitions:
        old = _finite(before.get(name))
        new = _finite(after.get(name))
        if old is None or new is None:
            continue
        if name == "angle_deg":
            old, new = abs(old), abs(new)
        delta = new - old
        changes[name] = round(delta, 4)
        signed_gain = -delta if lower_is_better else delta
        if signed_gain >= threshold:
            improvements += 1
        elif signed_gain <= -threshold:
            regressions += 1

    if not changes:
        outcome = "insufficient"
        summary = "当前没有足够的可比条纹指标。"
        recommendation = "保持相机、ROI 和曝光不变后重新进行一次小步调整。"
    elif improvements and not regressions:
        outcome = "improved"
        summary = "本次小步调整使条纹指标整体改善。"
        recommendation = "可以沿相同方向再微调一个小步，达到完成判据后立即停止。"
    elif regressions and not improvements:
        outcome = "worsened"
        summary = "本次调整使条纹指标整体变差。"
        recommendation = "退回上一个较好位置；方向未经标定时不要继续沿当前方向调整。"
    else:
        outcome = "mixed"
        summary = "本次调整的效果不一致，部分指标改善、部分指标变差。"
        recommendation = "先停止连续调整，确认主要目标后一次只改变一个旋钮。"
    return {
        "outcome": outcome,
        "summary": summary,
        "recommendation": recommendation,
        "changes": changes,
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
