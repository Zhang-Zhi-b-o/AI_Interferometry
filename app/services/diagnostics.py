"""同一快照产生同一诊断；只读，不调用模型、不驱动硬件。"""
from __future__ import annotations

import math

from app.agent.guidance import guidance


def diagnose(payload: dict) -> dict:
    state = payload.get("state", {})
    findings = []

    def add(code, observation, action):
        findings.append({"code": code, "observation": observation, "action": action,
                         "basis": "本地状态/相对趋势；不是仪器精度判据"})

    visible = state.get("camera_open") or state.get("frame_size")
    if not visible:
        add("no_frame", "没有可用画面。", "打开干涉相机或载入图像。")
    elif state.get("detecting") and not state.get("detect_count") and state.get("center_x") is None:
        add("no_fringe", "当前没有条纹识别结果，尚不能判断光程差。", "检查光路与视野，再重新识别。")
    width = (state.get("frame_size") or [0, 0])[0]
    center, band = state.get("center_x"), state.get("fringe_width_px")
    spacing, tilt = state.get("fringe_spacing_px"), state.get("tilt_deg")
    if visible and width and center is not None and band and abs(center - width / 2) > band:
        add("off_center", "中心条纹偏离画面中心超过自身宽度。", "核对参考位置与方向，再决定是否寻中。")
    if visible and tilt is not None and math.isfinite(tilt) and tilt != 0:
        add("tilted", "条纹倾角测量不为零；是否超差尚无仪器判据。", "观察倾角变化，必要时微调反射镜。")
    if visible and spacing and band and spacing < band:
        add("dense", "测得条纹间距小于中心条纹宽度。", "检查条纹分离与测量区域，重新测量。")
    contrasts = [e["result"].get("contrast") for e in payload.get("timeline", [])
                 if e.get("kind") == "fringe" and isinstance(e.get("result"), dict)
                 and e["result"].get("roi") == payload.get("roi")]
    contrasts = [v for v in contrasts if isinstance(v, (int, float)) and math.isfinite(v)]
    if visible and len(contrasts) >= 3 and contrasts[-3] > contrasts[-2] > contrasts[-1]:
        add("contrast_falling", "同一 ROI 最近三次图像亮度对比指标连续下降。",
            "检查曝光与照明；该趋势不能单独确定光路故障原因。")
    if state.get("reading_mm") is not None:
        if payload.get("reading_stable") is False:
            add("reading_unstable", "微分表服务报告读数尚不稳定。", "等待读数稳定后再记录。")
        if (payload.get("reading_age_seconds") or 0) > 1.5:
            add("reading_stale", "读数超过控制链已有的新鲜度窗口。", "重新采集读数，避免沿用历史值。")
    failures = 0
    for event in reversed(payload.get("timeline", [])):
        if event.get("kind") != "search_finished":
            continue
        if not (event.get("result") or {}).get("failed"):
            break
        failures += 1
    if failures >= 2:
        add("search_failures", "自动寻中连续失败。", "先核对画面、模型与运动方向，再尝试。")
    if payload.get("map_mode") == "relative":
        add("uncalibrated_map", "当前厚度分布来自未标定的相对模式。",
            "仅解释相对起伏；定量结果需要颜色标定与独立参照验证。")
    return {"stage": guidance(payload), "findings": findings,
            "search_failure_count": failures,
            "limitations": "诊断基于本地快照，不代表物理精度验收；缺失量不推断。"}


def contrast_metric(frame) -> float | None:
    """BGR 亮度的分位数对比指标，只用于同一区域的趋势比较。"""
    import numpy as np

    if frame is None or not frame.size:
        return None
    gray = frame.astype(float)
    if gray.ndim == 3:
        gray = gray[..., 0] * 0.114 + gray[..., 1] * 0.587 + gray[..., 2] * 0.299
    low, high = np.percentile(gray, [5, 95])
    return float((high - low) / (high + low)) if high + low > 0 else 0.0
