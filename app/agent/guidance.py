"""唯一的四阶段操作建议；不触发设备动作。"""
from __future__ import annotations


def guidance(payload: dict) -> str:
    state = payload["state"]
    if not state.get("camera_open") and not state.get("frame_size"):
        return "观察：请先打开干涉相机，或载入演示图像。"
    if state.get("center_x") is None:
        return "定位：先观察并识别中心条纹，再在搜索页确认运动方向并启动寻中。"
    if state.get("reading_mm") is None or (payload.get("reading_age_seconds") or 0) > 1.5:
        return "读数：请启用微分表相机，取得新的稳定读数并记录两次位置。"
    return "计算：把两次微分表读数送入计算页，检查折射率后计算厚度与不确定度。"
