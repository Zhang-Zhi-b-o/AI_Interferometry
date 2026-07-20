"""把实时设备状态整理为实验助手可读取的只读快照。"""
from __future__ import annotations

from typing import Any
import time


def _experiment_progress(context: dict[str, Any]) -> dict[str, Any]:
    """根据只读现场状态给出确定性的实验阶段和下一步指导。"""
    camera = context["camera"]
    vision = context["vision"]
    motor = context["motor"]
    meter = context["micrometer"]
    measurement = context["measurement"]

    if not camera["running"]:
        values = (5, "设备准备", "打开干涉画面摄像头",
                  "预览画面连续显示且 FPS 大于 0")
    elif not vision["model_loaded"]:
        values = (20, "视觉准备", "加载条纹识别模型",
                  "模型状态显示为已加载")
    elif not vision["prediction_running"]:
        values = (30, "视觉准备", "启动模型预测",
                  "实时画面开始产生识别结果")
    elif not vision["fringe_present"] and not vision["detections"]:
        values = (42, "条纹观察", "检查光路重合、画面清晰度和 ROI",
                  "画面出现稳定条纹或模型检测框")
    elif vision["center_x_px"] is None:
        values = (55, "寻找中心条纹", "观察条纹线索；需要自动寻中时连接电机并启动寻中",
                  "识别到中心条纹并显示中心横坐标")
    elif motor["auto_enabled"]:
        values = (68, "中心定位", "等待自动寻中完成，必要时关注电机方向和识别稳定性",
                  "自动寻中停车并报告中心稳定")
    elif motor["auto_control_state"] == "centered":
        values = (78, "中心已定位", "启动微分表视觉读数并等待稳定",
                  "微分表显示带时间戳的稳定读数")
    elif not meter["connected"]:
        values = (72, "中心条纹确认", "确认中心条纹位于画面中央，然后启动微分表读数",
                  "中心位置稳定且微分表相机已连接")
    elif meter["reading_mm"] is None:
        values = (82, "测量准备", "保持仪器稳定，等待微分表 OCR 读数稳定",
                  "出现有效读数和采集时间")
    elif measurement["record_count"] == 0:
        values = (92, "测量记录", "记录中心位置、微分表读数和实验现象",
                  "测量记录数量大于 0")
    else:
        values = (100, "数据已记录", "核对数据后进行计算、误差分析或生成实验报告",
                  "数据、单位、时间戳和实验现象均已核对")
    progress, stage, next_action, criterion = values
    return {
        "progress_percent": progress,
        "stage": stage,
        "next_action": next_action,
        "completion_criterion": criterion,
        "updated_at": context["snapshot_at"],
    }


def build_runtime_context(
    *, camera_running: bool, fps: float, model_loaded: bool,
    prediction_running: bool,
    detections: dict[str, float], center_x_px: float | None,
    fringe_motion: dict[str, Any], motor_connected: bool,
    motor_mode: str, auto_enabled: bool, auto_state: str,
    auto_control_state: str,
    micrometer_connected: bool, micrometer_reading_mm: float | None,
    micrometer_reading_at: float | None, scale_factor: float | None,
    record_count: int,
) -> dict[str, Any]:
    now = time.time()
    context = {
        "snapshot_at": now,
        "camera": {"running": camera_running, "fps": round(fps, 1)},
        "vision": {
            "model_loaded": model_loaded,
            "prediction_running": prediction_running,
            "detections": detections,
            "center_x_px": round(center_x_px, 2) if center_x_px is not None else None,
            "fringe_present": bool(fringe_motion.get("has_fringe", False)),
            "fringe_movement": fringe_motion.get("movement", "unknown"),
            "fringe_delta_x_px": fringe_motion.get("delta_x_px"),
        },
        "motor": {
            "connected": motor_connected, "mode": motor_mode,
            "auto_enabled": auto_enabled, "auto_state": auto_state,
            "auto_control_state": auto_control_state,
        },
        "micrometer": {
            "connected": micrometer_connected,
            "reading_mm": micrometer_reading_mm,
            "reading_captured_at": micrometer_reading_at,
            "reading_age_seconds": (
                round(max(0.0, now - micrometer_reading_at), 2)
                if micrometer_reading_at is not None else None),
            "scale_factor": scale_factor,
        },
        "measurement": {"record_count": record_count},
    }
    context["experiment_progress"] = _experiment_progress(context)
    return context
