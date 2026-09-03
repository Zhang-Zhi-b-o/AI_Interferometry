"""把实时设备状态整理为实验助手可读取的只读快照。"""
from __future__ import annotations

from typing import Any
import time

from src.agent.experiment_guidance import (
    ExperimentIntent,
    build_guidance_decision,
)


def _experiment_progress(context: dict[str, Any]) -> dict[str, Any]:
    """根据只读现场状态给出确定性的实验阶段和下一步指导。"""
    camera = context["camera"]
    vision = context["vision"]
    motor = context["motor"]
    setup_started = bool(
        camera["interferometer_running"] or camera["micrometer_running"]
        or motor["connected"] or context.get("instrument_ready", False))
    devices_ready = bool(
        camera["interferometer_running"]
        and camera["micrometer_running"]
        and motor["connected"])
    visual_ready = bool(
        camera["preview_adjusted"] and vision["roi_defined"])
    analysis_ready = bool(
        vision["model_loaded"] and vision["prediction_running"]
        and vision["auto_analysis_enabled"])
    centered = motor["auto_control_state"] == "centered"
    guidance = vision.get("fringe_guidance") or {}
    quality_gate_passed = bool(
        not guidance or guidance.get("measurement_ready", False))

    if not setup_started:
        values = (1, 5, "调整仪器并放置白光光源",
                  "调整仪器，放置白光光源",
                  "白光光源已放置，干涉仪光路已完成初步调整")
    elif not devices_ready:
        missing = []
        if not camera["interferometer_running"]:
            missing.append("干涉画面摄像头")
        if not camera["micrometer_running"]:
            missing.append("微分表摄像头")
        if not motor["connected"]:
            missing.append("电机")
        values = (2, 25, "连接双摄像头和电机",
                  "打开两个摄像头并连接电机；当前缺少：" + "、".join(missing),
                  "两路摄像头均有实时画面，电机串口显示已连接")
    elif not visual_ready:
        actions = []
        if not camera["preview_adjusted"]:
            actions.append("矫正预览画面")
        if not vision["roi_defined"]:
            actions.append("在条纹区域标注 ROI")
        values = (3, 45, "画面矫正与 ROI 标注", "，然后".join(actions),
                  "画面方向与条纹观察一致，并显示有效 ROI 框")
    elif not analysis_ready:
        actions = []
        if not vision["model_loaded"]:
            actions.append("加载模型")
        if not vision["prediction_running"]:
            actions.append("开始预测")
        if not vision["auto_analysis_enabled"]:
            actions.append("开启中心条纹自动分析")
        values = (4, 70, "模型预测与条纹分析", "，然后".join(actions),
                  "模型已加载、预测正在运行且自动条纹分析已开启")
    elif motor["auto_enabled"]:
        values = (5, 90, "自动寻中进行中",
                  "保持光路和设备稳定，观察自动寻中状态",
                  "中心条纹稳定到达画面中心并可靠停车")
    elif centered and quality_gate_passed:
        values = (5, 100, "自动寻中完成",
                  "核对中心位置、YOLO结果和微分表读数，随后记录数据",
                  "中心位置、微分表读数、时间戳与实验现象均已核对")
    elif centered:
        values = (5, 95, "中心已到位，等待测量质量门",
                  "按实时条纹诊断改善清晰度、稳定性或间距质量",
                  "条纹稳定清晰且间距质量检查通过，界面显示可测量")
    else:
        values = (5, 85, "准备自动寻中", "开始自动寻中",
                  "自动寻中进入运行状态并持续报告中心偏差")
    step_number, progress, stage, next_action, criterion = values
    return {
        "step_number": step_number,
        "total_steps": 5,
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
    interferometer_camera_index: int | None = None,
    micrometer_camera_index: int | None = None,
    preview_adjusted: bool = False,
    correction: dict[str, Any] | None = None,
    roi_xywh: tuple[int, int, int, int] | None = None,
    auto_analysis_enabled: bool = False,
    detection_details: list[dict[str, Any]] | None = None,
    center_confidence: float = 0.0,
    frame_width: int | None = None,
    micrometer_ocr: dict[str, Any] | None = None,
    motor_details: dict[str, Any] | None = None,
    recent_logs: list[dict[str, Any]] | None = None,
    instrument_ready: bool = False,
    measurement_records: list[dict[str, Any]] | None = None,
    temporary_measurement: dict[str, Any] | None = None,
    thickness_measurement: dict[str, Any] | None = None,
    experiment_assistant: dict[str, Any] | None = None,
    fringe_band_overlay: list[dict[str, Any]] | None = None,
    fringe_count_overlay: dict[str, Any] | None = None,
    fringe_realtime_active: bool = False,
    texture_analysis: dict[str, Any] | None = None,
    fringe_guidance: dict[str, Any] | None = None,
    laser_alignment_active: bool = False,
    adaptive_response: dict[str, Any] | None = None,
    guidance_execution_stage: str = "advisory",
    auto_direction_mapping: str = "learning",
    live_measurement: dict[str, Any] | None = None,
    live_measurement_active: bool = False,
    calibration_rows: list[dict[str, Any]] | None = None,
    experiment_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.time()
    context = {
        "snapshot_at": now,
        "instrument_ready": bool(instrument_ready),
        "camera": {
            "running": camera_running, "fps": round(fps, 1),
            "interferometer_running": camera_running,
            "interferometer_index": interferometer_camera_index,
            "micrometer_running": micrometer_connected,
            "micrometer_index": micrometer_camera_index,
            "preview_adjusted": bool(preview_adjusted),
            "correction": correction or {},
        },
        "vision": {
            "model_loaded": model_loaded,
            "prediction_running": prediction_running,
            "detections": detections,
            "center_x_px": round(center_x_px, 2) if center_x_px is not None else None,
            "fringe_present": bool(fringe_motion.get("has_fringe", False)),
            "fringe_movement": fringe_motion.get("movement", "unknown"),
            "fringe_delta_x_px": fringe_motion.get("delta_x_px"),
            "roi_defined": roi_xywh is not None,
            "roi_xywh": list(roi_xywh) if roi_xywh is not None else None,
            "auto_analysis_enabled": bool(auto_analysis_enabled),
            "detection_count": len(detection_details or []),
            "detection_details": detection_details or [],
            "center_confidence": round(float(center_confidence), 4),
            "frame_width": frame_width,
            "center_offset_px": (
                round(float(center_x_px) - float(frame_width) / 2.0, 2)
                if center_x_px is not None and frame_width else None),
            "fringe_band_overlay": fringe_band_overlay or [],
            "fringe_count_overlay": fringe_count_overlay or {},
            "fringe_realtime_active": bool(fringe_realtime_active),
            "texture_analysis": texture_analysis or {},
            "fringe_guidance": fringe_guidance or {},
            "laser_alignment_active": bool(laser_alignment_active),
            "adaptive_response": adaptive_response or {},
            "guidance_execution_stage": str(guidance_execution_stage),
        },
        "motor": {
            "connected": motor_connected, "mode": motor_mode,
            "auto_enabled": auto_enabled, "auto_state": auto_state,
            "auto_control_state": auto_control_state,
            "auto_direction_mapping": auto_direction_mapping,
            **(motor_details or {}),
        },
        "micrometer": {
            "connected": micrometer_connected,
            "reading_mm": micrometer_reading_mm,
            "reading_captured_at": micrometer_reading_at,
            "reading_age_seconds": (
                round(max(0.0, now - micrometer_reading_at), 2)
                if micrometer_reading_at is not None else None),
            "scale_factor": scale_factor,
            "ocr": micrometer_ocr or {},
        },
        "measurement": {
            "record_count": record_count,
            "records": measurement_records or [],
            "temporary": temporary_measurement or {},
            "thickness": thickness_measurement or {},
            "experiment_assistant": experiment_assistant or {},
            "calibration": calibration_rows or [],
            "live_measurement": live_measurement or {},
            "live_measurement_active": bool(live_measurement_active),
        },
        "recent_logs": recent_logs or [],
        "experiment_intent": ExperimentIntent.from_mapping(
            experiment_intent).as_dict(),
    }
    context["experiment_progress"] = _experiment_progress(context)
    context["assistant_guidance"] = build_guidance_decision(context).as_dict()
    return context
