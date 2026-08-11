"""视频演示模式默认参数加载。"""
from __future__ import annotations

from pathlib import Path

import yaml

from src import PROJECT_ROOT
from src.config import ConfigError


PRESET_PATH = PROJECT_ROOT / "config" / "video_demo.yaml"

REQUIRED_KEYS = {
    "main_camera": {
        "index", "resolution", "fps", "angle_deg", "zoom", "clarity_assist"},
    "reading_camera": {
        "model_path", "camera_index", "resolution", "fps", "interval_ms",
        "auto_roi", "roi", "min_score", "decimal_places", "stable_window",
        "stable_required", "max_step_mm", "jump_required",
        "scale_ratio_tolerance", "scale_factor"},
    "yolo": {
        "model_path", "device", "confidence_threshold", "iou_threshold",
        "imgsz", "auto_detect_center", "center_search_expand_ratio",
        "center_search_radius_ratio", "center_search_margin_ratio",
        "fringe_motion_window", "fringe_motion_threshold_px",
        "fringe_history_size", "fringe_missing_hold_frames",
        "fringe_visual_threshold", "fringe_assisted_threshold"},
    "motor": {"port", "baudrate", "timeout", "safety"},
    "auto_center": {
        "search_direction", "search_mode", "invert_direction",
        "auto_learn_direction", "show_center_line", "search_gear", "fast_gear",
        "slow_gear", "slow_zone_px", "tolerance_px", "stable_frames",
        "dropout_hold_frames", "center_confirm_frames",
        "command_refresh_frames", "learning_delta_px", "guide_min_confidence",
        "guide_loss_confirm_frames", "search_initial_span_turns",
        "search_expansion_factor", "search_max_span_turns", "search_min_gear",
        "search_acceleration_step", "blur_slowdown_frames", "blur_safe_gear",
        "blur_recovery_clear_frames", "stop_detect_move_seconds",
        "stop_detect_settle_seconds", "stop_detect_frames", "guide_worsening_px",
        "guide_trend_window", "guide_focus_confirm_frames",
        "guide_focus_shift_ratio", "guide_focus_min_shift_turns",
        "guide_focus_max_shift_turns"},
}
NESTED_REQUIRED_KEYS = {
    ("main_camera", "clarity_assist"): {
        "enabled", "motion_enabled_by_default", "preview_exposure",
        "preview_gain", "motion_exposure", "motion_gain", "min_exposure",
        "max_gain", "blur_ratio", "check_frames", "trigger_checks",
        "min_brightness_for_shorter_exposure", "software_enhancement"},
    ("main_camera", "clarity_assist", "software_enhancement"): {
        "enabled", "sharpen_strength", "max_sharpen_strength",
        "stripe_contrast_strength", "max_stripe_contrast_strength",
        "color_gain", "max_color_gain", "original_mix",
        "horizontal_kernel_size", "vertical_smooth_size",
        "background_kernel_size", "contrast_gain"},
    ("motor", "safety"): {
        "max_run_seconds", "black_confirm_frames", "max_missing_frames"},
}


def load_recording_preset(path: Path = PRESET_PATH) -> dict:
    """读取并校验视频预设；禁止静默回退到其他默认参数。"""
    if not path.exists():
        raise ConfigError(f"缺少视频演示唯一默认参数文件: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取视频演示参数 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config/video_demo.yaml 顶层必须是键值映射")
    errors: list[str] = []
    for section, required in REQUIRED_KEYS.items():
        value = data.get(section)
        if not isinstance(value, dict):
            errors.append(f"{section} 必须是键值映射")
            continue
        missing = sorted(required - value.keys())
        if missing:
            errors.append(f"{section} 缺少参数: {', '.join(missing)}")
    for path_parts, required in NESTED_REQUIRED_KEYS.items():
        value = data
        for part in path_parts:
            value = value.get(part) if isinstance(value, dict) else None
        path_text = ".".join(path_parts)
        if not isinstance(value, dict):
            errors.append(f"{path_text} 必须是键值映射")
            continue
        missing = sorted(required - value.keys())
        if missing:
            errors.append(f"{path_text} 缺少参数: {', '.join(missing)}")
    if data.get("auto_center", {}).get("search_direction") not in {
        "forward", "reverse",
    }:
        errors.append("auto_center.search_direction 必须是 forward 或 reverse")
    if data.get("auto_center", {}).get("search_mode") not in {
            "single_direction", "stop_and_detect", "bidirectional"}:
        errors.append(
            "auto_center.search_mode 必须是 bidirectional、single_direction "
            "或 stop_and_detect")
    for section in ("main_camera", "reading_camera"):
        resolution = data.get(section, {}).get("resolution")
        if (not isinstance(resolution, list) or len(resolution) != 2
                or not all(isinstance(v, int) and v > 0 for v in resolution)):
            errors.append(f"{section}.resolution 必须是两个正整数")
    for section in ("reading_camera", "yolo"):
        model_path = data.get(section, {}).get("model_path")
        if (not isinstance(model_path, str)
                or not (PROJECT_ROOT / model_path).is_file()):
            errors.append(f"{section}.model_path 文件不存在: {model_path}")
    yolo = data.get("yolo", {})
    for key in ("fringe_visual_threshold", "fringe_assisted_threshold"):
        value = yolo.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"yolo.{key} 必须在 0～1 之间")
    for key in ("fringe_history_size", "fringe_missing_hold_frames"):
        value = yolo.get(key)
        if not isinstance(value, int) or value < 0:
            errors.append(f"yolo.{key} 必须是非负整数")
    if (isinstance(yolo.get("fringe_history_size"), int)
            and yolo["fringe_history_size"] < 3):
        errors.append("yolo.fringe_history_size 必须至少为 3")
    if errors:
        raise ConfigError(
            "config/video_demo.yaml 配置校验失败：\n- "
            + "\n- ".join(errors))
    return data
