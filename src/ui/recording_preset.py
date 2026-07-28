"""视频演示模式默认参数加载。"""
from __future__ import annotations

from pathlib import Path

import yaml

from src import PROJECT_ROOT
from src.config import ConfigError


PRESET_PATH = PROJECT_ROOT / "config" / "video_demo.yaml"


def load_recording_preset(path: Path = PRESET_PATH) -> dict:
    """读取独立视频预设；缺失文件时返回空配置以兼容旧项目。"""
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取视频演示参数 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config/video_demo.yaml 顶层必须是键值映射")
    for section in ("main_camera", "reading_camera", "yolo", "auto_center"):
        value = data.get(section, {})
        if not isinstance(value, dict):
            raise ConfigError(f"video_demo.yaml 的 {section} 必须是键值映射")
    return data
