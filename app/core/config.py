"""配置读取：在项目根目录的 config.yaml 上做一层薄封装。

Qt 默认值与旧设备预设合并，再由用户配置覆盖。
新增参数时，请同时在 ``app/config.v2.yaml`` 里写默认值和一行注释。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # yaml 写错不该让程序起不来
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class AppConfig:
    """扁平化的配置视图：``cfg.get("vision.confidence", 0.25)``。"""

    data: dict = field(default_factory=dict)
    source: Path | None = None

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def section(self, path: str) -> dict:
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}

    # ---- 常用参数的语义化访问（改这里就能改全局默认值）----
    @property
    def app_title(self) -> str:
        return str(self.get("app.name", "白光干涉测量工作台"))

    @property
    def window_size(self) -> tuple[int, int]:
        size = self.get("ui.window_size", [1600, 1000])
        try:
            return int(size[0]), int(size[1])
        except Exception:
            return 1600, 1000

    @property
    def theme(self) -> str:
        return str(self.get("ui.theme", "light"))

    @property
    def font_scale(self) -> float:
        try:
            return max(0.8, min(1.6, float(self.get("ui.font_scale", 1.0))))
        except Exception:
            return 1.0

    @property
    def camera_index(self) -> int:
        return int(self.get("camera.index", 1))

    @property
    def camera_resolution(self) -> tuple[int, int]:
        size = self.get("camera.resolution", [1280, 1024])
        try:
            return int(size[0]), int(size[1])
        except Exception:
            return 1280, 1024

    @property
    def model_path(self) -> Path:
        raw = str(self.get("vision.model_path", "models/releases/yolov8n_full2954_20261027.pt"))
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    @property
    def refractive_index(self) -> float:
        return float(self.get("measurement.refractive_index", 1.4586))

    @property
    def pixel_to_mm_ratio(self) -> float:
        return float(self.get("measurement.pixel_to_mm_ratio", 0.000444))


def load_config() -> AppConfig:
    """Qt 默认值 + 旧演示预设 + 用户覆盖；复用双相机和控制参数。"""
    defaults = _read_yaml(ROOT / "config.v2.yaml")
    preset = _read_yaml(ROOT / "config" / "video_demo.yaml")
    if preset:
        camera = dict(preset.get("main_camera") or {})
        yolo = dict(preset.get("yolo") or {})
        vision = dict(yolo)
        # 旧演示预设的权重不覆盖 Qt 版当前默认模型；用户配置仍可覆盖。
        vision.pop("model_path", None)
        for old, new in (("confidence_threshold", "confidence"), ("iou_threshold", "iou")):
            if old in yolo:
                vision[new] = yolo[old]
        motor = dict(preset.get("motor") or {})
        search = dict(preset.get("auto_center") or {})
        search["max_missing_frames"] = (motor.get("safety") or {}).get("max_missing_frames", 90)
        defaults = _deep_merge(defaults, {
            "camera": camera, "vision": vision, "motor": motor,
            "reading": dict(preset.get("reading_camera") or {}), "search": search,
        })
    user = _read_yaml(ROOT / "config.yaml")
    merged = _deep_merge(defaults, user if isinstance(user, dict) else {})
    return AppConfig(data=merged, source=ROOT / "config.yaml")
