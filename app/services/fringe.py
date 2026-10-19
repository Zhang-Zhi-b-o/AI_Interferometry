"""条纹定位与测量：包装 ``src.vision`` 里的三个纯函数。

一次 ``measure()`` 同时给出「位置 + 宽度 + 间距 + 倾斜角」，页面只需要一张
指标卡，不用分别调三次。任何一路失败都只把对应字段留空，不整体报错。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class FringeMetrics:
    center_x: float | None = None
    confidence: float | None = None
    width_px: float | None = None
    period_px: float | None = None
    spacing_px: float | None = None
    tilt_deg: float | None = None
    num_bands: int | None = None
    band_kind: str | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FringeService:
    """无状态；直接调用即可，可安全地在后台线程里跑。"""

    def measure(self, frame: np.ndarray, center_x: float | None = None) -> FringeMetrics:
        metrics = FringeMetrics()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            metrics.errors.append("没有画面")
            return metrics

        # 1) 中心条纹位置（零光程差包络中心）
        try:
            from src.vision.fringe_center import find_center_in_region

            center = find_center_in_region(frame, expected_center_x=center_x)
            metrics.center_x = _round(center.get("center_x"))
            metrics.confidence = _round(center.get("confidence"))
            metrics.period_px = _round(center.get("period"))
        except Exception as exc:
            metrics.errors.append(f"中心定位失败：{exc}")

        # 2) 宽度（以定位结果或用户给的参考位置为准）
        reference = metrics.center_x if metrics.center_x is not None else center_x
        try:
            from src.vision.fringe_width import measure_center_fringe_width

            width = measure_center_fringe_width(frame, center_x=reference)
            metrics.num_bands = int(width.get("num_bands") or 0)
            metrics.period_px = metrics.period_px or _round(width.get("period_px"))
            band = width.get("center_band") or {}
            metrics.width_px = _round(band.get("width"))
            metrics.band_kind = band.get("kind")
        except Exception as exc:
            metrics.errors.append(f"宽度测量失败：{exc}")

        # 3) 条纹间距（2D 法向投影；比周期估计更稳）
        try:
            from src.vision.fringe_width import measure_fringe_spacing_robust

            spacing = measure_fringe_spacing_robust(frame)
            metrics.spacing_px = _round(spacing.get("spacing_px"))
        except Exception as exc:
            metrics.errors.append(f"间距测量失败：{exc}")

        # 4) 倾斜角
        try:
            from src.vision.fringe_angle import estimate_fringe_angle_2d

            angle = estimate_fringe_angle_2d(frame)
            metrics.tilt_deg = _round(angle.get("tilt_deg"))
        except Exception as exc:
            metrics.errors.append(f"倾角估计失败：{exc}")

        return metrics


def _round(value, digits: int = 2) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return round(number, digits)
    except Exception:
        return None
