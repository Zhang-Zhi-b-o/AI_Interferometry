"""条纹定位与测量：包装 ``app.engine.vision`` 里的三个纯函数。

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
    method: str = "default"
    width_mm: float | None = None
    spacing_mm: float | None = None
    details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FringeService:
    """无状态；直接调用即可，可安全地在后台线程里跑。"""

    def measure(self, frame: np.ndarray, center_x: float | None = None, *,
                method: str = "default", roi=None, mm_per_px: float | None = None) -> FringeMetrics:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return FringeMetrics(errors=["没有画面"])
        methods = {"default", "width_1d", "width_2d", "count", "robust", "spacing_2d"}
        if method not in methods:
            raise ValueError("未知条纹算法")
        if mm_per_px is not None and (not np.isfinite(mm_per_px) or mm_per_px <= 0):
            raise ValueError("像素标定须为正的有限数 (mm/px)")
        offset = 0
        if roi is not None:
            x, y, w, h = map(int, roi)
            height, width = frame.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("ROI 超出画面")
            frame = frame[y0:y1, x0:x1]
            offset = x0
            if center_x is not None:
                center_x -= offset
        metrics = self._measure_local(frame, center_x, method)
        if metrics.center_x is not None:
            metrics.center_x += offset
        metrics.width_mm = None if mm_per_px is None or metrics.width_px is None else metrics.width_px * mm_per_px
        metrics.spacing_mm = None if mm_per_px is None or metrics.spacing_px is None else metrics.spacing_px * mm_per_px
        metrics.details.update(roi=list(roi) if roi else None, mm_per_px=mm_per_px,
                               coordinate_system="full_frame", detail_coordinates="roi_local")
        return metrics

    def _measure_local(self, frame, center_x, method):
        metrics = FringeMetrics(method=method)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            metrics.errors.append("没有画面")
            return metrics

        # 1) 中心条纹位置（零光程差包络中心）
        try:
            from app.engine.vision.fringe_center import find_center_in_region

            center = find_center_in_region(frame, expected_center_x=center_x)
            metrics.center_x = _round(center.get("center_x"))
            metrics.confidence = _round(center.get("confidence"))
            metrics.period_px = _round(center.get("period"))
        except Exception as exc:
            metrics.errors.append(f"中心定位失败：{exc}")

        # 2) 宽度（以定位结果或用户给的参考位置为准）
        reference = metrics.center_x if metrics.center_x is not None else center_x
        try:
            from app.engine.vision.fringe_width import measure_center_fringe_width

            if method == "width_2d":
                from app.engine.vision.fringe_width import measure_center_fringe_width_2d
                width = measure_center_fringe_width_2d(frame, center_x=reference)
            else:
                width = measure_center_fringe_width(frame, center_x=reference)
            metrics.details["width"] = width
            metrics.num_bands = int(width.get("num_bands") or 0)
            metrics.period_px = metrics.period_px or _round(width.get("period_px"))
            band = width.get("center_band") or {}
            metrics.width_px = _round(band.get("width"))
            metrics.band_kind = band.get("kind")
        except Exception as exc:
            metrics.errors.append(f"宽度测量失败：{exc}")

        # 3) 条纹间距（2D 法向投影；比周期估计更稳）
        try:
            from app.engine.vision.fringe_width import measure_fringe_spacing_robust

            if method == "count":
                from app.engine.vision.fringe_width import measure_fringe_width_by_count
                spacing = measure_fringe_width_by_count(frame)
                metrics.spacing_px = _round(spacing.get("fringe_width"))
            elif method == "spacing_2d":
                from app.engine.vision.fringe_width import measure_fringe_spacing_2d
                spacing = measure_fringe_spacing_2d(frame)
                metrics.spacing_px = _round(spacing.get("spacing_px"))
            else:
                spacing = measure_fringe_spacing_robust(frame)
                metrics.spacing_px = _round(spacing.get("spacing_px"))
            metrics.details["spacing"] = spacing
        except Exception as exc:
            metrics.errors.append(f"间距测量失败：{exc}")

        # 4) 倾斜角
        try:
            from app.engine.vision.fringe_angle import estimate_fringe_angle_2d

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
