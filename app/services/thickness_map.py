"""单帧厚度分布：包装 ``app.engine.vision.thickness_distribution.analyze_thickness_distribution``。

两种模式：
- **相对模式**（不传标定表）：按颜色级次解包相位，结果只是相对起伏；
- **标定模式**（传 ``opd_um,r,g,b`` CSV）：标定后的相对起伏；绝对厚度另需独立锚点。

返回值里 ``overlay`` / ``heatmap`` 是 BGR 图像，交给界面直接显示。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import ROOT

def analyze_thickness_distribution(*args, **kwargs):
    try:
        from app.engine.vision.thickness_distribution import analyze_thickness_distribution as analyze
    except ImportError as exc:
        raise RuntimeError("厚度分布模块不可用（缺少依赖）") from exc
    return analyze(*args, **kwargs)


def sample_colour_band(*args, **kwargs):
    try:
        from app.engine.vision.thickness_distribution import sample_colour_band as sample
    except ImportError as exc:
        raise RuntimeError("取色模块不可用（缺少依赖）") from exc
    return sample(*args, **kwargs)


class ThicknessMapService:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        wavelength_nm: float = 589.3,
        refractive_index: float = 1.523,
        calibration: str | Path | None = None,
        invert: bool = False,
        reference_image: np.ndarray | None = None,
        reference_thickness_um: float | None = None,
        whole_region: bool = False,
        smooth_px: float = 2.0,
    ) -> dict:
        if frame is None or frame.size == 0:
            raise ValueError("没有可分析的画面")
        result = analyze_thickness_distribution(
            frame,
            wavelength_nm=float(wavelength_nm),
            refractive_index=float(refractive_index),
            calibration=None if calibration in (None, "") else str(calibration),
            invert=bool(invert),
            reference_thickness_um=reference_thickness_um,
            reference_image=reference_image,
            whole_region=bool(whole_region),
            smooth_px=float(smooth_px),
        )
        result["metrics"] = dict(result.get("metrics") or {})
        result["mode_label"] = "标定模式" if calibration else "相对模式"
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def save(result: dict, out_dir: str | Path | None = None) -> Path:
        directory = Path(out_dir) if out_dir else ROOT / "exports"
        directory.mkdir(parents=True, exist_ok=True)
        import time

        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = directory / f"thickness_map_{stamp}.png"
        overlay = result.get("overlay")
        if overlay is None:
            raise ValueError("分析结果中没有可保存的伪彩图")
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("伪彩图 PNG 编码失败")
        encoded.tofile(target)  # OpenCV imwrite 在 Windows 中文路径下不可靠
        return target

    @staticmethod
    def sample_colour(frame: np.ndarray, center_x: float | None = None) -> tuple[int, int, int]:
        """采样中心线附近的颜色（标定采集用）。"""
        reference = frame.shape[1] / 2.0 if center_x is None else float(center_x)
        r, g, b = sample_colour_band(frame, reference)
        return int(r), int(g), int(b)
