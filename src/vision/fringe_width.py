"""白光干涉条纹宽度测量 — 分析当前画面并给出中心条纹宽度。

把 ``Fringe width analysis/analyze_fringes.py`` 里的条纹分段算法（一维亮度
剖面 + 周期估计 + 明暗极值切分）收敛成可在实时画面复用的纯函数：输入一张
BGR 帧和一个参考位置（中心条纹 x，缺省用画面中心），输出该位置处条纹的
宽度、类型与边界，以及整体条纹周期。

竖直条纹沿水平方向变化最剧烈，因此取横向亮度剖面；只聚合横向纹理最清楚
的行，避免圆形光斑外的黑背景淹没条纹。所有坐标均为像素、相对整张输入图。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """BGR -> 亮度（Rec.601）。"""
    image = np.asarray(bgr, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    )


def _luma_profile(bgr: np.ndarray) -> np.ndarray:
    """取水平亮度剖面，只聚合横向纹理最清楚的行。"""
    gray = _luminance(bgr)
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        raise ValueError("图像过小，无法提取条纹剖面")
    gx = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
    keep = gx >= np.percentile(gx, 45)
    if np.count_nonzero(keep) < max(4, h // 5):
        keep[:] = True
    return np.mean(gray[keep], axis=0)


def _estimate_period(luma: np.ndarray) -> float:
    """用自相关估计条纹横向周期（亮纹中心到下一个亮纹中心的像素距离）。"""
    n = len(luma)
    if n < 12:
        return max(3.0, n / 4.0)
    y = luma - ndimage.gaussian_filter1d(luma, sigma=max(4.0, n / 12.0))
    y -= y.mean()
    corr = signal.correlate(y, y, mode="full", method="fft")[n - 1:]
    if corr[0] <= 1e-12:
        return max(3.0, n / 8.0)
    corr /= corr[0]
    min_lag = max(3, n // 60)
    max_lag = max(min_lag + 1, min(n // 3, 100))
    peaks, _ = signal.find_peaks(corr[min_lag:max_lag], prominence=0.04)
    if len(peaks):
        return float(peaks[0] + min_lag)
    return float(np.clip(n / 10.0, 4.0, 40.0))


def _fwhm(luma: np.ndarray, peak_x: int, left: int, right: int) -> float | None:
    """亮纹的半高全宽，相对局部基线测量。"""
    lo, hi = max(int(left), 0), min(int(right), len(luma) - 1)
    if hi - lo < 2:
        return None
    seg = luma[lo:hi + 1].astype(np.float64)
    baseline = float(np.percentile(seg, 20))
    peak = float(seg.max())
    half = baseline + 0.5 * (peak - baseline)
    if peak - baseline < 1e-6:
        return None
    above = np.where(seg >= half)[0]
    if len(above) == 0:
        return None
    return float(above[-1] - above[0] + 1)


def _find_bands(luma: np.ndarray, period: float) -> list[dict]:
    """把一维亮度剖面切分成一段段明/暗条纹，返回 ``{kind, center_x, left,
    right, width, peak_x, fwhm}`` 列表。"""
    width = len(luma)
    smooth = ndimage.gaussian_filter1d(luma, sigma=max(0.8, period * 0.06))
    distance = max(2, int(round(period * 0.4)))
    rng = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    prominence = max(0.5, rng * 0.02)

    bright_x, _ = signal.find_peaks(smooth, distance=distance, prominence=prominence)
    dark_x, _ = signal.find_peaks(-smooth, distance=distance, prominence=prominence)

    # 贴边伪条纹剔除
    border = max(3, int(round(period * 0.15)))
    bright_x = [int(x) for x in bright_x if border <= x <= width - 1 - border]
    dark_x = [int(x) for x in dark_x if border <= x <= width - 1 - border]

    # 亮度过滤：真正的亮纹明显高于暗背景
    baseline = float(np.percentile(smooth, 10))
    if bright_x:
        heights = [float(smooth[x]) - baseline for x in bright_x]
        max_h = max(heights)
        bright_x = [x for x, h in zip(bright_x, heights) if h >= 0.25 * max_h]

    extrema = [(float(x), 1.0, float(smooth[x])) for x in bright_x]
    extrema += [(float(x), -1.0, float(smooth[x])) for x in dark_x]

    # 单条纹兜底：有亮峰但无暗谷，把暗背景边缘补成暗锚点
    if bright_x and not dark_x:
        extrema.append((0.0, -1.0, float(smooth[0])))
        extrema.append((float(width - 1), -1.0, float(smooth[-1])))

    extrema.sort(key=lambda t: t[0])

    # 合并同类型且靠得过近的极值（保留更极端的那个），保证明暗交替
    merged: list[tuple[float, float, float]] = []
    for x, kind, val in extrema:
        if merged and merged[-1][1] == kind and x - merged[-1][0] < distance:
            prev_x, _, prev_val = merged[-1]
            if (kind > 0 and val > prev_val) or (kind < 0 and val < prev_val):
                merged[-1] = (x, kind, val)
            continue
        merged.append((x, kind, val))

    # 剔除没有被暗谷夹住的亮峰（边缘单调抬升的伪亮纹）
    kept: list[tuple[float, float, float]] = []
    for i, (x, kind, val) in enumerate(merged):
        if kind > 0:
            has_left = any(k < 0 for _, k, _ in merged[:i])
            has_right = any(k < 0 for _, k, _ in merged[i + 1:])
            if not (has_left and has_right):
                continue
        kept.append((x, kind, val))
    merged = kept

    if len(merged) < 2:
        return []

    bands: list[dict] = []
    for i, (x, kind, val) in enumerate(merged):
        left = 0.0 if i == 0 else (merged[i - 1][0] + x) / 2.0
        right = float(width - 1) if i == len(merged) - 1 else (x + merged[i + 1][0]) / 2.0
        kind_name = "bright" if kind > 0 else "dark"
        fwhm = _fwhm(smooth, int(x), int(left), int(right)) if kind_name == "bright" else None
        bands.append({
            "kind": kind_name,
            "center_x": float((left + right) / 2.0),
            "left": float(left),
            "right": float(right),
            "width": float(right - left),
            "peak_x": float(x),
            "fwhm": fwhm,
        })
    return bands


def measure_center_fringe_width(
    bgr: np.ndarray,
    center_x: float | None = None,
) -> dict:
    """分析一张 BGR 画面，给出中心条纹（``center_x`` 处）的宽度。

    参考位置缺省用画面中心。返回可 JSON 化的结构；识别不到条纹时
    ``center_band`` 为 None、``num_bands`` 为 0。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    h, w = bgr.shape[:2]

    luma = _luma_profile(bgr)
    period = _estimate_period(luma)
    bands = _find_bands(luma, period)

    ref = float(np.clip(center_x, 0, w - 1)) if (
        center_x is not None and np.isfinite(center_x)) else w / 2.0

    center_band = None
    if bands:
        # 优先取边界包含参考位置的那段，否则取中心最近的。
        containing = [b for b in bands if b["left"] <= ref <= b["right"]]
        center_band = (
            containing[0] if containing
            else min(bands, key=lambda b: abs(b["center_x"] - ref)))

    return {
        "frame_width": w,
        "frame_height": h,
        "period_px": round(float(period), 2),
        "num_bands": len(bands),
        "num_bright": sum(1 for b in bands if b["kind"] == "bright"),
        "num_dark": sum(1 for b in bands if b["kind"] == "dark"),
        "reference_x": round(ref, 2),
        "center_band": center_band,
    }
