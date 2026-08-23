"""单帧白光干涉彩色条纹的薄膜厚度分布估计。

从 Thickness_variation_analysis/analyze.py 移植的核心算法，去掉命令行与文件 I/O，
改为纯内存接口：传入一帧矫正后的 BGR 画面，返回逐像素厚度图、置信度图、掩膜、
统计指标与可直接显示的伪彩叠加图。

算法边界（重要）：
- 默认「相对」模式把 Lab 色彩空间的循环色彩坐标解包为连续相位，假设一个完整
  颜色周期对应一个有效干涉级次，厚度步长 Δt = λ_eff / (2(n-1))。这只是模型依赖
  的相对厚度估计，不是可溯源的绝对厚度。
- 「标定」模式用 opd_um,r,g,b 颜色标定表把像素颜色匹配到已扣除基准的 OPD，
  再按 t = opd_um / (n-1) 换算，接近绝对厚度，但仍依赖标定表与无膜基准。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase


def largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8)
    if count <= 1:
        return binary.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def sample_mask(image: np.ndarray) -> np.ndarray:
    """分割亮的薄膜区域，排除大部分黑色背景。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(light, (0, 0), 2.0)
    nonzero = blurred[blurred > 3]
    if nonzero.size == 0:
        return np.ones(light.shape, dtype=bool)
    threshold, _ = cv2.threshold(
        np.clip(blurred, 0, 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Otsu 对偏暗样品可能过严，保留中等照度像素。
    floor = max(8.0, min(float(threshold), float(np.percentile(nonzero, 38))))
    mask = blurred >= floor
    radius = max(2, int(round(min(image.shape[:2]) * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=max(1, radius // 2))
    return mask.astype(bool)


def sample_colour(image: np.ndarray) -> tuple[int, int, int]:
    """返回样品区域的中位颜色 (r, g, b)；无有效掩膜时退回全图中位。"""
    mask = sample_mask(image)
    if int(mask.sum()) < 50:
        mask = np.ones(image.shape[:2], dtype=bool)
    bgr = image[mask]
    med = np.median(bgr, axis=0)
    return int(round(med[2])), int(round(med[1])), int(round(med[0]))


def sample_colour_band(
    image: np.ndarray, x: float, half_width: int = 5
) -> tuple[int, int, int]:
    """返回图像中 x 附近一条竖直窄带内「条纹像素」的中位颜色 (r, g, b)。

    供颜色→光程差标定取「画面中心线所在条纹」的颜色：等厚干涉的彩色条纹近似
    竖直、颜色沿水平方向变化，取中心线 x±half_width 的窄带；但中心线往往比条纹
    更长（上下延伸到黑色背景），故先用亮区域掩膜筛掉背景，只对条纹像素取中位数。
    """
    height, width = image.shape[:2]
    cx = int(round(float(x)))
    x0 = max(0, cx - half_width)
    x1 = min(width, cx + half_width + 1)
    if x1 <= x0:
        raise ValueError(f"采样列 ({cx}) 超出图像范围")
    band = image[:, x0:x1]
    mask = sample_mask(image)[:, x0:x1]
    bgr = band[mask]
    if bgr.shape[0] < 50:
        # 亮条纹像素太少（几乎全黑），退回整条窄带，避免取到空结果。
        bgr = band.reshape(-1, 3)
    med = np.median(bgr, axis=0)
    return int(round(med[2])), int(round(med[1])), int(round(med[0]))


def colour_phase_map(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """估计循环色彩坐标（已包裹相位）与置信度。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    # 去除缓慢的照度/色偏；残差色彩在白光连续色带间旋转。
    sigma = max(5.0, min(image.shape[:2]) / 18.0)
    a_res = a - ndimage.gaussian_filter(a, sigma=sigma)
    b_res = b - ndimage.gaussian_filter(b, sigma=sigma)
    wrapped = np.arctan2(b_res, a_res)

    chroma = np.hypot(a_res, b_res)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    local = gray - ndimage.gaussian_filter(gray, sigma=max(3.0, sigma / 3.0))
    texture = ndimage.gaussian_filter(np.abs(local), 1.2)
    c_scale = max(float(np.percentile(chroma[mask], 90)), 1e-6)
    t_scale = max(float(np.percentile(texture[mask], 90)), 1e-6)
    confidence = np.clip(0.72 * chroma / c_scale + 0.28 * texture / t_scale, 0, 1)
    confidence[~mask] = 0
    return wrapped, confidence


def unwrap_relative_phase(
    wrapped: np.ndarray, confidence: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """对掩膜相位解包并做置信度加权平滑，输出相对相位（中位数为 0）。"""
    phase = unwrap_phase(np.ma.array(wrapped, mask=~mask)).filled(np.nan)
    finite = np.isfinite(phase) & mask
    if not np.any(finite):
        raise ValueError("No valid phase pixels")
    weights = confidence.copy()
    weights[~finite] = 0
    values = np.nan_to_num(phase) * weights
    sigma = max(1.2, min(wrapped.shape) / 180.0)
    num = ndimage.gaussian_filter(values, sigma=sigma)
    den = ndimage.gaussian_filter(weights, sigma=sigma)
    smooth = np.divide(num, den, out=np.nan_to_num(phase), where=den > 1e-4)
    smooth[~mask] = np.nan
    smooth -= float(np.nanmedian(smooth))
    return smooth


def load_colour_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取列 opd_um,r,g,b（OPD，须已扣除基准）→ (opd, lab)。"""
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (float(row["opd_um"]), float(row["r"]), float(row["g"]), float(row["b"])))
    if len(rows) < 4:
        raise ValueError("Colour calibration needs at least four rows")
    data = np.asarray(rows, dtype=np.float32)
    rgb = data[:, 1:4].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return data[:, 0].astype(np.float64), lab


def calibrated_opd_map(
    image: np.ndarray, mask: np.ndarray, calibration: Path
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    opd, cal_lab = load_colour_calibration(calibration)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    tree = cKDTree(cal_lab)
    distance, index = tree.query(lab.reshape(-1, 3), k=1)
    mapped = opd[index].reshape(mask.shape)
    scale = max(float(np.percentile(distance[mask], 90)), 1e-6)
    confidence = np.clip(1.0 - distance.reshape(mask.shape) / (1.5 * scale), 0, 1)
    mapped = ndimage.median_filter(mapped, size=3)
    mapped[~mask] = np.nan
    confidence[~mask] = 0
    return mapped, confidence


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, (2, 98))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return float(lo), float(hi)


def compute_metrics(thickness: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(thickness)
    if not np.any(valid):
        raise ValueError("No finite thickness pixels")
    values = thickness[valid]
    p2, p5, p95, p98 = np.percentile(values, (2, 5, 95, 98))
    clipped = values[(values >= p2) & (values <= p98)]
    gy, gx = np.gradient(np.nan_to_num(thickness, nan=float(np.nanmedian(thickness))))
    gradient = np.hypot(gx, gy)
    gradient[~valid] = np.nan
    return {
        "valid_pixels": int(valid.sum()),
        "mean_um": float(np.mean(clipped)),
        "median_um": float(np.median(clipped)),
        "min_robust_um": float(p2),
        "max_robust_um": float(p98),
        "pv_robust_um": float(p98 - p2),
        "rms_um": float(np.sqrt(np.mean((clipped - np.mean(clipped)) ** 2))),
        "p90_span_um": float(p95 - p5),
        "median_confidence": float(np.median(confidence[valid])),
        "median_gradient_um_per_px": float(np.nanmedian(gradient)),
        "p95_gradient_um_per_px": float(np.nanpercentile(gradient, 95)),
    }


def heatmap_bgr(values: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """把厚度图渲染成伪彩 BGR 图（NaN 处显示为灰色底）。"""
    lo, hi = robust_limits(values)
    filled = np.nan_to_num(values, nan=lo)
    normalized = np.clip((filled - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap(np.uint8(normalized * 255), cmap)


def overlay_bgr(image: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """在原图上叠加半透明厚度伪彩。"""
    lo, hi = robust_limits(thickness)
    normalized = np.clip((np.nan_to_num(thickness, nan=lo) - lo) / (hi - lo), 0, 1)
    heat = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    valid = np.isfinite(thickness)
    overlay = image.copy()
    overlay[valid] = cv2.addWeighted(image, 0.38, heat, 0.62, 0)[valid]
    return overlay


def _smooth_thickness(values: np.ndarray, sigma: float) -> np.ndarray:
    """对厚度图做归一化高斯平滑，保持掩膜外为 NaN。

    用有效像素掩膜做加权归一化，避免边界像素被背景 0 值拉低；用于消除标定
    模式下最近邻匹配带来的量化像素拼接感，让过渡更柔和。
    """
    valid = np.isfinite(values)
    if not np.any(valid):
        return values
    filled = np.nan_to_num(values, nan=0.0)
    smooth = ndimage.gaussian_filter(filled, sigma=sigma)
    weight = ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma)
    out = np.divide(smooth, weight, out=filled, where=weight > 1e-3)
    out[~valid] = np.nan
    return out


def _thickness_map(
    image: np.ndarray,
    *,
    wavelength_nm: float,
    refractive_index: float,
    calibration: str | Path | None,
    invert: bool,
    whole_region: bool = False,
    smooth_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """单图厚度图：返回 (thickness, confidence, mask, mode, step_um)。

    ``whole_region=True`` 时把整幅（通常是用户框选后裁剪出的区域）都当作
    有效区域，跳过亮膜自动分割。用于已明确指定分析区域、画面不含大片黑背景
    的情形，避免 Otsu 在纯亮区把掩膜误收缩到局部。``smooth_px`` 控制最终
    厚度图的高斯平滑半径（像素），0 表示不平滑。
    """
    mask = (
        np.ones(image.shape[:2], dtype=bool)
        if whole_region else sample_mask(image)
    )
    if mask.sum() < 200:
        raise ValueError("自动检测到的薄膜区域过小，请框选或调整画面")

    if calibration:
        opd_um, confidence = calibrated_opd_map(image, mask, Path(calibration))
        thickness = opd_um / (refractive_index - 1.0)
        mode = "calibrated"
        step_um = float("nan")
    else:
        wrapped, confidence = colour_phase_map(image, mask)
        phase = unwrap_relative_phase(wrapped, confidence, mask)
        step_um = wavelength_nm / (2.0 * (refractive_index - 1.0)) / 1000.0
        thickness = phase / (2.0 * math.pi) * step_um
        mode = "relative"

    if invert:
        thickness = -thickness
    if smooth_px and smooth_px > 0:
        thickness = _smooth_thickness(thickness, smooth_px)
    valid = np.isfinite(thickness)
    thickness = thickness - float(np.nanmedian(thickness))
    thickness[~valid] = np.nan
    return thickness, confidence, mask, mode, step_um


def analyze_thickness_distribution(
    image: np.ndarray,
    *,
    wavelength_nm: float = 589.3,
    refractive_index: float = 1.523,
    calibration: str | Path | None = None,
    invert: bool = False,
    reference_thickness_um: float | None = None,
    reference_image: np.ndarray | None = None,
    whole_region: bool = False,
    smooth_px: float = 2.0,
) -> dict:
    """从单帧 BGR 画面估计厚度分布，返回内存结果字典。

    ``reference_image`` 提供无膜基准图时，用同一套参数计算其厚度图并做差，
    扣除系统固有光程差的空间分布。``reference_thickness_um`` 提供一个绝对
    厚度基准（μm，通常由中心条纹读数与初始读数之差按 |Δ|/20×1000/(n-1)
    得到），会在厚度图（中位数为 0 的相对分布）基础上平移，把分布锚定到
    绝对厚度。``whole_region=True`` 时把整幅（用户框选后裁剪出的区域）都
    当作有效区域，跳过亮膜自动分割。``smooth_px`` 控制最终厚度图的高斯
    平滑半径（像素，0 表示不平滑），用于柔和过渡、消除标定最近邻匹配的
    量化拼接感。返回键：thickness (float32 厚度图，掩膜外为 NaN)、
    confidence、mask、metrics、overlay(伪彩叠加 BGR)、heatmap(伪彩 BGR)、
    mode、wavelength_nm、refractive_index、step_um。
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if refractive_index <= 1.0:
        raise ValueError("refractive_index must be greater than 1")

    thickness, confidence, mask, mode, step_um = _thickness_map(
        image, wavelength_nm=wavelength_nm, refractive_index=refractive_index,
        calibration=calibration, invert=invert, whole_region=whole_region,
        smooth_px=smooth_px)

    if reference_image is not None:
        ref = reference_image
        if ref.shape[:2] != image.shape[:2]:
            raise ValueError("无膜基准图尺寸与当前画面不一致，请用同一相机同一分辨率重新捕获")
        ref_thickness, _, _, _, _ = _thickness_map(
            ref, wavelength_nm=wavelength_nm, refractive_index=refractive_index,
            calibration=calibration, invert=invert, whole_region=whole_region,
            smooth_px=smooth_px)
        thickness = thickness - ref_thickness
        valid = np.isfinite(thickness)
        if np.any(valid):
            thickness = thickness - float(np.nanmedian(thickness))
        thickness[~valid] = np.nan

    if reference_thickness_um is not None:
        thickness = thickness + float(reference_thickness_um)

    metrics = compute_metrics(thickness, confidence)
    metrics.update({
        "wavelength_nm": float(wavelength_nm),
        "refractive_index": float(refractive_index),
        "mode": mode,
        "has_reference": reference_image is not None,
    })

    return {
        "thickness": thickness.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "mask": mask,
        "metrics": metrics,
        "overlay": overlay_bgr(image, thickness),
        "heatmap": heatmap_bgr(thickness),
        "mode": mode,
        "wavelength_nm": float(wavelength_nm),
        "refractive_index": float(refractive_index),
        "step_um": float(step_um),
    }
