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
    if not np.all(np.isfinite(data)):
        raise ValueError("Colour calibration contains non-finite values")
    opd = data[:, 0].astype(np.float64)
    delta = np.diff(opd)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Colour calibration OPD values must be strictly monotonic")
    rgb = data[:, 1:4].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return opd, lab


def calibrated_opd_map(
    image: np.ndarray,
    mask: np.ndarray,
    calibration: Path,
    wavelength_nm: float = 589.3,
) -> tuple[np.ndarray, np.ndarray]:
    """用连续标定曲线和空间相位先验估计 OPD（μm）。

    标定点按 OPD 顺序连接成 Lab 色彩空间中的分段线性曲线，像素投影到线段
    内部而不是吸附到离散标定点。白光颜色会周期性重复，因此同时使用空间解包
    相位作为连续性先验，在颜色相近的多个周期分支中选择与邻域变化一致的一支。
    """
    opd, cal_lab = load_colour_calibration(calibration)
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    pixels = lab.reshape(-1, 3)
    flat_mask = mask.reshape(-1)

    starts = cal_lab[:-1].astype(np.float64)
    vectors = np.diff(cal_lab.astype(np.float64), axis=0)
    lengths2 = np.einsum("ij,ij->i", vectors, vectors)
    usable = lengths2 > 1e-9
    if int(np.count_nonzero(usable)) < 3:
        raise ValueError("Colour calibration has too few distinct adjacent colours")
    starts = starts[usable]
    vectors = vectors[usable]
    lengths2 = lengths2[usable]
    opd_start = opd[:-1][usable]
    opd_delta = np.diff(opd)[usable]

    # 颜色距离的稳健尺度只负责平衡数据项，不改变 OPD 的物理单位。
    from scipy.spatial import cKDTree
    nearest_distance, _ = cKDTree(cal_lab).query(pixels, k=1)
    colour_scale = max(float(np.percentile(nearest_distance[flat_mask], 90)), 1.0)

    wrapped, phase_confidence = colour_phase_map(image, mask)
    phase = unwrap_relative_phase(wrapped, phase_confidence, mask).reshape(-1)
    # 项目测量模型：一个完整颜色周期对应 OPD = λ/2；因此
    # OPD_prior = phase/(2π) * λ/2，λ 从 nm 显式换算为 μm。
    opd_period_um = wavelength_nm / 2000.0
    phase_opd = phase * opd_period_um / (2.0 * math.pi)
    opd_center = float(np.median(opd))

    mapped_by_sign = [np.empty(pixels.shape[0], dtype=np.float64) for _ in range(2)]
    distance_by_sign = [np.empty(pixels.shape[0], dtype=np.float64) for _ in range(2)]
    gap_by_sign = [np.empty(pixels.shape[0], dtype=np.float64) for _ in range(2)]
    score_by_sign = [np.empty(pixels.shape[0], dtype=np.float64) for _ in range(2)]
    chunk_size = 8192
    continuity_weight = 2.5
    for start in range(0, pixels.shape[0], chunk_size):
        stop = min(start + chunk_size, pixels.shape[0])
        points = pixels[start:stop]
        delta = points[:, None, :] - starts[None, :, :]
        fraction = np.einsum("nsi,si->ns", delta, vectors) / lengths2[None, :]
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
        residual = points[:, None, :] - projected
        colour_distance2 = np.einsum("nsi,nsi->ns", residual, residual)
        opd_candidates = opd_start[None, :] + fraction * opd_delta[None, :]

        for sign_index, sign in enumerate((1.0, -1.0)):
            prior = opd_center + sign * phase_opd[start:stop, None]
            continuity = (opd_candidates - prior) / max(opd_period_um, 1e-9)
            score = colour_distance2 / (colour_scale ** 2) + continuity_weight * continuity ** 2
            choice = np.argmin(score, axis=1)
            rows = np.arange(stop - start)
            mapped_by_sign[sign_index][start:stop] = opd_candidates[rows, choice]
            distance_by_sign[sign_index][start:stop] = np.sqrt(
                colour_distance2[rows, choice])
            score_by_sign[sign_index][start:stop] = score[rows, choice]
            two_best = np.partition(score, 1, axis=1)[:, :2]
            gap_by_sign[sign_index][start:stop] = two_best[:, 1] - two_best[:, 0]

    sign_quality = [
        float(np.median(scores[flat_mask])) for scores in score_by_sign
    ]
    selected = int(np.argmin(sign_quality))
    mapped = mapped_by_sign[selected].reshape(mask.shape)
    selected_distance = distance_by_sign[selected].reshape(mask.shape)
    selected_gap = gap_by_sign[selected].reshape(mask.shape)

    # 匹配质量不是统计覆盖概率。它综合四类可观察证据：
    # 1) 像素到连续标定曲线的 Lab 残差；尺度取标定相邻点的典型间距，避免
    #    用待测图自身的低对比度把全部像素错误压成低分；
    # 2) 所选 OPD 分支与空间解包相位先验的一致性；
    # 3) 原始条纹色度/纹理质量；4) 最佳与次佳分支的可分辨程度。
    calibration_spacing = max(float(np.median(np.sqrt(lengths2))), 5.0)
    fit_quality = np.exp(-0.5 * (selected_distance / calibration_spacing) ** 2)
    selected_sign = 1.0 if selected == 0 else -1.0
    prior_map = (opd_center + selected_sign * phase_opd).reshape(mask.shape)
    branch_error = np.abs(mapped - prior_map)
    branch_scale = max(0.45 * opd_period_um, 1e-9)
    continuity_quality = np.exp(-0.5 * (branch_error / branch_scale) ** 2)
    gap_scale = max(float(np.percentile(selected_gap[mask], 75)), 1e-6)
    ambiguity_quality = 1.0 - np.exp(-selected_gap / gap_scale)
    confidence = (
        0.45 * fit_quality
        + 0.25 * continuity_quality
        + 0.25 * phase_confidence
        + 0.05 * ambiguity_quality
    )

    # 邻域稳健化只降低单像素噪声，不跨越 ROI/样品掩膜传播。
    confidence[~mask] = 0.0
    numerator = ndimage.gaussian_filter(confidence * mask, sigma=1.0)
    denominator = ndimage.gaussian_filter(mask.astype(np.float64), sigma=1.0)
    confidence = np.divide(
        numerator, denominator, out=np.zeros_like(confidence), where=denominator > 1e-6)
    confidence = np.clip(confidence, 0.0, 1.0)
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


def compute_metrics(
    thickness: np.ndarray,
    confidence: np.ndarray,
    *,
    has_absolute_anchor: bool = False,
) -> dict[str, float | None]:
    valid = np.isfinite(thickness)
    if not np.any(valid):
        raise ValueError("No finite thickness pixels")
    values = thickness[valid]
    p2, p5, p95, p98 = np.percentile(values, (2, 5, 95, 98))
    clipped = values[(values >= p2) & (values <= p98)]
    gy, gx = np.gradient(np.nan_to_num(thickness, nan=float(np.nanmedian(thickness))))
    gradient = np.hypot(gx, gy)
    gradient[~valid] = np.nan

    mean_um = float(np.mean(clipped))
    median_um = float(np.median(clipped))
    pv_robust_um = float(p98 - p2)
    rms_um = float(np.sqrt(np.mean((clipped - mean_um) ** 2)))
    p90_span_um = float(p95 - p5)

    # 归一化：把体现不均匀度的量换算成「相对厚度」的百分比。参考厚度取稳健
    # 均值（绝对值）；绝对厚度锚定后约等于锚定值，相对模式（中位数为 0）下
    # 接近 0，此时百分比无意义，置为 None。
    thickness_ref_um = abs(mean_um) if has_absolute_anchor else 0.0
    if has_absolute_anchor and thickness_ref_um < 1e-9:
        thickness_ref_um = abs(median_um)
    if has_absolute_anchor and thickness_ref_um > 0.0:
        pv_robust_pct = pv_robust_um / thickness_ref_um * 100.0
        rms_pct = rms_um / thickness_ref_um * 100.0
        p90_span_pct = p90_span_um / thickness_ref_um * 100.0
    else:
        pv_robust_pct = rms_pct = p90_span_pct = None

    return {
        "valid_pixels": int(valid.sum()),
        "mean_um": mean_um,
        "median_um": median_um,
        "min_robust_um": float(p2),
        "max_robust_um": float(p98),
        "pv_robust_um": pv_robust_um,
        "rms_um": rms_um,
        "p90_span_um": p90_span_um,
        "median_confidence": float(np.median(confidence[valid])),
        "median_gradient_um_per_px": float(np.nanmedian(gradient)),
        "p95_gradient_um_per_px": float(np.nanpercentile(gradient, 95)),
        # —— 归一化（相对厚度百分比）版本 ——
        "thickness_ref_um": thickness_ref_um,
        "pv_robust_pct": pv_robust_pct,
        "rms_pct": rms_pct,
        "p90_span_pct": p90_span_pct,
    }


def heatmap_bgr(values: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """把厚度图渲染成伪彩 BGR 图（NaN 处显示为灰色底）。"""
    lo, hi = robust_limits(values)
    filled = np.nan_to_num(values, nan=lo)
    normalized = np.clip((filled - lo) / (hi - lo), 0, 1)
    heat = cv2.applyColorMap(np.uint8(normalized * 255), cmap)
    heat[~np.isfinite(values)] = (238, 238, 238)
    return heat


def heatmap_with_legend_bgr(
    values: np.ndarray,
    cmap: int = cv2.COLORMAP_TURBO,
    unit: str = "um",
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """渲染厚度热图，并在右侧添加连续色标、刻度和单位。"""
    heat = heatmap_bgr(values, cmap)
    return add_continuous_legend_bgr(
        heat, values, cmap=cmap, unit=unit, value_range=value_range)


def add_continuous_legend_bgr(
    image: np.ndarray,
    values: np.ndarray,
    *,
    cmap: int = cv2.COLORMAP_TURBO,
    unit: str = "um",
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """给与数值场对应的 BGR 图追加连续色标，不改动原图像素。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("legend image must be a BGR image")
    if values.shape[:2] != image.shape[:2]:
        raise ValueError("legend values must match image height and width")
    lo, hi = value_range if value_range is not None else robust_limits(values)
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("legend value range must be finite and increasing")
    heat = image
    height, width = heat.shape[:2]
    legend_width = 118
    canvas = np.full((height, width + legend_width, 3), 255, dtype=np.uint8)
    canvas[:, :width] = heat
    if height < 60:
        top, bottom = 1, max(2, height - 2)
    else:
        top = max(12, int(round(height * 0.06)))
        bottom = min(height - 18, int(round(height * 0.94)))
    bar_x = width + 18
    bar_w = 22
    gradient = np.linspace(255, 0, bottom - top + 1, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, bar_w, axis=1)
    canvas[top:bottom + 1, bar_x:bar_x + bar_w] = cv2.applyColorMap(gradient, cmap)
    cv2.rectangle(canvas, (bar_x, top), (bar_x + bar_w, bottom), (40, 40, 40), 1)
    for fraction in np.linspace(0.0, 1.0, 5):
        y = int(round(bottom - fraction * (bottom - top)))
        value = lo + fraction * (hi - lo)
        cv2.line(canvas, (bar_x + bar_w, y), (bar_x + bar_w + 5, y), (40, 40, 40), 1)
        cv2.putText(
            canvas, f"{value:.2f}", (bar_x + bar_w + 8, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, unit, (bar_x + 1, max(11, top - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


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
        opd_um, confidence = calibrated_opd_map(
            image, mask, Path(calibration), wavelength_nm=wavelength_nm)
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

    has_absolute_anchor = reference_thickness_um is not None
    metrics = compute_metrics(
        thickness, confidence, has_absolute_anchor=has_absolute_anchor)
    metrics.update({
        "wavelength_nm": float(wavelength_nm),
        "refractive_index": float(refractive_index),
        "mode": mode,
        "has_reference": reference_image is not None,
        "has_absolute_anchor": has_absolute_anchor,
    })

    # 归一化厚度分布：把厚度换算成相对稳健中位数的百分比偏差，使不均匀程度
    # 以相对厚度的百分比呈现（锚定后数值远小于原始 μm 波动，分布图更平缓）。
    # 相对模式（无绝对厚度基准、中位数≈0）下参考厚度为 0，不生成归一化图。
    normalized_thickness = None
    ref = metrics.get("thickness_ref_um")
    if ref:
        normalized_thickness = (
            (thickness - float(metrics["median_um"])) / float(ref) * 100.0
        ).astype(np.float32)

    return {
        "thickness": thickness.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "mask": mask,
        "metrics": metrics,
        "overlay": overlay_bgr(image, thickness),
        "heatmap": heatmap_bgr(thickness),
        "heatmap_legend": heatmap_with_legend_bgr(thickness, unit="um"),
        "normalized_thickness": normalized_thickness,
        "mode": mode,
        "wavelength_nm": float(wavelength_nm),
        "refractive_index": float(refractive_index),
        "step_um": float(step_um),
    }
