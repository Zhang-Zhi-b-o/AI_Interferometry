"""白光干涉竖条纹的零光程差中心定位。"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage, signal

from src.vision.fringe_width import locate_central_band


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values, (5, 95))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _estimate_period(profile: np.ndarray) -> float:
    """用自相关估计竖条纹横向周期，失败时给出保守值。"""
    y = np.asarray(profile, dtype=np.float64)
    n = len(y)
    if n < 12:
        return max(3.0, n / 4.0)
    y = y - ndimage.gaussian_filter1d(y, sigma=max(4.0, n / 12.0))
    y -= y.mean()
    corr = signal.correlate(y, y, mode="full", method="fft")[n - 1:]
    if corr[0] <= 1e-12:
        return max(3.0, n / 8.0)
    corr /= corr[0]
    min_lag = max(3, n // 60)
    max_lag = max(min_lag + 1, min(n // 3, 100))
    peaks, _ = signal.find_peaks(corr[min_lag:max_lag], prominence=0.04)
    if len(peaks):
        # 首个显著峰是基频周期；不选幅度最大的高次峰。
        return float(peaks[0] + min_lag)
    return float(np.clip(n / 10.0, 4.0, 40.0))


def _profiles_for_vertical_stripes(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """提取横向颜色剖面，并估计图像呈竖条纹的可信程度。"""
    image = np.asarray(bgr, dtype=np.float64)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.max() > 1.5:
        image /= 255.0

    # 只聚合横向纹理最清楚的行，避免圆形光斑外的黑背景淹没条纹。
    luminance = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
    gx = np.mean(np.abs(np.diff(luminance, axis=1)), axis=1)
    keep = gx >= np.percentile(gx, 45)
    if np.count_nonzero(keep) < max(4, image.shape[0] // 5):
        keep[:] = True
    profiles = np.mean(image[keep], axis=0).T

    x_energy = float(np.mean(np.abs(np.diff(luminance[keep], axis=1))))
    y_energy = float(np.mean(np.abs(np.diff(luminance[keep], axis=0))))
    verticality = x_energy / max(x_energy + y_energy, 1e-12)
    return profiles, float(np.clip(verticality, 0.0, 1.0))


def _symmetry_at(envelope: np.ndarray, center: int, radius: int) -> float:
    length = min(center, len(envelope) - center - 1, radius)
    if length < 4:
        return 0.0
    left = envelope[center - length:center]
    right = envelope[center + 1:center + length + 1][::-1]
    scale = max(float(np.std(np.r_[left, right])), 1e-6)
    return float(np.exp(-np.mean(np.abs(left - right)) / scale))


def find_center_in_region(
    bgr_roi: np.ndarray,
    expected_center_x: float | None = None,
    search_radius: float | None = None,
    search_bounds: tuple[float, float] | None = None,
) -> dict:
    """定位白光干涉竖条纹的相干包络中心。

    中心条纹可以是黑、白、蓝或其他颜色。算法不搜索某一种颜色，而是
    联合三个颜色通道的局部条纹能量、包络左右对称性和可选的 YOLO 位置
    先验。``search_bounds`` 用于指定 YOLO 零级框在 ROI 内的左右边界；框
    仅限定搜索区域，不假设框的几何中心就是物理中心条纹。所有坐标均相对
    于 ``bgr_roi``。
    """
    if not isinstance(bgr_roi, np.ndarray) or bgr_roi.size == 0:
        raise ValueError("中心检测输入为空")
    if bgr_roi.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr_roi.ndim}")
    height, width = bgr_roi.shape[:2]
    if min(height, width) < 20:
        raise ValueError(f"中心检测区域过小: {bgr_roi.shape[:2]}")

    profiles, verticality = _profiles_for_vertical_stripes(bgr_roi)
    dynamic_range = float(max(np.percentile(p, 95) - np.percentile(p, 5) for p in profiles))
    if dynamic_range < 0.03:
        raise ValueError("图像对比度过低，未检测到有效条纹")

    luma = 0.114 * profiles[0] + 0.587 * profiles[1] + 0.299 * profiles[2]
    period = _estimate_period(luma)
    background_sigma = max(period * 1.8, width / 18.0, 5.0)
    high_pass = np.vstack([
        channel - ndimage.gaussian_filter1d(channel, background_sigma)
        for channel in profiles
    ])

    # 相干区的多通道振荡能量高；中心颜色改变不会改变这一几何特征。
    energy = np.sum(high_pass ** 2, axis=0)
    gradient = np.sum(np.gradient(high_pass, axis=1) ** 2, axis=0)
    envelope_sigma = max(2.0, period * 0.85)
    energy = ndimage.gaussian_filter1d(energy, envelope_sigma)
    gradient = ndimage.gaussian_filter1d(gradient, envelope_sigma)
    coherence = 0.68 * _normalise(energy) + 0.32 * _normalise(gradient)

    margin = max(3, int(round(period / 2)))
    lo, hi = margin, width - margin
    expected = None
    if expected_center_x is not None and np.isfinite(expected_center_x):
        expected = float(np.clip(expected_center_x, 0, width - 1))

    bounded_by_zero_box = search_bounds is not None
    if search_bounds is not None:
        bound_left, bound_right = sorted((float(search_bounds[0]), float(search_bounds[1])))
        # 略微避开框边缘，防止把检测框外的结构边界当成条纹中心。
        inset = max(
            2,
            int(round(period * 0.08)),
            int(round((bound_right - bound_left) * 0.04)),
        )
        lo = max(lo, int(np.floor(bound_left)) + inset)
        hi = min(hi, int(np.ceil(bound_right)) - inset + 1)
    elif expected is not None:
        requested_radius = float(
            search_radius if search_radius is not None else max(8.0, width * 0.12)
        )
        # 无零级框边界约束时，预期中心仅作软锚定；
        # 搜索半径放宽以允许图像特征主导定位，避免追逐不稳定框。
        radius = max(requested_radius, max(8.0, width * 0.10))
        lo = max(lo, int(np.floor(expected - radius)))
        hi = min(hi, int(np.ceil(expected + radius)) + 1)
    if hi <= lo:
        raise ValueError("中心搜索范围为空")

    symmetry = np.zeros(width, dtype=np.float64)
    symmetry_radius = max(8, int(round(period * 2.5)))
    for x in range(lo, hi):
        symmetry[x] = _symmetry_at(coherence, x, symmetry_radius)

    # 在完整零级框内对“条纹本体”评分。颜色离散度只衡量饱和程度，不偏好
    # 蓝、红或绿；相对暗谷也只是软特征，因此不会退化为“寻找黑线”。
    smooth_profiles = ndimage.gaussian_filter1d(
        profiles, sigma=max(0.8, period * 0.06), axis=1)
    smooth_luma = (
        0.114 * smooth_profiles[0]
        + 0.587 * smooth_profiles[1]
        + 0.299 * smooth_profiles[2]
    )
    local_background = ndimage.gaussian_filter1d(
        smooth_luma, sigma=max(3.0, period * 0.45))
    darkness = _normalise(np.maximum(local_background - smooth_luma, 0.0))
    chroma = _normalise(np.std(smooth_profiles, axis=0))
    stripe_strength = _normalise(np.sqrt(np.sum(high_pass ** 2, axis=0)))

    colour_gradient = np.sqrt(
        np.sum(np.gradient(smooth_profiles, axis=1) ** 2, axis=0))
    edge_pair = np.zeros(width, dtype=np.float64)
    edge_radius = max(3, int(round(period * 0.55)))
    for x in range(max(lo, edge_radius), min(hi, width - edge_radius)):
        left_edge = float(np.max(colour_gradient[x - edge_radius:x]))
        right_edge = float(np.max(colour_gradient[x + 1:x + edge_radius + 1]))
        balance = min(left_edge, right_edge) / max(left_edge, right_edge, 1e-9)
        edge_pair[x] = np.sqrt(left_edge * right_edge) * balance
    edge_pair = _normalise(edge_pair)

    if expected is None:
        prior = np.zeros(width, dtype=np.float64)
        prior[lo:hi] = 0.5
    else:
        # 只作弱平局判据；允许物理中心偏离 YOLO 框中心多个条纹周期。
        prior_sigma = max(3.0, (hi - lo) * (0.50 if bounded_by_zero_box else 0.28))
        xx = np.arange(width, dtype=np.float64)
        prior = np.exp(-0.5 * ((xx - expected) / prior_sigma) ** 2)

    if bounded_by_zero_box:
        box_interior = np.zeros(width, dtype=np.float64)
        box_width = max(bound_right - bound_left, 1.0)
        relative_x = np.clip((np.arange(width) - bound_left) / box_width, 0.0, 1.0)
        box_interior = np.sqrt(np.maximum(np.sin(np.pi * relative_x), 0.0))

        # --- YOLO-prior-adaptive scoring ---
        # 当条纹接近竖直时 (verticality >= 0.55)，图像特征主导评分，
        # YOLO prior 权重 12%。当条纹弯曲/旋转时 (verticality < 0.55)，
        # 水平剖面平均会模糊相干包络信号，YOLO 框中心更可靠，prior
        # 权重逐步提升至最高 30%。
        base_prior = 0.12
        base_darkness = 0.22
        base_chroma = 0.26
        base_stripe = 0.15
        base_edge = 0.10
        base_symmetry = 0.09
        base_box_interior = 0.06

        if verticality < 0.55:
            vert_factor = max(0.0, (0.55 - verticality) / 0.55)
            prior_boost = 0.18 * vert_factor
            prior_w = base_prior + prior_boost
            scale = (1.0 - prior_w) / (1.0 - base_prior)
            darkness_w = base_darkness * scale
            chroma_w = base_chroma * scale
            stripe_w = base_stripe * scale
            edge_w = base_edge * scale
            symmetry_w = base_symmetry * scale
            box_interior_w = base_box_interior * scale
        else:
            prior_w = base_prior
            darkness_w = base_darkness
            chroma_w = base_chroma
            stripe_w = base_stripe
            edge_w = base_edge
            symmetry_w = base_symmetry
            box_interior_w = base_box_interior

        score = (
            darkness_w * darkness
            + chroma_w * chroma
            + stripe_w * stripe_strength
            + edge_w * edge_pair
            + symmetry_w * symmetry
            + prior_w * prior
            + box_interior_w * box_interior
        )
    elif expected is None:
        score = 0.78 * coherence + 0.22 * symmetry
    else:
        # 无零级框边界时，预期中心仅作弱平局判据（15%），
        # 图像特征（相干度 + 对称性）主导中心定位。
        score = 0.43 * coherence + 0.42 * symmetry + 0.15 * prior

    best = lo + int(np.argmax(score[lo:hi]))
    # 三点抛物线作亚像素精修，避免整数像素来回跳。
    center = float(best)
    if 0 < best < width - 1:
        a, b, c = score[best - 1:best + 2]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            center += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))

    peak = float(score[best])
    baseline = float(np.median(score[lo:hi]))
    prominence = np.clip((peak - baseline) / max(1.0 - baseline, 1e-6), 0.0, 1.0)
    contrast_quality = np.clip(dynamic_range / 0.18, 0.0, 1.0)
    orientation_quality = np.clip((verticality - 0.30) / 0.35, 0.0, 1.0)
    prior_quality = 0.85 if expected is not None else 0.65
    confidence = contrast_quality * (0.35 + 0.65 * prominence) * (
        0.45 + 0.55 * orientation_quality
    ) * prior_quality

    # 只有明显由横向结构主导时才拒绝；轻微倾斜的竖条纹仍视为有效。
    orientation = "horizontal" if verticality < 0.30 else "vertical"

    # 用条纹宽度分析的明暗轮廓标定中心：以 YOLO 零级框为边界、以分数定位
    # 结果为参考，把中心吸附到最近一段条纹的极值（亮峰/暗谷），比纯分数
    # 更贴合实际条纹轮廓。仅在轮廓与分数一致（落在该段条纹附近）时采纳，
    # 避免单帧切分错误把中心拉偏。
    band = locate_central_band(
        bgr_roi, center_x=center, search_bounds=search_bounds)
    band_corrected = False
    if band is not None and band["confidence"] >= 0.10:
        agree_radius = max(band["width"] * 0.5, period * 0.25, 2.0)
        if abs(band["center_x"] - center) <= agree_radius:
            center = 0.65 * band["center_x"] + 0.35 * center
            band_corrected = True
            confidence = float(np.clip(
                confidence + 0.04 * band["confidence"], 0.0, 1.0))

    return {
        "center_main": center,
        "center_x": center,
        "center_y": float(height / 2),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "orientation": orientation,
        "verticality": verticality,
        "period": float(period),
        "roi_width": width,
        "roi_height": height,
        "band": band,
        "band_corrected": band_corrected,
        "methods": {
            "coherence": float(np.argmax(coherence[lo:hi]) + lo),
            "symmetry": float(np.argmax(symmetry[lo:hi]) + lo),
            "stripe_candidate": float(best),
            "darkness": float(darkness[best]),
            "chroma": float(chroma[best]),
            "edge_pair": float(edge_pair[best]),
            "prior": float(expected) if expected is not None else float("nan"),
        },
    }


class CenterTracker:
    """对中心位置做抗抖、跳变拒绝和短时丢帧保持。"""

    def __init__(self, hold_frames: int = 3, max_jump_px: float = 60.0):
        self.hold_frames = max(0, int(hold_frames))
        self.max_jump_px = float(max_jump_px)
        self._history: deque[float] = deque(maxlen=5)
        self._center: float | None = None
        self._confidence = 0.0
        self._misses = 0

    def reset(self) -> None:
        self._history.clear()
        self._center = None
        self._confidence = 0.0
        self._misses = 0

    @property
    def center(self) -> float | None:
        return self._center

    def update(self, center: float | None, confidence: float = 0.0) -> dict:
        if center is None or not np.isfinite(center):
            self._misses += 1
            if self._center is None or self._misses > self.hold_frames:
                self.reset()
                return {"center": None, "confidence": 0.0, "held": False, "accepted": False}
            return {
                "center": self._center,
                "confidence": self._confidence * (0.78 ** self._misses),
                "held": True,
                "accepted": False,
            }

        value = float(center)
        conf = float(np.clip(confidence, 0.0, 1.0))
        if self._center is not None and abs(value - self._center) > self.max_jump_px and conf < 0.65:
            return self.update(None, 0.0)

        self._misses = 0
        self._history.append(value)
        median = float(np.median(self._history))
        self._center = median if self._center is None else 0.35 * self._center + 0.65 * median
        self._confidence = conf
        return {"center": self._center, "confidence": conf, "held": False, "accepted": True}

    def reset_from_yolo(self, yolo_center: float, yolo_confidence: float) -> dict:
        """用 YOLO 零级框中心重置追踪器。

        当精细中心线检测失败但 YOLO 框仍在时使用此方法，
        YOLO 框位置成为新的锚点，后续帧可平滑过渡回中心线检测。

        为避免中值滤波在过渡期间将 YOLO 种子与真实测量混合导致偏移，
        前 3 个历史位置均填入 YOLO 中心值，确保前几帧 EMA 贴近锚点。
        """
        self._history.clear()
        self._center = float(yolo_center)
        self._confidence = float(np.clip(yolo_confidence, 0.0, 1.0))
        self._misses = 0
        seed_copies = min(3, self._history.maxlen)
        for _ in range(seed_copies):
            self._history.append(float(yolo_center))
        return {"center": self._center, "confidence": self._confidence,
                "held": False, "accepted": True}
