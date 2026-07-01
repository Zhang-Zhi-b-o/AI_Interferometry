"""白光干涉条纹中心检测 — 在扩展区域内定位零光程差位置"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from scipy.fft import fft, fftfreq, ifft


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _detect_fringe_orientation(gray: np.ndarray) -> str:
    """判断条纹方向: "vertical"(||||) 或 "horizontal"(====)"""
    H, W = gray.shape
    # 行方差 → 水平条纹变化大；列方差 → 垂直条纹变化大
    row_var = np.var(gray.mean(axis=1))  # 每行的均值再取方差
    col_var = np.var(gray.mean(axis=0))  # 每列的均值再取方差
    return "horizontal" if row_var > col_var * 1.3 else "vertical"


def _profile_1d(gray: np.ndarray, orientation: str) -> np.ndarray:
    """沿条纹方向做平均，得到正交方向的 1D 强度分布"""
    if orientation == "horizontal":
        # 条纹水平(=) → 纵向变化 → 水平方向平均
        return gray.mean(axis=1)
    else:
        # 条纹垂直(|||) → 横向变化 → 垂直方向平均
        return gray.mean(axis=0)


def _detrend(profile: np.ndarray, order: int = 2) -> np.ndarray:
    """低阶多项式去趋势"""
    n = len(profile)
    if n < order + 2:
        return profile - profile.mean()
    x = np.arange(n)
    trend = np.polyval(np.polyfit(x, profile, order), x)
    return profile - trend


def _estimate_period(profile: np.ndarray) -> float:
    """
    估计条纹周期（像素）。
    优先用自相关（对短信号更鲁棒），失败则回退到 FFT。
    """
    n = len(profile)
    y = _detrend(profile, 2)

    # 自相关法
    y_centered = y - y.mean()
    autocorr = np.correlate(y_centered, y_centered, mode="full")
    autocorr = autocorr[n - 1:]  # 只取正滞后
    autocorr = autocorr / max(autocorr[0], 1e-12)

    # 找第一个峰值（排除 lag=0）
    min_lag = max(3, n // 20)
    # 找自相关的局部最大值
    peaks = []
    for i in range(min_lag, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] >= autocorr[i + 1]:
            if autocorr[i] > 0.2:  # 显著峰值
                peaks.append((i, autocorr[i]))

    if peaks:
        # 取第一个显著峰值
        period_ac = float(peaks[0][0])
        # 合理性检查：周期应在 [3, n/2] 范围内
        if 3 <= period_ac <= n / 2:
            return period_ac

    # 回退到 FFT
    y_windowed = y * np.hanning(n)
    Y = np.abs(fft(y_windowed))[: n // 2]
    freqs = fftfreq(n, d=1.0)[: n // 2]
    mask = (freqs > 2.0 / n) & (freqs < 0.4)
    if mask.any():
        peak_idx = np.argmax(Y[mask])
        peak_freq = freqs[mask][peak_idx]
        if peak_freq > 1e-9:
            return 1.0 / peak_freq

    return n / 4.0


# ═══════════════════════════════════════════════════════════════════════════
# 检测方法
# ═══════════════════════════════════════════════════════════════════════════

def _method_envelope(profile: np.ndarray) -> tuple[float, np.ndarray]:
    """Hilbert 包络峰值法 — 条纹可见度最大处 = 零光程差"""
    n = len(profile)
    detrended = _detrend(profile, 3)
    period = _estimate_period(detrended)

    if period < 3 or period > n / 3:
        # 周期检测不可靠，回退到简单的平滑后峰值检测
        sigma = max(n / 15, 2.0)
        smooth = ndimage.gaussian_filter1d(detrended, sigma=sigma)
        envelope = np.abs(smooth)
        envelope = ndimage.gaussian_filter1d(envelope, sigma=sigma * 1.5)
        return float(np.argmax(envelope)), envelope

    carrier_freq = 1.0 / period
    Y = fft(detrended)
    freqs = fftfreq(n, d=1.0)
    bw = carrier_freq * 0.8
    lo = max(carrier_freq - bw, carrier_freq * 0.15)
    hi = min(carrier_freq + bw, 0.45)
    mask = (np.abs(freqs) >= lo) & (np.abs(freqs) <= hi)
    filtered = np.real(ifft(Y * mask))

    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)
    sigma = max(period * 1.2, 3.0)
    envelope = ndimage.gaussian_filter1d(envelope, sigma=sigma)
    return float(np.argmax(envelope)), envelope


def _method_gradient_midpoint(profile: np.ndarray) -> float:
    """
    梯度中点法 — 找中心条纹的边界，取中点。

    对白光干涉：零光程差处条纹最暗（或最亮），两侧对称。
    找到强度最低点附近的梯度翻转位置，取其中点。
    """
    n = len(profile)
    detrended = _detrend(profile, 2)
    smooth = ndimage.gaussian_filter1d(detrended, sigma=max(n / 40, 1.5))

    # 找全局最低点（零光程差处通常是最暗的条纹）
    min_idx = int(np.argmin(smooth))

    # 从最低点向两侧找最陡梯度位置
    grad = np.gradient(smooth)

    # 向左找正梯度峰值（暗→亮过渡）
    left_search = max(0, min_idx - n // 8)
    left_end = max(0, min_idx - 2)
    if left_end > left_search:
        grad_left = grad[left_search:left_end]
        left_edge = left_search + int(np.argmax(grad_left))
    else:
        left_edge = max(0, min_idx - n // 10)

    # 向右找负梯度峰值（亮→暗过渡）
    right_start = min(n, min_idx + 2)
    right_search = min(n, min_idx + n // 8)
    if right_search > right_start:
        grad_right = grad[right_start:right_search]
        right_edge = right_start + int(np.argmin(grad_right))
    else:
        right_edge = min(n, min_idx + n // 10)

    return float((left_edge + right_edge) / 2.0)


def _method_contrast(profile: np.ndarray) -> tuple[float, np.ndarray]:
    """局部峰谷对比度最大处（向量化，O(n)）"""
    n = len(profile)
    period = int(_estimate_period(profile))
    half_win = max(period // 2, 4)
    half_win = min(half_win, n // 4)

    # 用 scipy 滤波器代替逐像素循环
    local_max = ndimage.maximum_filter1d(profile, size=half_win * 2 + 1)
    local_min = ndimage.minimum_filter1d(profile, size=half_win * 2 + 1)
    contrast = local_max - local_min

    sigma = max(period * 0.7, 2.0)
    contrast = ndimage.gaussian_filter1d(contrast, sigma=sigma)
    return float(np.argmax(contrast)), contrast


def _method_symmetry(profile: np.ndarray) -> tuple[float, np.ndarray]:
    """镜像对称轴位置（步进采样，控制耗时）"""
    n = len(profile)
    margin = max(n // 8, 5)
    half_w = min(n // 3, 80)

    # 步进采样：每 2 像素检查一次，大幅减少计算量
    scores = np.zeros(n)
    for x in range(margin, n - margin, 2):
        length = min(x, n - x, half_w)
        if length < 8:
            continue
        left = profile[x - length: x]
        right = profile[x: x + length][::-1]
        ln, rn = left - left.mean(), right - right.mean()
        denom = np.sqrt(np.sum(ln ** 2) * np.sum(rn ** 2))
        if denom > 1e-12:
            scores[x] = float(np.sum(ln * rn) / denom)

    # 线性插值回填未采样点
    sampled = np.where(scores > 0)[0]
    if len(sampled) > 0:
        scores = np.interp(np.arange(n), sampled, scores[sampled])

    return float(np.argmax(scores)), scores


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def find_center_in_region(bgr_roi: np.ndarray) -> dict:
    """
    在图像区域内检测条纹中心位置。

    参数
    ----------
    bgr_roi : np.ndarray
        BGR 图像区域（uint8 [0,255]），应包含多个条纹周期

    返回
    -------
    dict:
        "center_x"    : 条纹中心在区域内的 x（或 y）坐标（像素）
        "center_y"    : 同上（始终提供，由 orientation 决定哪个是主方向）
        "confidence"  : 检测置信度 [0,1]
        "orientation" : "vertical" 或 "horizontal"
        "roi_width"   : 区域宽度
        "roi_height"  : 区域高度
        "methods"     : {方法名: 中心坐标}，各方法单独结果
    """
    if bgr_roi.ndim == 3:
        gray = bgr_roi.mean(axis=2)
    else:
        gray = bgr_roi.astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0

    H, W = gray.shape
    orientation = _detect_fringe_orientation(gray)
    profile = _profile_1d(gray, orientation)

    # 方法 1: 包络线峰值
    cx_env, envelope = _method_envelope(profile)

    # 方法 2: 对比度
    cx_ctr, _ = _method_contrast(profile)

    # 方法 3: 对称性
    cx_sym, _ = _method_symmetry(profile)

    # 方法 4: 梯度中点（仅作参考，权重较低）
    cx_grad = _method_gradient_midpoint(profile)

    # 检查方法间一致性，排除离群值
    all_cx = {"envelope": cx_env, "contrast": cx_ctr,
              "symmetry": cx_sym, "gradient": cx_grad}

    # 加权综合（降权梯度法）
    weights = {"envelope": 0.40, "contrast": 0.20,
               "symmetry": 0.25, "gradient": 0.15}
    weighted_sum = sum(weights[k] * v for k, v in all_cx.items())
    center_main = weighted_sum / sum(weights.values())

    # 一致性检查：去除最远的一个重新计算
    sorted_cx = sorted(all_cx.items(), key=lambda kv: abs(kv[1] - center_main))
    # 取前三个最接近的方法
    top3 = sorted_cx[:3]
    trimmed_sum = sum(weights[k] * v for k, v in top3)
    trimmed_w = sum(weights[k] for k, _ in top3)
    center_main = trimmed_sum / trimmed_w if trimmed_w > 0 else center_main

    # 置信度
    cx_values = [v for _, v in top3]
    spread = max(cx_values) - min(cx_values)
    profile_len = len(profile)
    confidence = max(0.0, 1.0 - spread / max(profile_len * 0.12, 5.0))

    return {
        "center_main": float(center_main),
        "center_x": float(center_main) if orientation == "vertical" else float(W / 2),
        "center_y": float(H / 2) if orientation == "vertical" else float(center_main),
        "confidence": float(min(confidence, 1.0)),
        "orientation": orientation,
        "roi_width": W,
        "roi_height": H,
        "methods": {k: float(v) for k, v in all_cx.items()},
    }
