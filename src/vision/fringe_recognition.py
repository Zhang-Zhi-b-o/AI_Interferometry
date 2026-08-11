"""融合 YOLO、二维纹理和时间连续性的条纹场景识别。"""
from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np


def analyse_fringe_texture(bgr: np.ndarray) -> dict:
    """返回对倾斜、弯曲和变色条纹均较稳健的二维纹理证据。

    这里只判断画面中是否存在连续条纹及其大致水平位置，不承担零级中心
    的精确定位。算法先分离亮的样品区域，再在多个局部块中计算结构张量；
    因此不同局部块允许拥有不同方向，弯曲条纹不会被整幅投影互相抵消。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        return _empty_texture()
    if bgr.ndim == 2:
        gray = bgr.astype(np.uint8, copy=False)
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] >= 3:
        colour = bgr[:, :, :3]
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    else:
        return _empty_texture()

    original_height, original_width = gray.shape
    coordinate_scale = 1.0
    if original_width > 480:
        coordinate_scale = original_width / 480.0
        resized_height = max(32, int(round(original_height / coordinate_scale)))
        colour = cv2.resize(colour, (480, resized_height),
                            interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if min(height, width) < 32:
        return _empty_texture()

    # 数据中的样品明显亮于背景。最大亮连通域可排除画面边缘和黑背景，
    # 收缩后又可避免把样品外轮廓误认为条纹。
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    threshold = max(18.0, float(np.percentile(smooth, 72)) * 0.42)
    mask = (smooth >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return _empty_texture()
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < max(150, int(height * width * 0.008)):
        return _empty_texture()
    specimen = (labels == index).astype(np.uint8)
    inset = max(2, int(round(min(height, width) * 0.008)))
    specimen = cv2.erode(specimen, np.ones((2 * inset + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(specimen)
    if len(xs) < 100:
        return _empty_texture()

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    roi_mask = specimen[y1:y2, x1:x2].astype(bool)
    roi_gray = gray[y1:y2, x1:x2].astype(np.float32) / 255.0
    lab = cv2.cvtColor(colour[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)

    # 亮度与两个色度通道都参与。这样黑白、彩色、发生色相变化的条纹
    # 都能产生响应，而不会偏好某一种中心颜色。
    channels = [roi_gray, lab[:, :, 1] / 255.0, lab[:, :, 2] / 255.0]
    tensor_xx = np.zeros_like(roi_gray)
    tensor_yy = np.zeros_like(roi_gray)
    tensor_xy = np.zeros_like(roi_gray)
    detail = np.zeros_like(roi_gray)
    for channel in channels:
        local = channel - cv2.GaussianBlur(channel, (0, 0), 5.0)
        gx = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
        tensor_xx += cv2.GaussianBlur(gx * gx, (0, 0), 2.2)
        tensor_yy += cv2.GaussianBlur(gy * gy, (0, 0), 2.2)
        tensor_xy += cv2.GaussianBlur(gx * gy, (0, 0), 2.2)
        detail += np.sqrt(gx * gx + gy * gy)

    trace = tensor_xx + tensor_yy
    discriminant = np.sqrt(np.maximum(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy ** 2, 0.0))
    coherence = discriminant / np.maximum(trace, 1e-8)
    valid_detail = detail[roi_mask]
    energy_floor = max(float(np.percentile(valid_detail, 58)), 0.018)
    stripe_pixels = roi_mask & (detail >= energy_floor) & (coherence >= 0.38)

    # 局部块投票允许条纹方向沿空间变化。至少多个块同时有规则纹理才会
    # 得到高分，单个硬边缘无法独自触发视觉回退。
    rh, rw = roi_gray.shape
    block = max(12, min(rh, rw) // 7)
    votes: list[float] = []
    angles: list[float] = []
    for by in range(0, rh, block):
        for bx in range(0, rw, block):
            bm = roi_mask[by:by + block, bx:bx + block]
            if bm.size == 0 or np.mean(bm) < 0.55:
                continue
            bs = stripe_pixels[by:by + block, bx:bx + block]
            votes.append(float(np.mean(bs[bm])))
            xx = float(np.mean(tensor_xx[by:by + block, bx:bx + block][bm]))
            yy = float(np.mean(tensor_yy[by:by + block, bx:bx + block][bm]))
            xy = float(np.mean(tensor_xy[by:by + block, bx:bx + block][bm]))
            angles.append(float(0.5 * np.degrees(np.arctan2(2.0 * xy, xx - yy))))

    if not votes:
        return _empty_texture()
    votes_array = np.asarray(votes)
    active_ratio = float(np.mean(votes_array >= 0.10))
    # 对每个颜色通道沿水平方向作频谱分析。倾斜和弯曲主要改变各行的
    # 相位，不会抹去稳定的空间周期；普通表面细纹则通常没有突出的窄带
    # 谱峰。三个量的归一化区间由 dataset_pool_after_1240 的 646 张图标定。
    spectral_powers = []
    amplitudes = []
    horizontal_window = np.hanning(rw)[None, :]
    for channel in channels:
        high_frequency = (
            channel - cv2.GaussianBlur(channel, (0, 0), 6.0)
        ) * horizontal_window
        spectrum = np.abs(np.fft.rfft(high_frequency, axis=1)) ** 2
        frequency = np.fft.rfftfreq(rw)
        band = (frequency >= 1.0 / 50.0) & (frequency <= 1.0 / 4.0)
        spectral_powers.append(np.mean(spectrum[:, band], axis=0))
        amplitudes.append(float(np.sqrt(np.mean(high_frequency ** 2))))
    spectral_power = np.sum(spectral_powers, axis=0)
    total_power = max(float(np.sum(spectral_power)), 1e-12)
    median_power = max(float(np.median(spectral_power)), 1e-12)
    peak_power = float(np.max(spectral_power))
    oscillation_amplitude = max(amplitudes)
    peak_fraction = peak_power / total_power
    peak_snr = peak_power / median_power
    amplitude_score = np.clip((oscillation_amplitude - 0.018) / 0.025, 0.0, 1.0)
    periodicity_score = np.clip((peak_fraction - 0.08) / 0.12, 0.0, 1.0)
    spectral_snr_score = np.clip((peak_snr - 4.0) / 20.0, 0.0, 1.0)
    texture_score = float(np.clip(
        0.30 * amplitude_score
        + 0.45 * periodicity_score
        + 0.25 * spectral_snr_score,
        0.0, 1.0,
    ))

    # 拉普拉斯只用于识别“此帧是否模糊”，不作为条纹存在的硬门槛。
    lap_var = float(np.var(cv2.Laplacian(
        (roi_gray * 255).astype(np.uint8), cv2.CV_32F)[roi_mask]))
    sharpness = float(np.clip(lap_var / 55.0, 0.0, 1.0))
    # 模糊信息交给时序层决定保持多久，不能在这里直接抹掉仍可见的条纹。
    confidence = texture_score

    column_energy = np.sum(detail * stripe_pixels, axis=0)
    if float(column_energy.sum()) > 1e-9:
        position_x = x1 + float(np.sum(
            np.arange(rw, dtype=np.float64) * column_energy) / column_energy.sum())
    else:
        position_x = x1 + rw / 2.0
    angle_spread = float(np.std(angles)) if angles else 90.0
    return {
        "confidence": confidence,
        "position_x": position_x * coordinate_scale,
        "sharpness": sharpness,
        "blurred": sharpness < 0.30,
        "active_block_ratio": active_ratio,
        "angle_spread_deg": angle_spread,
        "specimen_bounds": tuple(int(round(v * coordinate_scale)) for v in (
            x1, y1, x2, y2)),
        "oscillation_amplitude": oscillation_amplitude,
        "peak_fraction": peak_fraction,
        "peak_snr": peak_snr,
    }


def _empty_texture() -> dict:
    return {
        "confidence": 0.0, "position_x": None, "sharpness": 0.0,
        "blurred": True, "active_block_ratio": 0.0,
        "angle_spread_deg": 90.0, "specimen_bounds": None,
        "oscillation_amplitude": 0.0, "peak_fraction": 0.0,
        "peak_snr": 0.0,
    }


class FringeRecognitionTracker:
    """用位置连续性、速度和模糊状态融合逐帧条纹证据。"""

    def __init__(self, history_size: int = 8, missing_hold_frames: int = 4,
                 visual_threshold: float = 0.40,
                 assisted_threshold: float = 0.25):
        self.history_size = max(3, int(history_size))
        self.missing_hold_frames = max(0, int(missing_hold_frames))
        self.visual_threshold = float(visual_threshold)
        self.assisted_threshold = float(assisted_threshold)
        self._history: deque[tuple[float, float]] = deque(maxlen=self.history_size)
        self._confidence = 0.0
        self._misses = 0
        self._last_source = ""

    def reset(self) -> None:
        self._history.clear()
        self._confidence = 0.0
        self._misses = 0
        self._last_source = ""

    def update(self, *, yolo_has_fringe: bool,
               yolo_position_x: float | None = None,
               yolo_confidence: float = 0.0,
               texture: dict | None = None,
               now: float | None = None) -> dict:
        now = time.monotonic() if now is None else float(now)
        texture = texture or _empty_texture()
        predicted, velocity = self._predict(now)
        visual_x = texture.get("position_x")
        visual_conf = float(texture.get("confidence", 0.0))
        blurred = bool(texture.get("blurred", False))

        source = ""
        position = None
        confidence = 0.0
        if yolo_has_fringe:
            source, position = "yolo", yolo_position_x
            confidence = max(0.55, float(yolo_confidence))
        else:
            threshold = (
                self.assisted_threshold if self._history
                else self.visual_threshold)
            plausible = self._is_plausible(visual_x, predicted, velocity, now)
            if visual_conf >= threshold and plausible:
                source, position, confidence = "visual", visual_x, visual_conf

        if source:
            self._misses = 0
            if position is not None and np.isfinite(position):
                self._history.append((now, float(position)))
            self._confidence = float(np.clip(confidence, 0.0, 1.0))
            self._last_source = source
            predicted, velocity = self._predict(now)
            return self._result(True, position, confidence, source, False,
                                velocity, texture)

        self._misses += 1
        # 电机转动造成模糊时多保留两帧，但置信度持续衰减；没有任何近期
        # 轨迹时绝不凭“模糊”本身猜测存在条纹。
        hold_limit = self.missing_hold_frames + (2 if blurred else 0)
        if self._history and self._misses <= hold_limit:
            confidence = self._confidence * (0.76 ** self._misses)
            return self._result(True, predicted, confidence, "history", True,
                                velocity, texture)

        if self._misses > hold_limit:
            self.reset()
        return self._result(False, None, 0.0, "", False, 0.0, texture)

    def _predict(self, now: float) -> tuple[float | None, float]:
        if not self._history:
            return None, 0.0
        if len(self._history) == 1:
            return self._history[-1][1], 0.0
        times = np.asarray([item[0] for item in self._history], dtype=np.float64)
        positions = np.asarray([item[1] for item in self._history], dtype=np.float64)
        dt = np.diff(times)
        valid = dt > 1e-4
        speeds = np.diff(positions)[valid] / dt[valid]
        velocity = float(np.median(speeds)) if len(speeds) else 0.0
        velocity = float(np.clip(velocity, -1200.0, 1200.0))
        horizon = float(np.clip(now - times[-1], 0.0, 0.25))
        return float(positions[-1] + velocity * horizon), velocity

    def _is_plausible(self, value: float | None, predicted: float | None,
                      velocity: float, now: float) -> bool:
        if value is None or not np.isfinite(value):
            return False
        if predicted is None:
            return True
        last_time = self._history[-1][0]
        allowance = 55.0 + min(140.0, abs(velocity) * max(0.0, now - last_time))
        return abs(float(value) - predicted) <= allowance

    @staticmethod
    def _result(has_fringe: bool, position: float | None, confidence: float,
                source: str, held: bool, velocity: float,
                texture: dict) -> dict:
        return {
            "has_fringe": has_fringe,
            "position_x": position,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "source": source,
            "held": held,
            "velocity_px_s": float(velocity),
            "blurred": bool(texture.get("blurred", False)),
            "texture_confidence": float(texture.get("confidence", 0.0)),
        }
