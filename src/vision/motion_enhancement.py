"""连续旋转时的轻量级条纹画面增强。"""
from __future__ import annotations

import cv2
import numpy as np


class MotionFrameEnhancer:
    """增强竖条纹的水平边缘，同时限制噪声和反卷积振铃。"""

    def __init__(self, settings: dict | None = None):
        settings = dict(settings or {})
        self.enabled = bool(settings.get("enabled", True))
        self.base_strength = float(settings.get("sharpen_strength", 0.65))
        self.max_strength = max(
            self.base_strength,
            float(settings.get("max_sharpen_strength", 1.35)),
        )
        self.stripe_strength = float(settings.get(
            "stripe_contrast_strength", 1.6))
        self.max_stripe_strength = max(
            self.stripe_strength,
            float(settings.get("max_stripe_contrast_strength", 3.0)),
        )
        self.color_gain = max(1.0, float(settings.get("color_gain", 1.45)))
        self.max_color_gain = max(
            self.color_gain, float(settings.get("max_color_gain", 1.9)))
        self.original_mix = max(
            0.0, min(1.0, float(settings.get("original_mix", 0.35))))
        kernel_size = max(3, int(settings.get("horizontal_kernel_size", 5)))
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        vertical_size = max(1, int(settings.get("vertical_smooth_size", 9)))
        self.vertical_size = vertical_size if vertical_size % 2 else vertical_size + 1
        background_size = max(5, int(settings.get("background_kernel_size", 31)))
        self.background_size = (
            background_size if background_size % 2 else background_size + 1)
        self.contrast_gain = max(0.5, float(settings.get("contrast_gain", 1.06)))
        self.last_strength = 0.0
        self.last_stripe_strength = 0.0
        self.last_color_gain = 1.0

    def apply(
        self,
        frame: np.ndarray,
        *,
        clarity_score: float | None = None,
        clarity_baseline: float | None = None,
        blur_ratio: float = 0.55,
    ) -> np.ndarray:
        """返回同尺寸 BGR 增强帧；输入帧不会被原地修改。"""
        if not self.enabled or frame is None or frame.size == 0:
            self.last_strength = 0.0
            self.last_stripe_strength = 0.0
            self.last_color_gain = 1.0
            return frame

        blur_amount = self._blur_amount(
            clarity_score, clarity_baseline, blur_ratio)
        strength = self.base_strength + (
            self.max_strength - self.base_strength) * blur_amount
        stripe_strength = self.stripe_strength + (
            self.max_stripe_strength - self.stripe_strength) * blur_amount
        color_gain = self.color_gain + (
            self.max_color_gain - self.color_gain) * blur_amount

        # 先沿竖直条纹方向平均噪声，再减去较宽的水平背景。
        # 这样能从不均匀照明中提取低对比度、近似周期性的竖条纹。
        vertical_smooth = cv2.GaussianBlur(
            frame, (1, self.vertical_size), sigmaX=0, sigmaY=0)
        horizontal_background = cv2.GaussianBlur(
            vertical_smooth, (self.background_size, 1), sigmaX=0)
        stripe_layer = cv2.addWeighted(
            vertical_smooth, 1.0 + stripe_strength,
            horizontal_background, -stripe_strength, 0)
        restored = cv2.addWeighted(
            frame, self.original_mix,
            stripe_layer, 1.0 - self.original_mix, 0)

        # 白光干涉条纹主要为竖向结构，旋转时拖影主要沿水平方向。
        # 仅恢复水平方向高频，比二维锐化更少放大上下边缘和传感器噪声。
        horizontal_blur = cv2.GaussianBlur(
            restored, (self.kernel_size, 1), sigmaX=0)
        restored = cv2.addWeighted(
            restored, 1.0 + strength, horizontal_blur, -strength, 0)

        # 放大相对于灰度分量的颜色差异，使白光干涉的弱彩色条纹更明显。
        gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        restored = cv2.addWeighted(
            restored, color_gain, gray_bgr, 1.0 - color_gain, 0)
        if abs(self.contrast_gain - 1.0) > 0.001:
            restored = cv2.convertScaleAbs(restored, alpha=self.contrast_gain)

        self.last_strength = strength
        self.last_stripe_strength = stripe_strength
        self.last_color_gain = color_gain
        return restored

    @staticmethod
    def _blur_amount(
        score: float | None,
        baseline: float | None,
        blur_ratio: float,
    ) -> float:
        if score is None or baseline is None or baseline <= 0:
            return 0.0
        ratio = max(0.0, float(score) / float(baseline))
        threshold = max(0.05, min(1.0, float(blur_ratio)))
        return max(0.0, min(1.0, (threshold - ratio) / threshold))
