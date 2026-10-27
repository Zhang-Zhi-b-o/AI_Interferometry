"""画面旋转校正 & 缩放平移"""
from __future__ import annotations
import cv2
import numpy as np


class FrameCorrector:
    """管理画面手动旋转、缩放和平移参数。"""

    def __init__(self):
        self.manual_offset = 0.0    # 手动偏置角度
        self.zoom = 1.0             # 缩放系数
        self.pan_x = 0              # X 平移（像素）
        self.pan_y = 0              # Y 平移（像素）

    # ------------------------------------------------------------------
    # 有效角度
    # ------------------------------------------------------------------
    @property
    def effective_angle(self) -> float:
        """最终用于旋转的角度"""
        return self.manual_offset

    def set_manual_offset(self, offset: float):
        self.manual_offset = offset

    def reset_all(self):
        self.manual_offset = 0.0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

    def apply_zoom_pan(self, frame: np.ndarray) -> np.ndarray:
        """缩放 + 平移（裁剪方式）"""
        if self.zoom <= 0:
            return frame
        h, w = frame.shape[:2]
        new_w, new_h = int(w / self.zoom), int(h / self.zoom)
        cx, cy = w // 2 + self.pan_x, h // 2 + self.pan_y
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        if x2 <= x1 or y2 <= y1:
            return frame
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h))
