"""画面旋转校正 & 缩放平移"""
from __future__ import annotations
import cv2
import numpy as np


class FrameCorrector:
    """管理画面旋转角度、缩放和平移参数"""

    def __init__(self, max_angle: float = 60.0, gain: float = 0.70):
        self.max_angle = max_angle  # 自动校正角度限幅（度）
        self.gain = gain            # 校正增益
        self.auto_angle = 0.0       # 自动估计的角度
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
        limited = max(-self.max_angle, min(self.max_angle, self.auto_angle))
        return limited + self.manual_offset

    def update_auto_angle(self, estimated: float, smooth: float = 0.5):
        """指数平滑更新自动角度"""
        self.auto_angle = smooth * estimated + (1 - smooth) * self.auto_angle

    def set_manual_offset(self, offset: float):
        self.manual_offset = offset

    def reset_all(self):
        self.auto_angle = 0.0
        self.manual_offset = 0.0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

    # ------------------------------------------------------------------
    # 帧变换
    # ------------------------------------------------------------------
    def apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """旋转 + 扩展画布"""
        angle = self.effective_angle
        if abs(angle) < 0.01:
            return frame

        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(rot_mat[0, 0])
        sin = abs(rot_mat[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rot_mat[0, 2] += new_w / 2 - center[0]
        rot_mat[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(frame, rot_mat, (new_w, new_h), borderValue=(0, 0, 0))

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
