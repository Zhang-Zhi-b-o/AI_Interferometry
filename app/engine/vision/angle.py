"""图像旋转并扩展画布。"""
from __future__ import annotations
import cv2
import numpy as np


def rotate_expand(
    img: np.ndarray,
    angle_deg: float,
    border_value: tuple = (0, 0, 0),
) -> np.ndarray:
    """旋转图像并自动扩展画布防止裁剪"""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 计算新画布大小
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rot_mat[0, 2] += new_w / 2 - center[0]
    rot_mat[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(img, rot_mat, (new_w, new_h), borderValue=border_value)
