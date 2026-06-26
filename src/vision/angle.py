"""FFT 条纹角度估计 & 图像旋转"""
from __future__ import annotations
import cv2
import numpy as np


def estimate_stripe_angle(img_bgr: np.ndarray, n_angles: int = 360) -> float:
    """
    通过 2D FFT 频域分析估计干涉条纹的倾斜角度
    步骤：定位亮区 → 特征提取(灰度+LAB+DoG) → 2D FFT → 极坐标能量投影 → 峰值检测
    返回角度（-90 到 +90 度）
    """
    h, w = img_bgr.shape[:2]

    # 1. 定位条纹 ROI
    roi = _locate_bright_roi(img_bgr)
    if roi is None:
        return 0.0
    rx, ry, rw, rh = roi
    roi_bgr = img_bgr[ry:ry + rh, rx:rx + rw]

    # 2. 亮区掩膜
    mask = _bright_mask(roi_bgr)
    if mask is None or np.count_nonzero(mask) < 50:
        return 0.0

    # 3. 特征图：灰度 + LAB a/b + DoG
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    blur2 = cv2.GaussianBlur(gray, (0, 0), 3.0)
    dog = blur1 - blur2  # Difference of Gaussians

    features = np.stack([
        gray,
        lab[:, :, 1],  # a 通道
        lab[:, :, 2],  # b 通道
        dog,
    ], axis=-1)

    # 4. 零均值 + 软掩膜
    features -= np.mean(features, axis=(0, 1), keepdims=True)
    mask_f = mask.astype(np.float32) / 255.0
    features *= mask_f[..., None]

    # 5. 填充到正方形
    size = max(rw, rh)
    canvas = np.zeros((size, size), dtype=np.float32)
    y0, x0 = (size - rh) // 2, (size - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = np.mean(features, axis=-1)

    # 6. 2D FFT
    fft = np.fft.fftshift(np.fft.fft2(canvas))
    magnitude = np.abs(fft)

    # 7. 带通滤波
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_min, r_max = size * 0.03, size * 0.45
    mask_ring = (radius >= r_min) & (radius <= r_max)
    magnitude *= mask_ring

    # 8. 极坐标能量投影
    bins = 180
    energy = np.zeros(bins)
    for yi in range(size):
        for xi in range(size):
            if not mask_ring[yi, xi]:
                continue
            dx, dy = xi - cx, yi - cy
            angle = np.arctan2(dy, dx)  # -pi .. pi
            angle_deg = np.degrees(angle)
            if angle_deg < 0:
                angle_deg += 180
            bin_idx = int(angle_deg * bins / 180) % bins
            energy[bin_idx] += magnitude[yi, xi]

    # 9. 平滑 + 峰值
    energy = np.convolve(energy, np.ones(5) / 5, mode="same")
    peak_idx = int(np.argmax(energy))
    angle = peak_idx * 180.0 / bins

    # 归一化到 -90 ~ +90
    if angle > 90:
        angle -= 180
    return angle


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


# ------------------------------------------------------------------
# 内部辅助函数（提取自旧项目）
# ------------------------------------------------------------------

def _locate_bright_roi(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """定位图像中亮度最高的区域"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def _bright_mask(roi_bgr: np.ndarray) -> np.ndarray | None:
    """提取 ROI 中最亮的连通域作为掩膜"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask
