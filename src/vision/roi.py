"""干涉条纹有效区域搜索"""
from __future__ import annotations
import cv2
import numpy as np


def locate_fringe_roi(img_bgr: np.ndarray, pad_ratio: float = 0.10) -> tuple[int, int, int, int] | None:
    """
    通过亮度定位干涉条纹区域
    使用 OTSU 阈值 + 形态学 + 最大连通域
    """
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
    x, y, w, h = cv2.boundingRect(largest)

    # 加 padding
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_bgr.shape[1] - x, w + 2 * pad_x)
    h = min(img_bgr.shape[0] - y, h + 2 * pad_y)

    return (x, y, w, h)


def bright_mask(roi_bgr: np.ndarray) -> np.ndarray | None:
    """提取 ROI 中最亮的连通域"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask
