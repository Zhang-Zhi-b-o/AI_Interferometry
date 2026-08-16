"""打印三张图 1D 亮度剖面的峰/谷结构（数值真相，不靠看图）。

用法: python profile_dump.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage, signal


def load(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(path)
    return img


def luminance(bgr: np.ndarray) -> np.ndarray:
    return (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1]
            + 0.299 * bgr[:, :, 2]).astype(np.float64)


def profile(img: np.ndarray) -> np.ndarray:
    gray = luminance(img)
    h, w = gray.shape
    gx = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
    keep = gx >= np.percentile(gx, 45)
    if np.count_nonzero(keep) < max(4, h // 5):
        keep[:] = True
    return gray[keep].mean(axis=0)


FILES = ["约0度.png", "约30度.png", "约40度.png"]
DIR = Path(__file__).resolve().parent

for f in FILES:
    img = load(DIR / f)
    prof = profile(img)
    w = len(prof)
    sm = ndimage.gaussian_filter1d(prof, sigma=4.0)
    rng = float(np.percentile(sm, 95) - np.percentile(sm, 5))
    dist = max(3, int(round(w / 25)))
    prom = max(0.5, rng * 0.02)
    pk, _ = signal.find_peaks(sm, distance=dist, prominence=prom)
    vl, _ = signal.find_peaks(-sm, distance=dist, prominence=prom)
    print(f"=== {f}  (w={w}, 亮度范围 {prof.min():.0f}~{prof.max():.0f}) ===")
    print(f"  平滑剖面 min/max = {sm.min():.0f}/{sm.max():.0f}")
    # 合并峰谷按位置排序
    ev = [(int(x), "P", float(sm[x])) for x in pk]
    ev += [(int(x), "V", float(sm[x])) for x in vl]
    ev.sort()
    print("  位置: 强度序列 (P=亮峰, V=暗谷):")
    line = "   "
    for x, t, v in ev:
        line += f"{x}{t}={v:.0f}  "
        if len(line) > 110:
            print(line)
            line = "   "
    if line.strip():
        print(line)
    print(f"  亮峰数={len(pk)} 暗谷数={len(vl)}")
    print()
