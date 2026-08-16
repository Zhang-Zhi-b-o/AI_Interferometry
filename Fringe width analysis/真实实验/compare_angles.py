"""比较三个角度下的条纹间隔，判断间隔与角度的关系。

用法: python compare_angles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.fringe_width import measure_center_fringe_width_2d


def load(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(path)
    return img


FILES = ["约0度.png", "约30度.png", "约40度.png"]
DIR = Path(__file__).resolve().parent

for f in FILES:
    img = load(DIR / f)
    r = measure_center_fringe_width_2d(img)
    bright = [b for b in r["bands"] if b["kind"] == "bright"]
    bright.sort(key=lambda b: b["center_x"])
    xs = [b["center_x"] for b in bright]
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    widths = [b["width"] for b in bright]
    med = float(np.median(diffs)) if diffs else float("nan")
    span = (xs[-1] - xs[0]) if len(xs) >= 2 else float("nan")
    print(f"--- {f}  ({r['frame_width']}x{r['frame_height']}) ---")
    print(f"  period_px={r['period_px']}  num_bright={r['num_bright']}  "
          f"num_dark={r['num_dark']}")
    print(f"  bright centers x = {[round(x, 1) for x in xs]}")
    print(f"  bright-bright 间隔 = {[round(d, 1) for d in diffs]}")
    print(f"  中位间隔 = {med:.1f} px   均值间隔 = {np.mean(diffs):.1f} px   "
          f"跨度 = {span:.1f} px")
    print(f"  亮纹宽度(边界) = {[round(w, 1) for w in widths]}")
    print(f"  亮纹 FWHM = {[round(b['fwhm'], 1) if b['fwhm'] else None for b in bright]}")
    print()
