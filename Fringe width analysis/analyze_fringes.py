"""白光干涉竖条纹的逐段分析 — 位置 / 宽度 / 轮廓 / 颜色。

针对 YOLO 预标注图像（``20260811-0513-10.9599711-wan`` 等）里的白光干涉条纹，
把图像中每一段明、暗条纹分别识别出来，并给出：

- **位置**：该条纹在图像中的横向中心坐标（像素）。
- **宽度**：条纹的横向跨度（相邻极值中点之间的边界宽度），亮纹额外给出半高全宽 FWHM。
- **轮廓**：条纹横截面的亮度剖面（每个像素的灰度 / 三通道强度），用于刻画条纹形状。
- **颜色**：条纹中心区域的代表性 RGB 颜色及其色相命名。

条纹是竖直走向，因此算法沿水平方向取亮度剖面（只聚合纹理最清楚的行），
用自相关估计条纹周期后，通过平滑剖面找亮/暗极值，再用相邻极值的中点
作为条纹边界，从而把每一段条纹切分出来。

运行：``python analyze_fringes.py [图片目录] [输出目录] [示例张数]``
默认读取 ``images`` 子目录，处理前 4 张有有效条纹的图。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, signal

# Windows 控制台默认 GBK，强制 UTF-8 输出以便中文正常显示
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# matplotlib 中文字体（Windows 自带微软雅黑），否则剖面图中文会变方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """以 Unicode 安全的方式读图（OpenCV 的 imread 在 Windows 上不认中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """BGR -> 亮度（Rec.601）。"""
    return 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]


def extract_profiles(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """取水平亮度剖面（BGR 三通道 + 亮度），只聚合横向纹理最清楚的行。

    竖直条纹在水平方向（x）变化最剧烈，因此把图像沿行平均成一维剖面；
    为避免圆形光斑外的黑背景淹没条纹，只保留横向梯度较强的行。
    """
    image = img.astype(np.float64)
    h, w = image.shape[:2]
    gray = _luminance(image)
    gx = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)  # 每行的水平梯度
    keep = gx >= np.percentile(gx, 45)
    if np.count_nonzero(keep) < max(4, h // 5):
        keep[:] = True

    profiles = np.mean(image[keep], axis=0).T  # (3, w)，BGR 各一行
    luma = np.mean(gray[keep], axis=0)          # (w,)
    return profiles, luma


def estimate_period(luma: np.ndarray) -> float:
    """用自相关估计条纹横向周期（亮纹中心到下一个亮纹中心的像素距离）。"""
    n = len(luma)
    if n < 12:
        return max(3.0, n / 4.0)
    y = luma - ndimage.gaussian_filter1d(luma, sigma=max(4.0, n / 12.0))
    y -= y.mean()
    corr = signal.correlate(y, y, mode="full", method="fft")[n - 1:]
    if corr[0] <= 1e-12:
        return max(3.0, n / 8.0)
    corr /= corr[0]
    min_lag = max(3, n // 60)
    max_lag = max(min_lag + 1, min(n // 3, 100))
    peaks, _ = signal.find_peaks(corr[min_lag:max_lag], prominence=0.04)
    if len(peaks):
        return float(peaks[0] + min_lag)
    return float(np.clip(n / 10.0, 4.0, 40.0))


def _color_name(rgb: np.ndarray) -> str:
    """把 RGB 归到一组可读的色相名。

    白光干涉条纹饱和度低（粉彩），色相本身已足以区分紫红 / 蓝 / 黄绿等，
    因此仅在接近中性灰时加「浅」前缀，避免所有颜色都被标成「浅 X」。
    """
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    chroma = mx - mn
    if chroma < 18:  # 近灰
        if mx > 200:
            return "白"
        if mx < 90:
            return "黑"
        return "灰"
    # 色相（0~360）
    if mx == r:
        hue = (60 * ((g - b) / chroma)) % 360
    elif mx == g:
        hue = 60 * ((b - r) / chroma) + 120
    else:
        hue = 60 * ((r - g) / chroma) + 240
    if chroma < 30:
        sat = "浅"
    elif mn < 90:
        sat = "深"
    else:
        sat = ""
    names = [
        (15, "红"), (45, "橙"), (70, "黄"), (160, "绿"),
        (200, "青"), (250, "蓝"), (300, "紫"), (345, "洋红"), (360, "红"),
    ]
    name = next((nm for deg, nm in names if hue < deg), "红")
    return sat + name


@dataclass
class Band:
    """一段条纹的描述。"""
    kind: str                       # "bright" 亮纹 / "dark" 暗纹
    center_x: float                 # 条纹中心横向坐标（像素）
    left: float                     # 左边界（像素）
    right: float                    # 右边界（像素）
    width: float                    # 边界宽度 = right - left（像素）
    peak_valley_x: float            # 极值点坐标
    peak_valley_value: float        # 极值点亮度
    fwhm: float | None = None       # 亮纹的半高全宽（像素）
    color_rgb: tuple[int, int, int] = field(default=(0, 0, 0))
    color_name: str = ""
    contour: list[float] = field(default_factory=list)  # 边界内的亮度剖面（1D 横截面）
    centerline: list[list[float]] = field(default_factory=list)  # 2D 中心线 [[x, y], ...]

    @property
    def position(self) -> float:
        return self.center_x


def _fwhm(luma: np.ndarray, peak_x: int, left: int, right: int) -> float | None:
    """亮纹的半高全宽，相对局部基线测量。"""
    lo, hi = max(int(left), 0), min(int(right), len(luma) - 1)
    if hi - lo < 2:
        return None
    seg = luma[lo:hi + 1].astype(np.float64)
    baseline = float(np.percentile(seg, 20))          # 局部暗部作为基线
    peak = float(seg.max())
    half = baseline + 0.5 * (peak - baseline)
    if peak - baseline < 1e-6:
        return None
    above = np.where(seg >= half)[0]
    if len(above) == 0:
        return None
    return float(above[-1] - above[0] + 1)


def find_bands(luma: np.ndarray, period: float) -> list[Band]:
    """把一维亮度剖面切分成一段段明/暗条纹。

    明暗条纹是相对振荡（亮纹比左右邻居亮、暗纹比左右邻居暗），所以用
    ``find_peaks`` 找亮峰/暗谷，再用相邻极值中点作条纹边界，无需绝对阈值。
    额外处理三类易误判的情况：

    - 贴边伪条纹：图像左右边缘常有一两条很窄的亮边/反光（传感器边框等，
      不是干涉条纹），距边缘不足 15% 周期的极值直接剔除。
    - 边缘单调抬升：最左/最右的亮峰若没有暗谷夹在两侧，说明是朝边缘单调
      抬升的伪亮纹，一并剔除；只有真正被暗谷夹住的亮峰才算条纹。
    - 单条纹：只有一条宽亮纹、中间没有暗谷时，把暗背景边缘补成暗锚点。
    """
    width = len(luma)
    smooth = ndimage.gaussian_filter1d(luma, sigma=max(0.8, period * 0.06))
    distance = max(2, int(round(period * 0.4)))
    rng = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    prominence = max(0.5, rng * 0.02)  # 宽松：尽量不丢弱条纹，靠后续过滤兜底

    bright_x, _ = signal.find_peaks(smooth, distance=distance, prominence=prominence)
    dark_x, _ = signal.find_peaks(-smooth, distance=distance, prominence=prominence)

    # 贴边伪条纹剔除
    border = max(3, int(round(period * 0.15)))
    bright_x = [int(x) for x in bright_x if border <= x <= width - 1 - border]
    dark_x = [int(x) for x in dark_x if border <= x <= width - 1 - border]

    # 亮度过滤：真正的亮纹明显高于暗背景；暗背景里的微小噪声凸起不算亮纹。
    baseline = float(np.percentile(smooth, 10))
    if bright_x:
        heights = [float(smooth[x]) - baseline for x in bright_x]
        max_h = max(heights)
        bright_x = [x for x, h in zip(bright_x, heights) if h >= 0.25 * max_h]

    extrema = [(float(x), 1.0, float(smooth[x])) for x in bright_x]
    extrema += [(float(x), -1.0, float(smooth[x])) for x in dark_x]

    # 单条纹兜底：有亮峰但无暗谷，把暗背景边缘补成暗锚点
    if bright_x and not dark_x:
        extrema.append((0.0, -1.0, float(smooth[0])))
        extrema.append((float(width - 1), -1.0, float(smooth[-1])))

    extrema.sort(key=lambda t: t[0])

    # 合并同类型且靠得过近的极值（保留更极端的那个），保证明暗交替
    merged: list[tuple[float, float, float]] = []
    for x, kind, val in extrema:
        if merged and merged[-1][1] == kind and x - merged[-1][0] < distance:
            prev_x, _, prev_val = merged[-1]
            if (kind > 0 and val > prev_val) or (kind < 0 and val < prev_val):
                merged[-1] = (x, kind, val)
            continue
        merged.append((x, kind, val))

    # 剔除没有被暗谷夹住的亮峰（边缘单调抬升的伪亮纹）
    kept: list[tuple[float, float, float]] = []
    for i, (x, kind, val) in enumerate(merged):
        if kind > 0:
            has_left = any(k < 0 for _, k, _ in merged[:i])
            has_right = any(k < 0 for _, k, _ in merged[i + 1:])
            if not (has_left and has_right):
                continue
        kept.append((x, kind, val))
    merged = kept

    if len(merged) < 2:
        return []

    bands: list[Band] = []
    for i, (x, kind, val) in enumerate(merged):
        left = 0.0 if i == 0 else (merged[i - 1][0] + x) / 2.0
        right = float(width - 1) if i == len(merged) - 1 else (x + merged[i + 1][0]) / 2.0
        center = (left + right) / 2.0
        band_width = right - left
        kind_name = "bright" if kind > 0 else "dark"
        fwhm = _fwhm(smooth, int(x), int(left), int(right)) if kind_name == "bright" else None
        bands.append(Band(
            kind=kind_name,
            center_x=center,
            left=left,
            right=right,
            width=band_width,
            peak_valley_x=x,
            peak_valley_value=val,
            fwhm=fwhm,
        ))
    return bands


def band_color(img: np.ndarray, band: Band) -> None:
    """采样条纹本体的代表性 RGB 颜色。

    条纹可能很细、过曝、或只占画面中央一小块，简单取矩形均值会被背景
    稀释。这里在峰值列附近取一个窄横向窗口，再按亮度取最亮/最暗的一批
    像素（亮纹取前 20%，暗纹取后 20%），用它们代表条纹真实颜色。
    """
    h, w = img.shape[:2]
    half = max(2, min(band.width * 0.25, 8.0))
    x0 = int(np.clip(band.peak_valley_x - half, 0, w - 1))
    x1 = int(np.clip(band.peak_valley_x + half, 0, w - 1)) + 1
    patch = img[:, x0:x1].reshape(-1, 3).astype(np.float64)
    lum = 0.114 * patch[:, 0] + 0.587 * patch[:, 1] + 0.299 * patch[:, 2]
    q = 80 if band.kind == "bright" else 20
    thr = np.percentile(lum, q)
    sel = patch[lum >= thr] if band.kind == "bright" else patch[lum <= thr]
    if len(sel) == 0:
        sel = patch
    b, g, r = sel.mean(axis=0).astype(int)
    band.color_rgb = (int(r), int(g), int(b))
    band.color_name = _color_name(np.array([r, g, b], dtype=np.float64))


# ---------------------------------------------------------------------------
# 单图分析
# ---------------------------------------------------------------------------


def _band_peak_positions(
    gray: np.ndarray, period: float, *, bright: bool = True, n_bands: int | None = None,
) -> list[tuple[float, list[int]]]:
    """把图像沿竖直方向切成若干水平带，在每个带里找亮峰/暗谷。

    返回 ``[(y_center, [峰x, ...]), ...]``。条纹倾斜或弯曲时，不同带里的峰
    在 x 上会错开——这正是一条条纹中心线的骨架，下一步按相邻带最近邻连成
    2D 中心线，从而不再依赖「条纹必须竖直笔直」的假设。
    """
    h, w = gray.shape[:2]
    if n_bands is None:
        n_bands = max(6, min(30, h // 18))
    sigma = max(1.0, period * 0.08)
    distance = max(3, int(round(period * 0.4)))
    results: list[tuple[float, list[int]]] = []
    for i in range(n_bands):
        y0 = i * h // n_bands
        y1 = (i + 1) * h // n_bands
        yc = (y0 + y1) / 2.0
        prof = gray[y0:y1].mean(axis=0)
        sm = ndimage.gaussian_filter1d(prof, sigma=sigma)
        rng = float(np.percentile(sm, 95) - np.percentile(sm, 5))
        target = sm if bright else -sm
        pk, _ = signal.find_peaks(target, distance=distance, prominence=max(0.5, rng * 0.02))
        pk = list(pk)
        # 只有亮峰需要强度过滤：暗背景里的微小噪声凸起不算亮纹。
        if bright and pk:
            baseline = float(np.percentile(sm, 10))
            hs = [float(sm[p]) - baseline for p in pk]
            mh = max(hs)
            if mh > 0:
                pk = [p for p, h in zip(pk, hs) if h >= 0.25 * mh]
        results.append((yc, sorted(int(p) for p in pk)))
    return results


def _nearest_peak(pks: list[int], x: float, max_shift: float) -> float | None:
    if not pks:
        return None
    best = min(pks, key=lambda p: abs(p - x))
    return float(best) if abs(best - x) <= max_shift else None


def _chain_centerlines(
    band_peaks: list[tuple[float, list[int]]], period: float,
) -> list[list[tuple[float, float]]]:
    """把逐带的峰连成 2D 中心线。

    从峰最多的带（通常是零级条纹所在的中段）出发，向上下用最近邻匹配扩展，
    每条线即一条条纹的中心线 ``[[x, y], ...]``。倾斜条纹得到斜线、弯曲条纹
    得到曲线，都能如实反映。
    """
    if not band_peaks:
        return []
    max_shift = period * 0.4
    seed_i = max(range(len(band_peaks)), key=lambda i: len(band_peaks[i][1]))
    if not band_peaks[seed_i][1]:
        return []
    lines: list[list[tuple[float, float]]] = []
    for seed_x in band_peaks[seed_i][1]:
        line = [(float(seed_x), band_peaks[seed_i][0])]
        # 向下扩展
        cx = float(seed_x)
        for i in range(seed_i + 1, len(band_peaks)):
            nx = _nearest_peak(band_peaks[i][1], cx, max_shift)
            if nx is None:
                break
            line.append((nx, band_peaks[i][0]))
            cx = nx
        # 向上扩展
        cx = float(seed_x)
        for i in range(seed_i - 1, -1, -1):
            nx = _nearest_peak(band_peaks[i][1], cx, max_shift)
            if nx is None:
                break
            line.insert(0, (nx, band_peaks[i][0]))
            cx = nx
        if len(line) >= 2:
            lines.append(line)
    # 保持种子带的峰序（升序），而非按平均 x 排序：扇形/汇聚条纹下，种子带
    # 的左右顺序才是条纹的真实左右顺序（平均 x 会因上下汇聚而错乱）。
    return lines


def _line_extremum(gray: np.ndarray, line: list[tuple[float, float]], bright: bool):
    """中心线上最亮（bright）或最暗（dark）的点，返回 (x, y, 亮度)。"""
    h, w = gray.shape[:2]
    best = None
    for x, y in line:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        y0, y1 = max(0, yi - 2), min(h, yi + 3)
        v = float(gray[y0:y1, xi].mean())
        if best is None or (bright and v > best[2]) or (not bright and v < best[2]):
            best = (x, y, v)
    return best


def _band_color_2d(img: np.ndarray, line: list[tuple[float, float]], kind: str):
    """沿中心线采样条纹本体的代表颜色（避开背景稀释）。"""
    h, w = img.shape[:2]
    patches = []
    for x, y in line:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        x0, x1 = max(0, xi - 2), min(w, xi + 3)
        y0, y1 = max(0, yi - 3), min(h, yi + 4)
        patches.append(img[y0:y1, x0:x1].reshape(-1, 3))
    if not patches:
        return (0, 0, 0)
    patch = np.concatenate(patches).astype(np.float64)
    lum = 0.114 * patch[:, 0] + 0.587 * patch[:, 1] + 0.299 * patch[:, 2]
    q = 80 if kind == "bright" else 20
    thr = np.percentile(lum, q)
    sel = patch[lum >= thr] if kind == "bright" else patch[lum <= thr]
    if len(sel) == 0:
        sel = patch
    b, g, r = sel.mean(axis=0).astype(int)
    return (int(r), int(g), int(b))


def _mid_line(line_a: list[tuple[float, float]],
              line_b: list[tuple[float, float]],
              tol: float = 3.0) -> list[tuple[float, float]]:
    """两条相邻亮纹中心线的逐点中点，作为夹在中间的暗纹中心线。

    干涉暗纹就是相邻亮纹之间的极值，用两条亮纹在相同高度 y 处 x 坐标的中点
    即暗纹中心，天然继承亮纹的倾斜/弯曲形状，且无需单独在低对比度下找暗谷。
    """
    if not line_a or not line_b:
        return []
    b_sorted = sorted(line_b, key=lambda p: p[1])
    out = []
    for x, y in line_a:
        best = min(b_sorted, key=lambda p: abs(p[1] - y))
        if abs(best[1] - y) <= tol:
            out.append(((x + best[0]) / 2.0, (y + best[1]) / 2.0))
    return out


def analyze_image(path: Path) -> dict:
    """分析一张图片，返回所有条纹的描述。

    核心思路：**只追踪亮纹的 2D 中心线，暗纹由相邻亮纹的中点推导**。

    - 亮纹对比度高、逐水平带找峰稳定，连成中心线后如实反映倾斜/弯曲；
    - 暗纹是相邻亮纹之间的极值，取两条亮纹中心线的逐点中点即暗纹中心，
      无需在低对比度下独立找暗谷，也就不可能把黑背景误判成暗纹；
    - 明暗天然交替，边界用相邻条纹中心的中点，与一维 ``find_bands`` 语义一致。
    """
    img = load_image(path)
    h, w = img.shape[:2]
    gray = _luminance(img.astype(np.float64))
    profiles, luma = extract_profiles(img)      # 梯度过滤后的 1D 亮度剖面
    period = estimate_period(luma)

    bright_peaks = _band_peak_positions(gray, period, bright=True)
    bright_lines = _chain_centerlines(bright_peaks, period)
    # 线级贴边剔除（逐带已剔除贴边峰，这里兜底防止半贴边的线残留）
    border = max(3, int(round(period * 0.15)))
    bright_lines = [
        ln for ln in bright_lines
        if border <= sum(p[0] for p in ln) / len(ln) <= w - 1 - border
    ]
    if not bright_lines:
        return {
            "image": path.name,
            "size": {"width": w, "height": h},
            "period_px": round(float(period), 2),
            "num_bright": 0,
            "num_dark": 0,
            "bands": [],
        }
    # 构造明暗交替的中心序列：B0, D0, B1, D1, ..., D_{n-2}, B_{n-1}
    # 亮纹已按种子带峰序排列（扇形下这才是真实左右顺序）；暗纹取相邻亮纹
    # 中心线的逐点中点，暗纹位置也从这条中点线算，而非亮纹平均位置的简单中点。
    centers: list[tuple[str, float, list]] = []   # (kind, mean_x, centerline)
    bright_means = [sum(p[0] for p in ln) / len(ln) for ln in bright_lines]
    for i, ln in enumerate(bright_lines):
        centers.append(("bright", bright_means[i], ln))
        if i + 1 < len(bright_lines):
            dline = _mid_line(ln, bright_lines[i + 1])
            if not dline:
                dmean = (bright_means[i] + bright_means[i + 1]) / 2.0
                dline = [(dmean, y) for _, y in ln]
            else:
                dmean = sum(p[0] for p in dline) / len(dline)
            centers.append(("dark", dmean, dline))

    bands: list[Band] = []
    for i, (kind, mean_x, line) in enumerate(centers):
        # 边界 = 相邻条纹中心的中点；首/末条纹只有内侧有邻居，外侧按内侧
        # 半宽镜像对称扩展（不拖到黑背景），避免最外侧亮纹宽度被背景撑大。
        if i == 0:
            right = (centers[0][1] + centers[1][1]) / 2.0
            left = 2.0 * mean_x - right
        elif i == len(centers) - 1:
            left = (centers[i - 1][1] + mean_x) / 2.0
            right = 2.0 * mean_x - left
        else:
            left = (centers[i - 1][1] + mean_x) / 2.0
            right = (mean_x + centers[i + 1][1]) / 2.0
        width = right - left
        lo = max(0, int(left))
        hi = min(w, int(right) + 1)
        contour = [round(float(v), 2) for v in luma[lo:hi]]

        centerline = [[round(float(x), 2), round(float(y), 2)] for x, y in line]
        peak_x, peak_y, peak_v = _line_extremum(gray, line, kind == "bright")
        rgb = _band_color_2d(img, line, kind)
        fwhm = None
        if kind == "bright":
            row = gray[int(np.clip(peak_y, 0, h - 1))]
            fwhm = _fwhm(row, int(peak_x), int(left), int(right))
        bands.append(Band(
            kind=kind,
            center_x=mean_x,
            left=left,
            right=right,
            width=width,
            peak_valley_x=peak_x,
            peak_valley_value=peak_v,
            fwhm=fwhm,
            color_rgb=rgb,
            color_name=_color_name(np.array(rgb, dtype=np.float64)),
            contour=contour,
            centerline=centerline,
        ))

    return {
        "image": path.name,
        "size": {"width": w, "height": h},
        "period_px": round(float(period), 2),
        "num_bright": sum(1 for b in bands if b.kind == "bright"),
        "num_dark": sum(1 for b in bands if b.kind == "dark"),
        "bands": [asdict(b) for b in bands],
    }


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

_FONT = "Microsoft YaHei"


def _annotate(img: np.ndarray, bands: list[Band], luma: np.ndarray, out: Path) -> None:
    """在图上画出每段条纹的中心线、颜色标签与颜色块。

    有 2D 中心线时直接画折线（如实反映倾斜/弯曲条纹的轮廓），否则退回画
    竖直中轴。注意 cv2.putText 的 Hershey 字体不支持中文，这里只用 ASCII 标签
    （B=亮纹 bright，D=暗纹 dark）；颜色由顶部色块直观呈现。
    """
    vis = img.copy()
    h, w = vis.shape[:2]
    y_lo, y_hi = int(h * 0.12), int(h * 0.88)
    for i, b in enumerate(bands, 1):
        color_bgr = tuple(reversed(b.color_rgb))  # RGB -> BGR
        if len(b.centerline) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in b.centerline], np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cx = int(b.center_x)
            cv2.line(vis, (cx, y_lo), (cx, y_hi), (0, 0, 255), 1)
        # 顶部颜色块（该条纹实际颜色）+ 编号
        x0 = int(b.left)
        cv2.rectangle(vis, (x0, 8), (x0 + 24, 28), color_bgr, -1)
        cv2.rectangle(vis, (x0, 8), (x0 + 24, 28), (255, 255, 255), 1)
        tag = f"{'B' if b.kind == 'bright' else 'D'}{i}"
        cv2.putText(vis, tag, (x0 + 26, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imencode(".png", vis)[1].tofile(str(out))


def _profile_plot(bands: list[Band], luma: np.ndarray, out: Path) -> None:
    """画亮度剖面，用颜色块标出每一段条纹的轮廓与颜色。"""
    x = np.arange(len(luma))
    plt.figure(figsize=(12, 5))
    plt.plot(x, luma, color="#444444", lw=1.2, label="亮度剖面")
    cmap = plt.get_cmap("jet")
    colors = {  # 与 _annotate 一致：亮纹暖色，暗纹冷色
        "bright": "#e4572e",
        "dark": "#2e5b8a",
    }
    for i, b in enumerate(bands, 1):
        lo, hi = max(0, int(b.left)), min(len(luma), int(b.right) + 1)
        plt.axvspan(lo, hi, color=colors[b.kind], alpha=0.16)
        # 在剖面曲线上叠加条纹颜色
        seg_x = np.arange(lo, hi)
        plt.plot(seg_x, luma[lo:hi], color=np.array(b.color_rgb) / 255.0, lw=2.2)
        plt.annotate(
            f"{i}{'亮' if b.kind == 'bright' else '暗'}\n{b.color_name}",
            xy=(b.center_x, luma[int(b.peak_valley_x)]),
            xytext=(b.center_x, max(luma) + 3),
            ha="center", fontsize=7, fontproperties=_FONT,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"),
        )
    plt.xlabel("横向位置 x（像素）")
    plt.ylabel("亮度")
    plt.title("条纹轮廓：逐段亮度剖面与颜色")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def _print_report(report: dict) -> None:
    """在控制台打印一段可读的识别结果。"""
    print(f"\n=== {report['image']}  "
          f"({report['size']['width']}×{report['size']['height']}px, "
          f"周期≈{report['period_px']}px) ===")
    print(f"  亮纹 {report['num_bright']} 段，暗纹 {report['num_dark']} 段")
    print(f"  {'#':>2} {'类型':>4} {'位置x':>7} {'边界宽':>7} {'FWHM':>6} "
          f"{'颜色RGB':>18} {'色相名':>6}")
    for i, b in enumerate(report["bands"], 1):
        fwhm = f"{b['fwhm']:.1f}" if b["fwhm"] else "—"
        rgb = f"({b['color_rgb'][0]},{b['color_rgb'][1]},{b['color_rgb'][2]})"
        kind = "亮" if b["kind"] == "bright" else "暗"
        print(f"  {i:>2} {kind:>4} {b['center_x']:>7.1f} {b['width']:>7.1f} "
              f"{fwhm:>6} {rgb:>18} {b['color_name']:>6}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="白光干涉条纹逐段分析")
    parser.add_argument(
        "source_dir", nargs="?", type=Path,
        default=Path(
            r"D:\Files\Work\光电\AI_Interferometry_prepare\make_dataset_new"
            r"\pre_annotation_results\20260811-0513-10.9599711-wan\images"),
        help="图片目录（含 .jpg 条纹图）",
    )
    parser.add_argument("output_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "output",
                        help="结果输出目录")
    parser.add_argument("-n", "--num", type=int, default=4,
                        help="最多处理的示例图片张数（默认 4）")
    args = parser.parse_args(argv)

    src = Path(args.source_dir)
    if not src.is_dir():
        print(f"图片目录不存在: {src}", file=sys.stderr)
        return 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if not images:
        print(f"目录下没有 jpg/png 图片: {src}", file=sys.stderr)
        return 1

    reports = []
    processed = 0
    for path in images:
        try:
            report = analyze_image(path)
        except ValueError as exc:
            print(f"[跳过] {path.name}: {exc}")
            continue
        if report["num_bright"] == 0:
            print(f"[跳过] {path.name}: 未识别到亮纹")
            continue

        img = load_image(path)
        _, luma = extract_profiles(img)
        bands = [Band(**b) for b in report["bands"]]
        _annotate(img, bands, luma, out / f"{path.stem}_annotated.png")
        _profile_plot(bands, luma, out / f"{path.stem}_profile.png")
        _print_report(report)

        reports.append(report)
        processed += 1
        if processed >= args.num:
            break

    summary_path = out / "report.json"
    summary_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已处理 {processed} 张示例图，结果写入: {out}")
    print(f"  - 标注图 *_annotated.png")
    print(f"  - 剖面图 *_profile.png")
    print(f"  - 汇总   report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
