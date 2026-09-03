"""激光/白光干涉条纹宽度测量 — 分析当前画面并给出中心条纹宽度。

把 ``Fringe width analysis/analyze_fringes.py`` 里的条纹分段算法（一维亮度
剖面 + 周期估计 + 明暗极值切分）收敛成可在实时画面复用的纯函数：输入一张
BGR 帧和一个参考位置（中心条纹 x，缺省用画面中心），输出该位置处条纹的
宽度、类型与边界，以及整体条纹周期。

竖直条纹沿水平方向变化最剧烈，因此取横向亮度剖面；只聚合横向纹理最清楚
的行，避免圆形光斑外的黑背景淹没条纹。所有坐标均为像素、相对整张输入图。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import interpolate, ndimage, signal

from src.vision.fringe_orientation import (
    estimate_global_fringe_orientation,
    fit_line_orientation,
    rotate_same_size,
    suppress_cyan_overlay,
    transform_points,
)


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """BGR -> 分析强度；单色激光优先采用调制度最高的颜色通道。"""
    image = np.asarray(bgr, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    luma = (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    )
    luma_range = float(np.percentile(luma, 95) - np.percentile(luma, 5))
    channel_ranges = [
        float(np.percentile(image[:, :, index], 95)
              - np.percentile(image[:, :, index], 5))
        for index in range(3)
    ]
    best_index = int(np.argmax(channel_ranges))
    # 红色激光在 Rec.601 灰度中只保留 29.9% 调制度。明显的单通道优势通常
    # 表示单色光；白光各通道接近时仍用标准亮度，避免改变彩色条纹形态。
    if channel_ranges[best_index] > max(2.0, 1.45 * luma_range):
        return image[:, :, best_index]
    return luma


def _luma_profile(bgr: np.ndarray) -> np.ndarray:
    """取水平亮度剖面，只聚合横向纹理最清楚的行。"""
    gray = _luminance(bgr)
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        raise ValueError("图像过小，无法提取条纹剖面")
    gx = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
    keep = gx >= np.percentile(gx, 45)
    if np.count_nonzero(keep) < max(4, h // 5):
        keep[:] = True
    return np.mean(gray[keep], axis=0)


def _estimate_period(luma: np.ndarray) -> float:
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
    max_lag = max(min_lag + 1, n // 3)
    peaks, properties = signal.find_peaks(
        corr[min_lag:max_lag], prominence=0.04)
    if len(peaks):
        # 实拍激光散斑可能在短延迟处产生小伪峰；取突出度最高的周期峰，避免
        # 固定 100 px 上限漏掉画面中仅有数条的宽条纹。
        best = int(np.argmax(properties["prominences"]))
        return float(peaks[best] + min_lag)
    return float(np.clip(n / 10.0, 4.0, 40.0))


def _fwhm(luma: np.ndarray, peak_x: int, left: int, right: int) -> float | None:
    """亮纹的半高全宽，相对局部基线测量。"""
    lo, hi = max(int(left), 0), min(int(right), len(luma) - 1)
    if hi - lo < 2:
        return None
    seg = luma[lo:hi + 1].astype(np.float64)
    baseline = float(np.percentile(seg, 20))
    peak = float(seg.max())
    half = baseline + 0.5 * (peak - baseline)
    if peak - baseline < 1e-6:
        return None
    above = np.where(seg >= half)[0]
    if len(above) == 0:
        return None
    return float(above[-1] - above[0] + 1)


def _find_bands(luma: np.ndarray, period: float) -> list[dict]:
    """把一维亮度剖面切分成一段段明/暗条纹，返回 ``{kind, center_x, left,
    right, width, peak_x, fwhm}`` 列表。"""
    width = len(luma)
    smooth = ndimage.gaussian_filter1d(luma, sigma=max(0.8, period * 0.06))
    distance = max(2, int(round(period * 0.4)))
    rng = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    prominence = max(0.5, rng * 0.02)

    bright_x, _ = signal.find_peaks(smooth, distance=distance, prominence=prominence)
    dark_x, _ = signal.find_peaks(-smooth, distance=distance, prominence=prominence)

    # 贴边伪条纹剔除
    border = max(3, int(round(period * 0.15)))
    bright_x = [int(x) for x in bright_x if border <= x <= width - 1 - border]
    dark_x = [int(x) for x in dark_x if border <= x <= width - 1 - border]

    # 亮度过滤：真正的亮纹明显高于暗背景
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

    bands: list[dict] = []
    for i, (x, kind, val) in enumerate(merged):
        left = 0.0 if i == 0 else (merged[i - 1][0] + x) / 2.0
        right = float(width - 1) if i == len(merged) - 1 else (x + merged[i + 1][0]) / 2.0
        kind_name = "bright" if kind > 0 else "dark"
        fwhm = _fwhm(smooth, int(x), int(left), int(right)) if kind_name == "bright" else None
        bands.append({
            "kind": kind_name,
            "center_x": float((left + right) / 2.0),
            "left": float(left),
            "right": float(right),
            "width": float(right - left),
            "peak_x": float(x),
            "fwhm": fwhm,
        })
    return bands


def measure_center_fringe_width(
    bgr: np.ndarray,
    center_x: float | None = None,
) -> dict:
    """分析一张 BGR 画面，给出中心条纹（``center_x`` 处）的宽度。

    参考位置缺省用画面中心。返回可 JSON 化的结构；识别不到条纹时
    ``center_band`` 为 None、``num_bands`` 为 0。``bands`` 是全部条纹的
    列表（每段含 ``kind``/``left``/``right``/``width``/``peak_x``），
    供“标注所有条纹宽度”使用。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    h, w = bgr.shape[:2]

    luma = _luma_profile(bgr)
    period = _estimate_period(luma)
    bands = _find_bands(luma, period)

    ref = float(np.clip(center_x, 0, w - 1)) if (
        center_x is not None and np.isfinite(center_x)) else w / 2.0

    center_band = None
    if bands:
        # 优先取边界包含参考位置的那段，否则取中心最近的。
        containing = [b for b in bands if b["left"] <= ref <= b["right"]]
        center_band = (
            containing[0] if containing
            else min(bands, key=lambda b: abs(b["center_x"] - ref)))

    return {
        "frame_width": w,
        "frame_height": h,
        "period_px": round(float(period), 2),
        "num_bands": len(bands),
        "num_bright": sum(1 for b in bands if b["kind"] == "bright"),
        "num_dark": sum(1 for b in bands if b["kind"] == "dark"),
        "reference_x": round(ref, 2),
        "bands": bands,
        "center_band": center_band,
    }


def locate_central_band(
    bgr: np.ndarray,
    center_x: float | None = None,
    search_bounds: tuple[float, float] | None = None,
    kind: str | None = None,
) -> dict | None:
    """定位中心条纹的精确轮廓，供中心条纹识别标定使用。

    复用一维亮度剖面 + 周期估计 + 明暗切分，返回距离 ``center_x`` 最近、
    且极值点落在 ``search_bounds``（通常为 YOLO 零级框）内的那段条纹。
    ``kind`` 可取 ``"bright"`` / ``"dark"`` / ``None``（任意明暗）。返回的
    ``center_x`` 是该段条纹的极值点（亮纹取峰、暗纹取谷），比几何中心更能
    代表实际条纹轮廓。未找到时返回 None。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        return None
    if bgr.ndim not in (2, 3):
        return None
    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        return None

    luma = _luma_profile(bgr)
    period = _estimate_period(luma)
    bands = _find_bands(luma, period)
    if not bands:
        return None

    ref = float(np.clip(center_x, 0, w - 1)) if (
        center_x is not None and np.isfinite(center_x)) else w / 2.0

    candidates = bands if kind is None else [b for b in bands if b["kind"] == kind]
    if search_bounds is not None:
        lo_b, hi_b = sorted((float(search_bounds[0]), float(search_bounds[1])))
        candidates = [b for b in candidates if lo_b <= b["peak_x"] <= hi_b]
    if not candidates:
        return None

    band = min(candidates, key=lambda b: abs(b["peak_x"] - ref))

    # 用局部对比度做置信度：亮纹看峰高、暗纹看谷深，均相对动态范围归一。
    smooth = ndimage.gaussian_filter1d(luma, sigma=max(0.8, period * 0.06))
    dynamic = max(
        float(np.percentile(smooth, 95) - np.percentile(smooth, 5)), 1e-6)
    peak_val = float(smooth[int(round(band["peak_x"]))])
    if band["kind"] == "bright":
        contrast = peak_val - float(np.percentile(smooth, 10))
    else:
        bright = [b for b in bands if b["kind"] == "bright"]
        neighbor = (
            max(float(smooth[int(round(b["peak_x"]))]) for b in bright)
            if bright else float(np.percentile(smooth, 90)))
        contrast = neighbor - peak_val
    confidence = float(np.clip(contrast / dynamic, 0.0, 1.0))

    return {
        "kind": band["kind"],
        "center_x": float(band["peak_x"]),
        "geometric_center_x": float(band["center_x"]),
        "left": float(band["left"]),
        "right": float(band["right"]),
        "width": float(band["width"]),
        "fwhm": band["fwhm"],
        "period_px": round(float(period), 2),
        "num_bands": len(bands),
        "num_bright": sum(1 for b in bands if b["kind"] == "bright"),
        "num_dark": sum(1 for b in bands if b["kind"] == "dark"),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 2D 轮廓版：适合倾斜 / 弯曲条纹
# ---------------------------------------------------------------------------
# 一维亮度剖面假设条纹竖直笔直，弯曲条纹会把轮廓模糊掉。这里把整幅 2D 平面
# 沿竖直方向切成若干水平带逐带找亮峰，挑中部最干净的一带做种子，再对每条
# 种子做「窗口追踪」得到亮纹的 2D 中心线，拟合成平滑曲线；暗纹取相邻两条亮
# 纹中心线的逐点中点，从而如实反映条纹的倾斜 / 弯曲轮廓。


def _band_peak_positions(
    gray: np.ndarray, period: float, *, bright: bool = True,
    n_bands: int = 48,
) -> list[tuple[float, list[int]]]:
    """把整幅 2D 平面沿竖直方向切成若干水平带，逐带找亮峰/暗谷。

    返回 ``[(y_center, [峰x, ...]), ...]``——这是 2D 轮廓分析的骨架：每个带
    里的峰 x 会随条纹倾斜/弯曲而在不同带间错开。用于给窗口追踪挑选种子带。
    """
    h, w = gray.shape[:2]
    sigma = max(1.0, period * 0.1)
    distance = max(3, int(round(period * 0.35)))
    results: list[tuple[float, list[int]]] = []
    for i in range(n_bands):
        y0 = i * h // n_bands
        y1 = (i + 1) * h // n_bands
        yc = (y0 + y1) / 2.0
        prof = gray[y0:y1].mean(axis=0)
        sm = ndimage.gaussian_filter1d(prof, sigma=sigma)
        rng = float(np.percentile(sm, 95) - np.percentile(sm, 5))
        target = sm if bright else -sm
        pk, _ = signal.find_peaks(target, distance=distance,
                                  prominence=max(0.5, rng * 0.03))
        pk = list(int(p) for p in pk)
        if bright and pk:
            baseline = float(np.percentile(sm, 10))
            hs = [float(sm[p]) - baseline for p in pk]
            mh = max(hs)
            if mh > 0:
                pk = [p for p, h in zip(pk, hs) if h >= 0.25 * mh]
        results.append((yc, sorted(pk)))
    return results


def _seed_strip_index(band_peaks: list[tuple[float, list[int]]]) -> int:
    """挑一个能覆盖全部条纹的带做种子：中部偏下区域里峰数最多的带。

    顶/底可能有黑边、噪声或条纹淡出（峰数异常多或少），取中部偏下 30%..55%
    高度这一段、峰数取最多的带：它既避开了顶部噪声，又保留了下半段才清晰的
    最左侧窄条纹，能一次性覆盖整幅 2D 平面上的所有亮纹。
    """
    n = len(band_peaks)
    lo, hi = (n * 3) // 10, (n * 11) // 20  # 0.30n .. 0.55n
    if hi <= lo:
        lo, hi = 0, n
    counts = [len(p) for _, p in band_peaks[lo:hi]]
    if not counts:
        return n // 2
    return lo + int(max(range(len(counts)), key=lambda i: counts[i]))


def _trace_centerline(
    gray: np.ndarray,
    period: float,
    seed_x: float,
    seed_y: float,
    *,
    bright: bool = True,
    n_bands: int = 48,
) -> list[tuple[float, float]]:
    """从种子 ``(seed_x, seed_y)`` 出发，沿竖直方向逐带窗口追踪局部极值。

    与「逐带全局找峰再最近邻连线」不同，这里用「窗口追踪」：每个带里只在
    上一带中心 x 附近一个小子窗口内取局部最亮（或最暗）点。窗口远小于条纹
    间距，因此只会跟着当前条纹走；从条纹最清楚的种子带向上下两侧延伸，遇
    到黑边或条纹消失（极值贴到窗口边缘）即停。
    """
    h, w = gray.shape[:2]
    window = max(3, int(round(period * 0.22)))
    sigma = max(1.0, period * 0.1)
    band_h = h / n_bands
    seed_i = int(np.clip(round(seed_y / band_h), 0, n_bands - 1))

    def _peak_at(i: int, cx: float) -> tuple[float, float] | None:
        y0 = i * h // n_bands
        y1 = (i + 1) * h // n_bands
        prof = gray[y0:y1].mean(axis=0)
        sm = ndimage.gaussian_filter1d(prof, sigma=sigma)
        lo = max(0, int(round(cx - window)))
        hi = min(w, int(round(cx + window)) + 1)
        if hi - lo < 3:
            return None
        seg = sm[lo:hi] if bright else -sm[lo:hi]
        j = int(np.argmax(seg))
        if j <= 0 or j >= len(seg) - 1:
            return None  # 极值贴到窗口边缘：这一带没有可追踪的峰
        return float(lo + j), float((y0 + y1) / 2.0)

    down: list[tuple[float, float]] = []
    cx = float(seed_x)
    for i in range(seed_i, n_bands):
        hit = _peak_at(i, cx)
        if hit is None:
            break
        down.append(hit)
        cx = hit[0]
    up: list[tuple[float, float]] = []
    cx = float(seed_x)
    for i in range(seed_i - 1, -1, -1):
        hit = _peak_at(i, cx)
        if hit is None:
            break
        up.append(hit)
        cx = hit[0]
    return up[::-1] + down


def _smooth_centerline(
    line: list[tuple[float, float]], num: int = 64,
) -> list[tuple[float, float]]:
    """把中心线点列拟合成平滑曲线（保形 PCHIP 样条），返回密集采样点。

    中心线按 y 单调递增，故对 x(y) 插值；PCHIP 保形不越界，避免样条在条纹
    弯曲处上下振荡。点数不足时原样返回（折线本身已足够平滑）。
    """
    if len(line) < 3:
        return line
    pts = np.asarray(line, dtype=np.float64)
    y = pts[:, 1]
    x = pts[:, 0]
    if float(np.ptp(y)) < 1e-6:
        return line
    try:
        spl = interpolate.PchipInterpolator(y, x, extrapolate=False)
    except Exception:
        return line
    ynew = np.linspace(float(y.min()), float(y.max()), max(num, len(line)))
    xnew = spl(ynew)
    ok = np.isfinite(xnew)
    return [[float(xi), float(yi)] for xi, yi in zip(xnew[ok], ynew[ok])]


def _mid_line(
    line_a: list[tuple[float, float]], line_b: list[tuple[float, float]],
    tol: float = 3.0,
) -> list[tuple[float, float]]:
    """两条相邻亮纹中心线的逐点中点，作为夹在中间的暗纹中心线。"""
    if not line_a or not line_b:
        return []
    b_sorted = sorted(line_b, key=lambda p: p[1])
    out = []
    for x, y in line_a:
        best = min(b_sorted, key=lambda p: abs(p[1] - y))
        if abs(best[1] - y) <= tol:
            out.append(((x + best[0]) / 2.0, (y + best[1]) / 2.0))
    return out


def _line_extremum(
    gray: np.ndarray, line: list[tuple[float, float]], bright: bool,
) -> tuple[float, float, float]:
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


def _colour_name_zh(hue: float, saturation: float, value: float) -> str:
    """把 OpenCV HSV 外观统计映射为可读色名；不表示光学相位。"""
    if value < 22:
        return "黑色"
    if saturation < 28:
        return "白色" if value >= 190 else ("灰色" if value >= 65 else "暗灰色")
    if hue < 8 or hue >= 172:
        name = "红色"
    elif hue < 22:
        name = "橙色"
    elif hue < 36:
        name = "黄色"
    elif hue < 82:
        name = "绿色"
    elif hue < 100:
        name = "青色"
    elif hue < 135:
        name = "蓝色"
    elif hue < 158:
        name = "紫色"
    else:
        name = "品红色"
    return ("暗" + name) if value < 70 else name


def _sample_line_colour(
    bgr: np.ndarray, centerline: list[list[float]], radius_px: int,
) -> dict:
    """采样中心线附近的相机颜色，返回 JSON 可序列化的 BGR/RGB/HSV 描述。"""
    image = np.asarray(bgr)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[:, :, :3]
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.asarray(centerline, dtype=np.float64)
    if points.size:
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        cv2.polylines(mask, [np.rint(points).astype(np.int32)], False, 255,
                      thickness=max(1, 2 * int(radius_px) + 1))
    pixels = image[mask > 0]
    if not len(pixels):
        return {"name_zh": "未知", "rgb": None, "bgr": None,
                "hsv_opencv": None, "hex": None, "sample_count": 0,
                "meaning": "camera_appearance"}
    bgr_median = np.median(pixels, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_median.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    rgb = [int(bgr_median[2]), int(bgr_median[1]), int(bgr_median[0])]
    return {
        "name_zh": _colour_name_zh(float(hsv[0]), float(hsv[1]), float(hsv[2])),
        "rgb": rgb,
        "bgr": [int(value) for value in bgr_median],
        "hsv_opencv": [int(value) for value in hsv],
        "hex": "#" + "".join(f"{value:02X}" for value in rgb),
        "sample_count": int(len(pixels)),
        "meaning": "camera_appearance",
    }


def _enrich_band_details(bgr: np.ndarray, bands: list[dict]) -> list[dict]:
    """为每条明/暗纹补充序号、位置、形状及颜色外观。"""
    h, w = bgr.shape[:2]
    enriched = []
    for index, source in enumerate(bands):
        band = dict(source)
        line = band.get("centerline") or []
        points = np.asarray(line, dtype=np.float64)
        if len(points):
            center = np.mean(points, axis=0)
            x0, y0 = np.min(points, axis=0)
            x1, y1 = np.max(points, axis=0)
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        else:
            center = np.asarray([float(band.get("center_x", 0.0)), h / 2.0])
            x0 = x1 = center[0]
            y0 = y1 = center[1]
            length = 0.0
        fit = fit_line_orientation(line)
        width_px = max(1.0, abs(float(band.get("width") or 1.0)))
        band.update({
            "index": index,
            "id": f"{band.get('kind', 'unknown')}-{index:02d}",
            "position": {
                "center_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
                "center_normalized": [
                    round(float(center[0]) / max(1, w - 1), 5),
                    round(float(center[1]) / max(1, h - 1), 5),
                ],
                "bbox_xyxy_px": [round(float(x0), 2), round(float(y0), 2),
                                   round(float(x1), 2), round(float(y1), 2)],
            },
            "shape": {
                "representation": "centerline_polyline",
                "centerline": line,
                "length_px": round(length, 2),
                "tilt_deg_from_vertical": round(float(fit[0]), 3) if fit else None,
                "fit_residual_px": round(float(fit[1]), 3) if fit else None,
            },
            "color": _sample_line_colour(
                bgr, line, max(1, min(5, int(round(width_px * 0.18))))),
        })
        enriched.append(band)
    return enriched


def measure_center_fringe_width_2d(
    bgr: np.ndarray,
    center_x: float | None = None,
    *,
    auto_orient: bool = True,
    period_hint_px: float | None = None,
) -> dict:
    """2D 轮廓版条纹宽度分析，适合任意方向的直条纹及近竖直弯曲条纹。

    与 :func:`measure_center_fringe_width` 返回同样的结构，区别在于：
    - 每段条纹额外带 ``centerline``（平滑曲线 ``[[x, y], ...]``），如实反映
      条纹的倾斜 / 弯曲轮廓，供画面上绘制曲线而非直线；
    - 只追踪被暗纹夹住的亮纹中心线，暗纹由相邻亮纹中点推导，天然把左右
      黑背景等「贴边伪条纹」排除在外。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    h, w = bgr.shape[:2]

    # 激光预调时条纹可能接近水平，原有“沿 y 追踪 x 峰”的算法会退化。先在
    # 缩小图上估计全局方向；大角度时刚性校正到近竖直，复用成熟的中心线
    # 提取，再把中心线坐标映射回原图。宽度/周期是刚性旋转不变量。
    orientation = None
    if auto_orient:
        cleaned, _ = suppress_cyan_overlay(bgr)
        orientation = estimate_global_fringe_orientation(cleaned)
        tilt = orientation.get("tilt_deg")
        confidence = float(orientation.get("confidence") or 0.0)
        if tilt is not None and confidence >= 0.25 and abs(float(tilt)) >= 35.0:
            corrected, matrix = rotate_same_size(cleaned, -float(tilt))
            ref_x = float(np.clip(center_x, 0, w - 1)) if (
                center_x is not None and np.isfinite(center_x)) else w / 2.0
            ref_corrected = transform_points(
                [[ref_x, h / 2.0]], matrix)[0]
            result = measure_center_fringe_width_2d(
                corrected,
                center_x=ref_corrected[0],
                auto_orient=False,
                period_hint_px=orientation.get("period_px"),
            )
            inverse = cv2.invertAffineTransform(matrix)
            mapped_bands = []
            for band in result.get("bands", []):
                mapped_line = transform_points(
                    band.get("centerline") or [], inverse)
                mapped_line = [
                    [round(x, 2), round(y, 2)] for x, y in mapped_line
                    if -1.0 <= x <= w and -1.0 <= y <= h
                ]
                line_fit = fit_line_orientation(mapped_line)
                angle_error = (
                    abs((line_fit[0] - float(tilt) + 90.0) % 180.0 - 90.0)
                    if line_fit is not None else 180.0)
                if (
                    len(mapped_line) < 3
                    or line_fit is None
                    or line_fit[2] < 0.20 * min(h, w)
                    or angle_error > 20.0
                ):
                    continue
                mean_x = float(np.mean([point[0] for point in mapped_line]))
                width_px = float(band.get("width") or 0.0)
                mapped = dict(band)
                mapped.update({
                    "center_x": mean_x,
                    "left": mean_x - width_px / 2.0,
                    "right": mean_x + width_px / 2.0,
                    "peak_x": mean_x,
                    "centerline": mapped_line,
                })
                mapped_bands.append(mapped)

            mapped_bands = _enrich_band_details(cleaned, mapped_bands)

            reference = np.asarray([ref_x, h / 2.0], dtype=np.float64)
            center_band = None
            if mapped_bands:
                center_band = min(
                    mapped_bands,
                    key=lambda band: min(
                        float(np.sum((np.asarray(point) - reference) ** 2))
                        for point in band["centerline"]),
                )
            return {
                **result,
                "frame_width": w,
                "frame_height": h,
                "reference_x": round(ref_x, 2),
                "reference_point": [round(ref_x, 2), round(h / 2.0, 2)],
                "bands": mapped_bands,
                "center_band": center_band,
                "num_bands": len(mapped_bands),
                "num_bright": sum(
                    band["kind"] == "bright" for band in mapped_bands),
                "num_dark": sum(
                    band["kind"] == "dark" for band in mapped_bands),
                "orientation": orientation,
                "auto_oriented": True,
            }
        bgr = cleaned
        normal_period = orientation.get("period_px")
        if (
            confidence >= 0.25
            and tilt is not None
            and normal_period is not None
        ):
            cosine = abs(float(np.cos(np.radians(float(tilt)))))
            if cosine >= 0.25:
                period_hint_px = float(normal_period) / cosine

    gray = _luminance(bgr)
    # 激光条纹常叠加细颗粒散斑；平滑尺度随画面尺寸增长但限制在 8 px 内，
    # 保留条纹主体并抑制会被误追踪成窄条纹的高频颗粒。
    gray = ndimage.gaussian_filter(
        gray, sigma=min(8.0, max(1.0, min(h, w) / 160.0)))
    luma = _luma_profile(gray)
    period = (
        float(period_hint_px)
        if period_hint_px is not None
        and np.isfinite(period_hint_px)
        and 3.0 <= float(period_hint_px) <= max(h, w)
        else _estimate_period(luma)
    )

    # 2D 种子：在整幅平面上逐带找亮峰，挑中部最干净的一带做种子集，再对每
    # 条种子做窗口追踪，得到真实反映条纹倾斜/弯曲的 2D 中心线（不缩成一维）。
    n_bands = max(20, min(60, h // 4))
    band_peaks = _band_peak_positions(gray, period, bright=True, n_bands=n_bands)
    seed_i = _seed_strip_index(band_peaks)
    seed_y, seed_xs = band_peaks[seed_i]
    bright_lines = [
        _trace_centerline(gray, period, sx, seed_y, bright=True, n_bands=n_bands)
        for sx in seed_xs
    ]
    bright_lines = [ln for ln in bright_lines if len(ln) >= 3]
    border = max(3, int(round(period * 0.15)))
    bright_lines = [
        ln for ln in bright_lines
        if border <= sum(p[0] for p in ln) / len(ln) <= w - 1 - border
    ]

    # 合并过近的亮纹中心线：白光色散会在同一物理条纹上产生相邻亚峰（相距仅
    # 几像素），被追踪成两条几乎重合的线。阈值取 0.2 周期，只合并这种「贴得
    # 极近」的重复线，避免把左侧真实存在、间距约 8px 的窄条纹误并。
    bright_lines.sort(key=lambda ln: sum(p[0] for p in ln) / len(ln))
    merged_lines: list[list[tuple[float, float]]] = []
    for ln in bright_lines:
        mx = sum(p[0] for p in ln) / len(ln)
        if merged_lines:
            prev = merged_lines[-1]
            prev_mx = sum(p[0] for p in prev) / len(prev)
            if mx - prev_mx < period * 0.2:
                if _line_extremum(gray, ln, True)[2] > \
                        _line_extremum(gray, prev, True)[2]:
                    merged_lines[-1] = ln
                continue
        merged_lines.append(ln)
    bright_lines = merged_lines

    if not bright_lines:
        return {
            "frame_width": w,
            "frame_height": h,
            "period_px": round(float(period), 2),
            "num_bands": 0,
            "num_bright": 0,
            "num_dark": 0,
            "reference_x": round(w / 2.0, 2),
            "bands": [],
            "center_band": None,
            "orientation": orientation,
            "auto_oriented": False,
        }

    # 明暗交替的中心序列：B0, D0, B1, D1, ..., D_{n-2}, B_{n-1}
    centers: list[tuple[str, float, list]] = []
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

    bands: list[dict] = []
    for i, (kind, mean_x, line) in enumerate(centers):
        # 边界 = 相邻条纹中心的中点；首/末条纹外侧按内侧半宽镜像对称扩展。
        if len(centers) == 1:
            left = max(0.0, mean_x - period / 2.0)
            right = min(float(w - 1), mean_x + period / 2.0)
        elif i == 0:
            right = (centers[0][1] + centers[1][1]) / 2.0
            left = 2.0 * mean_x - right
        elif i == len(centers) - 1:
            left = (centers[i - 1][1] + mean_x) / 2.0
            right = 2.0 * mean_x - left
        else:
            left = (centers[i - 1][1] + mean_x) / 2.0
            right = (mean_x + centers[i + 1][1]) / 2.0
        width = right - left

        peak_x, peak_y, _ = _line_extremum(gray, line, kind == "bright")
        fwhm = None
        if kind == "bright":
            row = gray[int(np.clip(peak_y, 0, h - 1))]
            fwhm = _fwhm(row, int(peak_x), int(left), int(right))

        smooth_line = _smooth_centerline(line)
        bands.append({
            "kind": kind,
            "center_x": float(mean_x),
            "left": float(left),
            "right": float(right),
            "width": float(width),
            "peak_x": float(peak_x),
            "fwhm": fwhm,
            "centerline": [[round(float(x), 2), round(float(y), 2)]
                           for x, y in smooth_line],
        })

    ref = float(np.clip(center_x, 0, w - 1)) if (
        center_x is not None and np.isfinite(center_x)) else w / 2.0
    bands = _enrich_band_details(bgr, bands)
    center_band = None
    if bands:
        containing = [b for b in bands if b["left"] <= ref <= b["right"]]
        center_band = (
            containing[0] if containing
            else min(bands, key=lambda b: abs(b["center_x"] - ref)))

    return {
        "frame_width": w,
        "frame_height": h,
        "period_px": round(float(period), 2),
        "num_bands": len(bands),
        "num_bright": sum(1 for b in bands if b["kind"] == "bright"),
        "num_dark": sum(1 for b in bands if b["kind"] == "dark"),
        "reference_x": round(ref, 2),
        "bands": bands,
        "center_band": center_band,
        "orientation": orientation,
        "auto_oriented": False,
    }


def measure_fringe_width_by_count(
    bgr: np.ndarray,
    x_range: tuple[float, float] | None = None,
    *,
    fringe: str = "bright",
    mm_per_px: float | None = None,
) -> dict:
    """用「视场条纹数」估算条纹宽度（间隔）：宽度 = 视场宽度 / 条纹数量。

    相比逐段边界宽度，这种方式对单条条纹的峰/谷定位误差不敏感，更适合在
    动镜移动过程中跟踪条纹间隔的稳定与否（找介质片垂直位置的判据：移动
    千分尺时条纹宽度不变）。

    参数:
        x_range:   可选的横向视场区间 ``(x0, x1)``（像素）——即「效果较好的
                   视场」。缺省时自动取所有识别到的条纹峰、两端各留半周期，
                   作为该视场。
        fringe:    计数的条纹类型 ``"bright"`` / ``"dark"`` / ``"all"``。
        mm_per_px: 像素—物理长度标定系数（mm/px）。给定后额外输出
                   ``span_mm`` / ``fringe_width_mm``，把结果换算成毫米。

    返回:
        ``fringe_width`` = ``span_px`` / ``fringe_count``（条纹间隔的近似
        值），另有 ``region`` / ``span_px`` / ``fringe_count`` / ``period_px``
        / ``peak_positions`` 供核对。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    if fringe not in ("bright", "dark", "all"):
        raise ValueError(f"fringe 必须是 bright/dark/all，收到 {fringe!r}")
    h, w = bgr.shape[:2]

    luma = _luma_profile(bgr)
    period = _estimate_period(luma)
    bands = _find_bands(luma, period)

    selected = bands if fringe == "all" else [b for b in bands if b["kind"] == fringe]
    all_peak_x = [b["peak_x"] for b in selected]

    if x_range is None:
        # 自动「效果较好的视场」：覆盖所有识别到的条纹峰，两端各留半周期，
        # 使跨度约等于 count 个周期，宽度即平均条纹间隔。
        if not all_peak_x:
            return {
                "frame_width": w, "frame_height": h,
                "region": None, "span_px": 0.0, "fringe_count": 0,
                "fringe_width": None, "period_px": round(float(period), 2),
                "peak_positions": [], "kind": fringe,
            }
        half = float(period) / 2.0
        x0 = float(np.clip(min(all_peak_x) - half, 0, w - 1))
        x1 = float(np.clip(max(all_peak_x) + half, 0, w - 1))
    else:
        x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
        x0 = float(np.clip(x0, 0, w - 1))
        x1 = float(np.clip(x1, 0, w - 1))

    xs = [x for x in all_peak_x if x0 <= x <= x1]
    span = x1 - x0
    count = len(xs)

    if span <= 0 or count == 0:
        return {
            "frame_width": w, "frame_height": h,
            "region": [round(x0, 2), round(x1, 2)],
            "span_px": round(span, 2), "fringe_count": 0,
            "fringe_width": None, "period_px": round(float(period), 2),
            "peak_positions": [], "kind": fringe,
        }

    out = {
        "frame_width": w, "frame_height": h,
        "region": [round(x0, 2), round(x1, 2)],
        "span_px": round(span, 2),
        "fringe_count": count,
        "fringe_width": round(span / count, 2),
        "period_px": round(float(period), 2),
        "peak_positions": [round(float(x), 2) for x in xs],
        "kind": fringe,
    }
    if mm_per_px is not None and mm_per_px > 0:
        out["mm_per_px"] = round(float(mm_per_px), 6)
        out["span_mm"] = round(span * mm_per_px, 4)
        out["fringe_width_mm"] = round((span / count) * mm_per_px, 4)
    return out


def _fit_line_tilt_deg(line: list[tuple[float, float]]) -> float | None:
    """正交拟合中心线，返回相对竖直方向的倾角（正=``\\``）。"""
    fit = fit_line_orientation(line)
    return fit[0] if fit is not None else None


def _estimate_tilt_deg(lines: list[list[tuple[float, float]]]) -> float | None:
    """由多条中心线按长度加权取倾角中位数，得到整体平均倾角（度）。"""
    tilts: list[float] = []
    lengths: list[float] = []
    for line in lines:
        t = _fit_line_tilt_deg(line)
        if t is None:
            continue
        tilts.append(t)
        fit = fit_line_orientation(line)
        lengths.append(max(1.0, fit[2] if fit is not None else 1.0))
    if not tilts:
        return None
    arr = np.asarray(tilts, dtype=np.float64)
    w = np.asarray(lengths, dtype=np.float64)
    order = np.argsort(arr)
    ts = arr[order]
    ws = w[order]
    total = float(ws.sum())
    if total <= 0:
        return float(np.median(arr))
    idx = int(np.searchsorted(np.cumsum(ws), total / 2.0))
    return float(ts[min(idx, len(ts) - 1)])


def measure_fringe_spacing_2d(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    angle_deg: float | None = None,
    pixel_scale_mm: float | None = None,
    *,
    fringe: str = "bright",
    mad_scale: float = 3.0,
    max_cv_percent: float = 10.0,
    analysis_2d: dict | None = None,
) -> dict:
    """沿条纹法向计算相邻亮纹中心间距（项目主间距算法）。

    定义（与 PDF 报告统一）：相邻同类型条纹中心之间、沿条纹法向的平均距离。
    用 :func:`measure_center_fringe_width_2d` 取每条条纹的二维中心线，把中心
    线代表点投影到条纹法向 ``n=(cosθ, -sinθ)`` 得到 ``s_i``，再对相邻 ``s``
    的差值做中位数 + MAD 异常值剔除，主值取有效间隔中位数。条纹倾斜造成的
    水平放大被法向投影自动消除，不需要先旋转图像也能得到真实间距。

    参数:
        roi:            可选裁剪区域 ``(x, y, w, h)``（原图坐标），只分析该区域。
        angle_deg:      已知条纹倾角（度，正=``\\``）。缺省由二维中心线自动估计。
        pixel_scale_mm: 像素—物理长度标定系数（mm/px），给定后输出毫米结果。
        fringe:         间距统计的条纹类型 ``"bright"`` / ``"dark"`` / ``"all"``。
        mad_scale:      异常值剔除倍数（相对 1.4826×MAD 的稳健标准差）。
        max_cv_percent: 判定「有效」的最大变异系数（%）。

    返回结构含 ``spacing_px``（主值）、``mean/std/cv``、``spacing_first_last_px``
    （首末条纹 ``(s_last-s_first)/(N-1)``）、``count_estimate_px``（视场÷条纹数）、
    ``period_estimate_px``（自相关周期）、``angle_deg``、``confidence`` 与
    ``quality_valid`` 等。``count_estimate_px`` / ``period_estimate_px`` 与
    ``spacing_px`` 越一致，置信度越高。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    if fringe not in ("bright", "dark", "all"):
        raise ValueError(f"fringe 必须是 bright/dark/all，收到 {fringe!r}")

    full_h, full_w = bgr.shape[:2]

    # ROI 裁剪（原图坐标），裁剪后所有坐标相对 ROI 左上角。
    if roi is not None:
        rx, ry, rw, rh = (int(v) for v in roi)
        rx = max(0, min(rx, full_w - 1))
        ry = max(0, min(ry, full_h - 1))
        rw = max(1, min(rw, full_w - rx))
        rh = max(1, min(rh, full_h - ry))
        img = bgr[ry:ry + rh, rx:rx + rw]
        roi_rect = [rx, ry, rw, rh]
    else:
        img = bgr
        roi_rect = None

    h, w = img.shape[:2]

    # 调用方同时计算角度时可传入与当前 img 对应的二维结果，避免重复提取。
    result2d = analysis_2d if analysis_2d is not None else (
        measure_center_fringe_width_2d(img))
    bands = result2d.get("bands", [])
    period = float(result2d.get("period_px") or 0.0)

    # 倾角：亮纹中心线最可靠，统一用亮纹估计。
    bright_lines = [b.get("centerline") or [] for b in bands if b["kind"] == "bright"]
    if angle_deg is None:
        tilt_deg = _estimate_tilt_deg(bright_lines)
    else:
        tilt_deg = float(angle_deg)
    if tilt_deg is None or not np.isfinite(tilt_deg):
        tilt_deg = 0.0
    th = np.radians(tilt_deg)
    cos_t = float(np.cos(th))
    sin_t = float(np.sin(th))

    # 选定类型条纹的二维中心线，投影到法向得到代表坐标 s。
    selected = bands if fringe == "all" else [b for b in bands if b["kind"] == fringe]
    mid_y = h / 2.0
    off_x = rx if roi is not None else 0
    off_y = ry if roi is not None else 0
    entries: list[dict] = []
    for b in selected:
        line = b.get("centerline") or []
        pts = np.asarray(line, dtype=np.float64)
        if len(pts) == 0:
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        s = float(np.median(x * cos_t - y * sin_t))
        j = int(np.argmin(np.abs(y - mid_y)))
        fit = fit_line_orientation(line)
        entries.append({
            "s": s,
            "x": float(x[j]) + off_x,
            "y": float(y[j]) + off_y,
            "length": float(fit[2]) if fit is not None else 0.0,
        })

    count_result = measure_fringe_width_by_count(img, fringe=fringe)
    count_est = count_result.get("fringe_width")

    base = {
        "method": "2d_centerline_normal_projection",
        "frame_width": full_w, "frame_height": full_h,
        "roi": roi_rect, "kind": fringe,
        "angle_deg": round(float(tilt_deg), 2),
        "num_fringes": 0, "num_intervals": 0,
        "num_valid_intervals": 0, "num_rejected": 0,
        "individual_spacings_px": [], "rejected_intervals": [],
        "spacing_px": None, "median_spacing_px": None,
        "mean_spacing_px": None, "std_spacing_px": None,
        "cv_percent": None, "spacing_mad_px": None,
        "spacing_first_last_px": None,
        "count_estimate_px": round(count_est, 2) if count_est is not None else None,
        "period_estimate_px": round(period, 2),
        "fringe_centers": [], "interval_valid": [],
        "confidence": 0.0, "quality_valid": False,
        "min_fringes_ok": False, "min_intervals_ok": False,
        "cv_ok": False, "rejection_ok": False, "line_length_ok": False,
    }

    entries.sort(key=lambda e: e["s"])
    n = len(entries)
    base["num_fringes"] = n
    if n < 2:
        return base

    s_vals = np.asarray([e["s"] for e in entries], dtype=np.float64)
    intervals = np.diff(s_vals)
    d_med = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - d_med)))
    # 1.4826×MAD 是正态分布下的稳健标准差；加一个相对间距 1% 的下限，避免干净
    # 数据 MAD≈0 时把微小量化噪声也误判为异常。
    sigma_robust = 1.4826 * max(mad, 0.01 * abs(d_med))
    threshold = float(mad_scale) * sigma_robust
    valid_mask = np.abs(intervals - d_med) <= threshold
    valid_intervals = intervals[valid_mask]
    valid_count = int(valid_mask.sum())
    rejected_count = int((~valid_mask).sum())

    spacing = float(np.median(valid_intervals)) if valid_count > 0 else d_med
    mean_spacing = float(np.mean(valid_intervals)) if valid_count > 0 else d_med
    std_spacing = float(np.std(valid_intervals)) if valid_count > 1 else 0.0
    cv = (std_spacing / mean_spacing * 100.0) if mean_spacing > 0 else 0.0
    first_last = float((s_vals[-1] - s_vals[0]) / (n - 1))

    # 质量与置信度判据（§五）。
    min_fringes = 4
    min_intervals = 3
    max_rejected_ratio = 0.30
    min_len_frac = 0.40
    min_fringes_ok = n >= min_fringes
    min_intervals_ok = valid_count >= min_intervals
    cv_ok = cv <= max_cv_percent
    rejected_ratio = rejected_count / max(1, len(intervals))
    rejection_ok = rejected_ratio <= max_rejected_ratio
    avg_len = float(np.mean([e["length"] for e in entries]))
    len_frac = avg_len / max(h, 1.0)
    line_length_ok = len_frac >= min_len_frac
    quality_valid = bool(min_fringes_ok and min_intervals_ok and cv_ok
                         and rejection_ok and line_length_ok)

    count_c = min(1.0, n / 8.0)
    valid_c = min(1.0, valid_count / float(min_intervals))
    cv_c = max(0.0, 1.0 - cv / max_cv_percent)
    rej_c = max(0.0, 1.0 - rejected_ratio / max_rejected_ratio)
    len_c = min(1.0, len_frac / min_len_frac)
    confidence = float(np.clip(
        0.25 * count_c + 0.30 * valid_c + 0.20 * cv_c + 0.15 * rej_c + 0.10 * len_c,
        0.0, 1.0))

    out = {
        **base,
        "num_intervals": int(len(intervals)),
        "num_valid_intervals": valid_count,
        "num_rejected": rejected_count,
        "individual_spacings_px": [round(float(g), 2) for g in intervals],
        "rejected_intervals": [
            round(float(g), 2) for g, v in zip(intervals, valid_mask) if not v],
        "spacing_px": round(spacing, 2),
        "median_spacing_px": round(spacing, 2),
        "mean_spacing_px": round(mean_spacing, 2),
        "std_spacing_px": round(std_spacing, 2),
        "cv_percent": round(cv, 2),
        "spacing_mad_px": round(mad, 2),
        "spacing_first_last_px": round(first_last, 2),
        "fringe_centers": [
            {"s": round(e["s"], 2), "x": round(e["x"], 2), "y": round(e["y"], 2)}
            for e in entries],
        "interval_valid": [bool(v) for v in valid_mask],
        "confidence": round(confidence, 3),
        "quality_valid": quality_valid,
        "min_fringes_ok": min_fringes_ok,
        "min_intervals_ok": min_intervals_ok,
        "cv_ok": cv_ok,
        "rejection_ok": rejection_ok,
        "line_length_ok": line_length_ok,
    }
    if pixel_scale_mm is not None and pixel_scale_mm > 0:
        k = float(pixel_scale_mm)
        out["pixel_scale_mm"] = round(k, 6)
        out["spacing_mm"] = round(spacing * k, 4)
        out["median_spacing_mm"] = round(spacing * k, 4)
        out["mean_spacing_mm"] = round(mean_spacing * k, 4)
        out["std_spacing_mm"] = round(std_spacing * k, 4)
        out["spacing_mm_uncertainty"] = round(std_spacing * k, 4)
    return out


def measure_fringe_spacing_robust(
    bgr: np.ndarray,
    x_range: tuple[float, float] | None = None,
    *,
    fringe: str = "bright",
    mm_per_px: float | None = None,
    mad_scale: float = 2.5,
) -> dict:
    """旧接口：兼容早期调用方，委托给 :func:`measure_fringe_spacing_2d`。

    ``x_range`` 被转成横向裁剪 ROI；``mm_per_px`` 映射为 ``pixel_scale_mm``。
    返回值保留历史字段名（``gap_px`` / ``valid_count`` / ``rejected_count`` /
    ``spacing_first_last_px`` / ``spacing_mad_px`` 等）。
    """
    h, w = bgr.shape[:2]
    roi = None
    region = None
    if x_range is not None:
        x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
        x0 = float(np.clip(x0, 0, w - 1))
        x1 = float(np.clip(x1, 0, w - 1))
        roi = (int(x0), 0, max(1, int(x1 - x0)), h)
        region = [round(x0, 2), round(x1, 2)]
    m = measure_fringe_spacing_2d(
        bgr, roi=roi, pixel_scale_mm=mm_per_px, fringe=fringe,
        mad_scale=mad_scale)
    spacing = m.get("spacing_px")
    first_last = m.get("spacing_first_last_px")
    out = {
        "frame_width": m["frame_width"], "frame_height": m["frame_height"],
        "kind": fringe, "num_fringes": m["num_fringes"],
        "region": region,
        "spacing_px": spacing,
        "spacing_mad_px": m["spacing_mad_px"],
        "spacing_first_last_px": first_last,
        "gap_px": m["individual_spacings_px"],
        "valid_count": m["num_valid_intervals"],
        "rejected_count": m["num_rejected"],
        "confidence": m["confidence"],
        "fringe_center_x": [c["x"] for c in m["fringe_centers"]],
    }
    if mm_per_px is not None and mm_per_px > 0 and spacing is not None:
        out["mm_per_px"] = round(float(mm_per_px), 6)
        out["spacing_mm"] = round(spacing * mm_per_px, 4)
        out["spacing_mad_mm"] = round((m["spacing_mad_px"] or 0.0) * mm_per_px, 4)
        if first_last:
            out["spacing_first_last_mm"] = round(first_last * mm_per_px, 4)
    return out
