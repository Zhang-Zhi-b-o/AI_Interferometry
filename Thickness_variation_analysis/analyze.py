"""Generate a relative-thickness map from one white-light fringe image.

The default single-image mode is deliberately labelled *relative/estimated*:
it unwraps the cyclic colour coordinate of the fringes and assigns one colour
cycle to one effective interference order.  An optional colour calibration CSV
can replace that approximation with a measured colour -> OPD lookup table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def parse_roi(value: str | None, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if not value:
        return 0, 0, w, h
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,width,height")
    x, y, rw, rh = parts
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    if rw < 20 or rh < 20:
        raise ValueError("ROI is too small")
    return x, y, rw, rh


def largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return binary.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def sample_mask(image: np.ndarray) -> np.ndarray:
    """Segment the bright sample while excluding dark background."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(light, (0, 0), 2.0)
    nonzero = blurred[blurred > 3]
    if nonzero.size == 0:
        return np.ones(light.shape, dtype=bool)
    threshold, _ = cv2.threshold(
        np.clip(blurred, 0, 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Otsu can be too strict for a dim sample; retain moderately illuminated pixels.
    floor = max(8.0, min(float(threshold), float(np.percentile(nonzero, 38))))
    mask = blurred >= floor
    radius = max(2, int(round(min(image.shape[:2]) * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=max(1, radius // 2))
    return mask.astype(bool)


def colour_phase_map(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a wrapped cyclic colour coordinate and its confidence."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    # Remove slow illumination/colour cast. The residual colour rotates across
    # successive white-light colour bands.
    sigma = max(5.0, min(image.shape[:2]) / 18.0)
    a_res = a - ndimage.gaussian_filter(a, sigma=sigma)
    b_res = b - ndimage.gaussian_filter(b, sigma=sigma)
    wrapped = np.arctan2(b_res, a_res)

    chroma = np.hypot(a_res, b_res)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    local = gray - ndimage.gaussian_filter(gray, sigma=max(3.0, sigma / 3.0))
    texture = ndimage.gaussian_filter(np.abs(local), 1.2)
    c_scale = max(float(np.percentile(chroma[mask], 90)), 1e-6)
    t_scale = max(float(np.percentile(texture[mask], 90)), 1e-6)
    confidence = np.clip(0.72 * chroma / c_scale + 0.28 * texture / t_scale, 0, 1)
    confidence[~mask] = 0
    return wrapped, confidence


def unwrap_relative_phase(
    wrapped: np.ndarray, confidence: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    # Low-colour pixels are kept in the connected mask and interpolated by the
    # masked phase unwrapping plus confidence-weighted smoothing.
    phase = unwrap_phase(np.ma.array(wrapped, mask=~mask)).filled(np.nan)
    finite = np.isfinite(phase) & mask
    if not np.any(finite):
        raise ValueError("No valid phase pixels")
    weights = confidence.copy()
    weights[~finite] = 0
    values = np.nan_to_num(phase) * weights
    sigma = max(1.2, min(wrapped.shape) / 180.0)
    num = ndimage.gaussian_filter(values, sigma=sigma)
    den = ndimage.gaussian_filter(weights, sigma=sigma)
    smooth = np.divide(num, den, out=np.nan_to_num(phase), where=den > 1e-4)
    smooth[~mask] = np.nan
    smooth -= float(np.nanmedian(smooth))
    return smooth


def load_colour_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read CSV columns: opd_um,r,g,b (OPD must already be baseline-corrected)."""
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["opd_um"]), float(row["r"]), float(row["g"]), float(row["b"])))
    if len(rows) < 4:
        raise ValueError("Colour calibration needs at least four rows")
    data = np.asarray(rows, dtype=np.float32)
    rgb = data[:, 1:4].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return data[:, 0].astype(np.float64), lab


def calibrated_opd_map(
    image: np.ndarray, mask: np.ndarray, calibration: Path
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    opd, cal_lab = load_colour_calibration(calibration)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    tree = cKDTree(cal_lab)
    distance, index = tree.query(lab.reshape(-1, 3), k=1)
    mapped = opd[index].reshape(mask.shape)
    scale = max(float(np.percentile(distance[mask], 90)), 1e-6)
    confidence = np.clip(1.0 - distance.reshape(mask.shape) / (1.5 * scale), 0, 1)
    mapped = ndimage.median_filter(mapped, size=3)
    mapped[~mask] = np.nan
    confidence[~mask] = 0
    return mapped, confidence


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    lo, hi = np.percentile(valid, (2, 98))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return float(lo), float(hi)


def save_map_figure(
    path: Path,
    values: np.ndarray,
    title: str,
    unit: str,
    cmap: str = "turbo",
) -> None:
    lo, hi = robust_limits(values)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    shown = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("y / pixel")
    bar = fig.colorbar(shown, ax=ax, shrink=0.86)
    bar.set_label(unit)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(thickness: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(thickness)
    values = thickness[valid]
    p2, p5, p95, p98 = np.percentile(values, (2, 5, 95, 98))
    clipped = values[(values >= p2) & (values <= p98)]
    gy, gx = np.gradient(np.nan_to_num(thickness, nan=float(np.nanmedian(thickness))))
    gradient = np.hypot(gx, gy)
    gradient[~valid] = np.nan
    return {
        "valid_pixels": int(valid.sum()),
        "mean_um": float(np.mean(clipped)),
        "median_um": float(np.median(clipped)),
        "min_robust_um": float(p2),
        "max_robust_um": float(p98),
        "pv_robust_um": float(p98 - p2),
        "rms_um": float(np.sqrt(np.mean((clipped - np.mean(clipped)) ** 2))),
        "p90_span_um": float(p95 - p5),
        "median_confidence": float(np.median(confidence[valid])),
        "median_gradient_um_per_px": float(np.nanmedian(gradient)),
        "p95_gradient_um_per_px": float(np.nanpercentile(gradient, 95)),
    }


def save_profiles(path: Path, thickness: np.ndarray) -> None:
    h, w = thickness.shape
    rows = [int(h * q) for q in (0.25, 0.5, 0.75)]
    cols = [int(w * q) for q in (0.25, 0.5, 0.75)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for row in rows:
        axes[0].plot(thickness[row], label=f"y={row}")
    axes[0].set(title="Horizontal thickness profiles", xlabel="x / pixel", ylabel="Thickness / um")
    axes[0].legend()
    for col in cols:
        axes[1].plot(thickness[:, col], label=f"x={col}")
    axes[1].set(title="Vertical thickness profiles", xlabel="y / pixel", ylabel="Thickness / um")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_overlay(path: Path, image: np.ndarray, thickness: np.ndarray) -> None:
    lo, hi = robust_limits(thickness)
    normalized = np.clip((np.nan_to_num(thickness, nan=lo) - lo) / (hi - lo), 0, 1)
    heat = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    valid = np.isfinite(thickness)
    overlay = image.copy()
    overlay[valid] = cv2.addWeighted(image, 0.38, heat, 0.62, 0)[valid]
    write_image(path, overlay)


def write_report(
    output: Path,
    source: Path,
    mode: str,
    metrics: dict[str, float],
    wavelength_nm: float,
    refractive_index: float,
    has_reference: bool,
) -> None:
    kind = "标定光程差换算" if mode == "calibrated" else "单图彩色条纹级次插值"
    absolute = "近似绝对厚度" if has_reference or mode == "calibrated" else "相对厚度（中位数设为 0）"
    report = rf"""# 彩色薄膜干涉图厚度分布分析报告

## 1. 输入与方法

- 输入图像：`{source.name}`
- 分析模式：{kind}
- 输出量：{absolute}
- 有效波长：{wavelength_nm:.1f} nm
- 薄膜折射率：{refractive_index:.4f}
- 有效分析像素：{metrics['valid_pixels']}

程序先分割亮的薄膜区域，排除大部分黑色背景；再利用 Lab 色彩空间中的循环色彩坐标恢复连续条纹级次。默认模式假设一个完整颜色周期对应一个有效干涉级次，其厚度步长为

$$
\Delta t=\frac{{\lambda_{{eff}}}}{{2(n-1)}}.
$$

若使用颜色标定 CSV，程序将像素颜色匹配到已经基准扣除的实际光程差，再按

$$
t=\frac{{\Delta OPD}}{{2(n-1)}}
$$

换算厚度。

## 2. 结果

| 指标 | 数值 |
|---|---:|
| 稳健最小值（2%分位） | {metrics['min_robust_um']:.4f} μm |
| 稳健最大值（98%分位） | {metrics['max_robust_um']:.4f} μm |
| 稳健峰谷值 PV | {metrics['pv_robust_um']:.4f} μm |
| RMS 不均匀度 | {metrics['rms_um']:.4f} μm |
| 中间 90% 厚度跨度 | {metrics['p90_span_um']:.4f} μm |
| 中位置信度 | {metrics['median_confidence']:.3f} |
| 中位厚度梯度 | {metrics['median_gradient_um_per_px']:.5f} μm/pixel |
| 95%厚度梯度 | {metrics['p95_gradient_um_per_px']:.5f} μm/pixel |

![厚度分布图](thickness_map.png)

![厚度叠加图](thickness_overlay.png)

![横纵截面曲线](profiles.png)

![置信度图](confidence_map.png)

## 3. 图像现象分析

- 条纹明显弯曲且局部间距变化，说明薄膜厚度梯度的方向和大小随位置变化，不符合均匀平行薄膜模型。
- 条纹密集区域对应较大的相位/厚度梯度；条纹疏松区域对应较平缓的厚度变化。
- 高梯度且低置信度的位置可能来自薄膜真实突变，也可能来自样品边缘、反光、遮挡或条纹断裂，需要结合原图复核。
- 厚度图的整体正负方向存在符号约定；若已知哪一侧实际更厚，可使用 `--invert` 反转方向。

## 4. 结果边界

默认单图模式提供的是模型依赖的相对厚度估计，不是可溯源的绝对厚度测量。白光颜色还受到光源光谱、相机 RGB 响应、曝光、白平衡和折射率色散影响。若要用于定量结论，应在相同光源、相机和曝光条件下，采用停车后清晰图像制作 `opd_um,r,g,b` 颜色标定表，并使用无薄膜基准扣除系统原有光程差。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>薄膜厚度分布分析报告</title><style>body{{max-width:960px;margin:36px auto;font:16px/1.7 system-ui;color:#243142}}img{{max-width:100%;border:1px solid #ddd}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:6px 12px}}code{{background:#f2f4f7;padding:2px 5px}}</style>
<body><h1>彩色薄膜干涉图厚度分布分析报告</h1>
<p>输入：<code>{source.name}</code>；方法：{kind}；输出：{absolute}。</p>
<h2>主要结果</h2><table><tr><th>指标</th><th>数值</th></tr>
<tr><td>稳健峰谷值 PV</td><td>{metrics['pv_robust_um']:.4f} μm</td></tr>
<tr><td>RMS 不均匀度</td><td>{metrics['rms_um']:.4f} μm</td></tr>
<tr><td>中间 90% 厚度跨度</td><td>{metrics['p90_span_um']:.4f} μm</td></tr>
<tr><td>中位置信度</td><td>{metrics['median_confidence']:.3f}</td></tr></table>
<h2>厚度分布</h2><img src='thickness_map.png'><h2>原图叠加</h2><img src='thickness_overlay.png'>
<h2>截面曲线</h2><img src='profiles.png'><h2>置信度</h2><img src='confidence_map.png'>
<h2>解释与限制</h2><p>条纹弯曲和间距变化表明厚度梯度随位置变化。默认单图模式是模型依赖的相对厚度估计；没有颜色—光程差标定和无膜基准时，不能将其作为可溯源的绝对厚度。</p>
</body></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, float]:
    source = Path(args.image)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    full = read_image(source)
    x, y, w, h = parse_roi(args.roi, full.shape)
    image = full[y:y + h, x:x + w].copy()
    mask = sample_mask(image)
    if mask.sum() < 200:
        raise ValueError("The automatically detected sample region is too small; use --roi")

    if args.calibration:
        opd_um, confidence = calibrated_opd_map(image, mask, Path(args.calibration))
        thickness = opd_um / (args.refractive_index - 1.0)
        mode = "calibrated"
    else:
        wrapped, confidence = colour_phase_map(image, mask)
        phase = unwrap_relative_phase(wrapped, confidence, mask)
        step_um = args.wavelength_nm / (2.0 * (args.refractive_index - 1.0)) / 1000.0
        thickness = phase / (2.0 * math.pi) * step_um
        mode = "relative"

    if args.invert:
        thickness = -thickness
    valid = np.isfinite(thickness)
    thickness -= float(np.nanmedian(thickness))
    if args.reference_thickness_um is not None:
        thickness += float(args.reference_thickness_um)
    thickness[~valid] = np.nan

    metrics = compute_metrics(thickness, confidence)
    metrics.update({
        "wavelength_nm": float(args.wavelength_nm),
        "refractive_index": float(args.refractive_index),
        "roi_x": x, "roi_y": y, "roi_width": w, "roi_height": h,
        "mode": mode,
    })

    write_image(output / "input_roi.png", image)
    write_image(output / "sample_mask.png", np.uint8(mask) * 255)
    save_map_figure(output / "thickness_map.png", thickness, "Estimated thickness distribution", "um")
    save_map_figure(output / "confidence_map.png", np.where(mask, confidence, np.nan), "Analysis confidence", "0 - 1", "viridis")
    save_overlay(output / "thickness_overlay.png", image, thickness)
    save_profiles(output / "profiles.png", thickness)
    np.savetxt(output / "thickness_map_um.csv", thickness, delimiter=",", fmt="%.6f")
    (output / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        output, source, mode, metrics, args.wavelength_nm, args.refractive_index,
        args.reference_thickness_um is not None,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse a white-light fringe image")
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("--roi", help="Optional x,y,width,height")
    parser.add_argument("--wavelength-nm", type=float, default=589.3)
    parser.add_argument("--refractive-index", type=float, default=1.523)
    parser.add_argument("--reference-thickness-um", type=float)
    parser.add_argument("--calibration", help="CSV columns: opd_um,r,g,b")
    parser.add_argument("--invert", action="store_true", help="Reverse thickness direction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.refractive_index <= 1:
        raise SystemExit("--refractive-index must be greater than 1")
    metrics = analyze(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
