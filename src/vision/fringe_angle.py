"""条纹角度鲁棒估计 — 支持近竖直到接近水平的激光/白光直条纹。

PDF 报告中用 YOLO-OBB 逐条纹旋转框求倾角的方法成本高（需重新标注并训练
OBB 模型），且报告里用普通 ``atan`` 求角、反复旋转到 0° 的做法在近竖直时
不稳。这里改用项目已有的 :func:`src.vision.fringe_width.measure_center_fringe_width_2d`
取出的每条亮纹二维中心线：

1. 先用全局投影搜索发现任意方向的周期直条纹，并抑制实拍画面的青色辅助线；
2. 大角度条纹先刚性校正到近竖直，沿用二维中心线提取后再映射回原图；
3. 用正交 PCA 拟合中心线，在接近水平时也不存在回归奇点；
4. 对多条条纹的倾角取按长度加权的鲁棒中位数，并用拟合残差区分整体倾角
   与局部弯曲（弯曲
   条纹不能强行整体旋转竖直）。

返回的 ``correction_deg`` 可直接作为 :func:`src.vision.angle.rotate_expand`
的旋转角，把条纹转到竖直方向。``tilt_deg`` 正号表示条纹像 ``\\`` 那样顺时针
倾斜（``correction_deg`` 与其相反，是校正回竖直所需的旋转角）。
"""
from __future__ import annotations

import numpy as np

from src.vision.fringe_width import measure_center_fringe_width_2d
from src.vision.fringe_orientation import fit_line_orientation


def _fit_centerline_angle(
    line: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """正交拟合任意方向中心线，返回角度、残差和沿线长度。"""
    return fit_line_orientation(line)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """按权重取中位数：累计权重过半处的值。"""
    order = np.argsort(values)
    vs = values[order]
    ws = np.asarray(weights, dtype=np.float64)[order]
    total = float(ws.sum())
    if total <= 0:
        return float(np.median(values))
    idx = int(np.searchsorted(np.cumsum(ws), total / 2.0))
    return float(vs[min(idx, len(vs) - 1)])


def _median_abs_dev(values: np.ndarray, center: float) -> float:
    return float(np.median(np.abs(values - center)))


def estimate_fringe_angle_2d(
    bgr: np.ndarray,
    *,
    analysis_2d: dict | None = None,
) -> dict:
    """从二维亮纹中心线鲁棒估计画面倾斜角。

    复用 :func:`measure_center_fringe_width_2d` 提取的每条亮纹 ``centerline``，
    逐条拟合成直线后取「按长度加权的鲁棒中位数」，得到整体平均倾角，供相机
    安装偏斜校正。返回结构（可 JSON 化）：

    - ``tilt_deg``：整体平均倾角（度，正 = ``\\`` 形顺时针倾斜）；
    - ``correction_deg``：传给 ``rotate_expand`` 的校正旋转角（转回竖直）；
    - ``confidence``：0..1，倾角一致度 × 条纹数量 × 直线度综合；
    - ``curvature``：条纹弯曲程度（拟合残差相对长度），越大越弯曲；
    - ``num_lines`` / ``per_line``：参与拟合的中心线数量与逐条明细。

    对明显弯曲的条纹，``tilt_deg`` 只反映整体趋势、``curvature`` 会显著升高，
    此时不宜强行整体旋转竖直。识别不到条纹时 ``tilt_deg`` 为 None、置信度 0。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")

    # 实时诊断同时计算角度和间距时可复用同一次二维中心线提取。
    result = analysis_2d if analysis_2d is not None else (
        measure_center_fringe_width_2d(bgr))
    per_line: list[dict] = []
    for band in result.get("bands", []):
        if band.get("kind") != "bright":
            continue
        line = band.get("centerline") or []
        fit = _fit_centerline_angle(line)
        if fit is None:
            continue
        tilt_deg, residual_px, length_px = fit
        per_line.append({
            "tilt_deg": round(tilt_deg, 3),
            "residual_px": round(residual_px, 3),
            "length_px": round(length_px, 2),
        })

    empty = {
        "tilt_deg": None,
        "correction_deg": None,
        "confidence": 0.0,
        "curvature": None,
        "num_lines": 0,
        "per_line": per_line,
    }
    orientation = result.get("orientation") or {}
    global_tilt = orientation.get("tilt_deg")
    global_confidence = float(orientation.get("confidence") or 0.0)
    if not per_line:
        if global_tilt is not None and global_confidence >= 0.25:
            tilt = float(global_tilt)
            return {
                **empty,
                "tilt_deg": round(tilt, 3),
                "correction_deg": round(-tilt, 3),
                "confidence": round(global_confidence, 3),
                "method": "global_projection",
            }
        return empty

    tilts = np.array([p["tilt_deg"] for p in per_line], dtype=np.float64)
    residuals = np.array([p["residual_px"] for p in per_line], dtype=np.float64)
    lengths = np.array([p["length_px"] for p in per_line], dtype=np.float64)

    # 长度加权：长中心线拟合更可靠，权重更大；残差过大的弯曲线降权。
    weights = lengths / (1.0 + residuals)
    if weights.sum() <= 0:
        return empty
    tilt = _weighted_median(tilts, weights)
    curvature = float(np.median(residuals / np.maximum(lengths, 1e-6)))
    use_global = bool(
        global_tilt is not None
        and global_confidence >= 0.35
        and (
            abs(float(global_tilt)) >= 35.0
            or float(orientation.get("overlay_fraction") or 0.0) > 0.0
        )
    )
    if use_global:
        tilt = float(global_tilt)

    mad_deg = _median_abs_dev(tilts, tilt)

    # 置信度 = 倾角一致度（度数 MAD 越小越一致）× 条纹数量 × 直线度。
    agree = float(np.clip(1.0 - mad_deg / 4.0, 0.0, 1.0))
    count = float(np.clip(len(per_line) / 6.0, 0.0, 1.0))
    flat = float(np.clip(1.0 - curvature / 0.15, 0.0, 1.0))
    confidence = float(np.clip(
        0.45 * agree + 0.30 * count + 0.25 * flat, 0.0, 1.0))
    if use_global:
        confidence = max(confidence, 0.85 * global_confidence)

    return {
        "tilt_deg": round(float(tilt), 3),
        "correction_deg": round(float(-tilt), 3),
        "confidence": round(confidence, 3),
        "curvature": round(curvature, 4),
        "num_lines": len(per_line),
        "per_line": per_line,
        "method": (
            "global_projection+centerlines"
            if use_global else "centerlines"),
    }
