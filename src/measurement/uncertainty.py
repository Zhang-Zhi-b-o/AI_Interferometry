"""测量不确定度评定（GUM）与异常值检测。

供实验助手做**确定性**误差分析：A 类 / B 类、合成与扩展不确定度、
有效数字修约和 Grubbs 异常值检验，以及针对玻璃片厚度测量模型
``h = (d2 - d1) / [10 × (n - 1)]`` 的完整不确定度分析。

所有函数都是纯计算、无副作用，输入数值、输出可 JSON 化的结构，
便于大模型只做解释、不做数值计算（避免幻觉与计算错误）。
"""
from __future__ import annotations

from math import floor, isfinite, log10, sqrt
from typing import Iterable, Sequence

from src.constants import DEFAULT_CONFIDENCE_LEVEL, MICROMETER_ACCURACY
from src.measurement.thickness import GLASS_REFRACTIVE_INDEX

# 折射率手册允差的默认半宽（无量纲）。钠钙/光学玻璃常见 ±0.001 量级，
# 可按 config.yaml 的 uncertainty.refractive_index_tolerance 覆盖。
DEFAULT_REFRACTIVE_INDEX_TOLERANCE = 0.001

# 玻璃片厚度测量模型中的传动比系数：微分表读数差 10 mm 对应动镜实际
# 位移 1 mm（已由“条纹移动距离与螺旋测微器移动距离比例”标定）。
_THICKNESS_GEAR_RATIO = 10.0

# 学生 t 分布双侧 95% 置信的包含因子（自由度 -> k）。
# 自由度足够大（>=30）时趋近正态 1.96，教学场景按 k≈2 处理。
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

# Grubbs 检验单侧 α=0.05 临界值（NIST）。
_GRUBBS_TABLE = {
    3: 1.153, 4: 1.463, 5: 1.672, 6: 1.822, 7: 1.938, 8: 2.032,
    9: 2.110, 10: 2.176, 11: 2.234, 12: 2.285, 13: 2.331, 14: 2.371,
    15: 2.409, 16: 2.443, 17: 2.475, 18: 2.504, 19: 2.532, 20: 2.557,
    21: 2.580, 22: 2.603, 23: 2.624, 24: 2.644, 25: 2.663, 26: 2.681,
    27: 2.698, 28: 2.714, 29: 2.730, 30: 2.745,
}


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------


def sample_mean(values: Sequence[float]) -> float:
    values = _finite(values)
    if not values:
        raise ValueError("至少需要一次测量才能求平均值")
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    """样本标准差（贝塞尔公式，分母 n-1）。"""
    values = _finite(values)
    n = len(values)
    if n < 2:
        raise ValueError("样本标准差至少需要两次测量")
    mean = sum(values) / n
    return sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _finite(values: Iterable[float]) -> list[float]:
    out = [float(v) for v in values]
    if not all(isfinite(v) for v in out):
        raise ValueError("测量值必须是有限数值")
    return out


# ---------------------------------------------------------------------------
# A 类 / B 类 / 合成 / 扩展
# ---------------------------------------------------------------------------


def type_a_standard_uncertainty(values: Sequence[float]) -> float:
    """A 类标准不确定度 u_A = s / sqrt(n)。需要至少两次测量。"""
    values = _finite(values)
    return sample_std(values) / sqrt(len(values))


def type_b_rectangular(half_width: float) -> float:
    """B 类标准不确定度（矩形分布）：u_B = a / sqrt(3)。"""
    a = float(half_width)
    if a < 0:
        raise ValueError("半宽必须非负")
    return a / sqrt(3.0)


def type_b_normal(half_width: float, k: float = 2.0) -> float:
    """B 类标准不确定度（正态分布）：u_B = a / k。"""
    a = float(half_width)
    k = float(k)
    if a < 0 or k <= 0:
        raise ValueError("半宽必须非负，包含因子必须为正")
    return a / k


def combine_standard_uncertainties(*components: float) -> float:
    """各分量按方和根（RSS）合成为合成标准不确定度。"""
    parts = [float(c) for c in components]
    if any(not isfinite(c) or c < 0 for c in parts):
        raise ValueError("不确定度分量必须是非负有限数值")
    return sqrt(sum(c * c for c in parts))


def coverage_factor(degrees_of_freedom: int,
                    confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    """按自由度查 t 分布包含因子；自由度不足或超大时用正态近似。"""
    if confidence != DEFAULT_CONFIDENCE_LEVEL:
        # 仅内置 95% 表；其他置信度回退 k=2 并交给调用方说明。
        return 2.0
    nu = int(degrees_of_freedom)
    if nu < 1:
        return 2.0
    if nu in _T_TABLE:
        return _T_TABLE[nu]
    return 2.0  # nu >= 31 时 t(0.95) ≈ 1.96，教学取 2


def expanded_uncertainty(uc: float, degrees_of_freedom: int,
                         confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    """扩展不确定度 U = k * u_c。"""
    return coverage_factor(degrees_of_freedom, confidence) * float(uc)


# ---------------------------------------------------------------------------
# 有效数字修约
# ---------------------------------------------------------------------------


def format_measurement(value: float, uncertainty: float) -> str:
    """把测量结果与不确定度对齐到同一位，不确定度保留两位有效数字。"""
    value = float(value)
    uncertainty = float(uncertainty)
    if uncertainty <= 0 or not isfinite(uncertainty):
        return f"{value:.6g} ± 0"
    decimals = max(0, -floor(log10(uncertainty)) + 1)
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(value)} ± {fmt.format(uncertainty)}"


# ---------------------------------------------------------------------------
# 异常值检测
# ---------------------------------------------------------------------------


def grubbs_test(values: Sequence[float], alpha: float = 0.05) -> dict | None:
    """Grubbs 异常值检验（单侧，默认 α=0.05）。

    返回结构化结果；样本量小于 3 或大于 30（超出内置临界值表）时返回 None，
    表示无法用本表判定，可改用 3σ 准则。
    """
    values = _finite(values)
    n = len(values)
    if n < 3 or n not in _GRUBBS_TABLE:
        return None
    mean = sum(values) / n
    s = sample_std(values)
    if s == 0.0:
        return {
            "test": "Grubbs", "alpha": alpha, "sample_size": n,
            "is_outlier": False, "suspicious_index": None,
            "suspicious_value": None, "g_statistic": 0.0,
            "g_critical": _GRUBBS_TABLE[n],
            "note": "所有测量值完全相同，无异常值。",
        }
    g_crit = _GRUBBS_TABLE[n]
    deviations = [abs(v - mean) for v in values]
    index = max(range(n), key=deviations.__getitem__)
    g = deviations[index] / s
    return {
        "test": "Grubbs", "alpha": alpha, "sample_size": n,
        "is_outlier": g > g_crit,
        "suspicious_index": index + 1,  # 1 起编号，与轮次对齐
        "suspicious_value": values[index],
        "g_statistic": round(g, 4),
        "g_critical": g_crit,
        "note": (
            f"第 {index + 1} 次测量为可疑值（G={g:.3f} > G_crit={g_crit:.3f}），"
            f"建议复核或剔除后重算。" if g > g_crit else
            f"未检出显著异常值（G={g:.3f} <= G_crit={g_crit:.3f}）。"
        ),
    }


# ---------------------------------------------------------------------------
# 玻璃片厚度测量：完整不确定度分析
# ---------------------------------------------------------------------------


def analyze_glass_thickness(
    thickness_values: Sequence[float],
    d1_values: Sequence[float] | None = None,
    d2_values: Sequence[float] | None = None,
    refractive_index: float = GLASS_REFRACTIVE_INDEX,
    micrometer_accuracy_mm: float = MICROMETER_ACCURACY,
    refractive_index_tolerance: float = DEFAULT_REFRACTIVE_INDEX_TOLERANCE,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict:
    """对玻璃片厚度测量做完整的确定性不确定度分析。

    测量模型 ``h = (d2 - d1) / [10 × (n - 1)]``，输入输出均为 mm。

    - A 类：对多次独立测量得到的厚度序列做贝塞尔评定，u_A = s(h)/sqrt(N)；
    - B 类：微分表示值允差（矩形分布）经灵敏系数 1/[10(n-1)] 传导，
      以及折射率手册允差经灵敏系数 h/(n-1) 传导；
    - 合成标准不确定度按方和根，扩展不确定度取 t(0.95) 包含因子（N 大时≈2）。

    返回结构化 dict，字段均为可 JSON 化的数值或字符串。
    """
    h = _finite(thickness_values)
    n_index = float(refractive_index)
    if n_index <= 1.0:
        raise ValueError("折射率必须大于 1")
    if not h:
        raise ValueError("至少需要一次测量才能评定不确定度")

    mean_h = sum(h) / len(h)
    count = len(h)
    denominator = _THICKNESS_GEAR_RATIO * (n_index - 1.0)
    dhd = 1.0 / denominator            # |∂h/∂d|
    dhdn = mean_h / (n_index - 1.0)    # |∂h/∂n| = h/(n-1)

    # A 类（N>=2 时才有重复性）
    u_a = type_a_standard_uncertainty(h) if count >= 2 else 0.0

    # B 类：微分表允差（d1、d2 各一次，独立，RSS）
    u_d = type_b_rectangular(micrometer_accuracy_mm)
    u_b_micrometer = sqrt(2.0) * dhd * u_d

    # B 类：折射率允差
    u_n = type_b_rectangular(refractive_index_tolerance)
    u_b_index = dhdn * u_n

    u_b_total = combine_standard_uncertainties(u_b_micrometer, u_b_index)
    u_c = combine_standard_uncertainties(u_a, u_b_total)

    # 扩展不确定度：A 类自由度 N-1；单次测量无 A 类，只有 B 类，按正态取 k≈2。
    if count >= 2:
        nu = count - 1
        k_factor = coverage_factor(nu, confidence)
    else:
        nu = None
        k_factor = 2.0
    u_expanded = k_factor * u_c
    relative = (u_expanded / mean_h) if mean_h != 0.0 else float("inf")

    outlier = grubbs_test(h)

    return {
        "formula": "h = (d2 - d1) / [10 × (n - 1)]",
        "gear_ratio": _THICKNESS_GEAR_RATIO,
        "refractive_index": n_index,
        "count": count,
        "thickness_mean_mm": mean_h,
        "thickness_std_mm": sample_std(h) if count >= 2 else None,
        "type_a_mm": u_a,
        "type_b_mm": {
            "micrometer_accuracy_mm": micrometer_accuracy_mm,
            "micrometer_contribution_mm": u_b_micrometer,
            "refractive_index_tolerance": refractive_index_tolerance,
            "refractive_index_contribution_mm": u_b_index,
            "total": u_b_total,
        },
        "combined_uc_mm": u_c,
        "coverage_factor": k_factor,
        "degrees_of_freedom": nu,
        "expanded_U_mm": u_expanded,
        "relative_uncertainty": relative if isfinite(relative) else None,
        "result_text": format_measurement(mean_h, u_expanded) + " mm",
        "outlier": outlier,
        "warnings": _build_warnings(count, mean_h, d1_values, d2_values),
    }


def _build_warnings(
    count: int,
    mean_h: float,
    d1_values: Sequence[float] | None,
    d2_values: Sequence[float] | None,
) -> list[str]:
    """给出不依赖模型、由数据本身即可判定的确定性告警。"""
    warnings: list[str] = []
    if count < 2:
        warnings.append("仅有一次测量，无法评定 A 类重复性不确定度；结果只含 B 类分量。")
    if mean_h < 0:
        warnings.append("平均厚度为负，可能 d1 与 d2 顺序接反，请核对两次读数顺序。")
    if d1_values is not None and d2_values is not None:
        try:
            d1 = _finite(d1_values)
            d2 = _finite(d2_values)
        except ValueError:
            d1 = d2 = []
        if d1 and d2 and len(d1) == len(d2):
            diffs = [a - b for a, b in zip(d2, d1)]
            if any(d < 0 for d in diffs):
                warnings.append("存在 d2 < d1 的轮次（厚度为负），请核对该轮读数顺序。")
    return warnings
