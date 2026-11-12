"""钠双线平均波长：由跨两个可见度周期的微分表示值估算。"""
from math import isfinite, sqrt


MICROMETER_TO_MIRROR_RATIO = 20.0
SODIUM_DOUBLET_SEPARATION_NM = 0.6
VISIBILITY_PERIODS = 2


def calculate_mean_wavelength_nm(first_mm: float, second_mm: float) -> float:
    """固定模型：λ̄ ≈ sqrt(30000 * |d₂-d₁|)，读数为 mm，返回 nm。

    镜面移动 D=|d₂-d₁|/20；往返光程差为 2D。
    相邻模糊位置满足 2D_period * Δλ ≈ λ̄²。
    当前扫描为模糊→清晰→模糊→清晰→模糊，跨 m=2 个周期，
    因此 λ̄≈sqrt(2D * Δλ/m)。将 D 从 mm 换为 nm，固定
    Δλ=0.6 nm，得到上述公式。使用近双线近似，忽略 Δλ²/4 项。

    依据：LEOI-19 manual §5.1.5（可见度周期与谱线间隔关系）：
    https://lambdasys.com/uploads/info/LEOI-19-Content.pdf
    此结果依赖给定的谱线间隔、20:1传动比及完整的两周期端点识别。
    """
    first, second = float(first_mm), float(second_mm)
    if not all(isfinite(v) for v in (first, second)):
        raise ValueError("钠灯两端读数必须为有限数值")
    difference = abs(second - first)
    if not isfinite(difference) or difference <= 0:
        raise ValueError("钠灯两端读数差必须大于零，无法计算平均波长")
    mirror_nm = difference * (1e6 / MICROMETER_TO_MIRROR_RATIO)
    wavelength = sqrt(mirror_nm * (2 * SODIUM_DOUBLET_SEPARATION_NM / VISIBILITY_PERIODS))
    if not isfinite(wavelength):
        raise ValueError("钠灯读数超出可计算范围")
    return wavelength
