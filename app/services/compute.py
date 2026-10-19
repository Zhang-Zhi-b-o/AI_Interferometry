"""计算：玻璃片厚度、GUM 不确定度、多轮统计。全部包装 ``src.measurement``。

这里**只算数，不做解释**；任何文字解释交给未来的智能体，避免界面里长出一堆规则。
"""
from __future__ import annotations

from typing import Sequence

from src.measurement import thickness as _thickness
from src.measurement import uncertainty as _uncertainty


class ComputeService:
    def __init__(self, *, refractive_index: float = 1.4586,
                 micrometer_accuracy_mm: float = 0.0001,
                 refractive_index_tolerance: float = 0.001,
                 confidence: float = 0.95) -> None:
        self.refractive_index = float(refractive_index)
        self.micrometer_accuracy_mm = float(micrometer_accuracy_mm)
        self.refractive_index_tolerance = float(refractive_index_tolerance)
        self.confidence = float(confidence)

    # ------------------------------------------------------------------
    def thickness_mm(self, d1_mm: float, d2_mm: float) -> float:
        return _thickness.calculate_thickness_mm(
            float(d1_mm), float(d2_mm), self.refractive_index)

    def opd_um(self, d1_mm: float, d2_mm: float) -> float:
        """单次读数差对应的光程差：|Δ|/20 (mm) → μm。"""
        return abs(float(d2_mm) - float(d1_mm)) / 20.0 * 1000.0

    # ------------------------------------------------------------------
    def uncertainty(self, values: Sequence[float],
                    d1_values: Sequence[float] | None = None,
                    d2_values: Sequence[float] | None = None) -> dict:
        return _uncertainty.analyze_glass_thickness(
            [float(v) for v in values],
            d1_values=None if d1_values is None else [float(v) for v in d1_values],
            d2_values=None if d2_values is None else [float(v) for v in d2_values],
            refractive_index=self.refractive_index,
            micrometer_accuracy_mm=self.micrometer_accuracy_mm,
            refractive_index_tolerance=self.refractive_index_tolerance,
            confidence=self.confidence,
        )

    def statistics(self, values: Sequence[float]) -> dict:
        numbers = [float(v) for v in values]
        if not numbers:
            return {"count": 0}
        grubbs = _uncertainty.grubbs_test(numbers)
        return {
            "count": len(numbers),
            "mean": _uncertainty.sample_mean(numbers),
            "std": _uncertainty.sample_std(numbers) if len(numbers) > 1 else None,
            "min": min(numbers),
            "max": max(numbers),
            "type_a": _uncertainty.type_a_standard_uncertainty(numbers)
            if len(numbers) > 1 else None,
            "grubbs": grubbs,
        }

    @staticmethod
    def format(value: float | None, uncertainty: float | None = None,
               digits: int = 6) -> str:
        if value is None:
            return "—"
        if uncertainty is None or uncertainty <= 0:
            return f"{value:.{digits}g} mm"
        return _uncertainty.format_measurement(float(value), float(uncertainty))
