"""由两次中心条纹对应的微分表读数计算玻璃片厚度。"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time


GLASS_REFRACTIVE_INDEX = 1.4586


def calculate_thickness_mm(
    d1_mm: float,
    d2_mm: float,
    refractive_index: float = GLASS_REFRACTIVE_INDEX,
) -> float:
    """按 h=(d2-d1)/(10*(n-1)) 计算厚度，输入和返回值均为 mm。"""
    d1 = float(d1_mm)
    d2 = float(d2_mm)
    n = float(refractive_index)
    if not all(isfinite(value) for value in (d1, d2, n)):
        raise ValueError("读数和折射率必须是有限数值")
    if n <= 1.0:
        raise ValueError("折射率必须大于 1")
    return (d2 - d1) / (10.0 * (n - 1.0))


@dataclass(frozen=True)
class ThicknessReading:
    sequence: int
    value_mm: float
    captured_at: float

    @property
    def key(self) -> str:
        return f"R{self.sequence}"

    def as_dict(self) -> dict:
        return {
            "id": self.key,
            "sequence": self.sequence,
            "value_mm": self.value_mm,
            "captured_at": self.captured_at,
        }


class ThicknessMeasurement:
    """保存人工确认的中心条纹读数，并按记录编号计算厚度。"""

    def __init__(self, refractive_index: float = GLASS_REFRACTIVE_INDEX):
        if not isfinite(float(refractive_index)) or float(refractive_index) <= 1:
            raise ValueError("折射率必须大于 1")
        self.refractive_index = float(refractive_index)
        self._records: list[ThicknessReading] = []
        self._next_sequence = 1

    @property
    def records(self) -> tuple[ThicknessReading, ...]:
        return tuple(self._records)

    def add(self, value_mm: float, captured_at: float | None = None) -> ThicknessReading:
        value = float(value_mm)
        timestamp = time.time() if captured_at is None else float(captured_at)
        if not isfinite(value) or not isfinite(timestamp):
            raise ValueError("读数和采集时间必须是有限数值")
        if any(record.captured_at == timestamp for record in self._records):
            raise ValueError("该微分表采集帧已经记录，请等待新读数后再记录")
        record = ThicknessReading(self._next_sequence, value, timestamp)
        self._next_sequence += 1
        self._records.append(record)
        return record

    def get(self, key: str) -> ThicknessReading:
        for record in self._records:
            if record.key == key:
                return record
        raise KeyError(f"不存在读数记录: {key}")

    def remove(self, key: str) -> ThicknessReading:
        record = self.get(key)
        self._records.remove(record)
        return record

    def clear(self) -> None:
        self._records.clear()
        self._next_sequence = 1

    def calculate(self, d1_key: str, d2_key: str) -> float:
        if d1_key == d2_key:
            raise ValueError("d1 和 d2 必须选择两次不同的记录")
        d1 = self.get(d1_key)
        d2 = self.get(d2_key)
        return calculate_thickness_mm(
            d1.value_mm, d2.value_mm, self.refractive_index)
