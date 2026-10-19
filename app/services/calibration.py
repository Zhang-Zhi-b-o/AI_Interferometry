"""颜色 → 光程差标定表采集。

流程（对应界面「厚度分布 → 标定采集」）：:

    1. 设零点：记下当前微分表读数作为 d0；
    2. 每移动一段距离，采一个点：OPD=|d−d0|/20×1000 μm + 中心线颜色 (r,g,b)；
    3. 至少 4 个点、OPD 单调递增，才是可用的标定表；
    4. 存成 CSV（列：opd_um,r,g,b），在「分析」页选它即进入标定模式。

OPD 也可以手动输入（没有微分表读数时）。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CalibrationPoint:
    opd_um: float
    r: int
    g: int
    b: int

    def as_row(self) -> dict:
        return {"opd_um": round(float(self.opd_um), 4),
                "r": int(self.r), "g": int(self.g), "b": int(self.b)}


class CalibrationTable:
    MIN_POINTS = 4

    def __init__(self) -> None:
        self.points: list[CalibrationPoint] = []

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.points)

    @property
    def is_usable(self) -> bool:
        return not self.problems()

    def problems(self) -> list[str]:
        issues: list[str] = []
        if len(self.points) < self.MIN_POINTS:
            issues.append(f"至少需要 {self.MIN_POINTS} 个标定点（当前 {len(self.points)}）")
        opds = [p.opd_um for p in self.points]
        if len(opds) >= 2 and any(b <= a for a, b in zip(opds, opds[1:])):
            issues.append("OPD 必须严格递增")
        if any(not np.isfinite(p.opd_um) for p in self.points):
            issues.append("OPD 必须是有限数值")
        return issues

    # ------------------------------------------------------------------
    def add(self, *, opd_um: float, colour: tuple[int, int, int]) -> CalibrationPoint:
        point = CalibrationPoint(opd_um=float(opd_um), r=int(colour[0]),
                                 g=int(colour[1]), b=int(colour[2]))
        self.points.append(point)
        self.points.sort(key=lambda p: p.opd_um)
        return point

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def clear(self) -> None:
        self.points.clear()

    def rows(self) -> list[dict]:
        return [p.as_row() for p in self.points]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["opd_um", "r", "g", "b"])
            writer.writeheader()
            writer.writerows(self.rows())
        return target

    def load(self, path: str | Path) -> int:
        target = Path(path)
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.points = [
                CalibrationPoint(
                    opd_um=float(row["opd_um"]),
                    r=int(float(row["r"])),
                    g=int(float(row["g"])),
                    b=int(float(row["b"])),
                )
                for row in reader
            ]
        self.points.sort(key=lambda p: p.opd_um)
        return len(self.points)


def opd_um_from_readings(zero_mm: float, current_mm: float) -> float:
    """微分表读数差 → 光程差（μm）：|Δ|/20 mm × 1000。"""
    return abs(float(current_mm) - float(zero_mm)) / 20.0 * 1000.0
