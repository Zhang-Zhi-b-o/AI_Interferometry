"""实验助手 — 管理多次玻璃片厚度测量、求平均值与统计分析。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

from src.measurement.thickness import (
    GLASS_REFRACTIVE_INDEX,
    ThicknessReading,
    calculate_thickness_mm,
)


# ---------------------------------------------------------------------------
# 单次测量数据对象
# ---------------------------------------------------------------------------


@dataclass
class MeasurementRound:
    """一次完整的厚度测量：包含 d1、d2 两次读数与计算结果。"""

    sequence: int
    d1_mm: float
    d2_mm: float
    refractive_index: float
    thickness_mm: float
    d1_source: str = ""       # 读数来源标记，如 "R1" 或 "手动"
    d2_source: str = ""
    note: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at <= 0:
            self.created_at = time.time()

    @property
    def label(self) -> str:
        return f"第{self.sequence}次"

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "d1_mm": self.d1_mm,
            "d2_mm": self.d2_mm,
            "refractive_index": self.refractive_index,
            "thickness_mm": self.thickness_mm,
            "d1_source": self.d1_source,
            "d2_source": self.d2_source,
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MeasurementRound:
        return cls(
            sequence=int(data["sequence"]),
            d1_mm=float(data["d1_mm"]),
            d2_mm=float(data["d2_mm"]),
            refractive_index=float(
                data.get("refractive_index", GLASS_REFRACTIVE_INDEX)),
            thickness_mm=float(data["thickness_mm"]),
            d1_source=str(data.get("d1_source", "")),
            d2_source=str(data.get("d2_source", "")),
            note=str(data.get("note", "")),
            created_at=float(data.get("created_at", 0.0)),
        )


# ---------------------------------------------------------------------------
# 实验会话 — 管理多轮测量、导入读数、统计分析、存档
# ---------------------------------------------------------------------------


@dataclass
class SessionStats:
    """多轮厚度测量的统计汇总。"""
    count: int = 0
    mean_mm: float = 0.0
    std_mm: float = 0.0
    min_mm: float = 0.0
    max_mm: float = 0.0
    refractive_index: float = GLASS_REFRACTIVE_INDEX

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_mm": self.mean_mm,
            "std_mm": self.std_mm,
            "min_mm": self.min_mm,
            "max_mm": self.max_mm,
            "refractive_index": self.refractive_index,
        }


class ExperimentSession:
    """管理多次玻璃片厚度测量，支持手动输入、导入已有读数、统计分析、存档。"""

    _VERSION = 1

    def __init__(self, refractive_index: float = GLASS_REFRACTIVE_INDEX):
        if not isfinite(float(refractive_index)) or float(refractive_index) <= 1:
            raise ValueError("折射率必须大于 1")
        self.refractive_index = float(refractive_index)
        self._rounds: list[MeasurementRound] = []
        self._next_sequence = 1
        self.name: str = ""          # 实验名称（存档用）
        self.operator: str = ""      # 操作者
        self.sample_id: str = ""     # 样品编号
        self.created_at: float = time.time()

    # ---- 属性 ----

    @property
    def rounds(self) -> tuple[MeasurementRound, ...]:
        return tuple(self._rounds)

    @property
    def count(self) -> int:
        return len(self._rounds)

    # ---- 手动输入单次测量 ----

    def add_manual(
        self,
        d1_mm: float,
        d2_mm: float,
        note: str = "",
    ) -> MeasurementRound:
        """手动输入 d1、d2 读数并自动计算厚度，加入测量序列。"""
        d1 = float(d1_mm)
        d2 = float(d2_mm)
        if not all(isfinite(v) for v in (d1, d2)):
            raise ValueError("d1 和 d2 读数必须是有限数值")
        thickness = calculate_thickness_mm(d1, d2, self.refractive_index)
        round_ = MeasurementRound(
            sequence=self._next_sequence,
            d1_mm=d1,
            d2_mm=d2,
            refractive_index=self.refractive_index,
            thickness_mm=thickness,
            d1_source="手动",
            d2_source="手动",
            note=note,
        )
        self._next_sequence += 1
        self._rounds.append(round_)
        return round_

    # ---- 从已有读数记录导入 ----

    def add_from_readings(
        self,
        d1_reading: ThicknessReading,
        d2_reading: ThicknessReading,
        note: str = "",
    ) -> MeasurementRound:
        """从 ThicknessMeasurement 的两次已保存读数导入为一次测量。"""
        if d1_reading.key == d2_reading.key:
            raise ValueError("d1 和 d2 必须选择两次不同的记录")
        thickness = calculate_thickness_mm(
            d1_reading.value_mm, d2_reading.value_mm, self.refractive_index)
        round_ = MeasurementRound(
            sequence=self._next_sequence,
            d1_mm=d1_reading.value_mm,
            d2_mm=d2_reading.value_mm,
            refractive_index=self.refractive_index,
            thickness_mm=thickness,
            d1_source=d1_reading.key,
            d2_source=d2_reading.key,
            note=note,
        )
        self._next_sequence += 1
        self._rounds.append(round_)
        return round_

    # ---- 管理 ----

    def get(self, sequence: int) -> MeasurementRound:
        for round_ in self._rounds:
            if round_.sequence == sequence:
                return round_
        raise KeyError(f"不存在第 {sequence} 次测量")

    def remove(self, sequence: int) -> MeasurementRound:
        round_ = self.get(sequence)
        self._rounds.remove(round_)
        return round_

    def clear(self) -> None:
        self._rounds.clear()
        self._next_sequence = 1

    # ---- 统计分析 ----

    def statistics(self) -> SessionStats:
        values = [r.thickness_mm for r in self._rounds]
        n = len(values)
        if n == 0:
            return SessionStats(
                count=0, refractive_index=self.refractive_index)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = sqrt(variance)
        else:
            std = 0.0
        return SessionStats(
            count=n,
            mean_mm=mean,
            std_mm=std,
            min_mm=min(values),
            max_mm=max(values),
            refractive_index=self.refractive_index,
        )

    # ---- 序列化 ----

    def as_dict(self) -> dict:
        return {
            "version": self._VERSION,
            "name": self.name,
            "operator": self.operator,
            "sample_id": self.sample_id,
            "refractive_index": self.refractive_index,
            "created_at": self.created_at,
            "rounds": [r.as_dict() for r in self._rounds],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentSession:
        session = cls(
            refractive_index=float(
                data.get("refractive_index", GLASS_REFRACTIVE_INDEX)),
        )
        session.name = str(data.get("name", ""))
        session.operator = str(data.get("operator", ""))
        session.sample_id = str(data.get("sample_id", ""))
        session.created_at = float(data.get("created_at", time.time()))
        session._rounds = [
            MeasurementRound.from_dict(r)
            for r in data.get("rounds", [])
        ]
        if session._rounds:
            session._next_sequence = max(r.sequence for r in session._rounds) + 1
        return session

    def save(self, path: str | Path) -> None:
        """保存当前会话到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.as_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSession:
        """从 JSON 文件加载实验会话。"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = int(data.get("version", 0))
        if version > cls._VERSION:
            raise ValueError(
                f"存档文件版本 {version} 高于当前支持版本 {cls._VERSION}，"
                f"请升级软件后重试")
        return cls.from_dict(data)
