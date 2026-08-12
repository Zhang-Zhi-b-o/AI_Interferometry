"""实验助手模块的单元测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.measurement import (
    ExperimentSession,
    MeasurementRound,
    SessionStats,
    ThicknessReading,
    calculate_thickness_mm,
)
from src.measurement.experiment_assistant import GLASS_REFRACTIVE_INDEX


class MeasurementRoundTests(unittest.TestCase):
    def test_round_stores_all_fields(self):
        r = MeasurementRound(
            sequence=1, d1_mm=1.000, d2_mm=1.500,
            refractive_index=GLASS_REFRACTIVE_INDEX,
            thickness_mm=0.109, d1_source="手动", d2_source="手动",
            note="测试",
        )
        self.assertEqual(r.sequence, 1)
        self.assertEqual(r.label, "第1次")
        self.assertEqual(r.note, "测试")

    def test_round_as_dict_and_back(self):
        original = MeasurementRound(
            sequence=3, d1_mm=2.100, d2_mm=2.600,
            refractive_index=GLASS_REFRACTIVE_INDEX,
            thickness_mm=0.109, d1_source="R1", d2_source="R2",
            note="round trip",
        )
        restored = MeasurementRound.from_dict(original.as_dict())
        self.assertEqual(restored.sequence, original.sequence)
        self.assertAlmostEqual(restored.d1_mm, original.d1_mm, places=12)
        self.assertAlmostEqual(restored.d2_mm, original.d2_mm, places=12)
        self.assertAlmostEqual(restored.thickness_mm, original.thickness_mm, places=12)
        self.assertEqual(restored.d1_source, original.d1_source)
        self.assertEqual(restored.d2_source, original.d2_source)
        self.assertEqual(restored.note, original.note)

    def test_created_at_auto_populated(self):
        r = MeasurementRound(
            sequence=1, d1_mm=1.0, d2_mm=1.5,
            refractive_index=GLASS_REFRACTIVE_INDEX,
            thickness_mm=0.109,
        )
        self.assertGreater(r.created_at, 0)


class ExperimentSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ExperimentSession()

    # ---- 手动输入 ----

    def test_add_manual_computes_thickness(self):
        r = self.session.add_manual(d1_mm=1.000, d2_mm=1.500)
        expected = calculate_thickness_mm(1.000, 1.500, GLASS_REFRACTIVE_INDEX)
        self.assertAlmostEqual(r.thickness_mm, expected, places=12)
        self.assertEqual(r.d1_source, "手动")
        self.assertEqual(r.d2_source, "手动")

    def test_add_manual_increments_sequence(self):
        r1 = self.session.add_manual(1.0, 1.5)
        r2 = self.session.add_manual(2.0, 2.5)
        self.assertEqual(r1.sequence, 1)
        self.assertEqual(r2.sequence, 2)
        self.assertEqual(self.session.count, 2)

    def test_add_manual_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            self.session.add_manual(float("nan"), 1.5)
        with self.assertRaises(ValueError):
            self.session.add_manual(1.0, float("inf"))

    # ---- 从读数导入 ----

    def test_add_from_readings(self):
        reading1 = ThicknessReading(1, 1.000, 100.0)
        reading2 = ThicknessReading(2, 1.500, 101.0)
        r = self.session.add_from_readings(reading1, reading2, note="import")
        expected = calculate_thickness_mm(1.000, 1.500, GLASS_REFRACTIVE_INDEX)
        self.assertAlmostEqual(r.thickness_mm, expected)
        self.assertEqual(r.d1_source, "R1")
        self.assertEqual(r.d2_source, "R2")
        self.assertEqual(r.note, "import")

    def test_add_from_readings_rejects_same_reading(self):
        reading = ThicknessReading(1, 1.000, 100.0)
        with self.assertRaises(ValueError):
            self.session.add_from_readings(reading, reading)

    # ---- 折射率 ----

    def test_refractive_index_default(self):
        self.assertAlmostEqual(
            self.session.refractive_index, GLASS_REFRACTIVE_INDEX)

    def test_custom_refractive_index_affects_thickness(self):
        session = ExperimentSession(refractive_index=1.55)
        r = session.add_manual(1.000, 1.500)
        expected = calculate_thickness_mm(1.000, 1.500, 1.55)
        self.assertAlmostEqual(r.thickness_mm, expected)

    def test_refractive_index_must_be_greater_than_one(self):
        with self.assertRaises(ValueError):
            ExperimentSession(refractive_index=1.0)
        with self.assertRaises(ValueError):
            ExperimentSession(refractive_index=0.5)

    # ---- 管理 ----

    def test_get_and_remove(self):
        self.session.add_manual(1.0, 1.5)
        self.session.add_manual(2.0, 2.5)
        r = self.session.get(2)
        self.assertEqual(r.sequence, 2)
        self.session.remove(2)
        self.assertEqual(self.session.count, 1)
        with self.assertRaises(KeyError):
            self.session.get(2)

    def test_clear(self):
        self.session.add_manual(1.0, 1.5)
        self.session.add_manual(2.0, 2.5)
        self.session.clear()
        self.assertEqual(self.session.count, 0)
        # sequence 重置
        r = self.session.add_manual(3.0, 3.5)
        self.assertEqual(r.sequence, 1)

    # ---- 统计 ----

    def test_statistics_empty(self):
        stats = self.session.statistics()
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.mean_mm, 0.0)

    def test_statistics_single_round(self):
        self.session.add_manual(1.000, 1.500)
        stats = self.session.statistics()
        self.assertEqual(stats.count, 1)
        expected = calculate_thickness_mm(1.000, 1.500)
        self.assertAlmostEqual(stats.mean_mm, expected, places=12)
        self.assertAlmostEqual(stats.min_mm, expected, places=12)
        self.assertAlmostEqual(stats.max_mm, expected, places=12)
        self.assertEqual(stats.std_mm, 0.0)

    def test_statistics_multiple_rounds(self):
        rounds = [
            self.session.add_manual(1.000, 1.500),
            self.session.add_manual(2.000, 2.500),
            self.session.add_manual(3.000, 3.500),
        ]
        stats = self.session.statistics()
        self.assertEqual(stats.count, 3)
        values = [r.thickness_mm for r in rounds]
        expected_mean = sum(values) / 3
        self.assertAlmostEqual(stats.mean_mm, expected_mean, places=12)
        self.assertAlmostEqual(stats.min_mm, min(values), places=12)
        self.assertAlmostEqual(stats.max_mm, max(values), places=12)
        # 标准差：样本标准差
        variance = sum((v - expected_mean) ** 2 for v in values) / 2
        expected_std = variance ** 0.5
        self.assertAlmostEqual(stats.std_mm, expected_std, places=12)

    # ---- 序列化 ----

    def test_save_and_load_roundtrip(self):
        self.session.name = "测试实验"
        self.session.operator = "张三"
        self.session.sample_id = "BK7-001"
        self.session.add_manual(1.000, 1.500, note="round1")
        self.session.add_manual(2.000, 2.500, note="round2")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_session.json"
            self.session.save(path)
            loaded = ExperimentSession.load(path)

        self.assertEqual(loaded.name, "测试实验")
        self.assertEqual(loaded.operator, "张三")
        self.assertEqual(loaded.sample_id, "BK7-001")
        self.assertAlmostEqual(
            loaded.refractive_index, GLASS_REFRACTIVE_INDEX)
        self.assertEqual(loaded.count, 2)
        for orig, restored in zip(self.session.rounds, loaded.rounds):
            self.assertAlmostEqual(orig.thickness_mm, restored.thickness_mm, places=12)
            self.assertEqual(orig.note, restored.note)


class SessionStatsTests(unittest.TestCase):
    def test_stats_as_dict(self):
        stats = SessionStats(
            count=3, mean_mm=0.109, std_mm=0.001,
            min_mm=0.108, max_mm=0.110,
        )
        d = stats.as_dict()
        self.assertEqual(d["count"], 3)
        self.assertAlmostEqual(d["mean_mm"], 0.109)


if __name__ == "__main__":
    unittest.main()
