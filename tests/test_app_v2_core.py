"""v2 核心与服务层测试：不需要 Qt，也不需要硬件。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppConfig, load_config
from app.core.state import AppState
from app.services.calibration import CalibrationTable, opd_um_from_readings
from app.services.compute import ComputeService
from app.services.record import RecordService


# ----------------------------------------------------------------------
# 配置与状态
# ----------------------------------------------------------------------
def test_config_defaults_and_override():
    config = load_config()
    assert config.window_size[0] >= 1000
    assert config.refractive_index > 1.0
    assert config.get("search.direction_mode") == "single_direction"
    assert config.get("search.slow_gear") == 10
    custom = AppConfig(data={"measurement": {"refractive_index": 1.5}})
    assert custom.refractive_index == 1.5
    assert custom.get("does.not.exist", "fallback") == "fallback"


def test_state_update_and_snapshot():
    state = AppState()
    seen = []
    state.changed.subscribe(lambda s: seen.append(s.center_x))
    state.update(center_x=640.0, reading_mm=1.2345)
    state.update(center_x=640.0)  # 同值不应重复通知
    assert seen == [640.0]
    snapshot = state.snapshot()
    assert snapshot["reading_mm"] == 1.2345
    with pytest.raises(AttributeError):
        state.update(not_a_field=1)


# ----------------------------------------------------------------------
# 计算
# ----------------------------------------------------------------------
def test_thickness_matches_formula():
    service = ComputeService(refractive_index=1.4586)
    d1, d2 = 1.0000, 1.5000
    expected = (d2 - d1) / (20.0 * (1.4586 - 1.0))
    assert service.thickness_mm(d1, d2) == pytest.approx(expected)
    assert service.opd_um(d1, d2) == pytest.approx(abs(d2 - d1) / 20.0 * 1000.0)


def test_statistics_and_uncertainty():
    service = ComputeService()
    values = [1.230, 1.240, 1.220, 1.235]
    stats = service.statistics(values)
    assert stats["count"] == 4
    single = service.statistics([1.23])
    assert single["std"] is None and single["type_a"] is None
    assert stats["min"] == pytest.approx(1.220)
    report = service.uncertainty(values)
    assert isinstance(report, dict) and report


# ----------------------------------------------------------------------
# 标定表
# ----------------------------------------------------------------------
def test_calibration_roundtrip(tmp_path: Path):
    table = CalibrationTable()
    for index, opd in enumerate((0.0, 0.30, 0.60, 0.90)):
        table.add(opd_um=opd, colour=(10 * index, 20 * index, 30 * index))
    assert table.is_usable
    path = table.save(tmp_path / "calib.csv")
    other = CalibrationTable()
    assert other.load(path) == 4
    assert other.points[0].opd_um == 0.0
    assert other.rows()[0]["r"] == 0


def test_calibration_rejects_bad_tables():
    table = CalibrationTable()
    table.add(opd_um=0.0, colour=(1, 2, 3))
    table.add(opd_um=0.5, colour=(4, 5, 6))
    problems = table.problems()
    assert any("至少" in item for item in problems)

    duplicate = CalibrationTable()
    for opd in (0.0, 0.3, 0.3, 0.9):
        duplicate.add(opd_um=opd, colour=(1, 1, 1))
    assert any("递增" in item for item in duplicate.problems())


def test_opd_from_readings():
    assert opd_um_from_readings(1.0000, 1.5000) == pytest.approx(25.0)
    assert opd_um_from_readings(1.5000, 1.0000) == pytest.approx(25.0)


# ----------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------
def test_record_exports(tmp_path: Path):
    service = RecordService(out_dir=tmp_path)
    rows = [{"序号": 1, "厚度mm": 0.123456789}]
    csv_path = service.export_rows(rows, stem="test")
    assert csv_path.is_file() and csv_path.read_text(encoding="utf-8-sig").startswith("序号")
    json_path = service.export_json({"a": 1}, stem="test")
    assert '"a": 1' in json_path.read_text(encoding="utf-8")
    md_path = service.export_report(title="报告", sections=[("一", "内容")], stem="test")
    assert "# 报告" in md_path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 搜索闭环（用假设备，不碰硬件）
# ----------------------------------------------------------------------
class _FakeFringe:
    def __init__(self, center):
        self.center = center

    def measure(self, frame, center_x=None):
        from app.services.fringe import FringeMetrics

        return FringeMetrics(center_x=self.center)


class _FakeMotion:
    def __init__(self):
        import threading

        self.control_lock = threading.RLock()
        self.connected = True
        self.commands = []

    def set_speed(self, gear):
        self.commands.append(("speed", gear))

    def forward(self):
        self.commands.append(("forward",))

    def reverse(self):
        self.commands.append(("reverse",))

    def stop(self):
        self.commands.append(("stop",))


def test_search_converges_when_centered():
    from app.services.search import SearchService

    motion = _FakeMotion()
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    service = SearchService(
        frame_provider=lambda: frame,
        observe=lambda frame: {"center_x": 400.0, "confidence": 1.0},  # 正中央
        motion=motion,
    )
    ok, _ = service.start(tolerance_px=12, stable_frames=2, tick_ms=1, max_seconds=5)
    assert ok
    import time

    for _ in range(200):
        if not service.running:
            break
        time.sleep(0.01)
    assert not service.running
    assert ("stop",) in motion.commands


def test_search_moves_toward_center():
    from app.services.search import SearchService

    motion = _FakeMotion()
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    service = SearchService(
        frame_provider=lambda: frame,
        observe=lambda frame: {"center_x": 600.0, "confidence": 1.0},  # 偏右 200px
        motion=motion,
    )
    service.forward_positive = True
    ok, _ = service.start(tolerance_px=10, stable_frames=1, slow_zone_px=50,
                          fast_gear=9, slow_gear=10, tick_ms=1, max_seconds=0.05)
    assert ok
    service._thread.join(2)
    assert not service.active
    assert ("reverse",) in motion.commands  # 正转使 x 增大，偏右应反转
    assert motion.commands[-1] == ("stop",)
