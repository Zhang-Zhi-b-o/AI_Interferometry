"""完整演示集成回归；禁止触碰真实设备或模型接口。"""
import csv
import json
import os
import time
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.services.demo import DemoRecord, DemoRig  # noqa: E402
from app.ui.main_window import MainWindow, PAGES  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_theme(instance)
    return instance


def wait_for(app, condition, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    assert condition(), "演示操作未在预期时间内完成"


def test_virtual_motion_and_stop(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.services.demo.time.monotonic", lambda: clock[0])
    rig = DemoRig()
    rig.motion.reverse()
    clock[0] += 0.5
    value, center = rig.sample()
    assert value == pytest.approx(0.977)
    assert center == pytest.approx(136)
    rig.motion.stop()
    clock[0] += 20
    assert rig.sample()[0] == value
    assert rig.image().shape == (180, 640, 3)
    rig.close()


def test_complete_demo_without_hardware_or_paid_model(app, monkeypatch, tmp_path):
    for name in ("CameraService", "MotionService", "ReadingService", "DetectService"):
        monkeypatch.setattr("app.ui.main_window." + name,
                            Mock(side_effect=AssertionError("演示不应实例化真实服务")))
    monkeypatch.setattr("cv2.VideoCapture", Mock(side_effect=AssertionError("禁止打开相机")))
    monkeypatch.setattr("serial.Serial", Mock(side_effect=AssertionError("禁止打开串口")))
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=AssertionError("禁止请求模型")))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "demo-must-ignore-key")
    window = MainWindow(load_config(), demo_mode=True)
    try:
        assert "演示模式" in window.windowTitle()
        wait_for(app, lambda: window.state.reading_mm is not None and window.ctx.frame() is not None)
        for page_class in PAGES:
            window._navigate(page_class.key)
            window._apply_state()

        compute = window._pages["compute"]
        compute._take("d1")
        window._pages["reading"]._record()
        search = window._pages["search"]
        search._toggle_search()
        wait_for(app, lambda: not window.search.active, seconds=9)
        assert abs(window.state.center_x - 320) <= 12
        window._on_reading({"value_mm": window.reading.value_mm, "at": time.time(), "stable": True})
        compute._take("d2")
        compute._calculate()
        assert window.state.thickness_mm > 0
        compute._add_round()
        assert len(compute._rounds) == 1
        window.ctx.record = DemoRecord(tmp_path)
        for kind in ("csv", "json", "md"):
            compute._export(kind)
        csv_path = next(tmp_path.glob("*.csv"))
        with csv_path.open(encoding="utf-8-sig") as handle:
            assert next(csv.DictReader(handle))["data_source"] == "demo"
        assert json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))["data_source"] == "demo"
        assert "演示数据" in next(tmp_path.glob("*.md")).read_text(encoding="utf-8")

        search.target_input.setValue(1.005)
        search._toggle_target()
        wait_for(app, lambda: not window.target.active, seconds=3)
        assert abs(window.reading.value_mm - 1.005) <= 0.0011

        fringe = window._pages["fringe"]
        fringe._measure()
        wait_for(app, lambda: fringe.measure_button.isEnabled())
        assert window.state.fringe_width_px == 11
        page = window._pages["map"]
        page._analyze()
        wait_for(app, lambda: page.analyze_button.isEnabled(), seconds=8)
        assert page._last_result is not None
        assert page._last_result["metrics"]["data_source"] == "demo"
        assert np.isfinite(page._last_result["thickness"]).any()
        page._show_surface()
        assert "演示数据" in page._surface_dialog.windowTitle()
        assistant = window._pages["assistant"]
        assert not assistant.runtime.provider.available
        assert "离线模式" in assistant.runtime.ask("下一步")
    finally:
        window.close()
    assert not window._demo_rig._thread.is_alive()
    assert not window.motion.connected
    assert window.motion.direction == "stopped"
