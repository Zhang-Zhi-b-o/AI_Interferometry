"""界面冒烟测试：offscreen 起窗口，逐页切换，确认没有崩。

需要 PySide6；没装就跳过。运行::

    python -m pytest tests/test_app_v2_ui_smoke.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.ui import theme as theme_module  # noqa: E402
from app.ui.main_window import PAGES, MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    theme_module.apply_theme(instance, "light", 1.0)
    yield instance


def test_chinese_font_for_windows_offscreen(app):
    from PySide6.QtGui import QFontMetrics

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if not font_path.is_file():
        pytest.skip("本机没有 Windows 微软雅黑字体")
    assert QFontMetrics(app.font()).inFontUcs4(ord("厚"))


def test_main_window_builds_and_navigates(app):
    window = MainWindow(load_config())
    try:
        assert window.page_host.count() == len(PAGES)
        for page_class in PAGES:
            window._navigate(page_class.key)
            assert window.page_host.currentWidget().key == page_class.key
            # 每次切页都推一次状态，页面刷新不应抛异常
            window._apply_state()
        # 状态更新通道
        window.state.update(reading_mm=1.2345, center_x=640.0)
        window._apply_state()
        assert window.metric_reading._value.text() == "1.2345"
        # 急停不应抛异常（没有连接硬件）
        window._emergency_stop()
        window._toggle_theme()
        window._toggle_theme()
    finally:
        window.close()


def test_assistant_cancel_queued_turn(app):
    from unittest.mock import Mock

    window = MainWindow(load_config())
    try:
        page = window._pages["assistant"]
        provider = Mock(available=True)
        page.runtime.provider = provider
        queued = []
        window.ctx.run_async = lambda work, **kwargs: queued.append(work)
        page.input.setPlainText("开始寻中")
        page._send()
        assert not page.send.isEnabled()
        window._emergency_stop()
        assert "已取消" in queued[0]()
        provider.chat_with_tools.assert_not_called()
        page._done("已取消")
        assert page.send.isEnabled()
    finally:
        window.close()


def test_assistant_confirmation_rejected_on_shutdown(app):
    import threading

    window = MainWindow(load_config())
    page = window._pages["assistant"]
    request = {"ready": threading.Event(), "approved": False}
    page._confirm(request)
    assert not request["ready"].is_set()
    window.close()
    assert request["ready"].is_set()
    assert not request["approved"]


def test_assistant_streams_deltas_then_done(app):
    window = MainWindow(load_config())
    try:
        window._navigate("assistant")
        page = window._pages["assistant"]
        page.transcript.clear()
        page._streaming = False
        page._streamed = []
        page.activity.emit("delta", "先测条纹，")
        page.activity.emit("delta", "再读数。")
        app.processEvents()
        page._done("先测条纹，再读数。")
        text = page.transcript.toPlainText()
        assert text.count("先测条纹，") == 1
        assert text == "助手：先测条纹，再读数。"
        # 取消/未流入的结尾仍要显示出来。
        page._streamed = ["部分"]
        page._streaming = True
        page._done("已取消；已发出的模型请求可能仍由服务端处理。")
        assert page.transcript.toPlainText().endswith("已取消；已发出的模型请求可能仍由服务端处理。")
    finally:
        window.close()


def test_search_modes_and_background_detection_signal(app):
    from unittest.mock import Mock

    window = MainWindow(load_config())
    try:
        page = window._pages["search"]
        window.ctx.search.start = Mock(return_value=(False, "假设备未连接"))
        for direction in range(2):
            for recognition in range(2):
                page.direction_mode.setCurrentIndex(direction)
                page.recognition_mode.setCurrentIndex(recognition)
                page._toggle_search()
                args = window.ctx.search.start.call_args.kwargs
                assert args["direction_mode"] == page.direction_mode.currentData()
                assert args["recognition_mode"] == page.recognition_mode.currentData()
                assert 1 <= args["slow_gear"] <= 10
        window.bridge.detection.emit({
            "boxes_xyxy": [[10, 10, 30, 40]], "confs": [0.9],
            "class_names": ["zero_order"], "center": (20, 25)})
        assert window.state.detect_count == 1
        assert window.state.center_x is None  # 检测框中心不是物理中心条纹
        window._on_search_update({"center": 24, "confidence": 0.8, "running": True,
                                  "message": "扩大范围搜索",
                                  "searched_min_turns": -2, "searched_max_turns": 3})
        page.on_state(window.state)
        assert "估算已搜范围" in page.progress.text()
    finally:
        window.close()


def test_surface_dialog_and_unicode_image(app, tmp_path):
    import cv2
    import numpy as np

    from app.main import _load_image
    from app.ui.widgets.surface import SurfaceDialog

    y, x = np.mgrid[:12, :16]
    data = (x * 0.1 + y * 0.2).astype(float)
    data[4:6, 5:7] = np.nan
    result = {"thickness": data, "mask": np.isfinite(data), "mode": "relative",
              "metrics": {"has_absolute_anchor": False}}
    dialog = SurfaceDialog(result)
    try:
        assert "Relative" in dialog.axes.get_zlabel()
        target = tmp_path / "三维形貌.png"
        dialog.figure.savefig(target)
        assert target.stat().st_size > 1000
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :, 2] = 200
        image_path = tmp_path / "中文演示图.png"
        cv2.imencode(".png", image)[1].tofile(image_path)
        np.testing.assert_array_equal(_load_image(str(image_path)), image)
        assert _load_image(str(tmp_path / "不存在.png")) is None
    finally:
        dialog.close()


def test_escape_stops_with_text_field_focused(app):
    from unittest.mock import Mock

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    window = MainWindow(load_config())
    try:
        window.show()
        window.activateWindow()
        window._navigate("assistant")
        field = window._pages["assistant"].input
        field.setFocus()
        app.processEvents()
        window.motion.stop = Mock()
        QTest.keyClick(field, Qt.Key.Key_Escape)
        app.processEvents()
        window.motion.stop.assert_called_once()
    finally:
        window.close()


def test_calibration_table_view(app):
    window = MainWindow(load_config())
    try:
        page = window._pages["map"]
        window._navigate("map")
        window.ctx.calibration.add(opd_um=0.3, colour=(10, 20, 30))
        page._refresh_table()
        assert page.table.rowCount() == 1
        assert page.table.item(0, 0).text() == "0.300"
    finally:
        window.close()
