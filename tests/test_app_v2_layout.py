"""小窗口主操作可见性回归，使用虚拟设备且不滚动页面。"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_theme(instance)
    return instance


TARGETS = {
    "live": ("camera_button", "load_button", "detect_button"),
    "search": ("search_button", "direction_mode", "recognition_mode",
               "search_direction", "forward_positive"),
    "fringe": ("measure_button",),
    "reading": ("start_button", "record_button"),
    "compute": ("calc_button", "take_d1", "take_d2"),
    "map": ("analyze_button",),
    "assistant": ("send",),
}


@pytest.mark.parametrize("size", [(1366, 768), (1280, 720)])
def test_primary_actions_visible_without_scrolling(app, size, tmp_path):
    window = MainWindow(load_config(), demo_mode=True)
    try:
        window.resize(*size)
        window.show()
        app.processEvents()
        assert window.width() <= size[0] and window.height() <= size[1]

        def check(page, widgets):
            scroll = page.findChild(QScrollArea)
            scroll.verticalScrollBar().setValue(0)
            app.processEvents()
            viewport = scroll.viewport()
            for widget in widgets:
                rect = QRect(widget.mapTo(viewport, QPoint()), widget.size())
                if not viewport.rect().contains(rect):
                    window.grab().save(str(tmp_path / (page.key + ".png")))
                assert viewport.rect().contains(rect), (page.key, widget.text() if hasattr(widget, "text") else type(widget).__name__, rect, viewport.rect())

        for key, names in TARGETS.items():
            window._navigate(key)
            page = window._pages[key]
            app.processEvents()
            if key == "compute":
                page.d1.setValue(1)
                page.d2.setValue(1.5)
                page._calculate()  # 有结果时，长 OPD 文本也不能撑出横向边界。
            check(page, [getattr(page, name) for name in names])
            if key == "search":
                check(page, [page.motor.port, page.motor.baudrate, page.motor.button])
                page.tabs.setCurrentIndex(1)
                check(page, [page.target_input, page.target_tolerance,
                             page.use_current, page.target_button])
                page.tabs.setCurrentIndex(0)
    finally:
        window.close()
