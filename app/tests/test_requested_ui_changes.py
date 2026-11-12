import os
from types import SimpleNamespace as NS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.core.config import AppConfig, load_config
from app.core.state import AppState
from app.engine.vision.annotations import draw_detections, visible_detection_indices
from app.ui.main_window import MainWindow, PAGES
from app.ui.pages.reading import ReadingPage
from app.ui.theme import apply_theme


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def test_one_center_and_multiple_noncenter_boxes():
    boxes = [[10, 40, 40, 130], [60, 40, 90, 130],
             [110, 40, 140, 130], [160, 40, 190, 130]]
    names = ["zero_order", "near_fringe", "zero_order", "near_fringe"]
    confs = [.6, .8, .9, .7]
    assert visible_detection_indices(boxes, names, confs) == [1, 3, 2]
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    draw_detections(frame, boxes, names, confs)
    assert not frame[80, 10].any()
    assert frame[80, 110].any()
    assert frame[80, 60].any() and frame[80, 160].any()
    assert confs == [.6, .8, .9, .7]  # rendering preserves control inputs


def test_roi_and_noncenter_only_overlays_are_preserved():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert visible_detection_indices(boxes) == [0, 1]
    assert visible_detection_indices(boxes, ["near_fringe"] * 2, [.6, .7]) == [0, 1]


@pytest.mark.parametrize("expanded", [False, True])
def test_reading_content_fits_narrow_right_panel(qt_app, expanded):
    ctx = NS(config=AppConfig(), state=AppState(), reading=NS(connected=False), history=[])
    page = ReadingPage(ctx)
    try:
        page.resize(460, 720)
        page.advanced_toggle.setChecked(expanded)
        page.show()
        qt_app.processEvents()
        assert page.scroll.widget().width() <= page.scroll.viewport().width()
        assert page.scroll.horizontalScrollBar().maximum() == 0
    finally:
        page._signal_timer.stop()
        page.close()


def test_removed_pages_and_backlash_controls(qt_app):
    assert "map" not in [page.key for page in PAGES]
    assert [page.title[0] for page in PAGES] == list("①②③④⑤⑥⑦")
    window = MainWindow(load_config(), demo_mode=True)
    try:
        assert "map" not in window._nav_buttons
        assert window._nav_buttons["assistant"].text() == "⑥ 实验助手"
        search = window._pages["search"]
        assert [search.tabs.tabText(i) for i in range(search.tabs.count())] == ["自动寻中", "目标读数"]
        assert not hasattr(window, "backlash")
        search.on_state(window.state)
        window._navigate("reading")
    finally:
        window.close()
