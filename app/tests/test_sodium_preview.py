import os
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from app.core.config import AppConfig
from app.core.state import AppState
from app.ui.main_window import MainWindow
from app.ui.pages.sodium import SodiumPage
from app.engine.vision.annotations import detection_style


def test_load_and_preview_do_not_start_measurement_or_motor():
    app = QApplication.instance() or QApplication([])
    class Bridge(QObject):
        sodium_update = Signal(dict)
    sodium = NS(active=False, running=False, run_id=0, start=Mock(), snapshot=lambda: {})
    ctx = NS(config=AppConfig(), bridge=Bridge(), sodium=sodium, log=Mock(),
             camera=NS(is_open=True), state=AppState(),
             search=NS(active=False), target=NS(active=False), motion=Mock(),
             run_async=lambda work, on_done, **kwargs: on_done(work()))
    page = SodiumPage(ctx)
    detector = NS(load=Mock(return_value=(True, "loaded")),
                  class_names={0: "end_blur", 1: "middle_blur"}, configure=Mock())
    page.preview_detector = detector
    try:
        page._load_model()
        assert page.preview_ready
        assert page.recognition_button.isEnabled()
        assert detector.load.call_args.args[0].as_posix().endswith("models/sodium/best.pt")
        assert detector.configure.call_args.kwargs["standard_strategy"] is False
        page._toggle_recognition()
        assert page.preview_enabled
        assert page.recognition_button.text() == "停止识别"
        sodium.start.assert_not_called()
        assert ctx.motion.mock_calls == []
        page._toggle_recognition()
        assert not page.preview_enabled
        assert detection_style("end_blur", .99, chinese=True, compact=True)[0] == "端部模糊"
        assert detection_style("middle_blur", .8, chinese=True, compact=True)[0] == "中间模糊"
    finally:
        page.shutdown()
        page.close()
        app.processEvents()


@pytest.mark.parametrize("cancelled", [False, True])
def test_preview_uses_roi_and_discards_late_results(cancelled):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = {"boxes_xyxy": [[10, 20, 30, 40]], "class_names": ["end_blur"], "confs": [.9]}
    page = NS(preview_enabled=True, preview_generation=1,
              preview_detector=NS(detect=Mock(return_value=result)), model_status=Mock())
    ctx = NS(roi=(10, 20, 30, 40), run_async=Mock())
    window = NS(ctx=ctx, camera=NS(is_open=True, latest_sample=lambda: (frame, time.monotonic())),
                search=NS(active=False), target=NS(active=False), sodium=NS(active=False),
                _detect_busy=False, _vision_epoch=1, _apply_detection=Mock(), _on_message=Mock())
    MainWindow._pump_sodium_preview(window, page)
    work = ctx.run_async.call_args.args[0]
    returned = work()
    assert page.preview_detector.detect.call_args.kwargs["roi"] == (10, 20, 30, 40)
    if cancelled:
        page.preview_generation += 1
        page.preview_enabled = False
    ctx.run_async.call_args.kwargs["on_done"](returned)
    if cancelled:
        window._apply_detection.assert_not_called()
    else:
        window._apply_detection.assert_called_once_with(result)
    assert not window._detect_busy


def test_measurement_emits_boxes_in_full_frame_coordinates(tmp_path):
    import threading
    from app.services.sodium import SodiumService

    emitted = []
    service = SodiumService(camera=NS(), reading=NS(), motion=NS(control_lock=threading.RLock()),
                            model_path=tmp_path / "best.pt", on_detection=emitted.append)
    service.roi = (100, 200, 300, 400)
    service.device, service.threshold, service.run_id = "cpu", .75, 7
    box = NS(cls=np.array([0]), conf=np.array([.9]),
             xyxy=[NS(cpu=lambda: NS(tolist=lambda: [10, 20, 30, 40]))])
    model = NS(names={0: "end_blur", 1: "middle_blur"},
               predict=lambda *a, **k: [NS(boxes=[box])])
    assert service._classify(model, np.zeros((100, 100, 3)))[0] == "end"
    assert emitted[0]["result"]["boxes_xyxy"] == [[110, 220, 130, 240]]
    assert emitted[0]["result"]["class_names"] == ["end_blur"]
    applied = Mock()
    window = NS(sodium=NS(run_id=7, running=True), _apply_detection=applied)
    MainWindow._on_sodium_detection(window, emitted[0])
    applied.assert_called_once_with(emitted[0]["result"])
    applied.reset_mock()
    window.sodium.running = False
    MainWindow._on_sodium_detection(window, emitted[0])
    applied.assert_not_called()


def test_navigation_keeps_sodium_preview_and_white_results_cannot_overwrite_it():
    sodium_page = NS(preview_enabled=True, stop_preview=Mock())
    search_page = NS(on_show=Mock(), on_state=Mock())
    window = NS(_pages={"sodium": sodium_page, "search": search_page}, page_host=Mock(),
                _nav_buttons={}, state=AppState(), sodium=NS(active=False), _apply_detection=Mock())
    MainWindow._navigate(window, "search")
    sodium_page.stop_preview.assert_not_called()
    MainWindow._on_white_detection(window, {"class_names": ["zero_order"]})
    window._apply_detection.assert_not_called()
    sodium_page.preview_enabled = False
    MainWindow._on_white_detection(window, {"class_names": ["zero_order"]})
    window._apply_detection.assert_called_once()


def test_explicit_white_detection_switches_off_sodium_preview():
    from app.ui.pages.live import LivePage
    ctx = NS(state=AppState(), stop_sodium_preview=Mock())
    ctx.state.camera_open = True
    LivePage._toggle_detect(NS(ctx=ctx))
    ctx.stop_sodium_preview.assert_called_once()
    assert ctx.state.detecting
