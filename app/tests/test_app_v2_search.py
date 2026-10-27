"""寻中服务迁移回归：假帧、假检测、假串口，不驱动硬件。"""
import queue
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.services.motion import MotionService
from app.services.reading import ReadingService
from app.services.search import SearchService, TargetReadingService
from app.services.search_observation import SearchObservation

FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


class Motion:
    def __init__(self):
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


def make_search(motion, observe=lambda frame: {"center_x": None, "confidence": 0.0}, **kwargs):
    return SearchService(frame_provider=lambda: FRAME, motion=motion, observe=observe, **kwargs)


@pytest.mark.parametrize("direction_mode", ["single_direction", "bidirectional"])
@pytest.mark.parametrize("recognition_mode", ["continuous", "stop_and_detect"])
def test_four_modes_reach_old_machine(direction_mode, recognition_mode):
    motion = Motion()
    updated = threading.Event()
    states = []
    service = make_search(motion, on_update=lambda payload: (states.append(payload), updated.set()))
    ok, _ = service.start(direction_mode=direction_mode, recognition_mode=recognition_mode,
                          search_direction="reverse", tick_ms=10)
    assert ok
    assert updated.wait(2)
    service.stop()
    service._thread.join(2)
    assert ("reverse",) in motion.commands
    assert states[0]["search_phase"]
    assert motion.commands[-1] == ("stop",)


def test_blind_search_expands_and_reverses():
    motion = Motion()
    reversed_event = threading.Event()
    original_reverse = motion.reverse

    def reverse():
        original_reverse()
        reversed_event.set()

    motion.reverse = reverse
    service = make_search(motion)
    assert service.start(direction_mode="bidirectional", search_gear=1, blur_safe_gear=1,
                         search_min_gear=1, fast_gear=1, slow_gear=1,
                         search_initial_span_turns=0.1, tick_ms=5, max_seconds=2)[0]
    assert reversed_event.wait(2)
    service.stop()
    service._thread.join(2)
    assert ("forward",) in motion.commands
    assert ("reverse",) in motion.commands


def test_stop_during_inference_never_restarts_and_blocks_new_run():
    motion = Motion()
    entered, release = threading.Event(), threading.Event()

    def observe(frame):
        entered.set()
        assert release.wait(2)
        return {"center_x": None, "confidence": 0.0}

    service = make_search(motion, observe)
    assert service.start()[0]
    assert entered.wait(2)
    service.stop("急停")
    assert service.active
    assert not service.start()[0]
    release.set()
    service._thread.join(2)
    assert motion.commands and all(c == ("stop",) for c in motion.commands)


def test_watchdog_stops_when_frames_stale_during_inference():
    motion = Motion()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    stale = threading.Event()

    def observe(frame):
        entered.set()
        release.wait(2)
        return {"center_x": None, "confidence": 0.0}

    service = make_search(motion, observe,
                          ready=lambda: "画面过期" if stale.is_set() else "",
                          on_update=lambda p: stopped.set() if p.get("reason") == "画面过期" else None)
    assert service.start()[0]
    assert entered.wait(2)
    stale.set()
    assert stopped.wait(1)
    release.set()
    service._thread.join(2)
    assert all(c == ("stop",) for c in motion.commands)


def test_stale_inference_input_rejected_even_when_camera_is_fresh():
    motion = Motion()
    service = SearchService(
        frame_provider=lambda: (FRAME, time.monotonic() - 2),
        motion=motion, observe=lambda frame: {"center_x": None, "confidence": 0},
        ready=lambda: "")
    assert service.start()[0]
    service._thread.join(2)
    assert not service.active
    assert motion.commands and all(c == ("stop",) for c in motion.commands)


def test_single_search_direction_is_physical_even_with_inverted_mapping():
    motion = Motion()
    moved = threading.Event()
    service = make_search(motion, on_update=lambda p: moved.set() if p.get("gear") else None)
    service.forward_positive = False
    assert service.start(search_direction="forward")[0]
    assert moved.wait(2)
    service.stop()
    service._thread.join(2)
    assert ("forward",) in motion.commands
    assert ("reverse",) not in motion.commands


@pytest.mark.parametrize("fault", ["missing", "stale", "disconnected"])
def test_target_stops_on_invalid_live_reading(fault):
    motion = Motion()
    bad = threading.Event()
    moved = threading.Event()
    finished = threading.Event()
    reading = NS(connected=True)

    def latest():
        if bad.is_set():
            if fault == "disconnected":
                reading.connected = False
            return (None, time.time()) if fault == "missing" else (0, time.time() - 2)
        return 0, time.time()

    reading.latest = latest
    target = TargetReadingService(
        reading=reading, motion=motion,
        on_update=lambda p: moved.set() if p.get("direction") else None,
        on_finish=lambda reason: finished.set())
    assert target.start(target_mm=1, tick_ms=5)[0]
    assert moved.wait(2)
    bad.set()
    assert finished.wait(2)
    target._thread.join(1)
    assert motion.commands[-1] == ("stop",)


@pytest.mark.parametrize("failure", ["exception", "false"])
def test_motor_command_failure_marks_disconnected(failure):
    service = MotionService()
    stopped = threading.Event()

    def forward():
        if failure == "exception":
            raise OSError("串口断开")
        return False

    service._controller = NS(start_forward=forward, try_stop_on_close=stopped.set,
                             stop=lambda: pytest.fail("故障后常规 stop 不会写入串口"))
    service.connected = True
    service._running.set()
    service.forward()
    worker = threading.Thread(target=service._worker)
    worker.start()
    worker.join(1)
    assert not worker.is_alive()
    assert not service.connected
    assert stopped.is_set()


def test_held_ocr_value_keeps_its_original_timestamp():
    reading = ReadingService()
    old = time.time() - 10
    reading._handle(NS(stable_value_mm=1, value_mm=None, stable=False,
                       reading_held=True, stable_captured_at=old, captured_at=time.time()))
    assert reading.latest() == (1, old)
    reading.connected = True
    target = TargetReadingService(reading=reading, motion=Motion())
    assert not target.start(target_mm=2)[0]


def test_readiness_and_loop_exclusion():
    motion = Motion()
    assert not make_search(motion, ready=lambda: "没有模型").start()[0]
    assert not make_search(motion, busy=lambda: True).start()[0]
    target = TargetReadingService(reading=NS(connected=True), motion=motion, busy=lambda: True)
    assert not target.start(target_mm=1)[0]
    assert motion.commands == []


def test_zero_box_is_prior_not_measured_center(monkeypatch):
    result = {"boxes_xyxy": [[40, 10, 80, 90]], "confs": [0.9],
              "class_names": ["zero_order"]}
    detector = NS(detect=lambda *args, **kwargs: result, reset=lambda: None)
    observer = SearchObservation(detector)
    monkeypatch.setattr("app.services.search_observation.find_center_in_region",
                        lambda *args, **kwargs: {"center_main": 16, "confidence": 0.8,
                                                "orientation": "vertical"})
    payload = observer(FRAME)
    assert payload["zero_box_x"] == 60
    assert payload["center_x"] == 36  # 扩展 ROI 左端 20 + 实测 16，不能直接采用框中心 60


def test_near_fringe_never_becomes_zero_center():
    detector = NS(detect=lambda *args, **kwargs: {
        "boxes_xyxy": [[40, 10, 80, 90]], "confs": [0.9],
        "class_names": ["near_fringe"]}, reset=lambda: None)
    payload = SearchObservation(detector)(FRAME)
    assert payload["center_x"] is None
    assert payload["guide_x"] == 60
    assert payload["guide_count"] == 1


def test_motor_stop_invalidates_already_dequeued_command():
    dequeued, release, stopped = threading.Event(), threading.Event(), threading.Event()

    class PausingQueue(queue.Queue):
        first = True

        def get(self, *args, **kwargs):
            result = super().get(*args, **kwargs)
            if self.first:
                self.first = False
                dequeued.set()
                assert release.wait(2)
            return result

    calls = []
    motion = MotionService()
    motion._queue = PausingQueue()
    motion._controller = NS(start_forward=lambda: calls.append("forward"),
                            stop=lambda: (calls.append("stop"), stopped.set()))
    motion._running.set()
    motion.forward()
    worker = threading.Thread(target=motion._worker, daemon=True)
    worker.start()
    try:
        assert dequeued.wait(2)
        motion.stop()
        release.set()
        assert stopped.wait(2)
        assert calls == ["stop"]
    finally:
        release.set()
        motion._running.clear()
        worker.join(1)
