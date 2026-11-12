"""无硬件回归：序列、中间不停、停车确认、新读数及取消竞态。"""
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.services.sodium import SodiumSequence, SodiumService
from app.services.motion import MotionService


def feed(seq, kind, at):
    action = None
    for i in range(3):
        action = seq.observe(kind, at+i*.15) or action
    return action


def test_sequence_requires_middle_and_two_clear_intervals():
    s = SodiumSequence()
    assert feed(s, "end", 1) == "first"
    s.first_recorded()
    assert feed(s, "end", 2) is None
    assert s.phase == "leave_first"
    feed(s, "clear", 3)
    assert s.phase == "middle"
    assert feed(s, "end", 4) is None  # 中间前的误报不能停车
    assert feed(s, "middle", 5) is None
    assert s.phase == "leave_middle"
    assert feed(s, "end", 6) is None  # 未离开中间仍不停车
    feed(s, "clear", 7)
    assert feed(s, "end", 8) == "second"


def test_unknown_breaks_confirmation():
    s = SodiumSequence()
    s.observe("end", 1)
    s.observe("end", 1.2)
    s.observe("unknown", 1.3)
    assert s.observe("end", 1.4) is None
    assert s.phase == "first_end"


class Motion:
    def __init__(self):
        self.control_lock = threading.RLock()
        self.connected = True
        self.direction = "stopped"
        self.stop_written_at = 0
        self.commands = []
        self.ack = True

    def stop(self):
        self.commands.append("stop")
        self.direction = "stopped"
        self.stop_written_at = time.time() if self.ack else 0

    def set_speed(self, gear):
        self.commands.append(("speed", gear))

    def forward(self):
        self.direction = "forward"
        self.commands.append("forward")

    def reverse(self):
        self.direction = "reverse"
        self.commands.append("reverse")


class Camera:
    is_open = True
    frame_age = 0
    kind = "end"

    def latest_sample(self):
        return np.arange(30000, dtype=np.uint8).reshape(100, 100, 3), time.monotonic()


class Reading:
    connected = True
    value = 1.25
    frozen_at = None

    def latest_stable(self):
        return self.value, self.frozen_at if self.frozen_at is not None else time.time()


class Service(SodiumService):
    def _load_model(self):
        return None

    def _classify(self, model, frame):
        time.sleep(.03)
        return self.camera.kind, .95 if self.camera.kind == "end" else 0, .95 if self.camera.kind == "middle" else 0


def make(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"fake")
    s = Service(camera=Camera(), reading=Reading(), motion=Motion(), model_path=p)
    return s


def wait_phase(service, phase, limit=3):
    until = time.monotonic()+limit
    while time.monotonic() < until:
        if service.snapshot()["phase"] == phase:
            return
        time.sleep(.015)
    pytest.fail(str(service.snapshot()))


def close(service):
    service.stop()
    for t in (service._thread, service._watch):
        if t:
            t.join(2)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_full_scan_middle_does_not_stop_and_records_both(tmp_path, direction):
    s = make(tmp_path)
    try:
        assert s.start(direction=direction)[0]
        wait_phase(s, "leave_first")
        assert direction in s.motion.commands
        assert s.status["first_mm"] == 1.25
        s.camera.kind = "clear"
        wait_phase(s, "middle")
        stops = s.motion.commands.count("stop")
        s.camera.kind = "middle"
        wait_phase(s, "leave_middle")
        time.sleep(.1)
        assert s.motion.commands.count("stop") == stops
        assert s.motion.direction == direction
        s.camera.kind = "clear"
        wait_phase(s, "second_end")
        s.reading.value = 2.5
        s.camera.kind = "end"
        wait_phase(s, "done")
        assert s.motion.direction == "stopped"
        result = s.status["result"]
        assert result["second_mm"] == 2.5
        assert result["reading_difference_mm"] == 1.25
        assert result["mean_wavelength_nm"] == pytest.approx((30000 * 1.25)**.5)
        assert result["visibility_periods"] == 2
        assert result["doublet_separation_nm"] == .6
        assert result["micrometer_to_mirror_ratio"] == 20
        assert result["second_reading_at"] >= result["second_stop_written_at"]+.3
    finally:
        close(s)


def test_no_new_reading_after_stop_never_moves(tmp_path):
    s = make(tmp_path)
    s.reading.frozen_at = time.time()
    try:
        assert s.start()[0]
        wait_phase(s, "first_reading")
        wait_phase(s, "stopped", 3)
        assert "forward" not in s.motion.commands
        assert s.status["first_mm"] is None
    finally:
        close(s)


def test_camera_loss_stops_moving_motor(tmp_path):
    s = make(tmp_path)
    try:
        assert s.start()[0]
        wait_phase(s, "leave_first")
        s.camera.frame_age = 2
        wait_phase(s, "stopped")
        assert s.motion.direction == "stopped"
        assert s.status["result"] is None
    finally:
        close(s)


def test_cancel_during_inference_cannot_start_motor(tmp_path):
    s = make(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        release.wait(2)
        return "end", 1, 0
    s._classify = blocked
    try:
        assert s.start()[0]
        assert entered.wait(1)
        s.stop("急停")
        release.set()
        s._thread.join(2)
        assert "forward" not in s.motion.commands
        assert not s.running
    finally:
        release.set()
        close(s)


def test_busy_rejected_and_bad_roi_cannot_move(tmp_path):
    s = make(tmp_path)
    s._busy = lambda: True
    assert not s.start()[0]
    s._busy = lambda: False
    try:
        assert s.start(roi=(0, 0, 1000, 1000))[0]
        wait_phase(s, "stopped")
        assert "forward" not in s.motion.commands
    finally:
        close(s)


def test_middle_score_vetoes_end_detection(tmp_path):
    s = make(tmp_path)
    def box(cls, conf):
        return NS(cls=np.array([cls]), conf=np.array([conf]))
    model = NS(names={0: "end_blur", 1: "middle_blur"},
               predict=lambda *a, **k: [NS(boxes=[box(0, .99), box(1, .8)])])
    s.device = "cpu"
    assert SodiumService._classify(s, model, np.zeros((10, 10, 3)))[0] == "middle"


def test_stop_ack_is_written_only_after_controller_returns():
    s = MotionService()
    entered, release = threading.Event(), threading.Event()
    class Controller:
        is_connected = True
        def stop(self):
            entered.set()
            release.wait(2)
            return True
    s._controller = Controller()
    s.connected = True
    s._running.set()
    t = threading.Thread(target=s._worker, daemon=True)
    t.start()
    try:
        at = time.time()
        s.stop()
        assert entered.wait(1)
        assert s.stop_written_at == 0
        release.set()
        until = time.monotonic()+1
        while not s.stop_written_at and time.monotonic() < until:
            time.sleep(.01)
        assert s.stop_written_at >= at
    finally:
        release.set()
        s._running.clear()
        t.join(2)
