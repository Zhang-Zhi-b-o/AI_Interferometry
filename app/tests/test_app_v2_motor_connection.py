"""连接生命周期与页面异步行为，只使用假控制器，不访问串口。"""
import os
import threading
import time
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.services.motion import MotionService  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


class Controller:
    instances = []

    def __init__(self, port, baudrate):
        self.port, self.baudrate = port, baudrate
        self.closed = False
        self.instances.append(self)

    def connect(self):
        return True

    def try_stop_on_close(self):
        self.closed = True
        return True


@pytest.fixture
def controllers(monkeypatch):
    Controller.instances = []
    monkeypatch.setattr("app.engine.hardware.motor.MotorController", Controller)
    return Controller.instances


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_connect_disconnect_and_no_implicit_motion(controllers):
    motion = MotionService()
    assert motion.connect("COM9", 115200)[0]
    assert controllers[0].port == "COM9"
    assert controllers[0].baudrate == 115200
    assert motion.direction == "stopped"
    motion.begin_disconnect()
    assert not motion.connected
    motion.disconnect()
    assert controllers[0].closed
    assert motion._controller is None


def test_pre_cancelled_connect_never_constructs_controller(controllers):
    token = threading.Event()
    token.set()
    assert not MotionService().connect(cancel_event=token)[0]
    assert controllers == []


@pytest.mark.parametrize("cancel_with_stop", [False, True])
def test_late_connect_is_closed_after_disconnect_or_stop(monkeypatch, cancel_with_stop):
    entered, release = threading.Event(), threading.Event()
    instances = []

    class Delayed(Controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def connect(self):
            entered.set()
            assert release.wait(2)
            return True

    monkeypatch.setattr("app.engine.hardware.motor.MotorController", Delayed)
    motion = MotionService()
    token = threading.Event()
    results = []
    worker = threading.Thread(target=lambda: results.append(motion.connect(cancel_event=token)))
    worker.start()
    assert entered.wait(2)
    if cancel_with_stop:
        motion.stop()
    else:
        token.set()
        motion.disconnect()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert not results[0][0]
    assert not motion.connected and motion._controller is None
    assert instances[0].closed


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_failed_connection_closes_partial_controller(monkeypatch, failure):
    instances = []

    class Failed(Controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def connect(self):
            if failure == "exception":
                raise OSError("假串口连接失败")
            return False

    monkeypatch.setattr("app.engine.hardware.motor.MotorController", Failed)
    motion = MotionService()
    assert not motion.connect()[0]
    assert not motion.connected
    assert instances[0].closed


def test_panel_defers_io_handles_failure_and_disconnects(app, controllers):
    window = MainWindow(load_config())
    try:
        panel = window._pages["search"].motor
        queued = []
        window.ctx.run_async = lambda work, **callbacks: queued.append((work, callbacks))
        motion = window.motion
        motion.connect = Mock(return_value=(False, "端口不存在"))
        panel.port.setText("COM9")
        panel._toggle()
        assert panel.busy and not panel.button.isEnabled()
        motion.connect.assert_not_called()
        work, callbacks = queued.pop()
        callbacks["on_done"](work())
        assert not panel.busy and "端口不存在" in panel.status.text()
        panel.refresh()
        assert "端口不存在" in panel.status.text()

        def connected(*args, **kwargs):
            motion.connected = True
            return True, "假设备已连接"

        motion.connect.side_effect = connected
        panel._toggle()
        work, callbacks = queued.pop()
        callbacks["on_done"](work())
        assert panel.button.text() == "断开"
        assert not panel.port.isEnabled()
        window.ctx.stop_motion = Mock(wraps=window._emergency_stop)
        motion.disconnect = Mock()
        panel._toggle()
        window.ctx.stop_motion.assert_called_once()
        assert not motion.connected  # 后台关闭尚未执行，已经禁止新闭环
        motion.disconnect.assert_not_called()
        work, callbacks = queued.pop()
        callbacks["on_done"](work())
        motion.disconnect.assert_called_once()
        assert panel.button.text() == "连接"
    finally:
        window.close()


def test_disconnect_and_window_close_remain_responsive_during_serial_io(app, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Blocking(Controller):
        def start_forward(self):
            entered.set()
            assert release.wait(3)
            return True

        def try_stop_on_close(self):
            super().try_stop_on_close()
            closed.set()
            return True

    monkeypatch.setattr("app.engine.hardware.motor.MotorController", Blocking)
    window = MainWindow(load_config())
    try:
        assert window.motion.connect()[0]
        window.motion.forward()
        assert entered.wait(2)
        started = time.monotonic()
        window.motion.begin_disconnect()
        assert time.monotonic() - started < 0.2
        assert not window.motion.connected
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < 0.3
    finally:
        release.set()
        window.close()
    assert closed.wait(2)


def test_panel_window_close_cancels_queued_connect(app, controllers):
    window = MainWindow(load_config())
    panel = window._pages["search"].motor
    queued = []
    window.ctx.run_async = lambda work, **callbacks: queued.append((work, callbacks))
    panel._toggle()
    window.close()
    work, callbacks = queued.pop()
    result = work()
    assert not result[0]
    assert controllers == []
    callbacks["on_done"](result)
    assert not window.motion.connected
