"""Motor recovery checks use memory queues and mocked serial devices only."""
import importlib
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import serial

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from app.services.motion import MotionService


@pytest.fixture(autouse=True)
def quiet_motor_log(monkeypatch):
    monkeypatch.setattr('app.services.motion._logger', Mock())


@pytest.mark.parametrize('module_name', ['app.engine.hardware.motor'])
def test_serial_connect_and_reconnect_bound_writes(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, 'logger', Mock())
    factory = Mock()
    factory.return_value.is_open = True
    monkeypatch.setattr(module.serial, 'Serial', factory)
    monkeypatch.setattr(module.MotorController, 'detect_port', staticmethod(lambda port: 'COM_TEST'))
    controller = module.MotorController(timeout=.4)
    assert controller.connect()
    assert factory.call_args.kwargs['write_timeout'] == .4
    assert controller.reconnect()
    assert factory.call_args.kwargs['write_timeout'] == .4
    controller._ser.write.side_effect = serial.SerialTimeoutException('write timeout')
    assert not controller.start_forward()
    assert not controller.is_connected


def test_busy_io_query_reports_timeout_without_querying_device():
    motion = MotionService()
    motion.connected = True
    motion._controller = SimpleNamespace(query_status=Mock())
    # Simulate another worker holding the serial I/O lock.
    entered, release = threading.Event(), threading.Event()

    def hold():
        with motion._io_lock:
            entered.set()
            release.wait(4)

    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match='查询等待超时'):
            motion.query_status()
        motion._controller.query_status.assert_not_called()
    finally:
        release.set()
        worker.join(1)


def test_invalid_status_and_serial_errors_are_reported_then_valid_query_recovers():
    motion = MotionService()
    motion.connected = True
    query = Mock(side_effect=[{'valid': False, 'response': ''},
                            OSError('USB disconnected'), {'valid': True, 'speed': 10}])
    motion._controller = SimpleNamespace(query_status=query)
    with pytest.raises(RuntimeError, match='没有返回有效状态'):
        motion.query_status()
    with pytest.raises(OSError, match='USB disconnected'):
        motion.query_status()
    assert motion.query_status()['speed'] == 10


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_manual_retry_reposts_but_automatic_calls_still_deduplicate(direction):
    motion = MotionService()
    motion._controller = object()
    motion._running.set()
    command = getattr(motion, direction)
    command()
    command()
    assert motion._queue.qsize() == 1
    command(force=True)
    assert motion._queue.qsize() == 2
    motion.stop()
    assert motion._queue.get_nowait()[1] == ('stop',)
    assert motion._queue.empty()


def test_manual_buttons_force_retry_and_query_has_immediate_feedback():
    from PySide6.QtWidgets import QApplication
    from app.core.config import AppConfig
    from app.ui.widgets.manual_motor import ManualMotor
    app = QApplication.instance() or QApplication([])
    motion = SimpleNamespace(connected=True, control_lock=threading.RLock(),
                             forward=Mock(), reverse=Mock(), query_status=Mock())
    ctx = SimpleNamespace(motion=motion, config=AppConfig(), log=Mock(),
                          run_async=Mock(), stop_motion=Mock())
    panel = ManualMotor(ctx)
    try:
        panel.command('forward')
        panel.command('forward')
        assert motion.forward.call_count == 2
        motion.forward.assert_called_with(force=True)
        panel.query()
        ctx.log.assert_called_with('正在查询电机状态…')
        error_callback = ctx.run_async.call_args.kwargs['on_error']
        error_callback('设备无应答')
        assert '设备无应答' in ctx.log.call_args.args[0]
    finally:
        panel.close()
        app.processEvents()
