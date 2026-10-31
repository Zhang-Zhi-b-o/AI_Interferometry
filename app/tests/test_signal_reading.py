"""Signal-to-reading service, unit semantics, stop/release, and Qt history integration."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from app.engine.hardware.signal_micrometer import SignalMicrometerReader
from app.services.reading import ReadingService


class Device:
    hardware_id = "HID\\VID_C216&PID_010B"

    def __init__(self):
        self.keys = queue.Queue()
        self.closed = threading.Event()

    def open(self):
        pass

    def read(self, timeout):
        try:
            return self.keys.get(timeout=timeout / 1000)
        except queue.Empty:
            return None

    def close(self):
        self.closed.set()

    def send(self):
        for code in [28, 12, 4, 10, 52, 4, 7, 7, 28]:
            self.keys.put((code, 0))
            self.keys.put((code, 1))


class SignalReadingTests(unittest.TestCase):
    def test_main_window_discards_queued_previous_source_after_switch(self):
        from app.ui.main_window import MainWindow
        window = NS(reading=NS(generation=3), media=NS(submit=Mock()),
                    state=NS(update=Mock()), _pages={})
        MainWindow._on_reading(window, dict(value_mm=1, reading_generation=2))
        window.state.update.assert_not_called()
        MainWindow._on_reading(window, dict(value_mm=2, at=123, reading_generation=3))
        window.state.update.assert_called_once_with(reading_mm=2, reading_at=123)

    def test_signal_replaces_camera_and_keeps_receipt_time_and_releases(self):
        for unit, expected in [("mm", -39.366), ("um", -.039366), ("inch", -999.8964)]:
            with self.subTest(unit=unit):
                device = Device()
                received, payloads = threading.Event(), []
                def deliver(payload):
                    payloads.append(payload)
                    received.set()
                service = ReadingService(on_reading=deliver)
                def factory(**settings):
                    return SignalMicrometerReader(**settings, device_factory=lambda: device)
                with patch('app.engine.hardware.signal_micrometer.SignalMicrometerReader', side_effect=factory):
                    ok, message = service.start(source="signal", signal_unit=unit)
                self.assertTrue(ok, message)
                before = time.time()
                device.send()
                self.assertTrue(received.wait(2))
                value, at = service.latest_stable()
                self.assertAlmostEqual(value, expected)
                self.assertGreaterEqual(at, before)
                self.assertEqual(payloads[-1]['source'], 'signal')
                self.assertEqual(payloads[-1]['timestamp_kind'], 'received')
                self.assertIsNone(payloads[-1]['frame'])
                time.sleep(.02)
                self.assertEqual(service.latest_stable()[1], at)
                service.stop()
                self.assertTrue(device.closed.wait(2))
                self.assertEqual(service.latest(), (None, 0))

    def test_queued_start_can_be_cancelled_without_opening_device(self):
        service = ReadingService()
        with patch('app.services.reading.threading.Thread'):
            service.start_async(source='signal')
        generation = service._generation
        service.stop()
        with patch('app.engine.hardware.signal_micrometer.SignalMicrometerReader') as reader:
            ok, _ = service._start(generation, source='signal')
        self.assertFalse(ok)
        reader.assert_not_called()

    def test_page_button_and_auto_history_use_signal_as_micrometer_value(self):
        from PySide6.QtWidgets import QApplication
        from app.core.config import AppConfig
        from app.core.state import AppState
        from app.core.timeline import Timeline
        from app.ui.pages.reading import ReadingPage
        app = QApplication.instance() or QApplication([])
        service = NS(connected=False, connecting=False, source='camera', start_async=Mock(), stop=Mock(), stable=True)
        ctx = NS(config=AppConfig(), state=AppState(), reading=service, log=Mock(), history=[], timeline=Timeline())
        page = ReadingPage(ctx)
        try:
            page.signal_button.click()
            service.start_async.assert_called_once_with(source='signal', signal_unit='mm')
            now = time.time()
            payload = dict(source='signal', value_mm=-39.366, at=now, stable=True, text='-39.366',
                           received_monotonic=time.monotonic(), signal_unit='mm', hardware_id=Device.hardware_id)
            page.on_reading(payload)
            self.assertEqual(page.table.rowCount(), 1)
            self.assertEqual(page.table.item(0, 1).text(), '-39.366')
            self.assertEqual(ctx.history[-1]['value_mm'], -39.366)
            self.assertEqual(ctx.timeline.snapshot()[-1]['result']['captured_at'], now)
            self.assertEqual(ctx.timeline.snapshot()[-1]['result']['source'], 'signal')
            service.connected, service.source = True, 'signal'
            page._refresh_signal()
            self.assertIn('传输延迟', page.signal_status.text())
            page.signal_button.click()
            service.stop.assert_called_once()
            page.signal_auto_record.setChecked(False)
            page.on_reading(payload)
            self.assertEqual(page.table.rowCount(), 1)
        finally:
            page.close()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
