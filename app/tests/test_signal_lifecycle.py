"""Driver release must finish before reconnecting or terminating the process."""
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from app.engine.hardware.signal_device import Device
from app.engine.hardware.signal_micrometer import SignalMicrometerReader
from app.services.reading import ReadingService


class SignalLifecycleTests(unittest.TestCase):
    def test_process_exit_finishes_pending_release(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "released"
            script = """
import sys, time
from pathlib import Path
from app.services.reading import ReadingService
class Reader:
    def close(self):
        time.sleep(.2)
        Path(sys.argv[1]).write_text('released')
service = ReadingService()
service._reader = Reader()
service.stop()
"""
            result = subprocess.run([sys.executable, "-c", script, str(marker)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "released")

    def test_shutdown_waits_for_cancelled_connection(self):
        entered, release, closed, finished = [threading.Event() for _ in range(4)]
        reader = Mock()
        def connect():
            entered.set()
            release.wait(3)
        reader.connect.side_effect = connect
        reader.close.side_effect = closed.set
        service = ReadingService()
        with patch('app.engine.hardware.signal_micrometer.SignalMicrometerReader', return_value=reader):
            service.start_async(source='signal')
            self.assertTrue(entered.wait(2))
            def shutdown():
                service.shutdown()
                finished.set()
            worker = threading.Thread(target=shutdown)
            worker.start()
            try:
                self.assertFalse(finished.wait(.1))
            finally:
                release.set()
                worker.join(3)
        self.assertTrue(finished.is_set())
        self.assertTrue(closed.is_set())
        reader.start.assert_not_called()

    def test_reconnect_waits_for_previous_driver_release(self):
        entered, release, closed = [threading.Event() for _ in range(3)]
        previous, replacement = Mock(), Mock()
        def close():
            entered.set()
            release.wait(3)
            closed.set()
        previous.close.side_effect = close
        replacement.connect.side_effect = lambda: self.assertTrue(closed.is_set())
        service = ReadingService()
        service._reader = previous
        service.stop()
        self.assertTrue(entered.wait(2))
        with patch('app.engine.hardware.signal_micrometer.SignalMicrometerReader', return_value=replacement):
            try:
                service.start_async(source='signal')
                replacement.connect.assert_not_called()
            finally:
                release.set()
                # Wait for the startup thread before asking it to stop again.
                with service._lock:
                    threads = tuple(service._connect_threads)
                for thread in threads:
                    thread.join(3)
                service.shutdown()
        replacement.connect.assert_called_once()
        replacement.close.assert_called_once()

    def test_close_releases_handles_even_when_filter_reset_fails(self):
        device = Device.__new__(Device)
        device.k = Mock()
        device.handle, device.event = 123, 456
        device.ioctl = Mock(side_effect=OSError('disconnected'))
        with self.assertRaises(OSError):
            device.close()
        self.assertEqual([call.args[0] for call in device.k.CloseHandle.call_args_list],
                         [123, 456])
        device.close()  # idempotent after an error
        self.assertEqual(device.k.CloseHandle.call_count, 2)

    def test_thread_start_failure_still_closes_device(self):
        device = Mock()
        reader = SignalMicrometerReader(device_factory=lambda: device)
        with patch.object(threading.Thread, 'start', side_effect=RuntimeError('no thread')):
            with self.assertRaises(RuntimeError):
                reader.start(Mock())
        reader.close()
        device.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
