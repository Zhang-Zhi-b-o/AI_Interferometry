"""Inference workers are reused and live frames never accumulate in a queue."""
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from app.core.tasks import Background  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


class InferenceSchedulingTest(unittest.TestCase):
    def test_serial_worker_reused_and_survives_failure(self):
        tasks = Background()
        try:
            ids = []
            tasks.submit(lambda: ids.append(threading.get_ident()), serial=True).result(5)
            errors = []
            tasks.submit(lambda: 1 / 0, serial=True, on_error=errors.append).result(5)
            tasks.submit(lambda: ids.append(threading.get_ident()), serial=True).result(5)
            self.assertEqual(ids[0], ids[1])
            self.assertEqual(len(errors), 1)
        finally:
            tasks.shutdown()

    def test_live_busy_guard_and_duplicate_timestamp(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = Mock()
        camera.latest_sample.return_value = (frame, 10.)
        window = SimpleNamespace(
            search=SimpleNamespace(active=False), backlash=SimpleNamespace(active=False),
            state=Mock(detecting=True), detect=SimpleNamespace(is_loaded=True),
            _detect_busy=False, _current_frame=frame, camera=camera,
            _vision_epoch=0, _live_observe=Mock(return_value={}), ctx=Mock(),
        )
        MainWindow._pump_detection(window)
        self.assertTrue(window._detect_busy)
        self.assertTrue(window.ctx.run_async.call_args.kwargs["serial"])
        MainWindow._pump_detection(window)
        self.assertEqual(window.ctx.run_async.call_count, 1)
        window.ctx.run_async.call_args.kwargs["on_done"]({})
        MainWindow._pump_detection(window)
        self.assertFalse(window._detect_busy)
        self.assertEqual(window.ctx.run_async.call_count, 1)
        camera.latest_sample.return_value = (frame, 11.)
        MainWindow._pump_detection(window)
        self.assertEqual(window.ctx.run_async.call_count, 2)


if __name__ == "__main__":
    unittest.main()
