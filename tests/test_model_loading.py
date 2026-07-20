import threading
import unittest
from concurrent.futures import Future

from src.ui.app import YoloCamApp


class _RootStub:
    def __init__(self):
        self.calls = []

    def after(self, delay, callback):
        self.calls.append((threading.get_ident(), delay, callback))
        return "after-job"


class _LogStub:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


class ModelLoadingTests(unittest.TestCase):
    def _app(self, future):
        app = object.__new__(YoloCamApp)
        app.root = _RootStub()
        app.log = _LogStub()
        app._closing = False
        app._model_load_future = future
        app._model_load_job = "previous-job"
        app.status_lines = []
        app._set_status = app.status_lines.append
        return app

    def test_pending_load_is_polled_by_current_thread(self):
        app = self._app(Future())
        current_thread = threading.get_ident()

        app._poll_model_load()

        self.assertEqual(app.root.calls[0][0], current_thread)
        self.assertEqual(app.root.calls[0][1], 50)
        self.assertEqual(app._model_load_job, "after-job")

    def test_finished_load_updates_ui_from_poller(self):
        future = Future()
        future.set_result(True)
        app = self._app(future)

        app._poll_model_load()

        self.assertIsNone(app._model_load_future)
        self.assertEqual(app.status_lines, ["YOLO 模型已加载"])
        self.assertEqual(app.log.lines, ["YOLO 模型已加载"])
        self.assertEqual(app.root.calls, [])


if __name__ == "__main__":
    unittest.main()
