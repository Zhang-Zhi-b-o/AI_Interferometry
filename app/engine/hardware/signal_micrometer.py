"""HID micrometer adapter using the same parser/driver as the validated standalone tool."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from app.engine.hardware.signal_protocol import Frames
from app.engine.hardware.signal_device import Device


class SignalMicrometerReader:
    def __init__(self, *, unit="mm", on_error=None, device_factory=Device):
        self.scale_mm = {"mm": 1.0, "um": .001, "inch": 25.4}[unit]
        self.unit = unit
        self.on_error = on_error
        self.device = device_factory()
        self._stop = threading.Event()
        self._thread = None

    def connect(self):
        self.device.open()
        return True

    def start(self, callback):
        self._thread = threading.Thread(target=self._run, args=(callback,),
                                        name="micrometer-signal", daemon=True)
        self._thread.start()

    def _run(self, callback):
        frames = Frames()
        try:
            while not self._stop.is_set():
                key = self.device.read(100)
                if key is None:
                    continue
                value = frames.feed(*key)
                if value is None:
                    continue
                received_at = time.time()
                value_mm = float(value) * self.scale_mm
                callback(SimpleNamespace(
                    value_mm=value_mm, stable_value_mm=value_mm, stable=True,
                    captured_at=received_at, stable_captured_at=received_at,
                    received_monotonic=time.monotonic(), source="signal",
                    text=str(value), signal_unit=self.unit,
                    hardware_id=self.device.hardware_id,
                    message="信号读数（有传输延迟，时间为电脑接收时间）"))
        except Exception as exc:
            if self.on_error:
                self.on_error(str(exc))
        finally:
            self.device.close()

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(2)
        if self._thread is None:
            self.device.close()
