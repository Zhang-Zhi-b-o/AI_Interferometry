"""微分表读数：相机 OCR 与微分头信号共用一个读数服务。

一次读数包含：原始识别值、稳定值、是否稳定、识别文本、时间戳。
界面只消费 ``on_reading(dict)`` 回调。
"""
from __future__ import annotations

import threading
import time
from typing import Any


class ReadingService:
    def __init__(self, *, on_reading=None, on_status=None) -> None:
        self._on_reading = on_reading
        self._on_status = on_status
        self._reader: Any = None
        self._lock = threading.RLock()
        self._generation = 0
        self.connecting = False
        self.captured_at = 0.0
        self.value_mm: float | None = None
        self._stable_value_mm: float | None = None
        self._stable_at = 0.0
        self.stable: bool = False
        self.connected: bool = False
        self.source = "camera"
        self._closing = threading.Event()
        self._closing.set()

    # ------------------------------------------------------------------
    @property
    def generation(self) -> int:
        return self._generation

    def _begin(self) -> int:
        with self._lock:
            self.stop()
            self.connecting = True
            return self._generation

    def start_async(self, **settings) -> None:
        """Reserve the connection before scheduling, so an immediate stop cancels it."""
        generation = self._begin()
        self.source = settings.get("source", "camera")
        threading.Thread(target=self._start, args=(generation,), kwargs=settings,
                         name="reading-connect", daemon=True).start()

    def start(self, **settings) -> tuple[bool, str]:
        generation = self._begin()
        self.source = settings.get("source", "camera")
        return self._start(generation, **settings)

    def _start(self, generation: int, *, camera_index: int = 0,
               interval_ms: int = 200, resolution=(1280, 1024), fps: int = 15,
               auto_roi: bool = True, roi=(0.0, 0.0, 1.0, 1.0),
               model_path=None, min_score: float = 0.45, decimal_places: int = 3,
               stable_window: int = 7, stable_required: int = 3,
               max_step_mm: float = 0.05, jump_required: int = 6,
               scale_ratio_tolerance: float = 0.03, source: str = "camera",
               signal_unit: str = "mm") -> tuple[bool, str]:
        reader = None
        try:
            self._closing.wait()
            with self._lock:
                if generation != self._generation:
                    return False, "读数启动已取消"
            if source == "signal":
                from app.engine.hardware.signal_micrometer import SignalMicrometerReader
                reader = SignalMicrometerReader(
                    unit=signal_unit, on_error=lambda message: self._signal_error(generation, message))
                reader.connect()
                with self._lock:
                    if generation != self._generation:
                        return False, "读数启动已取消"
                    self._reader = reader
                    self.source = "signal"
                    self.connected = True
                    reader.start(lambda result: self._deliver(generation, result))
                    reader = None
                    self._status("信号读取与拦截已启动，请开启微分头连续传输")
                return True, "信号读数已启动"
            if source != "camera":
                raise ValueError(f"未知读数来源：{source}")
            from pathlib import Path
            from app.core.config import ROOT
            from app.engine.hardware.micrometer import MicrometerReader
            from app.engine.vision.micrometer_ocr import MicrometerOCR

            if model_path:
                path = Path(model_path)
                model_path = str(path if path.is_absolute() else ROOT / path)
            ocr = MicrometerOCR(
                model_path=model_path, min_score=min_score, decimal_places=decimal_places,
                stable_window=stable_window, stable_required=stable_required,
                max_step_mm=max_step_mm, jump_required=jump_required,
                scale_ratio_tolerance=scale_ratio_tolerance)
            ocr.load()
            with self._lock:
                if generation != self._generation:
                    return False, "读数启动已取消"
            reader = MicrometerReader(
                camera_index=int(camera_index), resolution=tuple(resolution),
                fps=int(fps), interval_ms=int(interval_ms), auto_roi=bool(auto_roi),
                manual_roi=tuple(roi), ocr=ocr)
            if not reader.connect():
                raise RuntimeError(f"微分表相机 {camera_index} 连接失败")
            with self._lock:
                if generation != self._generation:
                    return False, "读数启动已取消"
                self._reader = reader
                self.source = "camera"
                self.connected = True
                reader.start(lambda result: self._deliver(generation, result))
                reader = None  # ownership belongs to the service
                self._status("读数已启动")
            return True, "读数已启动"
        except Exception as exc:
            message = f"读数启动失败：{exc}"
            with self._lock:
                if generation == self._generation:
                    self.connected = False
                    self._reader = None
                    self._status(message)
            return False, message
        finally:
            if reader is not None:
                self._close_reader(reader)
            with self._lock:
                if generation == self._generation:
                    self.connecting = False

    def _deliver(self, generation: int, result: Any) -> None:
        with self._lock:
            if generation == self._generation and self.connected:
                self._handle(result)

    def _signal_error(self, generation: int, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self.stop()
            self._status(f"信号读数中断：{message}")

    @staticmethod
    def _close_reader(reader) -> None:
        try:
            reader.close()
        except Exception:
            pass

    def set_roi(self, *, auto_roi: bool, roi) -> None:
        with self._lock:
            if self._reader is not None:
                self._reader.manual_roi = tuple(roi)
                self._reader.auto_roi = bool(auto_roi)
                stabilizer = getattr(getattr(self._reader, "ocr", None), "stabilizer", None)
                if stabilizer is not None:
                    stabilizer.reset()

    def latest(self):
        with self._lock:
            return self.value_mm, self.captured_at

    def latest_stable(self):
        """可信读数及其原始采集时间；单帧候选仅供界面显示。"""
        with self._lock:
            return self._stable_value_mm, self._stable_at

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            reader, self._reader = self._reader, None
            self.connected = self.connecting = False
            self.value_mm, self.captured_at, self.stable = None, 0.0, False
            self._stable_value_mm, self._stable_at = None, 0.0
        if reader is not None:
            closed = self._closing = threading.Event()
            def close():
                try:
                    self._close_reader(reader)
                finally:
                    closed.set()
            threading.Thread(target=close,
                             name="reading-close", daemon=True).start()
        self._status("读数已停止")

    # ------------------------------------------------------------------
    def _handle(self, result: Any) -> None:
        value = getattr(result, "stable_value_mm", None)
        stable = bool(getattr(result, "stable", False))
        held = bool(getattr(result, "reading_held", False))
        if value is not None:
            # 保持显示的旧稳定值必须沿用它自己的采集时间，不能用本次失败帧刷新。
            captured_at = getattr(result, "stable_captured_at", None)
            if captured_at is None and stable and not held:
                captured_at = getattr(result, "captured_at", None)
            captured_at = float(captured_at or 0.0)
        else:
            value = getattr(result, "value_mm", None)
            captured_at = float(getattr(result, "captured_at", None) or time.time())
        with self._lock:
            self.value_mm = value
            self.stable = stable
            self.captured_at = captured_at
            if getattr(result, "stable_value_mm", None) is not None and captured_at > 0:
                self._stable_value_mm = value
                self._stable_at = captured_at
        payload = {
            "frame": getattr(result, "frame", None),
            "roi_xyxy": getattr(result, "roi_xyxy", None),
            "score": float(getattr(result, "score", 0.0) or 0.0),
            "raw_value_mm": getattr(result, "value_mm", None),
            "value_mm": value,
            "stable": stable,
            "reading_held": held,
            "text": str(getattr(result, "text", "") or ""),
            "message": str(getattr(result, "message", "") or ""),
            "at": captured_at,
            "source": getattr(result, "source", "camera"),
            "timestamp_kind": "received" if getattr(result, "source", "camera") == "signal" else "captured",
            "received_monotonic": getattr(result, "received_monotonic", None),
            "hardware_id": getattr(result, "hardware_id", None),
            "signal_unit": getattr(result, "signal_unit", None),
            "reading_generation": self._generation,
        }
        if self._on_reading is not None:
            try:
                self._on_reading(payload)
            except Exception:
                pass

    def _status(self, text: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                pass
