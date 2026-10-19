"""微分表读数：包装 ``src.hardware.micrometer.MicrometerReader``（第二相机 + PP-OCR）。

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
        self._lock = threading.Lock()
        self.captured_at = 0.0
        self.value_mm: float | None = None
        self.stable: bool = False
        self.connected: bool = False

    # ------------------------------------------------------------------
    def start(self, *, camera_index: int = 0, interval_ms: int = 200,
              resolution: tuple[int, int] = (1280, 1024), fps: int = 15) -> tuple[bool, str]:
        self.stop()
        try:
            from src.hardware.micrometer import MicrometerReader
        except Exception as exc:
            return False, f"无法导入读数模块：{exc}"
        try:
            reader = MicrometerReader(
                camera_index=int(camera_index),
                resolution=resolution,
                fps=int(fps),
                interval_ms=int(interval_ms),
            )
            if not reader.connect():
                return False, f"微分表相机 {camera_index} 连接失败"
            reader.start(self._handle)
        except Exception as exc:
            return False, f"读数启动异常：{exc}"
        self._reader = reader
        self.connected = True
        self._status("读数已启动")
        return True, "读数已启动"

    def latest(self):
        with self._lock:
            return self.value_mm, self.captured_at

    def stop(self) -> None:
        reader, self._reader = self._reader, None
        self.connected = False
        with self._lock:
            self.value_mm, self.captured_at, self.stable = None, 0.0, False
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
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
        payload = {
            "value_mm": value,
            "stable": stable,
            "reading_held": held,
            "text": str(getattr(result, "text", "") or ""),
            "message": str(getattr(result, "message", "") or ""),
            "at": captured_at,
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
