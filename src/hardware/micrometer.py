"""由独立 USB 摄像头和 OCR 驱动的数显微分表读数器。"""
from __future__ import annotations

from dataclasses import replace
import threading
import time
from typing import Callable

from src.camera import CameraManager
from src.logging import logger
from src.vision.micrometer_ocr import MicrometerOCR, MicrometerOCRResult


class MicrometerReader:
    """在后台线程采集第二摄像头，并发布经过稳定化的 OCR 读数。"""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        resolution: tuple[int, int] = (1280, 1024),
        fps: int = 15,
        interval_ms: int = 200,
        auto_roi: bool = True,
        manual_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        ocr: MicrometerOCR | None = None,
        camera_factory=CameraManager,
    ) -> None:
        self.camera_index = max(0, int(camera_index))
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.fps = max(1, int(fps))
        self.interval_s = max(0.05, int(interval_ms) / 1000.0)
        self.auto_roi = bool(auto_roi)
        self.manual_roi = tuple(float(v) for v in manual_roi)
        self.ocr = ocr or MicrometerOCR()
        self._camera_factory = camera_factory
        self._camera: CameraManager | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callback: Callable[[MicrometerOCRResult], None] | None = None
        self._value_lock = threading.Lock()
        self._stable_value: float | None = None
        self._connected = False
        self._pending_frame = None

    def connect(self) -> bool:
        if self._connected:
            return True
        # 第二路优先使用请求参数；若 USB 带宽不足或驱动只打开却不给帧，
        # 自动降到较轻的采集档位，避免影响主干涉相机。
        profiles = [
            (self.resolution, self.fps),
            ((960, 540), min(self.fps, 10)),
            ((640, 480), min(self.fps, 10)),
        ]
        tried = set()
        for resolution, fps in profiles:
            profile = (tuple(resolution), int(fps))
            if profile in tried:
                continue
            tried.add(profile)
            camera = self._camera_factory(
                index=self.camera_index, resolution=profile[0], fps=profile[1],
                owner="micrometer-camera")
            if not camera.start():
                continue
            frame = None
            # 部分 USB 摄像头在第二路同时工作时需要数秒才能给出首帧。
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                frame = camera.read()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is not None:
                self._camera = camera
                self.resolution, self.fps = profile
                self._pending_frame = frame
                self._connected = True
                return True
            camera.stop()
        return False

    def start(self, callback: Callable[[MicrometerOCRResult], None]) -> None:
        if not self._connected or self._camera is None:
            raise RuntimeError("微分表摄像头尚未连接")
        if self._thread is not None and self._thread.is_alive():
            return
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="micrometer-ocr", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._thread = None
        if self._camera is not None:
            self._camera.stop()
        self._camera = None
        self._pending_frame = None
        self._connected = False

    def read_value_mm(self) -> float | None:
        with self._value_lock:
            return self._stable_value

    @property
    def connected(self) -> bool:
        return self._connected

    @staticmethod
    def detect_cameras() -> list[int]:
        return CameraManager.detect_all()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            captured_at = time.time()
            try:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
                else:
                    frame = self._camera.read() if self._camera is not None else None
                if frame is None:
                    result = MicrometerOCRResult(
                        message="微分表摄像头读取失败",
                        captured_at=captured_at,
                        captured_monotonic=started,
                    )
                else:
                    result = self.ocr.recognize(
                        frame,
                        auto_roi=self.auto_roi,
                        manual_roi=self.manual_roi,
                    )
                    # OCR 负责返回识别区域，采集器补充完整画面供 UI 预览。
                    # 使用副本，避免相机下一次采集覆盖正在显示的帧。
                    result = replace(
                        result,
                        frame=frame.copy(),
                        captured_at=captured_at,
                        captured_monotonic=started,
                    )
                    if result.stable and result.stable_value_mm is not None:
                        with self._value_lock:
                            self._stable_value = result.stable_value_mm
                if self._callback is not None:
                    self._callback(result)
            except Exception as exc:  # 后台采集不能让 UI 因单帧错误退出
                logger.exception("微分表 OCR 失败: %s", exc)
                if self._callback is not None:
                    self._callback(MicrometerOCRResult(
                        message=f"OCR 失败：{exc}",
                        captured_at=captured_at,
                        captured_monotonic=started,
                    ))
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.0, self.interval_s - elapsed))
