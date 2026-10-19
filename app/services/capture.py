"""相机采集：cv2.VideoCapture + 后台取帧线程。

为什么不用旧的 ``src.camera.CameraManager``：它带清晰度补偿和相机注册表，
对 v2 的主流程是多余的；这里只有 100 行，换成 CameraManager 时继承本类即可。
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class CameraService:
    def __init__(self, *, on_frame=None, on_status=None, on_error=None) -> None:
        self._on_frame = on_frame
        self._on_status = on_status
        self._on_error = on_error
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_count = 0
        self._frame_at = 0.0
        self._fps = 0.0
        self.index: int | None = None

    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return bool(self._capture is not None and self._capture.isOpened())

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_age(self) -> float:
        with self._lock:
            return time.monotonic() - self._frame_at if self._frame_at else float("inf")

    def latest_sample(self):
        """原子读取图像与其采集时间，供闭环检查实际参与推理帧的时效。"""
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), self._frame_at)

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ------------------------------------------------------------------
    def open(self, index: int = 0, resolution: tuple[int, int] = (1280, 1024),
             fps: int = 30) -> tuple[bool, str]:
        self.close()
        backend = getattr(cv2, "CAP_DSHOW", 0) if hasattr(cv2, "CAP_DSHOW") else 0
        capture = None
        for candidate in (index, index, 0):  # 有些相机第一次打开失败，重试一次
            capture = cv2.VideoCapture(candidate, backend) if backend else cv2.VideoCapture(candidate)
            if capture is not None and capture.isOpened():
                index = candidate
                break
            if capture is not None:
                capture.release()
            capture = None
        if capture is None:
            self._status(f"相机 {index} 打开失败")
            return False, f"相机 {index} 打开失败"

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolution[0]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolution[1]))
        capture.set(cv2.CAP_PROP_FPS, float(fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self.index = index
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._status(f"相机 {index} 已打开 {size[0]}×{size[1]}")
        return True, f"相机 {index} 已打开"

    def close(self) -> None:
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.2)
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        with self._lock:
            self._frame = None
            self._frame_at = 0.0
        self._status("相机已关闭")

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        last_report = time.monotonic()
        counted = 0
        while self._running.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._frame_at = time.monotonic()
            counted += 1
            now = time.monotonic()
            if now - last_report >= 1.0:
                self._fps = counted / (now - last_report)
                counted = 0
                last_report = now
            self._frame_count += 1
            if self._on_frame is not None:
                try:
                    self._on_frame(frame)
                except Exception:
                    pass

    def _status(self, text: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                pass
