"""旧 CameraManager 的无 GUI 适配：共享所有权、清晰度和实际采集时间。"""
from __future__ import annotations

import math
import threading
import time


class CameraService:
    def __init__(self, *, on_frame=None, on_status=None, on_error=None, camera_factory=None):
        self._on_frame, self._on_status, self._on_error = on_frame, on_status, on_error
        self._factory = camera_factory
        self._camera = self._thread = None
        self._running = threading.Event()
        self._lock = threading.RLock()
        self._frame = self._raw_frame = None
        self._frame_at = self._fps = 0.0
        self._epoch = 0
        self._transform_epoch = 0
        self.index = None
        self.rotation = 0.0
        self.clarity_config = {}
        self._clarity_roi = None

    @property
    def is_open(self):
        return self._camera is not None and self._camera.is_opened

    @property
    def fps(self):
        return self._fps

    @property
    def frame_age(self):
        with self._lock:
            return time.monotonic() - self._frame_at if self._frame_at else float("inf")

    def latest_sample(self):
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), self._frame_at)

    def latest_frame(self):
        return self.latest_sample()[0]

    def raw_frame(self):
        with self._lock:
            return None if self._raw_frame is None else self._raw_frame.copy()

    @staticmethod
    def scan():
        from app.engine.camera import CameraManager

        return CameraManager.detect_all()

    def configure_transform(self, angle):
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("旋转角度必须为有限数")
        with self._lock:
            self.rotation = angle
            self._transform_epoch += 1
            self._frame = self._raw_frame = None
            self._frame_at = 0.0

    def configure_motion_clarity(self, *, exposure, gain, enabled):
        with self._lock:
            self.clarity_config.update(motion_exposure=float(exposure),
                                       motion_gain=float(gain), enabled=bool(enabled))
            camera = self._camera
            if camera is not None:
                # enabled 是当前用户选择，同时更新旧驱动配置中的总开关。
                camera.clarity_config["enabled"] = bool(enabled)
                camera.configure_motion_clarity(exposure=exposure, gain=gain, enabled=enabled)

    def set_clarity_region(self, roi=None):
        with self._lock:
            self._clarity_roi = tuple(roi) if roi is not None else None
            if self._camera is not None and hasattr(self._camera, "set_clarity_region"):
                self._camera.set_clarity_region(self._clarity_roi, self.rotation)

    @property
    def clarity_status(self):
        with self._lock:
            if self._camera is None:
                return {}
            status = self._camera.clarity_status
            return status() if callable(status) else dict(status)

    def open(self, index=0, resolution=(1280, 1024), fps=30, *, cancel_event=None, auto_mode=False):
        self.close()
        with self._lock:
            epoch = self._epoch
        if cancel_event is not None and cancel_event.is_set():
            return False, "相机连接已取消"
        if self._factory is None:
            from app.engine.camera import CameraManager

            factory = CameraManager
        else:
            factory = self._factory
        options = {"auto_mode": True} if auto_mode else {}
        camera = factory(index=int(index), resolution=tuple(resolution), fps=int(fps),
                         clarity_config=dict(self.clarity_config), owner="interferometer-camera",
                         **options)
        try:
            if not camera.start():
                camera.stop()
                detail = getattr(camera, "last_error", "") or "请检查占用和连接"
                return False, f"相机 {index} 打开失败（{detail}）"
            if self.clarity_config:
                camera.configure_motion_clarity(
                    exposure=self.clarity_config.get("motion_exposure", -6),
                    gain=self.clarity_config.get("motion_gain", 0),
                    enabled=self.clarity_config.get("enabled", False))
            with self._lock:
                cancelled = epoch != self._epoch or (cancel_event is not None and cancel_event.is_set())
                if not cancelled:
                    self._camera = camera
                    if hasattr(camera, "set_clarity_region"):
                        camera.set_clarity_region(self._clarity_roi, self.rotation)
                    self.index = int(index)
                    self._running = running = threading.Event()
                    running.set()
                    self._thread = threading.Thread(target=self._loop, args=(camera, running),
                                                    name="camera-publish", daemon=True)
                    self._thread.start()
            if cancelled:
                camera.stop()
                return False, "相机连接已取消"
        except Exception as exc:
            camera.stop()
            return False, f"相机打开异常：{exc}"
        message = f"相机 {index} 已打开"
        size = getattr(camera, "actual_resolution", None)
        if size:
            message += f" · 驱动回读 {size[0]}×{size[1]} @ {camera.actual_fps:.1f} fps"
        note = getattr(camera, "mode_note", "")
        if note:
            message += f"（{note}）"
        self._status(message)
        return True, message

    def close(self):
        with self._lock:
            self._epoch += 1
            self._running.clear()
            camera, self._camera = self._camera, None
            thread, self._thread = self._thread, None
            self._frame = self._raw_frame = None
            self._frame_at = self._fps = 0.0
        if thread is not None and thread is not threading.current_thread():
            thread.join(0.3)
        if camera is not None:
            camera.stop()

    def _loop(self, camera, running):
        counted, last_at, report_at = 0, 0.0, time.monotonic()
        while running.is_set():
            frame, at = camera.latest_sample()
            if frame is None or at <= last_at:
                time.sleep(0.005)
                continue
            last_at = at
            with self._lock:
                angle = self.rotation
                transform_epoch = self._transform_epoch
            if angle:
                from app.engine.vision.angle import rotate_expand

                corrected = rotate_expand(frame, angle)
            else:
                corrected = frame
            with self._lock:
                if not running.is_set() or camera is not self._camera:
                    break
                if transform_epoch != self._transform_epoch:
                    continue
                self._raw_frame, self._frame, self._frame_at = frame, corrected, at
            counted += 1
            now = time.monotonic()
            if now - report_at >= 1:
                self._fps = counted / (now - report_at)
                counted, report_at = 0, now
            if self._on_frame is not None:
                self._on_frame(corrected)

    def _status(self, message):
        if self._on_status is not None:
            self._on_status(message)
