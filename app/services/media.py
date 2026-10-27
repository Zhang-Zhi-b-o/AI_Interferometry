"""图像与视频记录；编码在后台执行，采集线程只投递有界队列。"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time


def save_image(path, frame):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", frame)
    if not ok:
        raise RuntimeError("图像编码失败")
    encoded.tofile(path)
    return path


class VideoRecorder:
    def __init__(self, *, on_status=None):
        self._status = on_status or (lambda text: None)
        self._lock = threading.RLock()
        self._queue = None
        self._thread = None
        self.active = False
        self.source = "corrected"
        self.frames = self.dropped = 0
        self.path = None
        self._stop_event = threading.Event()
        self._last_submitted_at = None
        self._interval = 0.0
        self.sampled_out = 0

    def start(self, path, *, fps=15, source="corrected", metadata=None):
        import math

        if not math.isfinite(float(fps)) or not 0 < float(fps) <= 120:
            return False, "录制帧率须在 0–120 之间"
        if source not in {"raw", "corrected", "annotated", "reading"}:
            return False, "无效录制来源"
        with self._lock:
            if self.active or self._thread is not None and self._thread.is_alive():
                return False, "录制正在运行或收尾，请稍候"
            self.path = Path(path)
            if self.path.suffix.lower() not in {".mp4", ".avi"}:
                return False, "录制文件须为 mp4 或 avi"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.frames = self.dropped = 0
            self.sampled_out = 0
            self._last_submitted_at = None
            self._interval = 1.0 / float(fps)
            self._stop_event = threading.Event()
            self.source = source
            self._queue = messages = queue.Queue(maxsize=4)
            self.active = True
            self._thread = threading.Thread(target=self._run, args=(messages, float(fps), dict(metadata or {})),
                                            name="video-encoder", daemon=True)
            self._thread.start()
        return True, "录制已开始；停止后保存视频与来源信息"

    def submit(self, frame, *, source, overlay=None):
        with self._lock:
            if not self.active or source != self.source or frame is None:
                return
            now = time.monotonic()
            if self._last_submitted_at is not None and now-self._last_submitted_at < self._interval:
                self.sampled_out += 1
                return
            try:
                self._queue.put_nowait((frame.copy(), dict(overlay or {})))
                self._last_submitted_at = now
            except queue.Full:
                self.dropped += 1

    def stop(self, *, wait=True):
        with self._lock:
            self.active = False
            thread = self._thread
            self._stop_event.set()
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(5)

    def _run(self, messages, fps, metadata):
        import cv2

        writer, size, error = None, None, None
        try:
            while True:
                try:
                    item = messages.get(timeout=0.05)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                frame, overlay = item
                if self.source == "annotated":
                    from app.engine.vision.annotations import draw_detections

                    draw_detections(frame, overlay.get("boxes", []),
                                    overlay.get("class_names", []), overlay.get("confs", []))
                    center = overlay.get("center_x")
                    if center is not None:
                        cv2.line(frame, (int(center), 0), (int(center), frame.shape[0]-1), (0, 255, 0), 2)
                if writer is None:
                    size = frame.shape[1::-1]
                    codec = "XVID" if self.path.suffix.lower() == ".avi" else "mp4v"
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), fps, size)
                    if not writer.isOpened():
                        raise RuntimeError("无法创建录像，请检查编码器和输出路径")
                if frame.shape[1::-1] != size:
                    frame = cv2.resize(frame, size)
                writer.write(frame)
                self.frames += 1
        except Exception as exc:
            error = str(exc)
            self._status("录制失败：" + error)
        finally:
            if writer is not None:
                writer.release()
            with self._lock:
                self.active = False
            report = dict(metadata, source=self.source, fps=fps, frames=self.frames,
                          dropped_frames=self.dropped, size=size, error=error,
                          sampled_out_frames=self.sampled_out,
                          finished_at=time.time(), timing="固定播放帧率；不是相机原始时间戳")
            try:
                self.path.with_suffix(self.path.suffix + ".json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                self._status(f"录制结束：{self.frames} 帧，丢弃 {self.dropped} 帧；{self.path}")
            except OSError as exc:
                self._status(f"录像元数据保存失败：{exc}")
