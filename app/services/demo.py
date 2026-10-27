"""纯内存演示设备，不继承真实硬件服务，也不打开相机、串口或网络。"""
from __future__ import annotations

import threading
import time

import numpy as np

from app.core.config import ROOT
from app.services.fringe import FringeMetrics
from app.services.record import RecordService
from app.services.thickness_map import ThicknessMapService


class DemoRig:
    width, height = 640, 180

    def __init__(self, *, on_frame=None, on_reading=None):
        self.lock = threading.RLock()
        self.reading_mm = 0.98
        self._at = time.monotonic()
        self.on_frame, self.on_reading = on_frame, on_reading
        self._closed = threading.Event()
        self._thread = None
        self.motion = DemoMotion(self)
        self.camera = DemoCamera(self)
        self.reading = DemoReading(self)
        self.detect = DemoDetector(self)
        self.fringe = DemoFringe(self)

    def sample(self):
        with self.lock:
            now = time.monotonic()
            direction = {"forward": 1, "reverse": -1}.get(self.motion.direction, 0)
            if self.motion.connected:
                self.reading_mm += direction * 0.006 * (11 - self.motion.gear) * (now - self._at)
                self.reading_mm = float(np.clip(self.reading_mm, 0.9, 1.1))
            self._at = now
            center = self.width / 2 + (self.reading_mm - 1.0) * 8000
            return self.reading_mm, center

    def image(self):
        _, center = self.sample()
        x = np.arange(self.width, dtype=float)
        phase = (x - center) * (2 * np.pi / 22)
        envelope = np.exp(-((x - center) / 130) ** 2)
        row = np.stack([128 + 100 * envelope * np.cos(phase + offset)
                        for offset in (-0.9, 0, 0.9)], axis=1)
        return np.broadcast_to(np.clip(row, 0, 255).astype(np.uint8),
                               (self.height, self.width, 3)).copy()

    def observe(self, frame):
        _, center = self.sample()
        visible = 0 <= center < self.width
        if self.detect.on_detection:
            self.detect.on_detection(self.detect.detect(frame))
        return {"center_x": center if visible else None,
                "confidence": 0.95 if visible else 0.0,
                "zero_box_x": center if visible else None,
                "zero_box_confidence": 0.95 if visible else 0.0}

    @staticmethod
    def reading_image(value):
        import cv2

        frame = np.full((160, 400, 3), 225, dtype=np.uint8)
        cv2.putText(frame, "SIMULATED MICROMETER", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 70, 70), 1)
        cv2.putText(frame, f"{value:.3f} mm", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 40, 20), 3)
        return frame

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="demo-rig", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._closed.is_set():
            if self.camera.is_open and self.on_frame:
                self.on_frame(self.camera.latest_frame())
            if self.reading.connected and self.on_reading:
                value, at = self.reading.latest()
                self.on_reading({"value_mm": value, "at": at, "stable": True,
                                 "frame": self.reading_image(value),
                                 "message": "演示微分表", "data_source": "demo"})
            self._closed.wait(0.08)

    def close(self):
        self._closed.set()
        self.motion.disconnect()
        self.camera.close()
        self.reading.stop()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(1)


class DemoMotion:
    def __init__(self, rig):
        self.rig = rig
        self.control_lock = rig.lock
        self.connected = True
        self.direction = "stopped"
        self.gear = 10

    def connect(self, *args, cancel_event=None, **kwargs):
        with self.control_lock:
            if self.rig._closed.is_set() or (cancel_event is not None and cancel_event.is_set()):
                return False, "连接已取消"
            self.connected = True
            return True, "演示电机已连接（不使用串口）"

    def disconnect(self):
        with self.control_lock:
            self.stop()
            self.connected = False

    def begin_disconnect(self):
        self.disconnect()

    def set_speed(self, gear):
        with self.control_lock:
            if not 1 <= int(gear) <= 10:
                raise ValueError("演示档位范围为 1–10")
            self.rig.sample()
            self.gear = int(gear)

    def forward(self):
        self._move("forward")

    def reverse(self):
        self._move("reverse")

    def _move(self, direction):
        with self.control_lock:
            self.rig.sample()
            self.direction = direction if self.connected else "stopped"

    def stop(self):
        self._move("stopped")

    def query_status(self):
        return {"direction": self.direction, "gear": self.gear, "data_source": "demo"}

    def toggle_direction(self):
        self._move("reverse" if self.direction == "forward" else "forward")

    def speed_up(self):
        self.set_speed(max(1, self.gear - 1))

    def speed_down(self):
        self.set_speed(min(10, self.gear + 1))

    @staticmethod
    def available_ports():
        return []


class DemoCamera:
    def __init__(self, rig):
        self.rig = rig
        self.is_open = True
        self.fps = 12.5
        self.rotation = 0.0

    @property
    def frame_age(self):
        return 0.0 if self.is_open else float("inf")

    def open(self, *args, **kwargs):
        self.is_open = True
        return True, "演示相机已打开（合成条纹）"

    def close(self):
        self.is_open = False

    def latest_sample(self):
        return self.latest_frame(), time.monotonic()

    def latest_frame(self):
        if not self.is_open:
            return None
        frame = self.rig.image()
        if self.rotation:
            from app.engine.vision.angle import rotate_expand

            frame = rotate_expand(frame, self.rotation)
        return frame

    def raw_frame(self):
        return self.rig.image() if self.is_open else None

    def configure_transform(self, angle):
        self.rotation = float(angle)

    def configure_motion_clarity(self, **settings):
        pass


class DemoReading:
    def __init__(self, rig):
        self.rig = rig
        self.connected = True
        self.stable = True

    @property
    def value_mm(self):
        return self.latest()[0]

    def latest(self):
        return (round(self.rig.sample()[0], 4), time.time()) if self.connected else (None, 0)

    def start(self, **kwargs):
        self.connected = True
        return True, "演示微分表已启动"

    def stop(self):
        self.connected = False


class DemoDetector:
    def __init__(self, rig):
        self.rig = rig
        self.is_loaded = True
        self.on_detection = None

    def load(self, *args, **kwargs):
        self.is_loaded = True
        return True, "演示检测已就绪（无需权重，不代表模型识别效果）"

    def reset(self):
        pass

    def detect(self, frame, roi=None):
        _, center = self.rig.sample()
        boxes = [[max(0, center - 40), 15, min(self.rig.width, center + 40), 165]]
        if not 0 <= center < self.rig.width:
            boxes = []
        return {"boxes_xyxy": np.array(boxes).reshape(-1, 4),
                "confs": np.array([0.95] if boxes else []),
                "class_names": ["zero_order"] if boxes else [],
                "error": None, "data_source": "demo"}

    @staticmethod
    def describe(result):
        return {"count": len(result["boxes_xyxy"]), "names": result["class_names"],
                "error": result.get("error")}


class DemoFringe:
    def __init__(self, rig):
        self.rig = rig

    def measure(self, frame, center_x=None):
        _, center = self.rig.sample()
        return FringeMetrics(center_x=round(center, 2), confidence=0.95,
                             width_px=11, spacing_px=22, period_px=22, tilt_deg=0,
                             num_bands=12, band_kind="bright")


class DemoRecord(RecordService):
    def __init__(self, out_dir=None):
        super().__init__(out_dir or ROOT / "exports" / "demo")

    def export_rows(self, rows, *args, **kwargs):
        return super().export_rows([dict(row, data_source="demo") for row in rows],
                                   *args, **kwargs)

    def export_json(self, payload, *args, **kwargs):
        return super().export_json(dict(payload, data_source="demo"), *args, **kwargs)

    def export_report(self, *, title, **kwargs):
        return super().export_report(title="演示数据 · " + title, **kwargs)


class DemoThickness(ThicknessMapService):
    def analyze(self, *args, **kwargs):
        result = super().analyze(*args, **kwargs)
        result["metrics"]["data_source"] = "demo"
        result["mode_label"] = "演示数据 · " + result["mode_label"]
        return result

    @staticmethod
    def save(result, out_dir=None):
        import cv2

        marked = result["overlay"].copy()
        cv2.putText(marked, "DEMO / SIMULATED", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return ThicknessMapService.save(dict(result, overlay=marked),
                                       out_dir or ROOT / "exports" / "demo")
