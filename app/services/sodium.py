"""钠灯双端模糊测量；独立模型、完整扫描顺序、新鲜读数、失联停车。"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from app.engine.measurement.sodium import (
    MICROMETER_TO_MIRROR_RATIO, SODIUM_DOUBLET_SEPARATION_NM,
    VISIBILITY_PERIODS, calculate_mean_wavelength_nm,
)


class SodiumSequence:
    """只对不同采集帧调用；连续稳定观测才推进，未知帧打断连续性。"""
    def __init__(self, frames=3, seconds=0.25):
        self.phase = "first_end"
        self.frames, self.seconds = frames, seconds
        self._kind, self._count, self._since = "", 0, 0.0

    def observe(self, kind, at):
        if kind != self._kind:
            self._kind, self._count, self._since = kind, 0, at
        self._count += 1
        if self._count < self.frames or at - self._since < self.seconds:
            return None
        transitions = {
            ("first_end", "end"): ("first_reading", "first"),
            ("leave_first", "clear"): ("middle", None),
            ("middle", "middle"): ("leave_middle", None),
            ("leave_middle", "clear"): ("second_end", None),
            ("second_end", "end"): ("second_reading", "second"),
        }
        item = transitions.get((self.phase, kind))
        if item is None:
            return None
        self.phase, action = item
        self._kind, self._count = "", 0
        return action

    def first_recorded(self):
        self.phase = "leave_first"
        self._kind, self._count = "", 0


class SodiumService:
    def __init__(self, *, camera, reading, motion, model_path, busy=lambda: False,
                 on_update=None, model_factory=None, on_detection=None):
        self.camera, self.reading, self.motion = camera, reading, motion
        self.model_path = Path(model_path)
        self._busy, self._on_update = busy, on_update
        self._factory = model_factory
        self._on_detection = on_detection
        self._lock = motion.control_lock
        self._thread = self._watch = None
        self._stop = threading.Event()
        self.running = False
        self.run_id = 0
        self.status = {"phase": "idle", "message": "待机", "running": False}

    @property
    def active(self):
        return self.running or bool(self._thread and self._thread.is_alive()) or bool(
            self._watch and self._watch.is_alive())

    def snapshot(self):
        with self._lock:
            return dict(self.status)

    def _reading_sample(self):
        return self.reading.latest_stable()

    @staticmethod
    def _fresh(value, at):
        return value is not None and math.isfinite(float(value)) and 0 <= time.time()-at <= 1.5

    def start(self, *, direction="forward", gear=10, threshold=0.75,
              max_seconds=300, roi=None):
        with self._lock:
            if self.active or self._busy():
                return False, "已有运动闭环运行或正在停止"
            if direction not in ("forward", "reverse") or not 1 <= int(gear) <= 10:
                return False, "请选择正转/反转及有效档位"
            if not all(math.isfinite(float(x)) for x in (threshold, max_seconds)) or not .5 <= threshold <= .99 or max_seconds <= 0:
                return False, "阈值或最长运行时间无效"
            if not self.motion.connected or not self.reading.connected:
                return False, "请先连接电机并启动微分表读数"
            if getattr(self.motion, "direction", "stopped") != "stopped":
                return False, "请先停止手动转动，再开始测量"
            if not self.camera.is_open or self.camera.frame_age > 1.5:
                return False, "请先打开干涉相机并等待新画面"
            if not self._fresh(*self._reading_sample()):
                return False, "请等待新鲜有效的微分表读数"
            if not self.model_path.is_file():
                return False, "钠灯模型文件不存在"
            self.direction, self.gear = direction, int(gear)
            self.threshold, self.max_seconds = float(threshold), float(max_seconds)
            self.roi = tuple(int(v) for v in roi) if roi else None
            self._shape = None
            self.sequence = SodiumSequence()
            self.run_id += 1
            self._stop = threading.Event()
            self.running = True
            self._started = self._last_vision = time.monotonic()
            self.status = dict(run_id=self.run_id, running=True, phase="loading",
                               message="正在加载钠灯模型，电机保持停止",
                               first_mm=None, second_mm=None, result=None,
                               direction=direction, gear=gear, threshold=threshold,
                               model_path=str(self.model_path), roi=self.roi)
            self._thread = threading.Thread(target=self._run, name="sodium-measure", daemon=True)
            self._watch = threading.Thread(target=self._watchdog, name="sodium-safety", daemon=True)
            self._thread.start()
            self._watch.start()
        self._emit()
        return True, "已开始识别第一端；确认后自动按所选方向转动"

    def stop(self, reason="手动停止"):
        with self._lock:
            if not self.running:
                return
            self.motion.stop()
            self.running = False
            self._stop.set()
            self.status.update(running=False, phase="stopped", message=reason)
        self._emit()

    def _emit(self):
        if self._on_update:
            self._on_update(self.snapshot())

    def _update(self, **kwargs):
        with self._lock:
            if not self.running:
                return
            self.status.update(kwargs)
        self._emit()

    def _watchdog(self):
        while not self._stop.wait(.1):
            try:
                reason = ""
                if time.monotonic()-self._started > self.max_seconds:
                    reason = "测量超时，已请求停车"
                elif not self.motion.connected or not self.reading.connected:
                    reason = "电机或读数设备失联，已请求停车"
                elif not self.camera.is_open or self.camera.frame_age > 1.5:
                    reason = "相机画面过期或断开，已请求停车"
                elif not self._fresh(*self._reading_sample()):
                    reason = "微分表读数超过1.5秒未更新，已请求停车"
                elif self.status["phase"] not in ("loading", "first_reading", "second_reading") and time.monotonic()-self._last_vision > 1.5:
                    reason = "识别超过1.5秒无新结果，已请求停车"
                if reason:
                    self.stop(reason)
            except Exception as exc:
                self.stop(f"测量监视异常：{exc}")

    def _load_model(self):
        if self._factory:
            return self._factory(self.model_path)
        import torch
        from ultralytics import YOLO
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(self.model_path))
        names = set(model.names.values())
        if names != {"end_blur", "middle_blur"}:
            raise ValueError("必须使用钠灯end_blur/middle_blur双类模型")
        model.predict(np.zeros((553, 980, 3), dtype=np.uint8), imgsz=960,
                      device=self.device, verbose=False)
        return model

    def _classify(self, model, frame):
        predictions = model.predict(frame, imgsz=960, device=self.device, conf=.15,
                                    max_det=10, verbose=False)[0]
        scores = {"end_blur": 0.0, "middle_blur": 0.0}
        for box in predictions.boxes:
            name = model.names[int(box.cls.item())]
            scores[name] = max(scores.get(name, 0), float(box.conf.item()))
        if self._on_detection:
            dx, dy = self.roi[:2] if self.roi else (0, 0)
            boxes, names, confs = [], [], []
            for box in predictions.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                boxes.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
                names.append(model.names[int(box.cls.item())])
                confs.append(float(box.conf.item()))
            self._on_detection({"run_id": self.run_id, "at": getattr(self, "_frame_at", time.monotonic()),
                "result": {"boxes_xyxy": boxes, "class_names": names, "confs": confs}})
        end, middle = scores["end_blur"], scores["middle_blur"]
        # 中间类别有可靠输出时否决端部停车；低置信度不强制当清晰。
        kind = ("middle" if middle >= .5 else "end" if end >= self.threshold
                else "clear" if max(end, middle) < .25 else "unknown")
        return kind, end, middle

    def _capture_reading(self, which):
        with self._lock:
            if not self.running:
                return False
            # stop()是异步串口请求；等待其写入确认后才接受新的读数。
            requested_at = time.time()
            self.motion.stop()
        self._update(phase=f"{which}_reading", message="已请求停车，等待新读数")
        limit = time.monotonic()+4
        while not self._stop.wait(.05):
            ack = self.motion.stop_written_at
            value, at = self._reading_sample()
            if ack >= requested_at and at >= ack+.3 and self._fresh(value, at):
                with self._lock:
                    if not self.running:
                        return False
                    self.status[f"{which}_mm"] = float(value)
                    self.status[f"{which}_reading_at"] = at
                    self.status[f"{which}_stop_written_at"] = ack
                    if which == "first":
                        self.sequence.first_recorded()
                        self.motion.set_speed(self.gear)
                        getattr(self.motion, self.direction)()
                        self._last_vision = time.monotonic()
                        self.status.update(phase="leave_first", message="已记录第一端，正在离开第一端")
                    else:
                        result = {k: self.status.get(k) for k in (
                            "first_mm", "second_mm", "first_reading_at", "second_reading_at",
                            "first_stop_written_at", "second_stop_written_at", "direction",
                            "gear", "threshold", "model_path", "roi")}
                        result["reading_difference_mm"] = result["second_mm"]-result["first_mm"]
                        result["mean_wavelength_nm"] = calculate_mean_wavelength_nm(
                            result["first_mm"], result["second_mm"])
                        result["micrometer_to_mirror_ratio"] = MICROMETER_TO_MIRROR_RATIO
                        result["doublet_separation_nm"] = SODIUM_DOUBLET_SEPARATION_NM
                        result["visibility_periods"] = VISIBILITY_PERIODS
                        result["reading_timestamp_kind"] = "received_or_captured_by_reading_service"
                        result["stopped_at"] = time.time()
                        self.status.update(result=result, phase="done", running=False,
                                           message="已到另一端并停车，平均波长已自动计算")
                        self.running = False
                        self._stop.set()
                self._emit()
                return True
            if time.monotonic() > limit:
                raise TimeoutError("停车指令未确认或没有停车后新读数；测量中止")
        return False

    def _run(self):
        try:
            self.device = "cpu"
            model = self._load_model()
            self._last_vision = time.monotonic()
            self._update(phase="first_end", message="等待第一端模糊稳定出现")
            last_at = None
            messages = {
                "first_end": "等待第一端模糊稳定出现",
                "leave_first": "已记录第一端，正在离开第一端",
                "middle": "等待中间模糊；提前出现的端部不停车",
                "leave_middle": "已识别中间模糊，继续转动，不停车",
                "second_end": "已越过中间模糊，等待另一端",
            }
            while not self._stop.is_set():
                frame, at = self.camera.latest_sample()
                if frame is None or at == last_at:
                    self._stop.wait(.03)
                    continue
                last_at = at
                if not 0 <= time.monotonic()-at <= 1.5:
                    raise RuntimeError("采集帧过期")
                shape = frame.shape[:2]
                if self._shape is not None and shape != self._shape:
                    raise RuntimeError("相机画面尺寸已改变")
                self._shape = shape
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    if not (0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]):
                        raise ValueError("ROI超出相机画面")
                    frame = frame[y1:y2, x1:x2]
                if frame.size == 0 or float(np.std(frame)) < 2:
                    raise RuntimeError("相机画面没有可识别的有效纹理")
                self._frame_at = at
                kind, end, middle = self._classify(model, frame)
                if time.monotonic()-at > 1.5:
                    raise RuntimeError("识别结果过期")
                with self._lock:
                    if not self.running:
                        break
                    self._last_vision = time.monotonic()
                    action = self.sequence.observe(kind, at)
                self._update(phase=self.sequence.phase,
                             message=messages.get(self.sequence.phase, "停车记录读数"),
                             observed=kind, end_score=end, middle_score=middle)
                if action and not self._capture_reading(action):
                    break
        except Exception as exc:
            self.stop(f"钠灯测量中止：{exc}")
        finally:
            with self._lock:
                if self.running:
                    self.stop("钠灯测量结束")
