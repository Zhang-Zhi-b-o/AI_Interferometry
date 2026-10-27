"""把当前帧的 YOLO 零级框、近场路标转成旧状态机的视觉输入。"""
from __future__ import annotations

import numpy as np
from collections import deque
import time

from app.engine.vision.fringe_motion import FringeMotionTracker


def find_center_in_region(*args, **kwargs):
    from app.engine.vision.fringe_center import find_center_in_region as locate

    return locate(*args, **kwargs)


def analyse_fringe_texture(frame):
    from app.engine.vision.fringe_recognition import analyse_fringe_texture as analyse

    return analyse(frame)


class SearchObservation:
    def __init__(self, detector, *, roi_provider=lambda: None, on_detection=None):
        self.detector = detector
        self.roi_provider = roi_provider
        self.on_detection = on_detection
        self.tracker = FringeMotionTracker()
        self.previous_box = None
        self.previous_center = None
        self.misses = 0
        self.recognition_tracker = self.center_tracker = None
        self.last_diagnostics = {}
        self._box_history = deque(maxlen=5)
        self._stable_votes = self._unstable_votes = 0
        self.box_stable = False

    def _ensure_trackers(self):
        if self.recognition_tracker is None:
            from app.engine.vision.fringe_recognition import FringeRecognitionTracker
            from app.engine.vision.fringe_center import CenterTracker

            self.recognition_tracker = FringeRecognitionTracker()
            self.center_tracker = CenterTracker()

    def reset(self):
        self.detector.reset()
        self.tracker.reset()
        self.previous_box = self.previous_center = None
        self.misses = 0
        if self.recognition_tracker is not None:
            self.recognition_tracker.reset()
            self.center_tracker.reset()
        self.last_diagnostics = {}
        self._box_history.clear()
        self._stable_votes = self._unstable_votes = 0
        self.box_stable = False

    def _update_box(self, box):
        self._box_history.append(((box[0] + box[2]) / 2, box[2] - box[0]))
        if len(self._box_history) < 5:
            return
        positions, widths = np.asarray(self._box_history).T
        stable = (np.std(positions) < np.mean(widths) * .15
                  and np.std(widths) < np.mean(widths) * .20)
        self._stable_votes = self._stable_votes + 1 if stable else 0
        self._unstable_votes = 0 if stable else self._unstable_votes + 1
        if self._stable_votes >= 3:
            self.box_stable = True
        elif self._unstable_votes >= 3:
            self.box_stable = False

    def __call__(self, frame):
        self._ensure_trackers()
        roi = self.roi_provider()
        result = self.detector.detect(frame, roi=roi)
        if result.get("error"):
            raise RuntimeError("寻中识别失败：" + str(result["error"]))
        if self.on_detection:
            self.on_detection(result)
        boxes = result.get("boxes_xyxy", [])
        confs = result.get("confs", [])
        names = result.get("class_names", [])
        zero, guides = [], []
        for box, conf, name in zip(boxes, confs, names):
            values = np.asarray(box, dtype=float)
            if (values.shape != (4,) or not np.isfinite(values).all()
                    or not np.isfinite(conf) or not 0 < conf <= 1):
                continue
            x1, y1, x2, y2 = values
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                continue
            if name == "zero_order":
                zero.append((float(conf), values))
            elif name == "near_fringe":
                guides.append((float(conf), (x1 + x2) / 2))
        zero.sort(key=lambda item: item[0], reverse=True)
        payload = {"center_x": None, "confidence": 0.0,
                   "zero_box_x": None, "zero_box_confidence": 0.0,
                   "zero_box_half_width": 0.0, "guide_x": None,
                   "guide_confidence": 0.0, "guide_count": len(guides)}
        if guides:
            weights, positions = zip(*guides)
            payload.update(guide_x=float(np.average(positions, weights=weights)),
                           guide_confidence=max(weights))
        if zero:
            confidence, box = zero[0]
            self.misses = 0
            self.previous_box = box
            self._update_box(box)
            payload.update(zero_box_x=float((box[0] + box[2]) / 2),
                           zero_box_confidence=confidence,
                           zero_box_half_width=float((box[2] - box[0]) / 2))
        else:
            self.misses += 1
            box = self.previous_box if self.misses <= 3 else None
            if self.misses >= 3:
                self.box_stable = False
                self._box_history.clear()
                self._stable_votes = self._unstable_votes = 0
        tracked = None
        # 缺框时只在最近零级区域内短时重定位，历史坐标本身不能冒充当前观测。
        if box is not None:
            x1, y1, x2, y2 = box
            expected = self.previous_center if not zero and self.previous_center is not None else (x1 + x2) / 2
            span = max(80, (x2 - x1) * 1.5)
            left, right = max(0, int(expected - span / 2)), min(frame.shape[1], int(expected + span / 2))
            top, bottom = max(0, int(y1)), min(frame.shape[0], int(y2))
            try:
                margin = max(6, int((x2 - x1) * .15))
                bounds = ((max(0, x1 - left - margin), min(right-left, x2-left+margin))
                          if self.box_stable else None)
                info = find_center_in_region(
                    frame[top:bottom, left:right],
                    expected_center_x=expected - left,
                    search_radius=max((x2 - x1) / 2, 15),
                    search_bounds=bounds,
                )
                measured = left + float(info["center_main"])
                confidence = float(info["confidence"])
                if (info["orientation"] == "vertical" and .18 <= confidence <= 1
                        and np.isfinite(measured) and left <= measured < right):
                    tracked = self.center_tracker.update(measured, confidence)
                    if tracked["accepted"] and not tracked["held"]:
                        self.previous_center = tracked["center"]
                        # 当前测量用于闭环；平滑/保持坐标仅作下一帧搜索先验和诊断。
                        payload.update(center_x=measured, confidence=confidence)
            except ValueError:
                pass  # 当前零级区域无可用条纹，状态机继续搜索或等待。
        if box is None:
            self.previous_center = None
        if tracked is None:
            tracked = self.center_tracker.update(None)
        texture_frame, offset = frame, 0
        if roi is not None:
            x, y, width, height = (int(v) for v in roi)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
            texture_frame, offset = frame[y1:y2, x1:x2], x1
        # 每次使用当前输入帧的纹理，不沿用上一帧纹理冒充新视觉证据。
        texture = dict(analyse_fringe_texture(texture_frame))
        if texture.get("position_x") is not None:
            texture["position_x"] += offset
        position = payload["center_x"]
        if position is None:
            position = payload["zero_box_x"] if zero else payload["guide_x"]
        recognition = self.recognition_tracker.update(
            yolo_has_fringe=bool(zero or guides), yolo_position_x=position,
            yolo_confidence=max([c for c, _ in zero + guides], default=0.0),
            texture=texture, now=time.monotonic())
        motion = self.tracker.update(has_fringe=recognition["has_fringe"],
                                     position_x=recognition["position_x"], source="fused")
        payload.update(fringe_movement=motion["movement"],
                       fringe_delta_x_px=motion["delta_x_px"],
                       fringe_velocity_px_s=recognition["velocity_px_s"],
                       scene_has_fringe=recognition["has_fringe"],
                       scene_position_x=recognition["position_x"],
                       scene_confidence=recognition["confidence"],
                       scene_source=recognition["source"],
                       scene_blurred=recognition["blurred"],
                       scene_held=recognition["held"])
        # 这些诊断名不是 CenterControlStateMachine.update 的参数，不能混入 payload。
        self.last_diagnostics = dict(
            has_fringe=recognition["has_fringe"], recognition_source=recognition["source"],
            confidence=recognition["confidence"], held=recognition["held"],
            velocity_px_s=recognition["velocity_px_s"], blurred=recognition["blurred"],
            position_x=recognition["position_x"], texture_confidence=recognition["texture_confidence"],
            tracked_center=tracked["center"], center_held=tracked["held"],
            box_stable=self.box_stable, zero_box_misses=self.misses)
        return payload
