"""把当前帧的 YOLO 零级框、近场路标转成旧状态机的视觉输入。"""
from __future__ import annotations

import numpy as np

from src.vision.fringe_center import find_center_in_region
from src.vision.fringe_motion import FringeMotionTracker


class SearchObservation:
    def __init__(self, detector, *, roi_provider=lambda: None, on_detection=None):
        self.detector = detector
        self.roi_provider = roi_provider
        self.on_detection = on_detection
        self.tracker = FringeMotionTracker()
        self.previous_box = None
        self.previous_center = None
        self.misses = 0

    def reset(self):
        self.detector.reset()
        self.tracker.reset()
        self.previous_box = self.previous_center = None
        self.misses = 0

    def __call__(self, frame):
        result = self.detector.detect(frame, roi=self.roi_provider())
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
            if values.shape != (4,) or not np.isfinite(values).all() or not np.isfinite(conf):
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
            payload.update(zero_box_x=float((box[0] + box[2]) / 2),
                           zero_box_confidence=confidence)
        else:
            self.misses += 1
            box = self.previous_box if self.misses <= 3 else None
        # 缺框时只在最近零级区域内短时重定位，历史坐标本身不能冒充当前观测。
        if box is not None:
            x1, y1, x2, y2 = box
            expected = self.previous_center if not zero and self.previous_center is not None else (x1 + x2) / 2
            span = max(80, (x2 - x1) * 1.5)
            left, right = max(0, int(expected - span / 2)), min(frame.shape[1], int(expected + span / 2))
            top, bottom = max(0, int(y1)), min(frame.shape[0], int(y2))
            try:
                info = find_center_in_region(
                    frame[top:bottom, left:right],
                    expected_center_x=expected - left,
                    search_radius=max((x2 - x1) / 2, 15),
                )
                if info["orientation"] == "vertical" and info["confidence"] >= 0.18:
                    self.previous_center = left + float(info["center_main"])
                    payload.update(center_x=self.previous_center,
                                   confidence=float(info["confidence"]))
            except ValueError:
                pass  # 当前零级区域无可用条纹，状态机继续搜索或等待。
        if box is None:
            self.previous_center = None
        position = payload["center_x"]
        if position is None:
            position = payload["guide_x"]
        motion = self.tracker.update(has_fringe=position is not None, position_x=position,
                                     source="center" if payload["center_x"] is not None else "guide")
        payload.update(fringe_movement=motion["movement"],
                       fringe_delta_x_px=motion["delta_x_px"])
        return payload
