"""YOLO 目标检测器 — 基于 Ultralytics YOLO"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from src.logging import logger


class YOLODetector:
    """YOLO 推理封装，支持 ROI 裁剪检测"""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._model: YOLO | None = None

    def load(self) -> bool:
        path = Path(self.model_path)
        if not path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        self._model = YOLO(str(path))
        logger.info(f"YOLO 模型已加载: {self.model_path}")
        return True

    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> dict:
        if self._model is None:
            logger.warning("YOLO 模型未加载")
            return self._empty_result()

        # ROI 裁剪
        target = frame
        roi_offset = (0, 0)
        if roi is not None:
            x, y, w, h = roi
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            target = frame[y:y + h, x:x + w]
            roi_offset = (x, y)

        results = self._model.predict(
            target, conf=self.confidence, iou=self.iou,
            imgsz=self.imgsz, verbose=False,
        )
        result = results[0]

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])
        class_names = [result.names.get(cid, "unknown") for cid in class_ids]

        # 坐标映射回全帧
        if roi_offset != (0, 0) and len(boxes_xyxy) > 0:
            ox, oy = roi_offset
            boxes_xyxy[:, [0, 2]] += ox
            boxes_xyxy[:, [1, 3]] += oy

        # 在全帧上画框
        annotated = frame.copy()
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{class_names[i]} {confs[i]:.2f}"
            cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 最高置信度框的中心
        center = None
        if len(boxes_xyxy) > 0:
            best_idx = int(np.argmax(confs))
            bx1, by1, bx2, by2 = boxes_xyxy[best_idx]
            center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        return {
            "boxes_xyxy": boxes_xyxy,
            "confs": confs,
            "class_ids": class_ids,
            "class_names": class_names,
            "annotated": annotated,
            "center": center,
        }

    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _empty_result() -> dict:
        return {
            "boxes_xyxy": np.array([]),
            "confs": np.array([]),
            "class_ids": np.array([]),
            "class_names": [],
            "annotated": None,
            "center": None,
        }

    @staticmethod
    def get_class_confidences(result: dict) -> dict[str, float]:
        if len(result["class_names"]) == 0:
            return {}
        class_conf = {}
        for name, conf in zip(result["class_names"], result["confs"]):
            if name not in class_conf or conf > class_conf[name]:
                class_conf[name] = float(conf)
        return class_conf
