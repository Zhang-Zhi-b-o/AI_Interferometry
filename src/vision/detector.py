"""YOLO 目标检测器 — 基于 Ultralytics YOLO"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from src.logging import logger
from src.vision.detection_strategy import (
    MODEL_INFERENCE_CONFIDENCE,
    MODEL_NMS_IOU,
    TemporalReference,
    apply_standard_detection_strategy,
)


class YOLODetector:
    """YOLO 推理封装，支持 ROI 裁剪检测"""

    def __init__(
        self,
        model_path: str,
        confidence: float = MODEL_INFERENCE_CONFIDENCE,
        iou: float = MODEL_NMS_IOU,
        imgsz: int = 640,
        device: str = "cuda",
        standard_strategy: bool = True,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.standard_strategy = bool(standard_strategy)
        self._model: YOLO | None = None
        self._reliable_detection_history: deque[list[TemporalReference]] = deque(
            maxlen=2)

    def load(self) -> bool:
        path = Path(self.model_path)
        if not path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        try:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，YOLO 将回退到 CPU")
                self.device = "cpu"
            self._model = YOLO(str(path))
            self.reset_temporal_history()
            logger.info(
                f"YOLO 模型已加载: {path.resolve()} "
                f"device={self.device} classes={self.class_names}"
            )
            return True
        except Exception as exc:
            self._model = None
            logger.exception(f"YOLO 模型加载失败: {exc}")
            return False

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
            height, width = frame.shape[:2]
            x, y = max(0, int(x)), max(0, int(y))
            if x >= width or y >= height:
                logger.warning(f"ROI 起点超出图像范围: {roi}")
                return self._empty_result(frame)
            w = min(max(0, int(w)), width - x)
            h = min(max(0, int(h)), height - y)
            if w < 10 or h < 10:
                logger.warning(f"ROI 尺寸无效: {(x, y, w, h)}")
                return self._empty_result(frame)
            target = frame[y:y + h, x:x + w]
            roi_offset = (x, y)

        if target.size == 0:
            return self._empty_result(frame)

        try:
            inference_confidence = (
                MODEL_INFERENCE_CONFIDENCE
                if self.standard_strategy
                else min(max(self.confidence, 0.0), 1.0)
            )
            inference_iou = (
                MODEL_NMS_IOU
                if self.standard_strategy
                else min(max(self.iou, 0.0), 1.0)
            )
            results = self._model.predict(
                target, conf=inference_confidence,
                iou=inference_iou,
                imgsz=max(32, int(self.imgsz)), device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception(f"YOLO 推理失败: {exc}")
            return self._empty_result(frame, error=str(exc))
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

        strategy_result = None
        if self.standard_strategy:
            temporal_references = [
                reference
                for frame_references in self._reliable_detection_history
                for reference in frame_references
            ]
            strategy_result = apply_standard_detection_strategy(
                boxes_xyxy, confs, class_ids, class_names, frame.shape,
                temporal_references=temporal_references,
            )
            self._reliable_detection_history.append(
                strategy_result.reliable_references)
            kept = np.asarray(strategy_result.kept_indices, dtype=int)
            boxes_xyxy = boxes_xyxy[kept]
            confs = confs[kept]
            class_ids = class_ids[kept]
            class_names = [class_names[index] for index in kept]

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
            "strategy": (
                strategy_result.as_dict() if strategy_result is not None else None),
        }

    def is_loaded(self) -> bool:
        return self._model is not None

    def reset_temporal_history(self) -> None:
        """在预测会话或视场切换时清空弱框的相邻帧依据。"""
        self._reliable_detection_history.clear()

    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            return {}
        names = getattr(self._model, "names", {})
        if isinstance(names, list):
            return dict(enumerate(names))
        return {int(k): str(v) for k, v in names.items()}

    def find_class_ids(self, *keywords: str) -> set[int]:
        wanted = tuple(k.lower() for k in keywords)
        return {
            cid for cid, name in self.class_names.items()
            if any(k in name.lower() for k in wanted)
        }

    @staticmethod
    def _empty_result(frame: np.ndarray | None = None, error: str | None = None) -> dict:
        return {
            "boxes_xyxy": np.array([]),
            "confs": np.array([]),
            "class_ids": np.array([]),
            "class_names": [],
            "annotated": frame.copy() if frame is not None else None,
            "center": None,
            "error": error,
            "strategy": None,
        }
