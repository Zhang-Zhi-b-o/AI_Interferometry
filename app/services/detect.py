"""YOLO 条纹检测：包装 ``app.engine.vision.detector.YOLODetector``。

保留的都是旧代码里已被测试覆盖的部分（类别名称映射、ROI、时间序列策略），
所以这里只做"加载 / 推理 / 结果规整"三件事。
"""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np


class DetectService:
    def __init__(self, *, on_status=None) -> None:
        self._on_status = on_status
        self._inference_lock = threading.RLock()
        self._detector: Any = None
        self._path: Path | None = None
        self._model_signature = None
        self._load_lock = threading.Lock()
        self.class_names: dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        return self._detector is not None

    @property
    def model_path(self) -> Path | None:
        return self._path

    # ------------------------------------------------------------------
    def load(self, path: str | Path, *, confidence: float = 0.25, iou: float = 0.45,
             imgsz: int = 640, device: str = "cpu") -> tuple[bool, str]:
        with self._load_lock:
            return self._load(path, confidence=confidence, iou=iou, imgsz=imgsz, device=device)

    @staticmethod
    def prepare_dependencies() -> None:
        """Import the inference stack on a worker after the first window paint."""
        from importlib import import_module

        import_module("ultralytics")
        import_module("app.engine.vision.detector")
        import_module("app.engine.vision.fringe_center")
        import_module("app.engine.vision.fringe_recognition")

    def _load(self, path, *, confidence, iou, imgsz, device):
        path = Path(path).resolve()
        if not path.is_file():
            return False, f"模型不存在：{path}"
        stat = path.stat()
        signature = (path, stat.st_mtime_ns, stat.st_size, str(device))
        with self._inference_lock:
            if self._detector is not None and signature == self._model_signature:
                self._detector.confidence = float(confidence)
                self._detector.iou = float(iou)
                self._detector.imgsz = int(imgsz)
                self._detector.reset_temporal_history()
                return True, f"模型已就绪（复用）：{path.name}"
        try:
            self.prepare_dependencies()
            from app.engine.vision.detector import YOLODetector
        except Exception as exc:  # ultralytics 未安装等
            return False, f"无法导入检测器：{exc}"
        try:
            detector = YOLODetector(
                str(path), confidence=float(confidence), iou=float(iou),
                imgsz=int(imgsz), device=device,
            )
            if not detector.load():
                return False, "模型加载失败（详见日志）"
            detector.warmup()
        except Exception as exc:
            return False, f"模型加载异常：{exc}"
        with self._inference_lock:
            self._detector = detector
            self._path = path
            self.class_names = dict(detector.class_names or {})
            self._model_signature = signature
        message = f"模型已加载：{path.name}（{detector.device}，已预热）"
        self._status(message)
        return True, message

    def detect(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
        if self._detector is None or frame is None:
            return self._empty()
        try:
            with self._inference_lock:
                result = self._detector.detect(frame, roi=roi)
        except Exception as exc:
            return self._empty(error=str(exc))
        result.setdefault("error", None)
        return result

    def reset(self) -> None:
        if self._detector is not None:
            try:
                with self._inference_lock:
                    self._detector.reset_temporal_history()
            except Exception:
                pass

    def configure(self, *, confidence, iou, imgsz, standard_strategy=True):
        """在同一推理锁中更新阈值，不重新读取权重。"""
        with self._inference_lock:
            if self._detector is not None:
                self._detector.confidence = float(confidence)
                self._detector.iou = float(iou)
                self._detector.imgsz = int(imgsz)
                self._detector.standard_strategy = bool(standard_strategy)
                self._detector.reset_temporal_history()

    # ------------------------------------------------------------------
    @staticmethod
    def describe(result: dict) -> dict:
        """把检测结果压成界面要用的几个数，避免页面直接摸 numpy 数组。"""
        boxes = result.get("boxes_xyxy")
        confs = result.get("confs")
        names = result.get("class_names") or []
        count = int(len(boxes)) if boxes is not None else 0
        best_conf = float(np.max(confs)) if confs is not None and len(confs) else None
        return {
            "count": count,
            "names": list(dict.fromkeys(str(n) for n in names)),
            "best_conf": best_conf,
            "error": result.get("error"),
        }

    @staticmethod
    def _empty(error: str | None = None) -> dict:
        return {
            "boxes_xyxy": np.zeros((0, 4), dtype=float),
            "confs": np.zeros((0,), dtype=float),
            "class_ids": np.zeros((0,), dtype=int),
            "class_names": [],
            "annotated": None,
            "center": None,
            "error": error,
            "strategy": None,
        }

    def _status(self, text: str) -> None:
        if self._on_status is not None:
            self._on_status(text)
