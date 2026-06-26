"""相机采集模块 — 基于 OpenCV 的摄像头管理"""
from __future__ import annotations
import cv2
import numpy as np
from src.logging import logger


class CameraManager:
    """管理 USB 摄像头的打开、帧采集、参数配置和释放"""

    def __init__(
        self,
        index: int = 1,
        resolution: tuple[int, int] = (1280, 1024),
        fps: int = 60,
    ):
        self.index = index
        self.resolution = resolution
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._opened = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """打开摄像头"""
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            logger.error(f"无法打开摄像头 (index={self.index})")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._opened = True
        logger.info(f"摄像头已打开: {self.resolution[0]}x{self.resolution[1]} @{self.fps}fps")
        return True

    def stop(self):
        """关闭摄像头"""
        if self._cap is not None:
            self._cap.release()
        self._opened = False
        logger.info("摄像头已关闭")

    def read(self) -> np.ndarray | None:
        """读取一帧，返回 BGR 图像"""
        if not self._opened or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    @property
    def is_opened(self) -> bool:
        return self._opened

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def detect_all(max_index: int = 9) -> list[int]:
        """扫描可用摄像头索引"""
        available = []
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(idx)
                cap.release()
        return available
