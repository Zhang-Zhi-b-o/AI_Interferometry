"""基于连续视觉定位结果判断画面中是否有条纹及其移动方向。"""
from __future__ import annotations

from collections import deque


class FringeMotionTracker:
    """用短时间位置窗口分析条纹存在性和水平移动方向。"""

    def __init__(self, window_size: int = 6, movement_threshold_px: float = 3.0,
                 missing_hold_frames: int = 3):
        self.window_size = max(2, int(window_size))
        self.movement_threshold_px = max(0.1, float(movement_threshold_px))
        self.missing_hold_frames = max(0, int(missing_hold_frames))
        self._positions: deque[float] = deque(maxlen=self.window_size)
        self._source = ""
        self._missing_frames = 0

    def reset(self) -> None:
        self._positions.clear()
        self._source = ""
        self._missing_frames = 0

    def update(self, *, has_fringe: bool, position_x: float | None,
               source: str = "") -> dict:
        if not has_fringe:
            self._missing_frames += 1
            if self._missing_frames > self.missing_hold_frames:
                self.reset()
            return {
                "has_fringe": False,
                "movement": "unknown",
                "movement_text": "未检测到条纹",
                "delta_x_px": None,
                "source": source or self._source,
            }

        self._missing_frames = 0
        if position_x is None:
            return {
                "has_fringe": True,
                "movement": "unknown",
                "movement_text": "检测到条纹，位置待确认",
                "delta_x_px": None,
                "source": source,
            }

        if source != self._source:
            self._positions.clear()
            self._source = source
        self._positions.append(float(position_x))
        delta = (
            self._positions[-1] - self._positions[0]
            if len(self._positions) >= 2 else 0.0
        )
        if len(self._positions) < 3:
            movement = "unknown"
            movement_text = "检测到条纹，正在分析移动方向"
        elif delta > self.movement_threshold_px:
            movement = "right"
            movement_text = f"条纹向右移动（{delta:+.1f} px）"
        elif delta < -self.movement_threshold_px:
            movement = "left"
            movement_text = f"条纹向左移动（{delta:+.1f} px）"
        else:
            movement = "stable"
            movement_text = f"条纹基本稳定（{delta:+.1f} px）"
        return {
            "has_fringe": True,
            "movement": movement,
            "movement_text": movement_text,
            "delta_x_px": delta,
            "source": source,
        }
