"""界面与服务的共同状态。

原则：**只有这里存"当前值"**。页面读状态、写状态，服务也算状态，
不允许页面之间互相调用（那样又会变成旧的 god object）。
"""
from __future__ import annotations

import threading
from typing import Any, Callable


class Signal:
    """极简订阅器：回调按注册顺序同步执行。线程安全够用即可。"""

    def __init__(self) -> None:
        self._subscribers: list[Callable[..., None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[..., None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def emit(self, *args: Any) -> None:
        with self._lock:
            targets = tuple(self._subscribers)
        for callback in targets:
            try:
                callback(*args)
            except Exception:  # 单个订阅者出错不影响其它人
                continue


class AppState:
    """全局单例状态。字段都是"最近一次有效值"，None 表示未知。"""

    #: 状态字段 → 中文名（用于调试输出与快照）
    FIELDS: dict[str, str] = {
        "camera_open": "相机已打开",
        "camera_index": "相机序号",
        "frame_size": "画面尺寸",
        "fps": "帧率",
        "model_loaded": "模型已加载",
        "detecting": "正在识别",
        "detect_count": "检测框数量",
        "detect_names": "检测到的类别",
        "center_x": "中心条纹横坐标",
        "center_conf": "中心条纹置信度",
        "fringe_width_px": "中心条纹宽度",
        "fringe_spacing_px": "条纹间距",
        "tilt_deg": "条纹倾斜角",
        "reading_mm": "微分表读数",
        "reading_at": "读数时间",
        "thickness_mm": "玻璃片厚度",
        "search_running": "自动寻中运行中",
        "search_gear": "当前档位",
        "search_message": "寻中阶段",
        "search_range_text": "估算搜索范围",
        "target_running": "目标读数运行中",
        "motor_connected": "电机已连接",
        "message": "状态提示",
    }

    def __init__(self) -> None:
        self.camera_open: bool = False
        self.camera_index: int | None = None
        self.frame_size: tuple[int, int] | None = None
        self.fps: float | None = None
        self.model_loaded: bool = False
        self.detecting: bool = False
        self.detect_count: int = 0
        self.detect_names: list[str] = []
        self.center_x: float | None = None
        self.center_conf: float | None = None
        self.fringe_width_px: float | None = None
        self.fringe_spacing_px: float | None = None
        self.tilt_deg: float | None = None
        self.reading_mm: float | None = None
        self.reading_at: float | None = None
        self.thickness_mm: float | None = None
        self.search_running: bool = False
        self.search_gear: int | None = None
        self.search_message = "待机"
        self.search_range_text = "搜索范围由标称转速积分估计，不是实测位置。"
        self.target_running: bool = False
        self.motor_connected: bool = False
        self.message: str = "就绪"
        #: 点号路径 → 相机 ROI（服务层共用同一个 ROI）
        self.roi: list[int] | None = None
        self.changed = Signal()

    # ------------------------------------------------------------------
    def update(self, **values: Any) -> None:
        touched = False
        for key, value in values.items():
            if not hasattr(self, key):
                raise AttributeError(f"AppState 没有字段 {key!r}")
            if getattr(self, key) != value:
                setattr(self, key, value)
                touched = True
        if touched:
            self.changed.emit(self)

    def snapshot(self) -> dict[str, Any]:
        """给日志 / 未来智能体用的纯 JSON 快照。"""
        return {
            key: (list(value) if isinstance(value, list) else value)
            for key in self.FIELDS
            if (value := getattr(self, key, None)) is not None
        }
