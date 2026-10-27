"""线程 → 界面 的桥。服务层回调都在工作线程，靠这里的 Qt 信号回到主线程。

规则：服务层不 import Qt，界面层不自己开线程，所有跨线程结果都过这个桥。
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class Bridge(QObject):
    frame = Signal(object)          # np.ndarray
    message = Signal(str)           # 底部状态栏
    state = Signal()                # 状态变化，页面自己来取快照
    detection = Signal(object)      # dict
    reading = Signal(object)        # dict
    search_update = Signal(object)  # dict
    target_update = Signal(object)  # dict
    backlash_update = Signal(object)
    task_ok = Signal(str, object)   # (ticket, result)
    task_err = Signal(str, str)     # (ticket, message)
