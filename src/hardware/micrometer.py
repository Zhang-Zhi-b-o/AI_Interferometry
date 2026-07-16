"""千分尺读数接口预留。

具体设备的通信方式、数据格式和读取协议尚未确定。后续确认硬件后，
在此实现连接、断开和读取逻辑，并由 UI 将读数写入实验流程状态。
"""
from __future__ import annotations


class MicrometerReader:
    """千分尺适配器占位类；当前不包含任何设备读取实现。"""

    def connect(self) -> bool:
        raise NotImplementedError("千分尺读数方式尚未确定")

    def close(self) -> None:
        raise NotImplementedError("千分尺读数方式尚未确定")

    def read_value_mm(self) -> float | None:
        raise NotImplementedError("千分尺读数方式尚未确定")
