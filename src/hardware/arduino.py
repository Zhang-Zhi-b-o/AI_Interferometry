"""Arduino 辅助模块 — 激光测距传感器 + OLED 显示"""
from __future__ import annotations
import serial
from src.logging import logger


class ArduinoReader:
    """通过串口读取 Arduino 传来的激光测距数据"""

    def __init__(self, port: str = "COM4", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self._connected = True
            logger.info(f"Arduino 已连接: {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Arduino 连接失败 {self.port}: {e}")
            return False

    def read(self) -> dict | None:
        """
        读取一帧数据
        返回 {"turns": float, "direction": str} 或 None
        """
        if not self._connected or self._ser is None:
            return None
        try:
            line = self._ser.readline().decode("ascii").strip()
            # 预期格式: "TURNS:5.2,DIR:CW"
            parts = line.split(",")
            data = {"turns": 0.0, "direction": "?"}
            for p in parts:
                if p.startswith("TURNS:"):
                    data["turns"] = float(p.split(":")[1])
                elif p.startswith("DIR:"):
                    data["direction"] = p.split(":")[1]
            return data
        except (serial.SerialException, ValueError, IndexError):
            return None

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
