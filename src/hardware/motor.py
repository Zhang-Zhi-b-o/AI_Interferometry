"""步进电机控制 — RS-232 串口通信"""
from __future__ import annotations
import time
from enum import Enum
import serial
import serial.tools.list_ports
from src.logging import logger


class MotorMode(Enum):
    MANUAL = "manual"        # 手动模式：方向键控制
    CONTINUOUS = "continuous" # 连续模式：持续旋转 + 检测
    STEP = "step"            # 步进模式：转→停→分析→再转


class MotorCommand(Enum):
    START = "R"
    STOP = "S"
    SPEED_UP = "+"
    SPEED_DOWN = "-"
    QUERY = "?"


class MotorController:
    """
    步进电机控制器（RS-232 串口）
    协议：
      R = 启动, S = 停止
      + = 加速, - = 减速（数值越小越快：10最慢, 1最快）
      ? = 查询状态，返回 "RUN,SPD:5,OMEGA:630deg/s"
    """

    def __init__(self, port: str = "COM3", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self._connected = True
            logger.info(f"电机已连接: {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"电机连接失败 {self.port}: {e}")
            return False

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False
        logger.info("电机已断开")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------------
    # 命令
    # ------------------------------------------------------------------
    def send_cmd(self, cmd: str, read_response: bool = False) -> str | None:
        """发送 ASCII 命令，可选读取一行响应"""
        if not self.is_connected:
            logger.warning("电机未连接")
            return None
        try:
            self._ser.write(cmd.encode("ascii"))
            if read_response:
                return self._ser.readline().decode("ascii").strip()
        except serial.SerialException as e:
            logger.error(f"电机命令失败: {e}")
            return None
        return None

    def start(self):
        self.send_cmd(MotorCommand.START.value)

    def stop(self):
        self.send_cmd(MotorCommand.STOP.value)

    def speed_up(self):
        self.send_cmd(MotorCommand.SPEED_UP.value)

    def speed_down(self):
        self.send_cmd(MotorCommand.SPEED_DOWN.value)

    def query_status(self) -> dict:
        """查询电机状态，返回 {running, speed, omega}"""
        resp = self.send_cmd(MotorCommand.QUERY.value, read_response=True)
        if resp is None:
            return {"running": False, "speed": 0, "omega": 0}
        return self._parse_status(resp)

    # ------------------------------------------------------------------
    # 高级控制
    # ------------------------------------------------------------------
    def set_speed(self, target_speed: int, max_attempts: int = 12) -> bool:
        """将电机速度调整到目标值"""
        for _ in range(max_attempts):
            status = self.query_status()
            current = status["speed"]
            if current == target_speed:
                return True
            if current < target_speed:
                self.speed_down()
            else:
                self.speed_up()
            time.sleep(0.15)
        return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    @staticmethod
    def _parse_status(status_str: str) -> dict:
        """解析 "RUN,SPD:5,OMEGA:630deg/s" 格式"""
        result = {"running": False, "speed": 0, "omega": 0}
        parts = status_str.split(",")
        for p in parts:
            if p == "RUN":
                result["running"] = True
            elif p.startswith("SPD:"):
                try:
                    result["speed"] = int(p.split(":")[1])
                except (ValueError, IndexError):
                    pass
            elif p.startswith("OMEGA:"):
                try:
                    omega_str = p.split(":")[1].replace("deg/s", "")
                    result["omega"] = int(omega_str)
                except (ValueError, IndexError):
                    pass
        return result
