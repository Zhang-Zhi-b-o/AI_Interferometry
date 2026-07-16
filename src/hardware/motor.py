"""步进电机控制 — RS-232 串口通信。"""
from __future__ import annotations

import json
import threading
from enum import Enum

import serial
import serial.tools.list_ports

from src.logging import logger


class MotorMode(Enum):
    """保留手动模式枚举，自动实验由独立流程状态机管理。"""

    MANUAL = "manual"


class MotorCommand(Enum):
    FORWARD = "R"
    REVERSE = "r"
    STOP = "S"
    TOGGLE_DIRECTION = "D"
    SPEED_UP = "+"
    SPEED_DOWN = "-"
    QUERY = "?"  # 协议规定任意其他字符均返回 JSON 状态。


MOTOR_GEAR_TABLE = {
    1: {"pulse_freq_hz": 5000, "omega_deg_s": 1125, "turn_seconds": 0.32},
    2: {"pulse_freq_hz": 4450, "omega_deg_s": 1001, "turn_seconds": 0.36},
    3: {"pulse_freq_hz": 3900, "omega_deg_s": 877, "turn_seconds": 0.41},
    4: {"pulse_freq_hz": 3350, "omega_deg_s": 754, "turn_seconds": 0.48},
    5: {"pulse_freq_hz": 2800, "omega_deg_s": 630, "turn_seconds": 0.57},
    6: {"pulse_freq_hz": 2250, "omega_deg_s": 506, "turn_seconds": 0.71},
    7: {"pulse_freq_hz": 1700, "omega_deg_s": 382, "turn_seconds": 0.94},
    8: {"pulse_freq_hz": 1150, "omega_deg_s": 259, "turn_seconds": 1.39},
    9: {"pulse_freq_hz": 800, "omega_deg_s": 180, "turn_seconds": 2.00},
    10: {"pulse_freq_hz": 500, "omega_deg_s": 112, "turn_seconds": 3.20},
}


class MotorController:
    """电机串口协议封装。

    ``R`` 正转启动，``r`` 反转启动，``S`` 停止，``D`` 在运行中换向，
    ``+`` 加速，``-`` 减速。发送 ``?`` 等其他字符可读取 JSON 状态。
    """

    def __init__(self, port: str = "COM3", baudrate: int = 9600,
                 timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._connected = False
        self._io_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self._connected = True
            logger.info("电机已连接: %s", self.port)
            return True
        except serial.SerialException as exc:
            logger.error("电机连接失败 %s: %s", self.port, exc)
            return False

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False
        logger.info("电机已断开")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ser is not None and self._ser.is_open

    def send_cmd(self, cmd: str, read_response: bool = False) -> str | bool | None:
        """发送单个 ASCII 字符，可选读取一行响应。"""
        if not self.is_connected:
            logger.warning("电机未连接")
            return None
        if not isinstance(cmd, str) or len(cmd) != 1 or ord(cmd) > 127:
            raise ValueError("电机命令必须是单个 ASCII 字符")
        try:
            with self._io_lock:
                self._ser.write(cmd.encode("ascii"))
                if read_response:
                    return self._ser.readline().decode("utf-8", errors="replace").strip()
            return True
        except serial.SerialException as exc:
            self._connected = False
            logger.error("电机命令失败: %s", exc)
            return None

    def start_forward(self) -> bool:
        return self.send_cmd(MotorCommand.FORWARD.value) is True

    def start_reverse(self) -> bool:
        return self.send_cmd(MotorCommand.REVERSE.value) is True

    def start(self) -> bool:
        """自动实验兼容入口，默认按正转方向启动。"""
        return self.start_forward()

    def stop(self) -> bool:
        return self.send_cmd(MotorCommand.STOP.value) is True

    def toggle_direction(self) -> bool:
        return self.send_cmd(MotorCommand.TOGGLE_DIRECTION.value) is True

    def speed_up(self) -> bool:
        return self.send_cmd(MotorCommand.SPEED_UP.value) is True

    def speed_down(self) -> bool:
        return self.send_cmd(MotorCommand.SPEED_DOWN.value) is True

    def query_status(self) -> dict:
        response = self.send_cmd(MotorCommand.QUERY.value, read_response=True)
        if not isinstance(response, str):
            return self._empty_status()
        return self._parse_status(response)

    def set_speed(self, target_speed: int, max_attempts: int = 12) -> bool:
        """供内部自动实验将档位调至 1~10；不在手动插件中暴露。"""
        target_speed = max(1, min(10, int(target_speed)))
        for _ in range(max_attempts):
            status = self.query_status()
            current = int(status.get("speed", 0))
            if current == target_speed:
                return True
            if current <= 0:
                return False
            if current < target_speed:
                self.speed_down()
            else:
                self.speed_up()
        return False

    @staticmethod
    def list_ports() -> list[str]:
        return [port.device for port in serial.tools.list_ports.comports()]

    @staticmethod
    def _empty_status() -> dict:
        return {
            "running": False,
            "speed": 0,
            "omega": 0,
            "direction": "unknown",
            "pulse_freq": 0,
            "raw": {},
        }

    @staticmethod
    def _parse_status(status_str: str) -> dict:
        """优先解析控制器 JSON，同时兼容旧版逗号文本。"""
        result = MotorController._empty_status()
        text = str(status_str).strip()
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None

        if isinstance(payload, dict):
            state = str(payload.get("state", payload.get("status", ""))).lower()
            running_value = payload.get("running", payload.get("run", None))
            if isinstance(running_value, str):
                running = running_value.lower() in ("1", "true", "run", "running")
            elif running_value is None:
                running = state in ("run", "running", "forward", "reverse")
            else:
                running = bool(running_value)

            def integer(*keys: str) -> int:
                for key in keys:
                    if key in payload:
                        try:
                            return int(float(payload[key]))
                        except (TypeError, ValueError):
                            pass
                return 0

            direction = payload.get("direction", payload.get("dir", "unknown"))
            result.update({
                "running": running,
                "speed": integer("speed", "gear", "level", "spd"),
                "omega": integer("omega", "omega_deg_s", "deg_s"),
                "direction": str(direction),
                "pulse_freq": integer("pulse_freq", "pulse_freq_hz", "frequency", "freq"),
                "raw": payload,
            })
            gear = MOTOR_GEAR_TABLE.get(result["speed"])
            if gear:
                result["omega"] = result["omega"] or gear["omega_deg_s"]
                result["pulse_freq"] = result["pulse_freq"] or gear["pulse_freq_hz"]
            return result

        # 兼容旧控制器：RUN,SPD:5,OMEGA:630deg/s
        for part in text.split(","):
            part = part.strip()
            if part == "RUN":
                result["running"] = True
            elif part.startswith("SPD:"):
                try:
                    result["speed"] = int(part.split(":", 1)[1])
                except ValueError:
                    pass
            elif part.startswith("OMEGA:"):
                try:
                    result["omega"] = int(
                        part.split(":", 1)[1].replace("deg/s", ""))
                except ValueError:
                    pass
        return result
