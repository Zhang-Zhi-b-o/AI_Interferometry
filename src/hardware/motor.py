"""步进电机控制 — RS-232 串口通信。"""
from __future__ import annotations

import json
import re
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

    def __init__(self, port: str = "auto", baudrate: int = 9600,
                 timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._connected = False
        self._io_lock = threading.Lock()

    def connect(self) -> bool:
        selected_port = self.detect_port(self.port)
        if selected_port is None:
            logger.error("未检测到可确定的电机串口")
            return False
        self.port = selected_port
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
        """发送单个 ASCII 字符，可选读取状态响应。

        部分控制器会先回显查询字符，再在下一行返回 JSON，因此查询时最多
        读取四行，并在获得可识别的状态内容后立即返回。
        """
        if not self.is_connected:
            logger.warning("电机未连接")
            return None
        if not isinstance(cmd, str) or len(cmd) != 1 or ord(cmd) > 127:
            raise ValueError("电机命令必须是单个 ASCII 字符")
        try:
            with self._io_lock:
                if read_response and hasattr(self._ser, "reset_input_buffer"):
                    self._ser.reset_input_buffer()
                self._ser.write(cmd.encode("ascii"))
                if read_response:
                    lines = []
                    for _ in range(4):
                        raw = self._ser.readline()
                        if not raw:
                            break
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        lines.append(line)
                        if self._looks_like_status(line):
                            break
                    return "\n".join(lines)
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
        """将档位调至 1~10，状态不可读时使用限位归档方式。

        ``-`` 在最慢的 10 档继续发送不会越界，因此可先发送足够次数的
        减速命令归档到 10 档，再用 ``+`` 到达目标档位。
        """
        target_speed = max(1, min(10, int(target_speed)))
        for _ in range(max_attempts):
            status = self.query_status()
            current = int(status.get("speed", 0))
            if current == target_speed:
                return True
            if current <= 0:
                return self._set_speed_from_slowest(target_speed)
            if current < target_speed:
                if not self.speed_down():
                    return False
            else:
                if not self.speed_up():
                    return False
        return False

    def _set_speed_from_slowest(self, target_speed: int) -> bool:
        """不依赖状态反馈，从最慢档位安全归档后设置目标档位。"""
        for _ in range(10):
            if not self.speed_down():
                return False
        for _ in range(10 - target_speed):
            if not self.speed_up():
                return False
        return True

    @staticmethod
    def list_ports() -> list[str]:
        return [port.device for port in serial.tools.list_ports.comports()]

    @staticmethod
    def detect_port(preferred: str | None = None) -> str | None:
        """自动选择电机串口。

        已配置端口仍存在时优先使用；否则优先识别常见 USB 转串口，只有
        一个串口时直接使用。多个无法区分的串口不会被随意选中。
        """
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return None

        preferred_text = str(preferred or "").strip()
        if preferred_text and preferred_text.lower() not in {"auto", "自动检测"}:
            for port in ports:
                if port.device.upper() == preferred_text.upper():
                    return port.device

        usb_keywords = (
            "CH340", "CH341", "USB-SERIAL", "USB SERIAL", "CP210", "FTDI",
            "1A86:7523",
        )
        likely = []
        for port in ports:
            description = f"{port.description} {port.hwid}".upper()
            if any(keyword in description for keyword in usb_keywords):
                likely.append(port.device)
        if len(likely) == 1:
            return likely[0]
        if len(ports) == 1:
            return ports[0].device
        return None

    @staticmethod
    def _empty_status() -> dict:
        return {
            "valid": False,
            "running": False,
            "speed": 0,
            "omega": 0,
            "direction": "unknown",
            "pulse_freq": 0,
            "raw": {},
            "response": "",
        }

    @staticmethod
    def _looks_like_status(text: str) -> bool:
        upper = str(text).upper()
        return "{" in upper or "SPD:" in upper or upper.startswith(("RUN", "STOP"))

    @staticmethod
    def _normalise_direction(value) -> str:
        raw = str(value).strip()
        if raw == "R":
            return "forward"
        if raw == "r":
            return "reverse"
        text = raw.lower()
        if text in {"1", "forward", "fwd", "cw", "正转", "正向"}:
            return "forward"
        if text in {"-1", "reverse", "rev", "ccw", "反转", "反向"}:
            return "reverse"
        if text in {"stop", "stopped", "停止", "idle"}:
            return "stopped"
        return text or "unknown"

    @staticmethod
    def _parse_status(status_str: str) -> dict:
        """解析控制器状态，兼容回显、嵌套 JSON 和常见字段命名。"""
        result = MotorController._empty_status()
        text = str(status_str).strip()
        result["response"] = text

        # 串口可能返回 ``?\r\n{...}`` 或在 JSON 前后附加提示文字。
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        json_text = json_match.group(0) if json_match else text
        try:
            payload = json.loads(json_text)
        except (json.JSONDecodeError, TypeError):
            payload = None

        if isinstance(payload, dict):
            root_payload = payload
            for nested_key in ("motor", "data", "status"):
                nested = payload.get(nested_key)
                if isinstance(nested, dict):
                    payload = {**payload, **nested}

            # 去掉大小写、下划线和驼峰差异，例如 speed_level/speedLevel。
            normalised = {
                re.sub(r"[^a-z0-9]", "", str(key).lower()): value
                for key, value in payload.items()
            }
            known_fields = {
                "running", "run", "isrunning", "motorrunning", "enabled", "state", "status",
                "motorstate", "speed", "gear", "level", "spd", "speedlevel",
                "currentgear", "currentlevel", "speedindex", "omega", "omegadegs",
                "degs", "angularspeed", "direction", "dir", "motordirection",
                "pulsefreq", "pulsefreqhz", "pulsefrequency", "frequency", "freq",
            }

            def value(*keys: str, default=None):
                for key in keys:
                    signature = re.sub(r"[^a-z0-9]", "", key.lower())
                    if signature in normalised:
                        return normalised[signature]
                return default

            state_value = value("state", "status", "motor_state", default="")
            state = str(state_value).strip().lower()
            running_value = value(
                "running", "run", "is_running", "motor_running", "enabled",
                default=None,
            )
            if isinstance(running_value, str):
                running = running_value.strip().lower() in (
                    "1", "true", "on", "run", "running", "forward", "reverse",
                    "正转", "反转", "运行",
                )
            elif running_value is None:
                running = state in (
                    "1", "true", "on", "run", "running", "forward", "reverse",
                    "正转", "反转", "运行",
                )
            else:
                running = bool(running_value)

            def integer(*keys: str) -> int:
                raw_value = value(*keys, default=None)
                if raw_value is not None:
                    try:
                        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw_value))
                        return int(float(match.group(0))) if match else 0
                    except (TypeError, ValueError):
                        pass
                return 0

            direction = value("direction", "dir", "motor_direction", default="unknown")
            speed = integer(
                "speed", "gear", "level", "spd", "speed_level", "current_gear",
                "current_level", "speed_index",
            )
            result.update({
                "valid": bool(known_fields.intersection(normalised)),
                "running": running,
                "speed": speed,
                "omega": integer("omega", "omega_deg_s", "deg_s", "angular_speed"),
                "direction": MotorController._normalise_direction(direction),
                "pulse_freq": integer(
                    "pulse_freq", "pulse_freq_hz", "pulse_frequency", "frequency",
                    "freq", "pulse_freq_hz_value",
                ),
                "raw": root_payload,
            })
            gear = MOTOR_GEAR_TABLE.get(result["speed"])
            if gear:
                result["omega"] = result["omega"] or gear["omega_deg_s"]
                result["pulse_freq"] = result["pulse_freq"] or gear["pulse_freq_hz"]
            return result

        # 兼容旧控制器：RUN,SPD:5,OMEGA:630deg/s
        recognised = False
        for part in text.upper().split(","):
            part = part.strip()
            if part == "RUN":
                result["running"] = True
                recognised = True
            elif part in ("STOP", "STOPPED", "IDLE"):
                result["running"] = False
                recognised = True
            elif part.startswith("SPD:"):
                try:
                    result["speed"] = int(part.split(":", 1)[1])
                    recognised = True
                except ValueError:
                    pass
            elif part.startswith("OMEGA:"):
                try:
                    result["omega"] = int(
                        part.split(":", 1)[1].replace("DEG/S", ""))
                    recognised = True
                except ValueError:
                    pass
        result["valid"] = recognised
        if recognised:
            gear = MOTOR_GEAR_TABLE.get(result["speed"])
            if gear:
                result["omega"] = result["omega"] or gear["omega_deg_s"]
                result["pulse_freq"] = gear["pulse_freq_hz"]
        return result
