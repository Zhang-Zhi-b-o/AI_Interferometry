"""电机：包装 ``app.engine.hardware.motor.MotorController``，并把串口操作串行化。

串口命令必须一条一条发（旧程序吃过这个苦），所以这里只有**一个**工作线程
和一个命令队列；``stop()`` 作废待执行指令并请求停车，已在驱动中执行的 I/O 等其返回。
"""
from __future__ import annotations

import queue
import threading
from typing import Any


class MotionService:
    def __init__(self, *, on_status=None) -> None:
        self._on_status = on_status
        self.control_lock = threading.RLock()  # 仅保护状态；不能在此锁内执行串口 I/O
        self._io_lock = threading.RLock()
        self._controller: Any = None
        self._queue: queue.Queue = queue.Queue()
        self._epoch = 0
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self.connected = False
        self.gear: int | None = None
        self.direction = "stopped"

    # ------------------------------------------------------------------
    def connect(self, port: str = "auto", baudrate: int = 9600,
                *, cancel_event: threading.Event | None = None) -> tuple[bool, str]:
        if cancel_event is not None and cancel_event.is_set():
            return False, "连接已取消"
        self.disconnect()
        with self.control_lock:
            token = self._epoch
        controller = None
        try:
            from app.engine.hardware.motor import MotorController

            if cancel_event is not None and cancel_event.is_set():
                return False, "连接已取消"
            controller = MotorController(port=port, baudrate=int(baudrate))
            if not controller.connect():
                self._close_controller(controller)
                return False, f"电机连接失败（{port}）"
            with self.control_lock:
                cancelled = token != self._epoch or (cancel_event is not None and cancel_event.is_set())
                if not cancelled:
                    self._controller = controller
                    self.connected = True
                    self._running = threading.Event()
                    self._running.set()
                    self._queue = queue.Queue()
                    self._thread = threading.Thread(target=self._worker,
                                                    args=(self._running, self._queue),
                                                    name="motor", daemon=True)
                    self._thread.start()
            if cancelled:
                self._close_controller(controller)
                return False, "连接已取消"
        except Exception as exc:
            if controller is not None:
                self._close_controller(controller)
            return False, f"电机连接异常：{exc}"
        message = f"电机已连接（{getattr(controller, 'port', port)}）"
        self._status(message)
        return True, message

    @staticmethod
    def _close_controller(controller):
        try:
            if hasattr(controller, "try_stop_on_close"):
                controller.try_stop_on_close()
            else:
                controller.stop()
                controller.close()
        except Exception:
            pass

    def begin_disconnect(self):
        """立即禁止新闭环，耗时的串口关闭随后由后台执行。"""
        with self.control_lock:
            self.connected = False
            self.stop()

    def disconnect(self) -> None:
        with self.control_lock:
            self.begin_disconnect()
            self._running.clear()
            thread, self._thread = self._thread, None
            controller, self._controller = self._controller, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        if controller is not None:
            with self._io_lock:
                self._close_controller(controller)

    # ---- 对外动作：全部非阻塞，进队列 ----
    def set_speed(self, gear: int) -> None:
        with self.control_lock:
            gear = int(gear)
            if not 1 <= gear <= 10:
                raise ValueError("电机档位范围为 1–10")
            self.gear = gear
            self._post(("speed", gear))

    def forward(self) -> None:
        with self.control_lock:
            if self.direction != "forward":
                self.direction = "forward"
                self._post(("forward",))

    def reverse(self) -> None:
        with self.control_lock:
            if self.direction != "reverse":
                self.direction = "reverse"
                self._post(("reverse",))

    def stop(self) -> None:
        """清空待发命令；已出队但尚未执行的旧命令也必须失效。"""
        with self.control_lock:
            self.direction = "stopped"
            self._epoch += 1
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._post(("stop",))

    def query_status(self) -> dict:
        controller = self._controller
        if controller is None:
            return {}
        try:
            with self._io_lock:
                return dict(controller.query_status() or {})
        except Exception:
            return {}

    def toggle_direction(self):
        with self.control_lock:
            self.direction = "reverse" if self.direction == "forward" else "forward"
            self._post(("toggle_direction",))

    def speed_up(self):
        with self.control_lock:
            if self.gear is not None:
                self.gear = max(1, self.gear - 1)
            self._post(("speed_up",))

    def speed_down(self):
        with self.control_lock:
            if self.gear is not None:
                self.gear = min(10, self.gear + 1)
            self._post(("speed_down",))

    @staticmethod
    def available_ports():
        from serial.tools.list_ports import comports

        return [port.device for port in comports()]

    # ------------------------------------------------------------------
    def _post(self, command: tuple) -> None:
        if self._running.is_set() and self._controller is not None:
            self._queue.put((self._epoch, command))

    def _worker(self, running=None, messages=None) -> None:
        running = self._running if running is None else running
        messages = self._queue if messages is None else messages
        while running.is_set():
            try:
                epoch, command = messages.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._io_lock:
                with self.control_lock:
                    controller = self._controller
                    if (controller is None or epoch != self._epoch
                            or not running.is_set()):
                        continue
                    # 此处接受指令；状态锁随即释放，驱动阻塞不能冻结 UI。
                name = command[0]
                try:
                    if name == "speed":
                        ok = controller.set_speed(int(command[1]))
                    elif name == "forward":
                        ok = controller.start_forward()
                    elif name == "reverse":
                        ok = controller.start_reverse()
                    elif name == "stop":
                        ok = controller.stop()
                    elif name in {"toggle_direction", "speed_up", "speed_down"}:
                        ok = getattr(controller, name)()
                    else:
                        continue
                    if ok is False or not getattr(controller, "is_connected", True):
                        raise RuntimeError("串口命令未成功发送")
                except Exception as exc:
                    with self.control_lock:
                        if self._controller is controller:
                            self.connected = False
                            self._epoch += 1
                            self.direction = "stopped"
                        running.clear()
                    try:
                        if hasattr(controller, "try_stop_on_close"):
                            controller.try_stop_on_close()
                        else:
                            controller.stop()
                    except Exception:
                        pass
                    self._status(f"电机命令 {name} 失败，连接已标记失效：{exc}")

    def _status(self, text: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                pass
