"""与 Tk 控件无关的应用关闭与硬件安全协调。"""
from __future__ import annotations

from dataclasses import dataclass

from src.hardware.command_queue import CommandResult, SerialCommandQueue


@dataclass(frozen=True)
class MotorShutdownReport:
    completed: bool
    stop_succeeded: bool
    error: str = ""


def shutdown_motor_safely(
    commands: SerialCommandQueue,
    controller,
    *,
    timeout: float = 3.0,
) -> MotorShutdownReport:
    """在队列线程中停车并关闭串口，等待动作真实结束。

    即使控制器此前因瞬态串口异常将 ``_connected`` 置为 False，
    只要串口句柄仍打开，就会尝试尽力写入停车命令。
    """
    if controller is None:
        commands.shutdown(timeout=timeout)
        return MotorShutdownReport(True, True)

    def stop_and_close() -> bool:
        try:
            if getattr(controller, "is_connected", True):
                return bool(controller.stop())
            # 控制器标记为未连接但串口可能仍可写：尽力停车
            # （try_stop_on_close 内部会关闭串口）
            try_stop = getattr(controller, "try_stop_on_close", None)
            if try_stop is not None:
                return try_stop()
            # 回退：直接关闭
            controller.close()
            return True
        except Exception:
            return False
        finally:
            # 正常连接路径下 stop() 不会关闭串口，这里兜底
            if getattr(controller, "is_connected", True):
                try:
                    controller.close()
                except Exception:
                    pass

    result: CommandResult | None = commands.shutdown(
        stop_and_close, timeout=timeout)
    if result is None:
        return MotorShutdownReport(False, False, "停车等待超时")
    if result.error is not None:
        return MotorShutdownReport(True, False, str(result.error))
    return MotorShutdownReport(True, bool(result.value))
