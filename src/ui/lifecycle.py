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
    """在队列线程中停车并关闭串口，等待动作真实结束。"""
    if controller is None:
        commands.shutdown(timeout=timeout)
        return MotorShutdownReport(True, True)

    def stop_and_close() -> bool:
        stopped = False
        try:
            stopped = bool(controller.stop())
            return stopped
        finally:
            controller.close()

    result: CommandResult | None = commands.shutdown(
        stop_and_close, timeout=timeout)
    if result is None:
        return MotorShutdownReport(False, False, "停车等待超时")
    if result.error is not None:
        return MotorShutdownReport(True, False, str(result.error))
    return MotorShutdownReport(True, bool(result.value))
