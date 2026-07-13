"""实时状态面板"""
from __future__ import annotations
import tkinter as tk

from src.ui.theme import BORDER, FONT, MUTED, PRIMARY, PRIMARY_SOFT, SURFACE, TEXT


class StatusPanel(tk.LabelFrame):
    """实时显示：FPS、电机连接状态、电机转速、电机档位"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="实时状态", bg=SURFACE, fg=TEXT,
                         relief=tk.FLAT, bd=0)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.motor_connected_var = tk.StringVar(value="电机: 未连接")
        self.motor_speed_var = tk.StringVar(value="转速: --")
        self.motor_gear_var = tk.StringVar(value="档位: --")

        self._build()

    def _build(self):
        grid = tk.Frame(self, bg=SURFACE)
        grid.pack(fill=tk.X, padx=4, pady=4)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        for index, var in enumerate([
            self.fps_var, self.motor_connected_var,
            self.motor_speed_var, self.motor_gear_var,
        ]):
            card = tk.Frame(grid, bg=PRIMARY_SOFT, highlightthickness=1,
                            highlightbackground=BORDER)
            card.grid(row=index // 2, column=index % 2, sticky="ew",
                      padx=3, pady=3)
            tk.Label(card, textvariable=var, bg=PRIMARY_SOFT,
                     fg=PRIMARY if index == 0 else MUTED, anchor="w",
                     font=(FONT, 9, "bold" if index == 0 else "normal")).pack(
                         fill=tk.X, padx=9, pady=8)

    # ------------------------------------------------------------------
    # 更新方法
    # ------------------------------------------------------------------
    def update_fps(self, value: float):
        self.fps_var.set(f"FPS: {value:.1f}")

    def update_motor_connected(self, connected: bool, port: str = ""):
        if connected:
            self.motor_connected_var.set(f"电机: 已连接 ({port})")
        else:
            self.motor_connected_var.set("电机: 未连接")

    def update_motor_speed(self, omega: int):
        self.motor_speed_var.set(f"转速: {omega} deg/s")

    def update_motor_gear(self, gear: int):
        self.motor_gear_var.set(f"档位: {gear}")
