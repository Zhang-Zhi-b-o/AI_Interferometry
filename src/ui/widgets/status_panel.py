"""实时状态面板"""
from __future__ import annotations
import tkinter as tk

from src.ui.theme import (
    BORDER,
    DANGER,
    DANGER_SOFT,
    FONT,
    MUTED,
    PRIMARY,
    PRIMARY_SOFT,
    SUCCESS,
    SUCCESS_SOFT,
    SURFACE,
    TEXT,
    WARNING,
    WARNING_SOFT,
)


class StatusPanel(tk.LabelFrame):
    """实时显示：FPS、电机连接状态、电机转速、电机档位"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="实时状态", bg=SURFACE, fg=TEXT,
                         relief=tk.FLAT, bd=0)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.motor_connected_var = tk.StringVar(value="电机: 未连接")
        self.motor_speed_var = tk.StringVar(value="转速: --")
        self.motor_gear_var = tk.StringVar(value="档位: --")
        self._cards: list[tk.Frame] = []
        self._value_labels: list[tk.Label] = []

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
            label = tk.Label(
                card,
                textvariable=var,
                bg=PRIMARY_SOFT,
                fg=PRIMARY if index == 0 else MUTED,
                anchor="w",
                font=(FONT, 9, "bold" if index == 0 else "normal"),
            )
            label.pack(fill=tk.X, padx=9, pady=8)
            self._cards.append(card)
            self._value_labels.append(label)

    def _set_tone(self, index: int, foreground: str, background: str) -> None:
        self._cards[index].configure(bg=background)
        self._value_labels[index].configure(bg=background, fg=foreground)

    # ------------------------------------------------------------------
    # 更新方法
    # ------------------------------------------------------------------
    def update_fps(self, value: float):
        self.fps_var.set(f"FPS: {value:.1f}")
        if value <= 0:
            self._set_tone(0, MUTED, PRIMARY_SOFT)
        elif value < 10:
            self._set_tone(0, WARNING, WARNING_SOFT)
        else:
            self._set_tone(0, SUCCESS, SUCCESS_SOFT)

    def update_motor_connected(self, connected: bool, port: str = ""):
        if connected:
            self.motor_connected_var.set(f"电机: 已连接 ({port})")
            self._set_tone(1, SUCCESS, SUCCESS_SOFT)
        else:
            self.motor_connected_var.set("电机: 未连接")
            self._set_tone(1, DANGER, DANGER_SOFT)

    def update_motor_speed(self, omega: int):
        self.motor_speed_var.set(f"转速: {omega} deg/s")
        self._set_tone(2, PRIMARY if omega else MUTED, PRIMARY_SOFT)

    def update_motor_gear(self, gear: int):
        self.motor_gear_var.set(f"档位: {gear}")
        self._set_tone(3, PRIMARY, PRIMARY_SOFT)
