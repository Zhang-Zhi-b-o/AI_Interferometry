"""实验进度识别、人工确认和自动实验控制面板。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ExperimentWorkflowPanel(tk.LabelFrame):
    BG = "#ffffff"
    TEXT = "#10233f"
    MUTED = "#5f6f82"
    BLUE = "#1677ff"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="实验流程", bg=self.BG, fg=self.TEXT)
        self.on_command = lambda _command, _payload=None: None

        self.auto_var = tk.BooleanVar(value=False)
        self.instrument_var = tk.BooleanVar(value=False)
        self.white_light_var = tk.BooleanVar(value=False)
        self.scale_var = tk.StringVar(value="0.05")
        self.stage_var = tk.StringVar(value="当前阶段：人工调整仪器")
        self.next_var = tk.StringVar(value="下一步：用红光完成光路调整，然后点击确认。")
        self.progress_var = tk.IntVar(value=10)
        self.micrometer_var = tk.StringVar(value="视觉微分表：尚未启动")
        self.reading_var = tk.StringVar(value="读数 -- mm  │  动镜位移 -- mm")

        intro = tk.Label(
            self,
            text="人工只需完成两项：调整仪器、放置白光光源。其余步骤由程序自动推进。",
            bg=self.BG,
            fg=self.MUTED,
            justify="left",
            wraplength=430,
            anchor="w",
        )
        intro.pack(fill=tk.X, padx=8, pady=(8, 5))

        manual = tk.Frame(self, bg=self.BG)
        manual.pack(fill=tk.X, padx=8, pady=2)
        tk.Checkbutton(
            manual,
            text="仪器已用红光调整完成",
            variable=self.instrument_var,
            command=lambda: self._emit("instrument", self.instrument_var.get()),
            bg=self.BG,
            activebackground=self.BG,
            selectcolor=self.BG,
        ).pack(anchor="w")
        tk.Checkbutton(
            manual,
            text="白光光源已放置完成",
            variable=self.white_light_var,
            command=lambda: self._emit("white_light", self.white_light_var.get()),
            bg=self.BG,
            activebackground=self.BG,
            selectcolor=self.BG,
        ).pack(anchor="w")

        tk.Frame(self, bg="#e4e9f0", height=1).pack(fill=tk.X, padx=8, pady=6)

        meter = tk.Frame(self, bg=self.BG)
        meter.pack(fill=tk.X, padx=8)
        tk.Label(meter, text="微分表读数：独立摄像头 OCR", bg=self.BG,
                 fg=self.MUTED).pack(side=tk.LEFT)
        tk.Label(meter, text="放缩系数", bg=self.BG, fg=self.TEXT).pack(
            side=tk.LEFT, padx=(12, 0))
        tk.Entry(meter, textvariable=self.scale_var, width=7).pack(side=tk.LEFT, padx=5)

        tk.Label(
            self, textvariable=self.micrometer_var, bg=self.BG, fg=self.MUTED,
            anchor="w").pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(
            self, textvariable=self.reading_var, bg=self.BG, fg=self.TEXT,
            anchor="w", font=("Consolas", 9)).pack(fill=tk.X, padx=8, pady=(1, 5))

        tk.Frame(self, bg="#e4e9f0", height=1).pack(fill=tk.X, padx=8, pady=5)

        auto_row = tk.Frame(self, bg="#eef5ff")
        auto_row.pack(fill=tk.X, padx=8, pady=2, ipady=4)
        tk.Checkbutton(
            auto_row,
            text="自动进行实验",
            variable=self.auto_var,
            command=lambda: self._emit("auto", self.auto_var.get()),
            bg="#eef5ff",
            activebackground="#eef5ff",
            selectcolor="#eef5ff",
            fg=self.BLUE,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            auto_row,
            text="重新开始",
            command=lambda: self._emit("reset"),
            bg="#dbe9ff",
            fg=self.TEXT,
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        tk.Label(
            self, textvariable=self.stage_var, bg=self.BG, fg=self.BLUE,
            font=("Microsoft YaHei UI", 10, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 1))
        self.progress = ttk.Progressbar(
            self, maximum=100, variable=self.progress_var, mode="determinate")
        self.progress.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(
            self, textvariable=self.next_var, bg=self.BG, fg=self.MUTED,
            justify="left", wraplength=430, anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    def _emit(self, command: str, payload=None) -> None:
        self.on_command(command, payload)

    def get_scale_factor(self) -> float:
        try:
            value = float(self.scale_var.get())
            return value if value > 0 else 1.0
        except ValueError:
            return 1.0

    def update_workflow(self, decision) -> None:
        self.stage_var.set(f"当前阶段：{decision.title}")
        prefix = "异常：" if decision.warning else "下一步："
        message = decision.warning or decision.next_action
        self.next_var.set(prefix + message)
        self.progress_var.set(decision.progress)

    def update_micrometer(
        self,
        connected: bool,
        reading_mm: float | None,
        displacement_mm: float | None,
    ) -> None:
        self.micrometer_var.set(
            "视觉微分表：已接入" if connected else "视觉微分表：尚未启动（不阻塞自动实验）")
        reading = "--" if reading_mm is None else f"{reading_mm:.6f}"
        displacement = "--" if displacement_mm is None else f"{displacement_mm:.6f}"
        self.reading_var.set(
            f"读数 {reading} mm  │  动镜位移 {displacement} mm")

    def reset_controls(self) -> None:
        self.instrument_var.set(False)
        self.white_light_var.set(False)
        self.auto_var.set(False)
