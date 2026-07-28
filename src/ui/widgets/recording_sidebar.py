"""视频演示使用的精简实验控制侧边栏。"""
from __future__ import annotations

import tkinter as tk

from src.ui.theme import BORDER, FONT, MUTED, NAVY, PRIMARY, SURFACE, TEXT


class RecordingSidebar(tk.Frame):
    """只暴露实验演示所需的核心操作。"""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=SURFACE)
        self.on_command = lambda _command: None
        self.same_direction_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="准备就绪")
        self._build()

    def _build(self) -> None:
        tk.Label(
            self, text="实验快捷控制", bg=SURFACE, fg=TEXT,
            font=(FONT, 12, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(14, 2))
        tk.Label(
            self, text="视频录制模式 · 仅保留核心实验操作",
            bg=SURFACE, fg=MUTED, font=(FONT, 8), anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(0, 10))

        self._section("01  设备与读数")
        self._primary_button(
            "启动第一相机（干涉画面）", "camera_1")
        self._primary_button(
            "启动第二相机（读数画面）", "camera_2")
        self._secondary_button("记录当前位置", "record_position")

        self._divider()
        self._section("02  YOLO 识别")
        self._primary_button("加载 YOLO 模型", "load_model")
        self._primary_button(
            "开始预测并检测中心条纹", "start_prediction")

        self._divider()
        self._section("03  自动寻找与寻中")
        tk.Checkbutton(
            self, text="沿同一方向旋转",
            variable=self.same_direction_var,
            bg="#eef5ff", activebackground="#eef5ff",
            selectcolor="#eef5ff", fg=NAVY,
            font=(FONT, 9, "bold"), anchor="w",
            padx=10, pady=7,
        ).pack(fill=tk.X, padx=12, pady=(2, 7))

        actions = tk.Frame(self, bg=SURFACE)
        actions.pack(fill=tk.X, padx=12)
        tk.Button(
            actions, text="自动寻找条纹并寻中",
            command=lambda: self._emit("start_auto_center"),
            bg=PRIMARY, fg="#ffffff", activebackground="#1d4ed8",
            activeforeground="#ffffff", relief=tk.FLAT, bd=0,
            cursor="hand2", font=(FONT, 9, "bold"), pady=8,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(
            actions, text="停止",
            command=lambda: self._emit("stop_auto_center"),
            bg="#fee4e2", fg="#b42318", activebackground="#ffd5d2",
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=(FONT, 9, "bold"), width=7, pady=8,
        ).pack(side=tk.RIGHT)

        tk.Label(
            self, textvariable=self.status_var, bg="#f5f8fc", fg=NAVY,
            font=(FONT, 9), anchor="w", justify=tk.LEFT,
            wraplength=410, padx=10, pady=8,
        ).pack(fill=tk.X, padx=12, pady=(12, 10))

    def _section(self, text: str) -> None:
        tk.Label(
            self, text=text, bg=SURFACE, fg=NAVY,
            font=(FONT, 9, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(3, 6))

    def _divider(self) -> None:
        tk.Frame(self, bg=BORDER, height=1).pack(
            fill=tk.X, padx=12, pady=12)

    def _primary_button(self, text: str, command: str) -> None:
        tk.Button(
            self, text=text, command=lambda: self._emit(command),
            bg="#eaf2ff", fg="#1e4f87",
            activebackground="#dbeafe", activeforeground="#163f70",
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=(FONT, 9, "bold"), pady=8,
        ).pack(fill=tk.X, padx=12, pady=3)

    def _secondary_button(self, text: str, command: str) -> None:
        tk.Button(
            self, text=text, command=lambda: self._emit(command),
            bg="#eef2f7", fg=TEXT,
            activebackground="#e2e8f0", activeforeground=TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=(FONT, 9, "bold"), pady=8,
        ).pack(fill=tk.X, padx=12, pady=3)

    def _emit(self, command: str) -> None:
        self.on_command(command)

    def set_status(self, text: str) -> None:
        self.status_var.set(str(text))
