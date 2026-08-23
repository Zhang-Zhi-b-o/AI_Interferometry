"""视频演示使用的精简实验控制侧边栏。"""
from __future__ import annotations

import tkinter as tk

from src.ui.theme import BORDER, FONT, MUTED, NAVY, PRIMARY, SURFACE, TEXT


class RecordingSidebar(tk.Frame):
    """只暴露实验演示所需的核心操作。"""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=SURFACE)
        self.on_command = lambda _command: None
        self.main_camera_index_var = tk.StringVar(value="1")
        self.reading_camera_index_var = tk.StringVar(value="0")
        self.motor_port_var = tk.StringVar(value="auto")
        self.search_direction_var = tk.StringVar(value="forward")
        self.status_var = tk.StringVar(value="准备就绪")
        self.meter_reading_var = tk.StringVar(value="读数：--")
        self._meter_photo = None
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
        self._camera_row(
            "第一相机（干涉画面）",
            self.main_camera_index_var, "camera_1")
        self._camera_row(
            "第二相机（读数画面）",
            self.reading_camera_index_var, "camera_2")
        self._motor_row()
        self.meter_preview = tk.Label(
            self, text="第二相机画面\n启动后在此显示读数",
            bg="#172033", fg="#dbeafe", height=9,
            anchor=tk.CENTER, compound=tk.CENTER,
            font=(FONT, 9),
        )
        self.meter_preview.pack(fill=tk.X, padx=12, pady=(5, 2))
        tk.Label(
            self, textvariable=self.meter_reading_var,
            bg="#f5f8fc", fg=NAVY, anchor="w",
            font=(FONT, 9, "bold"), padx=8, pady=5,
        ).pack(fill=tk.X, padx=12, pady=(0, 5))
        self._secondary_button("记录当前位置", "record_position")

        self._divider()
        self._section("02  YOLO 识别")
        self._primary_button("加载 YOLO 模型", "load_model")
        self._primary_button(
            "开始预测并检测中心条纹", "start_prediction")

        self._divider()
        self._section("03  自动寻找与寻中")
        direction_box = tk.Frame(self, bg="#eef5ff")
        direction_box.pack(fill=tk.X, padx=12, pady=(2, 7))
        tk.Label(
            direction_box, text="已知条纹方向", bg="#eef5ff", fg=NAVY,
            font=(FONT, 9, "bold"),
        ).pack(side=tk.LEFT, padx=(10, 8), pady=7)
        for text, value in (("正转", "forward"), ("反转", "reverse")):
            tk.Radiobutton(
                direction_box, text=text, value=value,
                variable=self.search_direction_var,
                bg="#eef5ff", activebackground="#eef5ff",
                selectcolor="#eef5ff", fg=NAVY,
                font=(FONT, 9),
            ).pack(side=tk.LEFT, padx=4)

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
            wraplength=360, padx=10, pady=8,
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

    def _camera_row(
        self, label: str, index_var: tk.StringVar, command: str,
    ) -> None:
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill=tk.X, padx=12, pady=3)
        tk.Button(
            row, text=f"启动{label}", command=lambda: self._emit(command),
            bg="#eaf2ff", fg="#1e4f87",
            activebackground="#dbeafe", activeforeground="#163f70",
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=(FONT, 9, "bold"), pady=8,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            row, text="索引", bg=SURFACE, fg=MUTED,
            font=(FONT, 8),
        ).pack(side=tk.LEFT, padx=(8, 3))
        tk.Spinbox(
            row, from_=0, to=20, textvariable=index_var,
            width=4, justify=tk.CENTER, font=(FONT, 9),
        ).pack(side=tk.RIGHT, fill=tk.Y)

    def _motor_row(self) -> None:
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill=tk.X, padx=12, pady=3)
        tk.Button(
            row, text="连接电机", command=lambda: self._emit("connect_motor"),
            bg="#eaf2ff", fg="#1e4f87",
            activebackground="#dbeafe", activeforeground="#163f70",
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=(FONT, 9, "bold"), pady=8,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            row, text="端口", bg=SURFACE, fg=MUTED,
            font=(FONT, 8),
        ).pack(side=tk.LEFT, padx=(8, 3))
        tk.Entry(
            row, textvariable=self.motor_port_var, width=9,
            justify=tk.CENTER, font=(FONT, 9),
        ).pack(side=tk.RIGHT, fill=tk.Y)

    @staticmethod
    def _camera_index(variable: tk.StringVar) -> int:
        try:
            return max(0, min(20, int(variable.get().strip())))
        except (ValueError, tk.TclError):
            return 0

    @property
    def main_camera_index(self) -> int:
        return self._camera_index(self.main_camera_index_var)

    @property
    def reading_camera_index(self) -> int:
        return self._camera_index(self.reading_camera_index_var)

    @property
    def motor_port(self) -> str:
        value = self.motor_port_var.get().strip()
        return value or "auto"

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

    def update_meter_preview(self, photo, reading_text: str) -> None:
        """显示第二相机预览；PhotoImage 由调用方在 Tk 主线程创建。"""
        if photo is not None:
            self._meter_photo = photo
            self.meter_preview.configure(image=photo, text="", height=0)
        self.meter_reading_var.set(reading_text or "读数：--")

    def reset_meter_preview(
        self, text: str = "第二相机画面\n启动后在此显示读数",
    ) -> None:
        self._meter_photo = None
        self.meter_preview.configure(image="", text=text, height=9)
        self.meter_reading_var.set("读数：--")
