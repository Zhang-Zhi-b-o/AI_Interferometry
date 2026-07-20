"""自动实验页中的设备选择和画面调试面板。"""
from __future__ import annotations

import tkinter as tk


class AutomaticDeviceDebugPanel(tk.LabelFrame):
    BG = "#ffffff"
    TEXT = "#10233f"
    MUTED = "#64748b"
    BLUE = "#1677ff"

    def __init__(self, parent: tk.Widget, *, camera_index: int = 1,
                 micrometer_index: int = 0, motor_port: str = "auto"):
        super().__init__(parent, text="自动实验设备调试", bg=self.BG, fg=self.TEXT)
        self.on_command = lambda _command, _payload=None: None
        self.camera_var = tk.StringVar(value=str(camera_index))
        self.micrometer_var = tk.StringVar(value=str(micrometer_index))
        self.motor_var = tk.StringVar(value=motor_port)
        self.zoom_var = tk.StringVar(value="2.0")
        self.angle_var = tk.StringVar(value="0")
        self.roi_mode_var = tk.BooleanVar(value=False)
        self.pan_mode_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="可在这里完成自动实验前的设备与画面调试")
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text="选择自动实验使用的两台摄像头和电机串口；所有设置与手动页面同步。",
            bg=self.BG, fg=self.MUTED, anchor="w", justify="left",
        ).pack(fill=tk.X, padx=10, pady=(8, 5))

        devices = tk.Frame(self, bg=self.BG)
        devices.pack(fill=tk.X, padx=8, pady=3)
        for column in range(3):
            devices.columnconfigure(column, weight=1)
        self._selector(devices, 0, "干涉相机", self.camera_var)
        self._selector(devices, 1, "微分表相机", self.micrometer_var)
        self._selector(devices, 2, "电机串口", self.motor_var)

        row = tk.Frame(self, bg=self.BG)
        row.pack(fill=tk.X, padx=8, pady=4)
        for text, command, colour in (
            ("检测设备", "detect", "#e8f1ff"),
            ("连接主相机", "camera_open", "#e8f1ff"),
            ("关闭主相机", "camera_close", "#f1f5f9"),
            ("启动微分表", "meter_start", "#ecfdf3"),
            ("停止微分表", "meter_stop", "#f1f5f9"),
            ("连接电机", "motor_connect", "#fff8e8"),
            ("断开电机", "motor_disconnect", "#f1f5f9"),
        ):
            tk.Button(
                row, text=text, command=lambda value=command: self._emit(value),
                bg=colour, fg=self.TEXT, relief=tk.FLAT, cursor="hand2",
                padx=8, pady=4,
            ).pack(side=tk.LEFT, padx=2)

        model_row = tk.Frame(self, bg=self.BG)
        model_row.pack(fill=tk.X, padx=8, pady=(2, 6))
        for text, command in (
            ("加载识别模型", "model_load"),
            ("开始实时识别", "model_start"),
            ("停止实时识别", "model_stop"),
        ):
            tk.Button(
                model_row, text=text,
                command=lambda value=command: self._emit(value),
                bg="#172033", fg="#ffffff", relief=tk.FLAT,
                cursor="hand2", padx=10, pady=4,
            ).pack(side=tk.LEFT, padx=2)

        tk.Frame(self, bg="#e4e9f0", height=1).pack(fill=tk.X, padx=8, pady=4)
        image_row = tk.Frame(self, bg=self.BG)
        image_row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(image_row, text="缩放", bg=self.BG, fg=self.TEXT).pack(side=tk.LEFT)
        tk.Spinbox(
            image_row, from_=1.0, to=8.0, increment=0.1,
            textvariable=self.zoom_var, width=6, format="%.1f",
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(image_row, text="应用", command=lambda: self._emit("zoom_apply"),
                  bg="#e8f1ff", relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(image_row, text="恢复1倍", command=lambda: self._emit("zoom_reset"),
                  bg="#f1f5f9", relief=tk.FLAT).pack(side=tk.LEFT, padx=(3, 12))
        tk.Label(image_row, text="旋转角度", bg=self.BG, fg=self.TEXT).pack(side=tk.LEFT)
        tk.Entry(image_row, textvariable=self.angle_var, width=6).pack(side=tk.LEFT, padx=4)
        tk.Button(image_row, text="应用", command=lambda: self._emit("angle_apply"),
                  bg="#e8f1ff", relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(image_row, text="归零", command=lambda: self._emit("angle_reset"),
                  bg="#f1f5f9", relief=tk.FLAT).pack(side=tk.LEFT, padx=3)

        interaction = tk.Frame(self, bg=self.BG)
        interaction.pack(fill=tk.X, padx=8, pady=4)
        tk.Checkbutton(
            interaction, text="在左侧预览拖拽画 ROI", variable=self.roi_mode_var,
            command=lambda: self._emit("roi_toggle", self.roi_mode_var.get()),
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT)
        tk.Button(interaction, text="清除 ROI", command=lambda: self._emit("roi_clear"),
                  bg="#f1f5f9", relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            interaction, text="拖拽移动画面", variable=self.pan_mode_var,
            command=lambda: self._emit("pan_toggle", self.pan_mode_var.get()),
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT, padx=(10, 2))
        for text, dx, dy in (
            ("←", -40, 0), ("→", 40, 0), ("↑", 0, -40), ("↓", 0, 40),
        ):
            tk.Button(
                interaction, text=text,
                command=lambda x=dx, y=dy: self._emit("pan_step", (x, y)),
                bg="#e8f1ff", fg=self.BLUE, relief=tk.FLAT, width=3,
            ).pack(side=tk.LEFT, padx=1)
        tk.Button(interaction, text="复位移动", command=lambda: self._emit("pan_reset"),
                  bg="#f1f5f9", relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        tk.Label(self, textvariable=self.status_var, bg=self.BG, fg=self.BLUE,
                 anchor="w").pack(fill=tk.X, padx=10, pady=(3, 8))

    def _selector(self, parent: tk.Widget, column: int, title: str,
                  variable: tk.StringVar) -> None:
        box = tk.Frame(parent, bg=self.BG)
        box.grid(row=0, column=column, sticky="ew", padx=3)
        tk.Label(box, text=title, bg=self.BG, fg=self.MUTED).pack(anchor="w")
        tk.Entry(box, textvariable=variable).pack(fill=tk.X)

    def _emit(self, command: str, payload=None) -> None:
        self.on_command(command, payload)

    def set_camera_list(self, cameras: list[int]) -> None:
        self.status_var.set(
            "可用摄像头：" + ("、".join(str(value) for value in cameras)
                          if cameras else "未检测到"))

    def set_motor_ports(self, ports: list[str]) -> None:
        if len(ports) == 1:
            self.motor_var.set(ports[0])
        suffix = "、".join(ports) if ports else "未检测到"
        self.status_var.set(f"可用电机串口：{suffix}")

    @property
    def camera_index(self) -> int:
        try:
            return max(0, int(self.camera_var.get()))
        except ValueError:
            return 0

    @property
    def micrometer_index(self) -> int:
        try:
            return max(0, int(self.micrometer_var.get()))
        except ValueError:
            return 1

    @property
    def zoom(self) -> float:
        try:
            return max(1.0, min(8.0, float(self.zoom_var.get())))
        except ValueError:
            return 2.0

    @property
    def angle(self) -> float:
        try:
            return max(-180.0, min(180.0, float(self.angle_var.get())))
        except ValueError:
            return 0.0
