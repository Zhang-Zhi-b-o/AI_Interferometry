"""电机手动控制面板。"""
from __future__ import annotations

import tkinter as tk


class MotorControlPanel(tk.LabelFrame):
    """只提供串口连接和协议定义的手动命令。"""

    UI_BG = "#ffffff"
    UI_TEXT = "#10233f"
    UI_MUTED = "#64748b"

    def __init__(self, parent: tk.Widget, default_port: str = "自动检测"):
        super().__init__(parent, text="电机手动控制", bg=self.UI_BG, fg=self.UI_TEXT)
        self.port_var = tk.StringVar(value=default_port or "自动检测")
        self.preferred_port = default_port or "自动检测"
        self.command_status_var = tk.StringVar(value="等待连接电机")
        self.on_refresh_ports = lambda: None
        self.on_connect = lambda _port: None
        self.on_disconnect = lambda: None
        self.on_manual_command = lambda _command: None
        self._build()

    def _build(self) -> None:
        dark_button = dict(
            relief=tk.FLAT,
            bd=0,
            bg="#172033",
            fg="#ffffff",
            activebackground="#26344e",
            activeforeground="#ffffff",
            cursor="hand2",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        light_button = dict(
            relief=tk.FLAT,
            bd=0,
            bg="#e8eef6",
            fg=self.UI_TEXT,
            activebackground="#d8e3f0",
            cursor="hand2",
        )

        port_row = tk.Frame(self, bg=self.UI_BG)
        port_row.pack(fill=tk.X, padx=8, pady=(7, 5))
        tk.Label(port_row, text="串口", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        self.port_menu = tk.OptionMenu(port_row, self.port_var, self.port_var.get())
        self.port_menu.configure(width=8)
        self.port_menu.pack(side=tk.LEFT, padx=(6, 5))
        tk.Button(port_row, text="刷新", command=lambda: self.on_refresh_ports(),
                  **light_button).pack(side=tk.LEFT, padx=2)
        tk.Button(port_row, text="连接", command=self._connect,
                  **dark_button).pack(side=tk.LEFT, padx=2)
        tk.Button(port_row, text="断开", command=lambda: self.on_disconnect(),
                  **light_button).pack(side=tk.LEFT, padx=2)

        tk.Label(
            self,
            text="R 正转  ·  r 反转  ·  S 停止  ·  D 运行中换向  ·  + 加速  ·  - 减速",
            bg=self.UI_BG,
            fg=self.UI_MUTED,
            anchor="w",
            justify="left",
            wraplength=430,
        ).pack(fill=tk.X, padx=8, pady=(2, 6))

        controls = tk.Frame(self, bg=self.UI_BG)
        controls.pack(fill=tk.X, padx=8, pady=2)
        buttons = [
            (0, 0, "正转启动\nR", "FORWARD", "#126c49"),
            (0, 1, "反转启动\nr", "REVERSE", "#155f91"),
            (1, 0, "停止\nS", "STOP", "#b42318"),
            (1, 1, "运行中换向\nD", "TOGGLE_DIRECTION", "#7a4e00"),
            (2, 0, "加速\n+", "SPEED_UP", "#172033"),
            (2, 1, "减速\n-", "SPEED_DOWN", "#172033"),
        ]
        for row, column, label, command, colour in buttons:
            tk.Button(
                controls,
                text=label,
                command=lambda value=command: self._send(value),
                bg=colour,
                fg="#ffffff",
                activebackground=colour,
                activeforeground="#ffffff",
                relief=tk.FLAT,
                bd=0,
                cursor="hand2",
                font=("Microsoft YaHei UI", 9, "bold"),
                height=2,
            ).grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        tk.Button(
            self,
            text="查询 JSON 状态",
            command=lambda: self._send("STATUS"),
            **light_button,
        ).pack(fill=tk.X, padx=11, pady=(4, 5), ipady=3)
        tk.Label(
            self,
            textvariable=self.command_status_var,
            bg=self.UI_BG,
            fg=self.UI_MUTED,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

    @property
    def mode(self) -> str:
        """兼容智能体上下文；插件始终为手动模式。"""
        return "manual"

    def _connect(self) -> None:
        self.on_connect(self.port_var.get())

    def _send(self, command: str) -> None:
        descriptions = {
            "FORWARD": "已发送正转启动命令 R",
            "REVERSE": "已发送反转启动命令 r",
            "STOP": "已发送停止命令 S",
            "TOGGLE_DIRECTION": "已发送运行中换向命令 D",
            "SPEED_UP": "已发送加速命令 +",
            "SPEED_DOWN": "已发送减速命令 -",
            "STATUS": "正在查询 JSON 状态",
        }
        self.command_status_var.set(descriptions.get(command, command))
        self.on_manual_command(command)

    def update_ports(self, ports: list[str]) -> None:
        menu = self.port_menu["menu"]
        menu.delete(0, "end")
        choices = ports or ["未检测到串口"]
        for port in choices:
            menu.add_command(label=port, command=lambda value=port: self.port_var.set(value))
        current = self.port_var.get()
        if current not in ports:
            if self.preferred_port in ports:
                self.port_var.set(self.preferred_port)
            elif len(ports) == 1:
                self.port_var.set(ports[0])
            elif not ports:
                self.port_var.set("未检测到串口")

    def update_command_status(self, text: str) -> None:
        self.command_status_var.set(text)
