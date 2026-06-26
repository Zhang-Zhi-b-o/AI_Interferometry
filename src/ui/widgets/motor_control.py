"""电机控制面板 — 手动 / 连续 / 步进 三种模式"""
from __future__ import annotations

import tkinter as tk


class MotorControlPanel(tk.LabelFrame):
    """电机控制：串口连接、模式切换、各模式独立参数"""

    UI_BG = "#ffffff"
    UI_TEXT = "#000000"
    UI_TEXT_SECONDARY = "#666666"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="电机控制", bg=self.UI_BG, fg=self.UI_TEXT)

        # 当前模式
        self._active_mode = "manual"

        # ---- Tk 变量 ----
        self.port_var = tk.StringVar(value="COM3")

        # 手动模式
        self.manual_speed_var = tk.StringVar(value="5")
        self.manual_auto_fix_var = tk.BooleanVar(value=False)

        # 连续模式
        self.cont_search_speed_var = tk.StringVar(value="10")
        self.cont_color_speed_var = tk.StringVar(value="5")
        self.cont_black_speed_var = tk.StringVar(value="10")
        self.cont_black_threshold_var = tk.StringVar(value="0.5")

        # 步进模式
        self.step_first_ms_var = tk.StringVar(value="1800")
        self.step_cycle_ms_var = tk.StringVar(value="1000")
        self.step_pause_ms_var = tk.StringVar(value="500")
        self.step_speed_var = tk.StringVar(value="5")
        self.step_black_threshold_var = tk.StringVar(value="0.5")

        # 状态显示
        self.auto_status_var = tk.StringVar(value="自动控制: 未启动")

        self._build()

    def _build(self):
        btn_cfg = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                       activebackground="#0b0b0b", cursor="hand2")
        mode_btn_cfg = dict(relief=tk.FLAT, bd=0, bg="#e6e6e6", fg=self.UI_TEXT,
                            activebackground="#d8d8d8", cursor="hand2")

        # -- 串口行 --
        port_row = tk.Frame(self, bg=self.UI_BG)
        port_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(port_row, text="串口号", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        self.port_menu = tk.OptionMenu(port_row, self.port_var, "COM3")
        self.port_menu.config(width=10)
        self.port_menu.pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(port_row, text="刷新串口", command=self._on_refresh,
                  relief=tk.FLAT, bd=0, bg="#444", fg="#fff",
                  activebackground="#333").pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(port_row, text="连接", command=self._on_connect,
                  **btn_cfg).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(port_row, text="断开", command=self._on_disconnect,
                  **btn_cfg).pack(side=tk.LEFT, padx=(8, 0))

        # -- 模式切换按钮 --
        mode_row = tk.Frame(self, bg=self.UI_BG)
        mode_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        for mode_key, label in [("manual", "手动转"), ("continuous", "边转边识别"), ("step", "短转后分析")]:
            btn = tk.Button(mode_row, text=label,
                            command=lambda m=mode_key: self._set_mode(m),
                            **mode_btn_cfg)
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        # -- 状态占位 --
        tk.Label(self, textvariable=self.auto_status_var, bg=self.UI_BG,
                 fg=self.UI_TEXT_SECONDARY, anchor="w").pack(fill=tk.X, padx=8, pady=(0, 4))

        # -- 模式容器 --
        self._mode_host = tk.Frame(self, bg=self.UI_BG)
        self._mode_host.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._manual_frame = self._build_manual_panel(self._mode_host, btn_cfg)
        self._continuous_frame = self._build_continuous_panel(self._mode_host, btn_cfg)
        self._step_frame = self._build_step_panel(self._mode_host, btn_cfg)

        self._set_mode("manual")

    # ==================================================================
    # 手动模式面板
    # ==================================================================
    def _build_manual_panel(self, parent, btn_cfg):
        frame = tk.Frame(parent, bg=self.UI_BG)

        tk.Label(frame, text="手动转动页面只显示手动速度和人工控制按钮，不会影响自动模式参数。",
                 bg=self.UI_BG, fg=self.UI_TEXT_SECONDARY, anchor="w", justify="left",
                 wraplength=420).pack(fill=tk.X, pady=(0, 6))

        # 速度
        sr = tk.Frame(frame, bg=self.UI_BG)
        sr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(sr, text="启动速度(1~10)", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.manual_speed_var, width=6).pack(side=tk.LEFT, padx=(8, 0))

        # 自动校准
        fr = tk.Frame(frame, bg=self.UI_BG)
        fr.pack(fill=tk.X, pady=(0, 6))
        tk.Checkbutton(fr, text="速度不一致时自动校准到预设",
                       variable=self.manual_auto_fix_var, onvalue=True, offvalue=False,
                       bg=self.UI_BG, fg=self.UI_TEXT, activebackground=self.UI_BG,
                       activeforeground=self.UI_TEXT, selectcolor=self.UI_BG).pack(side=tk.LEFT)
        tk.Button(fr, text="立即校准到预设", command=self._on_manual_calibrate,
                  **btn_cfg).pack(side=tk.LEFT, padx=(8, 0))

        # 按钮区
        pad = tk.Frame(frame, bg=self.UI_BG)
        pad.pack(fill=tk.X, pady=(0, 6))
        buttons = [
            (0, 1, "启动", "START"),
            (1, 0, "减速", "SPEED_DOWN"),
            (1, 1, "停止", "STOP"),
            (1, 2, "加速", "SPEED_UP"),
            (2, 1, "状态", "STATUS"),
        ]
        for row, col, label, cmd in buttons:
            tk.Button(pad, text=label, width=8,
                      command=lambda c=cmd: self._on_manual_cmd(c),
                      **btn_cfg).grid(row=row, column=col, padx=3, pady=3, sticky="ew")
        for i in range(3):
            pad.grid_columnconfigure(i, weight=1)

        return frame

    # ==================================================================
    # 连续模式面板
    # ==================================================================
    def _build_continuous_panel(self, parent, btn_cfg):
        frame = tk.Frame(parent, bg=self.UI_BG)

        tk.Label(frame, text="边转边识别：可分别设置无目标、彩条、黑条三个速度档位。",
                 bg=self.UI_BG, fg=self.UI_TEXT_SECONDARY, anchor="w", justify="left",
                 wraplength=420).pack(fill=tk.X, pady=(0, 6))

        sr = tk.Frame(frame, bg=self.UI_BG)
        sr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(sr, text="无目标速度", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.cont_search_speed_var, width=5).pack(side=tk.LEFT, padx=(6, 10))
        tk.Label(sr, text="彩条速度", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.cont_color_speed_var, width=5).pack(side=tk.LEFT, padx=(6, 10))
        tk.Label(sr, text="黑条速度", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.cont_black_speed_var, width=5).pack(side=tk.LEFT, padx=(6, 0))

        tr = tk.Frame(frame, bg=self.UI_BG)
        tr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(tr, text="停机阈值(%)", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(tr, textvariable=self.cont_black_threshold_var, width=6).pack(side=tk.LEFT, padx=(8, 10))
        tk.Button(tr, text="应用连续参数", command=self._on_apply_continuous,
                  **btn_cfg).pack(side=tk.LEFT)

        ar = tk.Frame(frame, bg=self.UI_BG)
        ar.pack(fill=tk.X, pady=(0, 6))
        tk.Button(ar, text="启动边转边识别", command=self._on_auto_start,
                  **btn_cfg).pack(side=tk.LEFT)
        tk.Button(ar, text="停止", command=self._on_auto_stop,
                  **btn_cfg).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(ar, text="查询状态", command=self._on_query_status,
                  **btn_cfg).pack(side=tk.LEFT, padx=(6, 0))

        return frame

    # ==================================================================
    # 步进模式面板
    # ==================================================================
    def _build_step_panel(self, parent, btn_cfg):
        frame = tk.Frame(parent, bg=self.UI_BG)

        tk.Label(frame, text="短转后分析：每次旋转一小段时间后暂停分析，循环执行。",
                 bg=self.UI_BG, fg=self.UI_TEXT_SECONDARY, anchor="w", justify="left",
                 wraplength=420).pack(fill=tk.X, pady=(0, 6))

        tr = tk.Frame(frame, bg=self.UI_BG)
        tr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(tr, text="首轮ms", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(tr, textvariable=self.step_first_ms_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(tr, text="循环ms", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(tr, textvariable=self.step_cycle_ms_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(tr, text="分析ms", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(tr, textvariable=self.step_pause_ms_var, width=6).pack(side=tk.LEFT, padx=(4, 0))

        sr = tk.Frame(frame, bg=self.UI_BG)
        sr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(sr, text="短转速度", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.step_speed_var, width=6).pack(side=tk.LEFT, padx=(8, 10))
        tk.Label(sr, text="停机阈值(%)", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(sr, textvariable=self.step_black_threshold_var, width=6).pack(side=tk.LEFT, padx=(8, 10))
        tk.Button(sr, text="应用短转参数", command=self._on_apply_step,
                  **btn_cfg).pack(side=tk.LEFT)

        ar = tk.Frame(frame, bg=self.UI_BG)
        ar.pack(fill=tk.X, pady=(0, 6))
        tk.Button(ar, text="启动短转后分析", command=self._on_auto_start,
                  **btn_cfg).pack(side=tk.LEFT)
        tk.Button(ar, text="停止", command=self._on_auto_stop,
                  **btn_cfg).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(ar, text="查询状态", command=self._on_query_status,
                  **btn_cfg).pack(side=tk.LEFT, padx=(6, 0))

        return frame

    # ==================================================================
    # 模式切换
    # ==================================================================
    def _set_mode(self, mode: str):
        self._active_mode = mode
        for f in [self._manual_frame, self._continuous_frame, self._step_frame]:
            f.pack_forget()
        if mode == "manual":
            self._manual_frame.pack(fill=tk.X)
        elif mode == "continuous":
            self._continuous_frame.pack(fill=tk.X)
        elif mode == "step":
            self._step_frame.pack(fill=tk.X)
        self.on_mode_change(mode)

    @property
    def mode(self) -> str:
        return self._active_mode

    # ==================================================================
    # 回调
    # ==================================================================
    def on_refresh_ports(self):
        pass

    def on_connect(self, port: str):
        pass

    def on_disconnect(self):
        pass

    def on_mode_change(self, mode: str):
        pass

    def on_manual_command(self, cmd: str):
        """cmd: START / STOP / SPEED_UP / SPEED_DOWN / STATUS"""
        pass

    def on_manual_calibrate(self, target_speed: int):
        pass

    def on_apply_continuous(self, search_speed: int, color_speed: int,
                            black_speed: int, threshold: float):
        pass

    def on_apply_step(self, first_ms: int, cycle_ms: int, pause_ms: int,
                      speed: int, threshold: float):
        pass

    def on_auto_start(self):
        pass

    def on_auto_stop(self):
        pass

    def on_query_status(self):
        pass

    # ==================================================================
    # 按钮事件
    # ==================================================================
    def _on_refresh(self):
        self.on_refresh_ports()

    def _on_connect(self):
        self.on_connect(self.port_var.get())

    def _on_disconnect(self):
        self.on_disconnect()

    def _on_manual_cmd(self, cmd: str):
        self.on_manual_command(cmd)

    def _on_manual_calibrate(self):
        try:
            speed = int(self.manual_speed_var.get())
            self.on_manual_calibrate(speed)
        except ValueError:
            pass

    def _on_apply_continuous(self):
        try:
            self.on_apply_continuous(
                int(self.cont_search_speed_var.get()),
                int(self.cont_color_speed_var.get()),
                int(self.cont_black_speed_var.get()),
                float(self.cont_black_threshold_var.get()),
            )
        except ValueError:
            pass

    def _on_apply_step(self):
        try:
            self.on_apply_step(
                int(self.step_first_ms_var.get()),
                int(self.step_cycle_ms_var.get()),
                int(self.step_pause_ms_var.get()),
                int(self.step_speed_var.get()),
                float(self.step_black_threshold_var.get()),
            )
        except ValueError:
            pass

    def _on_auto_start(self):
        self.auto_status_var.set("自动控制: 运行中")
        self.on_auto_start()

    def _on_auto_stop(self):
        self.auto_status_var.set("自动控制: 已停止")
        self.on_auto_stop()

    def _on_query_status(self):
        self.on_query_status()

    # ==================================================================
    # 更新方法
    # ==================================================================
    def update_ports(self, ports: list[str]):
        menu = self.port_menu["menu"]
        menu.delete(0, "end")
        for p in ports or ["COM3"]:
            menu.add_command(label=p, command=lambda v=p: self.port_var.set(v))

    def update_auto_status(self, text: str):
        self.auto_status_var.set(text)

    def get_manual_speed(self) -> int:
        try:
            return int(self.manual_speed_var.get())
        except ValueError:
            return 5

    def get_manual_auto_fix(self) -> bool:
        return self.manual_auto_fix_var.get()

    def get_continuous_params(self) -> dict:
        try:
            return {
                "search_speed": int(self.cont_search_speed_var.get()),
                "color_speed": int(self.cont_color_speed_var.get()),
                "black_speed": int(self.cont_black_speed_var.get()),
                "black_threshold": float(self.cont_black_threshold_var.get()),
            }
        except ValueError:
            return {"search_speed": 10, "color_speed": 5, "black_speed": 10, "black_threshold": 0.5}

    def get_step_params(self) -> dict:
        try:
            return {
                "first_ms": int(self.step_first_ms_var.get()),
                "cycle_ms": int(self.step_cycle_ms_var.get()),
                "pause_ms": int(self.step_pause_ms_var.get()),
                "speed": int(self.step_speed_var.get()),
                "black_threshold": float(self.step_black_threshold_var.get()),
            }
        except ValueError:
            return {"first_ms": 1800, "cycle_ms": 1000, "pause_ms": 500, "speed": 5, "black_threshold": 0.5}
