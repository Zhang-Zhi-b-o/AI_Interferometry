"""电机自动旋转与中心条纹闭环定位面板。"""
from __future__ import annotations

import tkinter as tk


class AutoCenterControlPanel(tk.LabelFrame):
    """配置并启动基于中心条纹位置的双向自动控制。"""

    BG = "#ffffff"
    TEXT = "#10233f"
    MUTED = "#64748b"
    BLUE = "#1677ff"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="电机自动寻中", bg=self.BG, fg=self.TEXT)
        self.on_command = lambda _command: None
        self.fast_gear_var = tk.IntVar(value=9)
        self.slow_gear_var = tk.IntVar(value=10)
        self.search_gear_var = tk.IntVar(value=9)
        self.slow_zone_var = tk.StringVar(value="160")
        self.tolerance_var = tk.StringVar(value="15")
        self.stable_frames_var = tk.StringVar(value="5")
        self.search_mode_var = tk.StringVar(value="bidirectional")
        self.search_direction_var = tk.StringVar(value="forward")
        self.invert_direction_var = tk.BooleanVar(value=False)
        self.auto_learn_direction_var = tk.BooleanVar(value=True)
        self.show_center_line_var = tk.BooleanVar(value=True)
        self.learning_delta_px = 8.0
        self.dropout_hold_frames = 3
        self.center_confirm_frames = 3
        self.command_refresh_frames = 10
        self.guide_min_confidence = 0.2
        self.guide_loss_confirm_frames = 10
        self.search_initial_span_var = tk.StringVar(value="6")
        self.search_expansion_var = tk.StringVar(value="1.6")
        self.search_max_span_var = tk.StringVar(value="0")
        self.search_min_gear_var = tk.IntVar(value=9)
        self.search_acceleration_step_var = tk.IntVar(value=0)
        self.blur_slowdown_frames = 3
        self.blur_safe_gear = 10
        self.blur_recovery_clear_frames = 5
        self.stop_detect_move_seconds = 0.6
        self.stop_detect_settle_seconds = 0.3
        self.stop_detect_frames = 2
        self.guide_worsening_px = 12.0
        self.guide_trend_window = 8
        self.guide_focus_confirm_frames = 3
        self.guide_focus_shift_ratio = 0.5
        self.guide_focus_min_shift_turns = 1.0
        self.guide_focus_max_shift_turns = 12.0
        self.status_var = tk.StringVar(value="自动寻中未启动")
        self.position_var = tk.StringVar(value="中心位置 --  │  目标 --  │  偏差 --")
        self.scene_analysis_var = tk.StringVar(value="画面分析：尚未检测")
        self.clarity_var = tk.StringVar(value="清晰度增强：未启动")
        self.search_range_var = tk.StringVar(value="搜索范围：尚未启动")
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text="自动旋转搜索中心条纹；识别后通过正反转将其移动到画面中央，接近目标时自动减速。",
            bg=self.BG, fg=self.MUTED, justify="left", anchor="w", wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(8, 6))

        grid = tk.Frame(self, bg=self.BG)
        grid.pack(fill=tk.X, padx=8, pady=2)
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        self._gear_field(grid, 0, "搜索档位", self.search_gear_var)
        self._gear_field(grid, 1, "快速档位", self.fast_gear_var)
        self._gear_field(grid, 2, "接近档位", self.slow_gear_var)

        numeric = tk.Frame(self, bg=self.BG)
        numeric.pack(fill=tk.X, padx=8, pady=3)
        for column in range(3):
            numeric.columnconfigure(column, weight=1)
        self._entry_field(numeric, 0, "减速距离(px)", self.slow_zone_var)
        self._entry_field(numeric, 1, "中心容差(px)", self.tolerance_var)
        self._entry_field(numeric, 2, "稳定帧数", self.stable_frames_var)

        search_range = tk.Frame(self, bg=self.BG)
        search_range.pack(fill=tk.X, padx=8, pady=3)
        for column in range(3):
            search_range.columnconfigure(column, weight=1)
        self._entry_field(
            search_range, 0, "初始范围(圈)", self.search_initial_span_var)
        self._entry_field(
            search_range, 1, "扩大倍数", self.search_expansion_var)
        self._entry_field(
            search_range, 2, "最大范围(0=不限)", self.search_max_span_var)

        acceleration = tk.Frame(self, bg=self.BG)
        acceleration.pack(fill=tk.X, padx=8, pady=3)
        for column in range(3):
            acceleration.columnconfigure(column, weight=1)
        self._gear_field(
            acceleration, 0, "最快搜索档位", self.search_min_gear_var)
        self._entry_field(
            acceleration, 1, "每轮加速档数", self.search_acceleration_step_var)
        tk.Label(
            acceleration, text="默认不逐轮加速，避免旋转时画面模糊",
            bg=self.BG, fg=self.MUTED, anchor="w", justify="left",
            wraplength=130,
        ).grid(row=0, column=2, sticky="ew", padx=3)

        strategy = tk.Frame(self, bg=self.BG)
        strategy.pack(fill=tk.X, padx=8, pady=(3, 1))
        tk.Label(strategy, text="搜索方案", bg=self.BG, fg=self.TEXT).pack(side=tk.LEFT)
        tk.Radiobutton(
            strategy, text="双向扩展搜索", value="bidirectional",
            variable=self.search_mode_var, bg=self.BG,
            activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT, padx=(5, 0))
        tk.Radiobutton(
            strategy, text="已知方向，单向找条纹", value="single_direction",
            variable=self.search_mode_var, bg=self.BG,
            activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT, padx=(5, 0))
        tk.Radiobutton(
            strategy, text="转停识别", value="stop_and_detect",
            variable=self.search_mode_var, bg=self.BG,
            activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT, padx=(5, 0))

        options = tk.Frame(self, bg=self.BG)
        options.pack(fill=tk.X, padx=8, pady=(3, 5))
        tk.Label(options, text="初始/固定方向", bg=self.BG, fg=self.TEXT).pack(side=tk.LEFT)
        tk.Radiobutton(
            options, text="正转", value="forward", variable=self.search_direction_var,
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT, padx=(5, 0))
        tk.Radiobutton(
            options, text="反转", value="reverse", variable=self.search_direction_var,
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            options, text="手动反向", variable=self.invert_direction_var,
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.RIGHT)

        tk.Checkbutton(
            self,
            text="自动判断正转/反转方向（推荐）",
            variable=self.auto_learn_direction_var,
            bg="#eef5ff", activebackground="#eef5ff", selectcolor="#eef5ff",
            fg=self.BLUE, anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 5), ipady=3)

        tk.Checkbutton(
            self,
            text="始终显示画面中心线",
            variable=self.show_center_line_var,
            command=lambda: self.on_command("toggle_center_line"),
            bg="#eefbf7", activebackground="#eefbf7", selectcolor="#eefbf7",
            fg="#087f5b", anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 5), ipady=3)

        tk.Label(
            self, text=("档位 1 最快，档位 10 最慢；单向方案只限定找到条纹前的"
                        "搜索方向，找到后使用原有闭环方法移到中心，丢失后再搜索。"),
            bg="#fff8e8", fg="#7a4e00", anchor="w", justify="left", wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(0, 6), ipady=4)

        buttons = tk.Frame(self, bg=self.BG)
        buttons.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(
            buttons, text="开始自动旋转并寻中", command=lambda: self.on_command("start"),
            bg=self.BLUE, fg="#ffffff", activebackground="#0f62d6",
            activeforeground="#ffffff", relief=tk.FLAT, bd=0, cursor="hand2",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=5)
        tk.Button(
            buttons, text="停止", command=lambda: self.on_command("stop"),
            bg="#fee4e2", fg="#b42318", activebackground="#ffd5d2",
            relief=tk.FLAT, bd=0, cursor="hand2", width=9,
        ).pack(side=tk.RIGHT, ipady=5)

        tk.Label(
            self, textvariable=self.status_var, bg=self.BG, fg=self.BLUE,
            font=("Microsoft YaHei UI", 9, "bold"), anchor="w", justify="left",
            wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(6, 1))
        tk.Label(
            self, textvariable=self.position_var, bg=self.BG, fg=self.MUTED,
            font=("Consolas", 9), anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            self, textvariable=self.scene_analysis_var, bg=self.BG, fg=self.MUTED,
            font=("Microsoft YaHei UI", 9), anchor="w", justify="left",
            wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            self, textvariable=self.clarity_var, bg=self.BG, fg=self.MUTED,
            font=("Consolas", 9), anchor="w", justify="left", wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            self, textvariable=self.search_range_var, bg=self.BG, fg=self.MUTED,
            font=("Consolas", 9), anchor="w", justify="left", wraplength=420,
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

    def _gear_field(self, parent: tk.Widget, column: int, label: str, variable) -> None:
        box = tk.Frame(parent, bg=self.BG)
        box.grid(row=0, column=column, sticky="ew", padx=3)
        tk.Label(box, text=label, bg=self.BG, fg=self.MUTED).pack(anchor="w")
        tk.Spinbox(box, from_=1, to=10, textvariable=variable, width=7,
                   justify="center").pack(fill=tk.X)

    def _entry_field(self, parent: tk.Widget, column: int, label: str, variable) -> None:
        box = tk.Frame(parent, bg=self.BG)
        box.grid(row=0, column=column, sticky="ew", padx=3)
        tk.Label(box, text=label, bg=self.BG, fg=self.MUTED).pack(anchor="w")
        tk.Entry(box, textvariable=variable, justify="center").pack(fill=tk.X)

    def load_settings(self, settings: dict) -> None:
        self.search_gear_var.set(int(settings.get("search_gear", 9)))
        self.fast_gear_var.set(int(settings.get("fast_gear", 9)))
        self.slow_gear_var.set(int(settings.get("slow_gear", 10)))
        self.slow_zone_var.set(str(settings.get("slow_zone_px", 160)))
        self.tolerance_var.set(str(settings.get("tolerance_px", 15)))
        self.stable_frames_var.set(str(settings.get("stable_frames", 5)))
        self.search_direction_var.set(str(settings.get("search_direction", "forward")))
        self.search_mode_var.set(str(settings.get("search_mode", "bidirectional")))
        self.invert_direction_var.set(bool(settings.get("invert_direction", False)))
        self.auto_learn_direction_var.set(
            bool(settings.get("auto_learn_direction", True)))
        self.show_center_line_var.set(
            bool(settings.get("show_center_line", True)))
        self.learning_delta_px = self._bounded_float(
            settings.get("learning_delta_px", 8), 1, 100, 8)
        self.dropout_hold_frames = self._bounded_int(
            settings.get("dropout_hold_frames", 3), 0, 30, 3)
        self.center_confirm_frames = self._bounded_int(
            settings.get("center_confirm_frames", 3), 1, 30, 3)
        self.command_refresh_frames = self._bounded_int(
            settings.get("command_refresh_frames", 10), 1, 100, 10)
        self.guide_min_confidence = self._bounded_float(
            settings.get("guide_min_confidence", 0.2), 0, 1, 0.2)
        self.guide_loss_confirm_frames = self._bounded_int(
            settings.get("guide_loss_confirm_frames", 10), 1, 60, 10)
        self.search_initial_span_var.set(str(
            settings.get("search_initial_span_turns", 6)))
        self.search_expansion_var.set(str(
            settings.get("search_expansion_factor", 1.6)))
        self.search_max_span_var.set(str(
            settings.get("search_max_span_turns", 0)))
        self.search_min_gear_var.set(int(settings.get("search_min_gear", 9)))
        self.search_acceleration_step_var.set(int(
            settings.get("search_acceleration_step", 0)))
        self.blur_slowdown_frames = self._bounded_int(
            settings.get("blur_slowdown_frames", 3), 1, 60, 3)
        self.blur_safe_gear = self._bounded_int(
            settings.get("blur_safe_gear", 10), 1, 10, 10)
        self.blur_recovery_clear_frames = self._bounded_int(
            settings.get("blur_recovery_clear_frames", 5), 1, 60, 5)
        self.stop_detect_move_seconds = self._bounded_float(
            settings.get("stop_detect_move_seconds", 0.6), 0.05, 10, 0.6)
        self.stop_detect_settle_seconds = self._bounded_float(
            settings.get("stop_detect_settle_seconds", 0.3), 0.05, 10, 0.3)
        self.stop_detect_frames = self._bounded_int(
            settings.get("stop_detect_frames", 2), 1, 30, 2)
        self.guide_worsening_px = self._bounded_float(
            settings.get("guide_worsening_px", 12), 1, 500, 12)
        self.guide_trend_window = self._bounded_int(
            settings.get("guide_trend_window", 8), 6, 30, 8)
        self.guide_focus_confirm_frames = self._bounded_int(
            settings.get("guide_focus_confirm_frames", 3), 1, 30, 3)
        self.guide_focus_shift_ratio = self._bounded_float(
            settings.get("guide_focus_shift_ratio", 0.5), 0.05, 2, 0.5)
        self.guide_focus_min_shift_turns = self._bounded_float(
            settings.get("guide_focus_min_shift_turns", 1), 0.1, 100, 1)
        self.guide_focus_max_shift_turns = self._bounded_float(
            settings.get("guide_focus_max_shift_turns", 12),
            self.guide_focus_min_shift_turns, 1000, 12)

    def get_params(self) -> dict:
        initial_span = self._bounded_float(
            self.search_initial_span_var.get(), 0.1, 100, 6)
        return {
            "search_gear": self._bounded_int(self.search_gear_var.get(), 1, 10, 9),
            "fast_gear": self._bounded_int(self.fast_gear_var.get(), 1, 10, 9),
            "slow_gear": self._bounded_int(self.slow_gear_var.get(), 1, 10, 10),
            "slow_zone_px": self._bounded_float(self.slow_zone_var.get(), 10, 2000, 160),
            "tolerance_px": self._bounded_float(self.tolerance_var.get(), 1, 500, 15),
            "stable_frames": self._bounded_int(self.stable_frames_var.get(), 1, 100, 5),
            "search_direction": self.search_direction_var.get(),
            "search_mode": self.search_mode_var.get(),
            "invert_direction": self.invert_direction_var.get(),
            "auto_learn_direction": self.auto_learn_direction_var.get(),
            "show_center_line": self.show_center_line_var.get(),
            "learning_delta_px": self.learning_delta_px,
            "dropout_hold_frames": self.dropout_hold_frames,
            "center_confirm_frames": self.center_confirm_frames,
            "command_refresh_frames": self.command_refresh_frames,
            "guide_min_confidence": self.guide_min_confidence,
            "guide_loss_confirm_frames": self.guide_loss_confirm_frames,
            "search_initial_span_turns": initial_span,
            "search_expansion_factor": self._bounded_float(
                self.search_expansion_var.get(), 1.1, 3, 1.6),
            "search_max_span_turns": self._bounded_float(
                self.search_max_span_var.get(), 0, 1_000_000_000, 0),
            "search_min_gear": self._bounded_int(
                self.search_min_gear_var.get(), 1, 10, 9),
            "search_acceleration_step": self._bounded_int(
                self.search_acceleration_step_var.get(), 0, 3, 0),
            "blur_slowdown_frames": self.blur_slowdown_frames,
            "blur_safe_gear": self.blur_safe_gear,
            "blur_recovery_clear_frames": self.blur_recovery_clear_frames,
            "stop_detect_move_seconds": self.stop_detect_move_seconds,
            "stop_detect_settle_seconds": self.stop_detect_settle_seconds,
            "stop_detect_frames": self.stop_detect_frames,
            "guide_worsening_px": self.guide_worsening_px,
            "guide_trend_window": self.guide_trend_window,
            "guide_focus_confirm_frames": self.guide_focus_confirm_frames,
            "guide_focus_shift_ratio": self.guide_focus_shift_ratio,
            "guide_focus_min_shift_turns": self.guide_focus_min_shift_turns,
            "guide_focus_max_shift_turns": self.guide_focus_max_shift_turns,
            "min_confidence": 0.18,
        }

    def update_control(self, decision, center_x=None, frame_width=None) -> None:
        self.status_var.set(decision.message)
        target = float(frame_width) / 2.0 if frame_width else None
        center_text = "--" if center_x is None else f"{float(center_x):.1f}px"
        target_text = "--" if target is None else f"{target:.1f}px"
        error_text = "--" if decision.error_px is None else f"{decision.error_px:+.1f}px"
        self.position_var.set(
            f"中心 {center_text}  │  目标 {target_text}  │  偏差 {error_text}  │  "
            f"{decision.direction_mapping}")
        if decision.search_position_turns is not None:
            if decision.search_phase in (
                    "centering", "confirming", "centered", "center_waiting",
                    "center_dropout"):
                phase_text = {
                    "centering": "中心精确定位",
                    "confirming": "中心稳定确认",
                    "centered": "中心已找到",
                    "center_waiting": "等待中心恢复",
                    "center_dropout": "中心短时丢失",
                }[decision.search_phase]
                self.search_range_var.set(
                    f"搜索范围 {decision.searched_min_turns:+.1f}~"
                    f"{decision.searched_max_turns:+.1f} 圈  │  "
                    f"搜索中心 {decision.search_center_turns:+.1f}  │  "
                    f"当前位置 {decision.search_position_turns:+.1f}  │  "
                    f"{phase_text}")
            else:
                self.search_range_var.set(
                    f"搜索范围 {decision.searched_min_turns:+.1f}~"
                    f"{decision.searched_max_turns:+.1f} 圈  │  "
                    f"搜索中心 {decision.search_center_turns:+.1f}  │  "
                    f"当前位置 {decision.search_position_turns:+.1f}  │  "
                    f"下一边界 {decision.search_target_turns:+.1f}  │  "
                    f"第 {decision.search_expansion_level + 1} 轮")
        elif decision.state in ("idle", "stopped"):
            self.search_range_var.set("搜索范围：尚未启动")

    def update_scene_analysis(self, analysis: dict) -> None:
        """显示连续画面的条纹存在性与水平移动方向。"""
        present = "有条纹" if analysis.get("has_fringe") else "无条纹"
        detail = str(analysis.get("movement_text") or "方向未知")
        source_names = {
            "yolo": "YOLO", "visual": "二维纹理", "history": "历史预测",
        }
        source = source_names.get(
            str(analysis.get("recognition_source") or ""), "--")
        confidence = float(analysis.get("recognition_confidence") or 0.0)
        velocity = float(analysis.get("velocity_px_s") or 0.0)
        blur = " │ 运动模糊" if analysis.get("blurred") else ""
        self.scene_analysis_var.set(
            f"画面分析：{present} │ {detail} │ {source} {confidence:.2f}"
            f" │ {velocity:+.1f} px/s{blur}")

    def update_clarity(self, status: dict) -> None:
        if not status:
            self.clarity_var.set("清晰度增强：不可用")
            return
        score = float(status.get("score") or 0.0)
        baseline = status.get("baseline")
        baseline_text = "--" if baseline is None else f"{float(baseline):.1f}"
        exposure = status.get("exposure")
        gain = status.get("gain")
        algorithm = "算法开" if status.get("software_enabled") else "算法关"
        strength = float(status.get("software_strength") or 0.0)
        stripe_strength = float(status.get("stripe_strength") or 0.0)
        color_gain = float(status.get("color_gain") or 1.0)
        self.clarity_var.set(
            f"清晰度增强：{'运动模式' if status.get('enabled') else '普通模式'} │ "
            f"清晰度 {score:.1f}/{baseline_text} │ "
            f"曝光 {exposure if exposure is not None else '--'} │ "
            f"增益 {gain if gain is not None else '--'} │ "
            f"{algorithm} 锐化{strength:.1f}x/条纹{stripe_strength:.1f}x/色彩{color_gain:.1f}x"
        )

    @staticmethod
    def _bounded_int(value, low: int, high: int, default: int) -> int:
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError, tk.TclError):
            return default

    @staticmethod
    def _bounded_float(value, low: float, high: float, default: float) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError, tk.TclError):
            return default
