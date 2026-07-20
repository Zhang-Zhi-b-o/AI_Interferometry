"""临时数据测量模块 — 目标读数定位 + 回程差测量。"""
from __future__ import annotations
import tkinter as tk


class TemporaryMeasurementPanel(tk.LabelFrame):
    """目标读数定位 & 回程差（ backlash / hysteresis ）测量。"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="临时数据测量", bg="#ffffff", fg="#000000")
        btn = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                   activebackground="#0b0b0b", cursor="hand2")
        sm_btn = dict(relief=tk.FLAT, bd=0, bg="#444444", fg="#ffffff",
                      activebackground="#333333", cursor="hand2")
        stop_btn = dict(relief=tk.FLAT, bd=0, bg="#c0392b", fg="#ffffff",
                        activebackground="#a93226", cursor="hand2")

        # ================================================================
        # 一、目标读数定位
        # ================================================================
        r1 = tk.Frame(self, bg="#fff")
        r1.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(r1, text="目标读数 (mm)", bg="#fff", fg="#000").pack(side=tk.LEFT)
        self.target_var = tk.StringVar(value="")
        tk.Entry(r1, textvariable=self.target_var, width=10).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Label(r1, text="（填入期望的微分表读数）", bg="#fff", fg="#888",
                 font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        r2 = tk.Frame(self, bg="#fff")
        r2.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.start_btn = tk.Button(
            r2, text="开始旋转到目标读数",
            command=lambda: self._emit("measurement_start"),
            **btn)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.stop_btn = tk.Button(
            r2, text="停止",
            command=lambda: self._emit("measurement_stop"),
            **stop_btn)
        self.stop_btn.pack(side=tk.RIGHT, padx=(2, 0))
        self.stop_btn.configure(state=tk.DISABLED)

        # ---- 状态 ----
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(self, textvariable=self.status_var, bg="#fff", fg="#333",
                 anchor="w", font=("Microsoft YaHei UI", 8),
                 ).pack(fill=tk.X, padx=9, pady=(6, 2))

        # ---- 当前读数 ----
        self.current_var = tk.StringVar(value="当前微分表读数: --")
        tk.Label(self, textvariable=self.current_var, bg="#fff", fg="#555",
                 anchor="w", font=("Consolas", 9),
                 ).pack(fill=tk.X, padx=9, pady=(0, 4))

        # ================================================================
        # 分隔线
        # ================================================================
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ================================================================
        # 二、回程差测量
        # ================================================================
        backlash_header = tk.Frame(self, bg="#fff")
        backlash_header.pack(fill=tk.X, padx=8, pady=(2, 4))
        tk.Label(backlash_header, text="回程差测量", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(backlash_header, text="（中心条纹对齐画面中心线时记录微分表读数）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))
        tk.Label(
            self,
            text="起点读数须小于终点；端点仅近似到达，过程不反向修正位置",
            bg="#fff", fg="#b45309", font=("Microsoft YaHei UI", 7),
            anchor="w",
        ).pack(fill=tk.X, padx=9, pady=(0, 3))

        # 起点
        r3 = tk.Frame(self, bg="#fff")
        r3.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r3, text="起点读数", bg="#fff", fg="#000", width=10, anchor="w").pack(side=tk.LEFT)
        self.backlash_start_var = tk.StringVar(value="")
        tk.Entry(r3, textvariable=self.backlash_start_var, width=10).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Button(r3, text="标定", command=lambda: self._emit("backlash_set_start"),
                  **sm_btn).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(r3, text="（填入或点击标定取当前读数）", bg="#fff", fg="#888",
                 font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        # 终点
        r4 = tk.Frame(self, bg="#fff")
        r4.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r4, text="终点读数", bg="#fff", fg="#000", width=10, anchor="w").pack(side=tk.LEFT)
        self.backlash_end_var = tk.StringVar(value="")
        tk.Entry(r4, textvariable=self.backlash_end_var, width=10).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Button(r4, text="标定", command=lambda: self._emit("backlash_set_end"),
                  **sm_btn).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(r4, text="（填入或点击标定取当前读数）", bg="#fff", fg="#888",
                 font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        # 开始测量 + 停止
        r5 = tk.Frame(self, bg="#fff")
        r5.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.backlash_start_btn = tk.Button(
            r5, text="开始回程差测量",
            command=lambda: self._emit("backlash_start"),
            **btn)
        self.backlash_start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.backlash_stop_btn = tk.Button(
            r5, text="停止",
            command=lambda: self._emit("backlash_stop"),
            **stop_btn)
        self.backlash_stop_btn.pack(side=tk.RIGHT, padx=(2, 0))
        self.backlash_stop_btn.configure(state=tk.DISABLED)

        # 结果摘要
        self.backlash_status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.backlash_status_var, bg="#fff", fg="#333",
                 anchor="w", font=("Microsoft YaHei UI", 8),
                 ).pack(fill=tk.X, padx=9, pady=(4, 0))

        # 结果详情
        self.backlash_detail_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.backlash_detail_var, bg="#fff", fg="#555",
                 anchor="w", font=("Consolas", 9),
                 justify=tk.LEFT).pack(fill=tk.X, padx=9, pady=(0, 2))

        # 中心条纹对齐状态
        self.center_align_var = tk.StringVar(value="中心条纹: --")
        tk.Label(self, textvariable=self.center_align_var, bg="#fff", fg="#888",
                 anchor="w", font=("Microsoft YaHei UI", 7),
                 ).pack(fill=tk.X, padx=9, pady=(0, 8))

    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """measurement_start / measurement_stop / backlash_set_start / backlash_set_end / backlash_start / backlash_stop"""
        pass

    def _emit(self, cmd: str):
        self.on_command(cmd)

    # ------------------------------------------------------------------
    # 目标读数定位
    # ------------------------------------------------------------------
    @property
    def target_mm(self) -> float | None:
        try:
            return float(self.target_var.get())
        except (ValueError, TypeError):
            return None

    def set_busy(self, busy: bool) -> None:
        if busy:
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    def set_current_reading(self, value_mm: float | None) -> None:
        if value_mm is None:
            self.current_var.set("当前微分表读数: --")
        else:
            self.current_var.set(f"当前微分表读数: {value_mm:.6f} mm")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    # ------------------------------------------------------------------
    # 回程差测量
    # ------------------------------------------------------------------
    @property
    def backlash_start_mm(self) -> float | None:
        try:
            return float(self.backlash_start_var.get())
        except (ValueError, TypeError):
            return None

    @property
    def backlash_end_mm(self) -> float | None:
        try:
            return float(self.backlash_end_var.get())
        except (ValueError, TypeError):
            return None

    def set_backlash_start(self, value_mm: float) -> None:
        self.backlash_start_var.set(f"{value_mm:.6f}")

    def set_backlash_end(self, value_mm: float) -> None:
        self.backlash_end_var.set(f"{value_mm:.6f}")

    def set_backlash_busy(self, busy: bool) -> None:
        if busy:
            self.backlash_start_btn.configure(state=tk.DISABLED)
            self.backlash_stop_btn.configure(state=tk.NORMAL)
            self.start_btn.configure(state=tk.DISABLED)
        else:
            self.backlash_start_btn.configure(state=tk.NORMAL)
            self.backlash_stop_btn.configure(state=tk.DISABLED)
            self.start_btn.configure(state=tk.NORMAL)

    def set_backlash_status(self, text: str) -> None:
        self.backlash_status_var.set(text)

    def set_backlash_result(self, reading_forward: float | None,
                            reading_backward: float | None) -> None:
        if reading_forward is not None and reading_backward is not None:
            diff = abs(reading_forward - reading_backward)
            self.backlash_detail_var.set(
                f"正向（起点→终点）对齐读数: {reading_forward:.6f} mm\n"
                f"反向（终点→起点）对齐读数: {reading_backward:.6f} mm\n"
                f"回程差: {diff:.6f} mm"
            )
        elif reading_forward is not None:
            self.backlash_detail_var.set(
                f"正向（起点→终点）对齐读数: {reading_forward:.6f} mm\n"
                f"反向（终点→起点）对齐读数: 等待中..."
            )
        elif reading_backward is not None:
            self.backlash_detail_var.set(
                f"正向（起点→终点）对齐读数: 等待中...\n"
                f"反向（终点→起点）对齐读数: {reading_backward:.6f} mm"
            )
        else:
            self.backlash_detail_var.set("")

    def set_center_align(self, aligned: bool, center_x: float | None,
                         frame_width: float | None) -> None:
        if center_x is not None and frame_width is not None:
            offset = center_x - frame_width / 2
            indicator = "◉ 已对齐" if aligned else "○ 未对齐"
            self.center_align_var.set(
                f"中心条纹: {indicator}  |  x={center_x:.1f}  "
                f"画面中心={frame_width / 2:.1f}  "
                f"偏差={offset:+.1f} px"
            )
        else:
            self.center_align_var.set("中心条纹: 未检测到")
