"""临时数据测量模块 — 目标读数定位 + 回程差测量。"""
from __future__ import annotations
import tkinter as tk


class TemporaryMeasurementPanel(tk.LabelFrame):
    """目标读数定位 & 回程差（ backlash / hysteresis ）测量。"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="临时数据测量", bg="#ffffff", fg="#000000")
        # 实时测量记录（微分表读数 + 中心条纹宽度，可命名）
        self.records: list[dict] = []
        self._record_seq = 0
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
                 ).pack(fill=tk.X, padx=9, pady=(0, 4))

        # ================================================================
        # 分隔线
        # ================================================================
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ================================================================
        # 三、中心条纹宽度测量
        # ================================================================
        fringe_header = tk.Frame(self, bg="#fff")
        fringe_header.pack(fill=tk.X, padx=8, pady=(2, 4))
        tk.Label(fringe_header, text="中心条纹宽度测量", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(fringe_header, text="（按中心条纹位置识别其横向宽度）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))

        r6 = tk.Frame(self, bg="#fff")
        r6.pack(fill=tk.X, padx=8, pady=2)
        self.fringe_width_btn = tk.Button(
            r6, text="分析当前画面",
            command=lambda: self._emit("fringe_width_analyze"),
            **btn)
        self.fringe_width_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.show_all_bands_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            r6, text="标注所有条纹宽度", variable=self.show_all_bands_var,
            bg="#fff", fg="#000", activebackground="#fff",
            activeforeground="#000", highlightthickness=0, anchor="w",
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.fringe_width_status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.fringe_width_status_var, bg="#fff",
                 fg="#333", anchor="w", font=("Microsoft YaHei UI", 8),
                 ).pack(fill=tk.X, padx=9, pady=(4, 0))

        self.fringe_width_detail_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.fringe_width_detail_var, bg="#fff",
                 fg="#555", anchor="w", font=("Consolas", 9),
                 justify=tk.LEFT).pack(fill=tk.X, padx=9, pady=(0, 4))

        # ================================================================
        # 分隔线
        # ================================================================
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ================================================================
        # 四、实时测量与记录
        # ================================================================
        live_header = tk.Frame(self, bg="#fff")
        live_header.pack(fill=tk.X, padx=8, pady=(2, 4))
        tk.Label(live_header, text="实时测量与记录", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(live_header, text="（开启后持续分析，可命名记录）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))
        self.live_toggle_btn = tk.Button(
            live_header, text="开始实时测量",
            command=lambda: self._emit("live_toggle"),
            **btn)
        self.live_toggle_btn.pack(side=tk.RIGHT)

        self.live_var = tk.StringVar(value="微分表: --    中心条纹宽度: --")
        tk.Label(self, textvariable=self.live_var, bg="#fff", fg="#333",
                 anchor="w", font=("Consolas", 9),
                 ).pack(fill=tk.X, padx=9, pady=(2, 4))

        # 命名 + 记录 / 清空
        r7 = tk.Frame(self, bg="#fff")
        r7.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r7, text="数据名称", bg="#fff", fg="#000", width=10,
                 anchor="w").pack(side=tk.LEFT)
        self.record_name_var = tk.StringVar(value="")
        tk.Entry(r7, textvariable=self.record_name_var, width=18).pack(
            side=tk.LEFT, padx=(8, 6))
        self.record_btn = tk.Button(
            r7, text="记录当前数据",
            command=lambda: self._emit("live_record"),
            **btn)
        self.record_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.record_btn.configure(state=tk.DISABLED)
        self.clear_btn = tk.Button(
            r7, text="清空",
            command=lambda: self._emit("live_clear"),
            **sm_btn)
        self.clear_btn.pack(side=tk.LEFT)

        self.live_status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.live_status_var, bg="#fff", fg="#888",
                 anchor="w", font=("Microsoft YaHei UI", 7),
                 ).pack(fill=tk.X, padx=9, pady=(2, 0))

        # 记录列表
        list_frame = tk.Frame(self, bg="#fff")
        list_frame.pack(fill=tk.X, padx=8, pady=(2, 8))
        self.record_list = tk.Text(
            list_frame, height=6, bg="#fafafa", fg="#333",
            font=("Consolas", 9), relief=tk.FLAT, highlightthickness=1,
            highlightbackground="#e5e5e5", wrap=tk.NONE, state=tk.DISABLED)
        scrollbar = tk.Scrollbar(list_frame, command=self.record_list.yview)
        self.record_list.configure(yscrollcommand=scrollbar.set)
        self.record_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """measurement_start / measurement_stop / backlash_set_start / backlash_set_end / backlash_start / backlash_stop / fringe_width_analyze / live_toggle / live_record / live_clear"""
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

    # ------------------------------------------------------------------
    # 中心条纹宽度测量
    # ------------------------------------------------------------------
    def set_fringe_width_status(self, text: str) -> None:
        self.fringe_width_status_var.set(text)

    @property
    def show_all_bands(self) -> bool:
        return bool(self.show_all_bands_var.get())

    def show_fringe_width_result(self, result: dict) -> None:
        """把中心条纹宽度测量结果渲染到面板。

        勾选「标注所有条纹宽度」时，除中心条纹外一并列出每一段条纹的
        边界与宽度；否则只显示中心条纹。
        """
        band = result.get("center_band")
        period = result.get("period_px")
        num = result.get("num_bands", 0)
        num_bright = result.get("num_bright", 0)
        num_dark = result.get("num_dark", 0)
        if not band:
            self.fringe_width_status_var.set("未识别到条纹")
            self.fringe_width_detail_var.set(
                f"周期≈{period}px  |  识别到 {num} 段条纹\n"
                f"请确认画面中有清晰、横向展开的干涉条纹。")
            return
        if self.show_all_bands:
            bands = result.get("bands", [])
            lines = [
                f"识别到 {num} 段条纹（亮 {num_bright} / 暗 {num_dark}），"
                f"周期≈{period}px",
                "— 各条纹边界 / 宽度（★=中心条纹）—",
            ]
            for i, b in enumerate(bands, 1):
                k = "亮" if b["kind"] == "bright" else "暗"
                mark = "★" if b is band else " "
                lines.append(
                    f"{mark}{i:>2} {k}  {b['left']:.1f}–{b['right']:.1f}px  "
                    f"宽 {b['width']:.1f}px")
            self.fringe_width_status_var.set(
                f"中心条纹（{'亮纹' if band['kind'] == 'bright' else '暗纹'}）"
                f"宽度 = {band['width']:.1f} px")
            self.fringe_width_detail_var.set("\n".join(lines))
        else:
            kind_text = "亮纹" if band["kind"] == "bright" else "暗纹"
            fwhm_text = f"  FWHM={band['fwhm']:.1f}px" if band.get("fwhm") else ""
            self.fringe_width_status_var.set(
                f"中心条纹（{kind_text}）宽度 = {band['width']:.1f} px")
            self.fringe_width_detail_var.set(
                f"类型: {kind_text}    中心 x={band['center_x']:.1f}px\n"
                f"左边界={band['left']:.1f}px  右边界={band['right']:.1f}px\n"
                f"宽度={band['width']:.1f}px{fwhm_text}    周期≈{period}px\n"
                f"识别到 {num} 段条纹（亮 {num_bright} / 暗 {num_dark}）")

    # ------------------------------------------------------------------
    # 实时测量与记录
    # ------------------------------------------------------------------
    def set_live_measurement(self, reading_mm: float | None,
                             width_px: float | None,
                             kind: str | None = None) -> None:
        """实时刷新微分表读数与中心条纹宽度的当前值。"""
        reading_text = "--" if reading_mm is None else f"{reading_mm:.6f} mm"
        if width_px is None:
            width_text = "--"
        else:
            kind_text = "暗纹" if kind == "dark" else (
                "亮纹" if kind == "bright" else "")
            width_text = f"{width_px:.1f} px"
            if kind_text:
                width_text += f"（{kind_text}）"
        self.live_var.set(f"微分表: {reading_text}    中心条纹宽度: {width_text}")

    def set_live_running(self, running: bool) -> None:
        """切换实时测量开关状态；停止时清空实时读数并禁用记录按钮。"""
        if running:
            self.live_toggle_btn.configure(text="停止实时测量")
            self.record_btn.configure(state=tk.NORMAL)
        else:
            self.live_toggle_btn.configure(text="开始实时测量")
            self.record_btn.configure(state=tk.DISABLED)
            self.live_var.set("微分表: --    中心条纹宽度: --")

    @property
    def record_name(self) -> str:
        name = self.record_name_var.get().strip()
        return name if name else f"数据 {self._record_seq + 1}"

    def set_live_status(self, text: str) -> None:
        self.live_status_var.set(text)

    def append_record(self, record: dict) -> None:
        self.records.append(record)
        self._record_seq = len(self.records)
        self._render_records()

    def clear_records(self) -> None:
        self.records.clear()
        self._record_seq = 0
        self._render_records()
        self.live_status_var.set("")

    def _render_records(self) -> None:
        self.record_list.configure(state=tk.NORMAL)
        self.record_list.delete("1.0", tk.END)
        if not self.records:
            self.record_list.insert(
                tk.END, "（尚无记录，点击“记录当前数据”保存一条）")
        else:
            for i, rec in enumerate(self.records, 1):
                kind = rec.get("kind")
                kind_text = "暗纹" if kind == "dark" else (
                    "亮纹" if kind == "bright" else "未知")
                reading_mm = rec.get("reading_mm")
                reading_text = "--" if reading_mm is None else f"{reading_mm:.6f} mm"
                width_px = rec.get("width_px")
                width_text = "--" if width_px is None else f"{width_px:.1f} px"
                line = (f"#{i:<2} {rec.get('name', ''):<14}  "
                        f"微分表 {reading_text}   宽度 {width_text}（{kind_text}）\n")
                self.record_list.insert(tk.END, line)
        self.record_list.configure(state=tk.DISABLED)
