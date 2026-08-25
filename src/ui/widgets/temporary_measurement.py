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
                 anchor="w", justify=tk.LEFT, wraplength=360,
                 font=("Microsoft YaHei UI", 7),
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

        r6b = tk.Frame(self, bg="#fff")
        r6b.pack(fill=tk.X, padx=8, pady=(4, 2))
        self.fringe_realtime_btn = tk.Button(
            r6b, text="开始实时分析条纹宽度",
            command=lambda: self._emit("fringe_realtime_toggle"),
            **btn)
        self.fringe_realtime_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.annotate_fringe_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            r6b, text="标注到画面", variable=self.annotate_fringe_var,
            bg="#fff", fg="#000", activebackground="#fff",
            activeforeground="#000", highlightthickness=0, anchor="w",
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.fringe_realtime_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.fringe_realtime_var, bg="#fff",
                 fg="#0b5bd3", anchor="w", font=("Consolas", 9),
                 ).pack(fill=tk.X, padx=9, pady=(0, 2))

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
            highlightbackground="#e5e5e5", wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = tk.Scrollbar(list_frame, command=self.record_list.yview)
        self.record_list.configure(yscrollcommand=scrollbar.set)
        self.record_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ================================================================
        # 分隔线
        # ================================================================
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ================================================================
        # 五、薄膜厚度分布（单帧）
        # ================================================================
        thick_header = tk.Frame(self, bg="#fff")
        thick_header.pack(fill=tk.X, padx=8, pady=(2, 4))
        tk.Label(thick_header, text="薄膜厚度分布（单帧）", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(thick_header, text="（用彩色条纹解包厚度，标定CSV留空为相对模式）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))

        # 波长 + 折射率
        rt = tk.Frame(self, bg="#fff")
        rt.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(rt, text="波长(nm)", bg="#fff", fg="#000", width=10,
                 anchor="w").pack(side=tk.LEFT)
        self.thickness_wavelength_var = tk.StringVar(value="589.3")
        tk.Entry(rt, textvariable=self.thickness_wavelength_var, width=10).pack(
            side=tk.LEFT, padx=(8, 10))
        tk.Label(rt, text="折射率", bg="#fff", fg="#000").pack(side=tk.LEFT)
        self.thickness_refractive_var = tk.StringVar(value="1.523")
        tk.Entry(rt, textvariable=self.thickness_refractive_var, width=10).pack(
            side=tk.LEFT, padx=(8, 0))

        # 标定 CSV + 浏览
        rc = tk.Frame(self, bg="#fff")
        rc.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(rc, text="标定CSV", bg="#fff", fg="#000", width=10,
                 anchor="w").pack(side=tk.LEFT)
        self.thickness_calibration_var = tk.StringVar(value="")
        tk.Entry(rc, textvariable=self.thickness_calibration_var, width=24).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Button(rc, text="浏览", command=lambda: self._emit("thickness_browse"),
                  **sm_btn).pack(side=tk.LEFT)

        # 分析按钮 + 反转方向
        ra = tk.Frame(self, bg="#fff")
        ra.pack(fill=tk.X, padx=8, pady=(4, 2))
        self.thickness_analyze_btn = tk.Button(
            ra, text="分析单帧厚度分布",
            command=lambda: self._emit("thickness_analyze"),
            **btn)
        self.thickness_analyze_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.thickness_invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ra, text="反转厚度方向", variable=self.thickness_invert_var,
            bg="#fff", fg="#000", activebackground="#fff",
            activeforeground="#000", highlightthickness=0, anchor="w",
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 0))

        # 框选分析区域（鼠标在视频上拖拽，只分析框内厚度）
        rr = tk.Frame(self, bg="#fff")
        rr.pack(fill=tk.X, padx=8, pady=(0, 2))
        self.thickness_roi_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            rr, text="鼠标框选分析区域", variable=self.thickness_roi_mode_var,
            command=lambda: self._emit("thickness_roi_mode"),
            bg="#fff", fg="#000", activebackground="#fff",
            activeforeground="#000", highlightthickness=0, anchor="w",
            cursor="hand2",
        ).pack(side=tk.LEFT)
        tk.Button(
            rr, text="清除区域", command=lambda: self._emit("thickness_roi_clear"),
            **sm_btn).pack(side=tk.LEFT, padx=(6, 0))
        self.thickness_roi_status_var = tk.StringVar(value="分析区域: 全画面")
        tk.Label(rr, textvariable=self.thickness_roi_status_var, bg="#fff",
                 fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(8, 0))

        # 无膜基准图
        rb = tk.Frame(self, bg="#fff")
        rb.pack(fill=tk.X, padx=8, pady=(0, 2))
        self.thickness_baseline_btn = tk.Button(
            rb, text="捕获无膜基准图",
            command=lambda: self._emit("thickness_capture_baseline"),
            **sm_btn)
        self.thickness_baseline_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.thickness_baseline_clear_btn = tk.Button(
            rb, text="清除基准",
            command=lambda: self._emit("thickness_clear_baseline"),
            **sm_btn)
        self.thickness_baseline_clear_btn.pack(side=tk.LEFT)
        self.thickness_baseline_var = tk.StringVar(value="无膜基准: 未设置")
        tk.Label(rb, textvariable=self.thickness_baseline_var, bg="#fff",
                 fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(8, 0))

        # 绝对厚度锚定（中心条纹读数 − 初始读数 → 基准厚度 μm）
        anchor_header = tk.Frame(self, bg="#fff")
        anchor_header.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Label(anchor_header, text="绝对厚度锚定", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(anchor_header, text="（厚度 = |中心条纹读数 − 初始读数| ÷ 20 × 1000 ÷ (n−1)）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))

        ra_initial = tk.Frame(self, bg="#fff")
        ra_initial.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(ra_initial, text="初始读数(mm)", bg="#fff", fg="#000", width=12,
                 anchor="w").pack(side=tk.LEFT)
        self.thickness_initial_var = tk.StringVar(value="")
        tk.Entry(ra_initial, textvariable=self.thickness_initial_var, width=12).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Button(ra_initial, text="读取",
                  command=lambda: self._emit("thickness_set_initial"),
                  **sm_btn).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(ra_initial, text="（捕获无膜基准时自动记录）", bg="#fff", fg="#888",
                 font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        ra_center = tk.Frame(self, bg="#fff")
        ra_center.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(ra_center, text="中心条纹读数(mm)", bg="#fff", fg="#000", width=12,
                 anchor="w").pack(side=tk.LEFT)
        self.thickness_center_var = tk.StringVar(value="")
        tk.Entry(ra_center, textvariable=self.thickness_center_var, width=12).pack(
            side=tk.LEFT, padx=(8, 6))
        tk.Button(ra_center, text="读取",
                  command=lambda: self._emit("thickness_set_center"),
                  **sm_btn).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(ra_center, text="（中心条纹对齐中心线时记录）", bg="#fff", fg="#888",
                 font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        # 状态 + 指标
        self.thickness_status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.thickness_status_var, bg="#fff",
                 fg="#333", anchor="w", font=("Microsoft YaHei UI", 8),
                 ).pack(fill=tk.X, padx=9, pady=(4, 0))
        self.thickness_detail_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.thickness_detail_var, bg="#fff",
                 fg="#555", anchor="w", font=("Consolas", 9),
                 justify=tk.LEFT).pack(fill=tk.X, padx=9, pady=(0, 2))

        # 结果缩略图
        self.thickness_image_label = tk.Label(self, bg="#fafafa", bd=1,
                                              relief=tk.SOLID)
        self.thickness_image_label.pack(fill=tk.X, padx=8, pady=(2, 8))
        self._thickness_photo = None  # 持有 PhotoImage 引用防止被回收

        # ================================================================
        # 分隔线
        # ================================================================
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ================================================================
        # 六、颜色→光程差标定表采集
        # ================================================================
        cal_header = tk.Frame(self, bg="#fff")
        cal_header.pack(fill=tk.X, padx=8, pady=(2, 4))
        tk.Label(cal_header, text="颜色→光程差标定表采集", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(cal_header, text="（逐点取颜色存为 opd_um,r,g,b）",
                 bg="#fff", fg="#888", font=("Microsoft YaHei UI", 7)).pack(
            side=tk.LEFT, padx=(6, 0))

        # OPD + 零点读数
        co = tk.Frame(self, bg="#fff")
        co.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(co, text="OPD(μm)", bg="#fff", fg="#000", width=10,
                 anchor="w").pack(side=tk.LEFT)
        self.calibration_opd_var = tk.StringVar(value="")
        tk.Entry(co, textvariable=self.calibration_opd_var, width=10).pack(
            side=tk.LEFT, padx=(8, 10))
        tk.Label(co, text="零点读数(mm)", bg="#fff", fg="#000").pack(side=tk.LEFT)
        self.calibration_zero_var = tk.StringVar(value="")
        tk.Entry(co, textvariable=self.calibration_zero_var, width=10).pack(
            side=tk.LEFT, padx=(8, 0))

        # 自动算 OPD + 取点
        ca = tk.Frame(self, bg="#fff")
        ca.pack(fill=tk.X, padx=8, pady=2)
        self.calibration_auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ca, text="由微分表自动算 OPD（=|当前−零点|÷20×1000）",
            variable=self.calibration_auto_var, bg="#fff", fg="#000",
            activebackground="#fff", activeforeground="#000",
            highlightthickness=0, anchor="w", cursor="hand2",
        ).pack(side=tk.LEFT)
        self.calibration_capture_btn = tk.Button(
            ca, text="标定当前点", command=lambda: self._emit("calibration_capture"),
            **btn)
        self.calibration_capture_btn.pack(side=tk.RIGHT, padx=(6, 0))

        # 保存 / 清空
        cs = tk.Frame(self, bg="#fff")
        cs.pack(fill=tk.X, padx=8, pady=(4, 2))
        self.calibration_save_btn = tk.Button(
            cs, text="保存CSV", command=lambda: self._emit("calibration_save"),
            **btn)
        self.calibration_save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                       padx=(0, 6))
        self.calibration_clear_btn = tk.Button(
            cs, text="清空", command=lambda: self._emit("calibration_clear"),
            **sm_btn)
        self.calibration_clear_btn.pack(side=tk.LEFT)

        # 状态
        self.calibration_status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.calibration_status_var, bg="#fff",
                 fg="#333", anchor="w", font=("Microsoft YaHei UI", 8),
                 ).pack(fill=tk.X, padx=9, pady=(4, 0))

        # 已采集列表
        cal_list_frame = tk.Frame(self, bg="#fff")
        cal_list_frame.pack(fill=tk.X, padx=8, pady=(2, 8))
        self.calibration_list = tk.Text(
            cal_list_frame, height=5, bg="#fafafa", fg="#333",
            font=("Consolas", 9), relief=tk.FLAT, highlightthickness=1,
            highlightbackground="#e5e5e5", wrap=tk.WORD, state=tk.DISABLED)
        cal_scroll = tk.Scrollbar(cal_list_frame, command=self.calibration_list.yview)
        self.calibration_list.configure(yscrollcommand=cal_scroll.set)
        self.calibration_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cal_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.calibration_rows: list[dict] = []

    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """measurement_start / measurement_stop / backlash_set_start / backlash_set_end / backlash_start / backlash_stop / fringe_width_analyze / fringe_realtime_toggle / live_toggle / live_record / live_clear / thickness_analyze / thickness_browse / thickness_capture_baseline / thickness_clear_baseline / thickness_roi_mode / thickness_roi_clear / thickness_set_initial / thickness_set_center / calibration_capture / calibration_save / calibration_clear"""
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

    @property
    def annotate_fringe(self) -> bool:
        return bool(self.annotate_fringe_var.get())

    def set_fringe_realtime_running(self, running: bool) -> None:
        self.fringe_realtime_btn.configure(
            text="停止实时分析" if running else "开始实时分析条纹宽度")

    def set_fringe_realtime_text(self, text: str) -> None:
        self.fringe_realtime_var.set(text)

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

    def show_fringe_width_by_count_result(self, result: dict) -> None:
        """把「视场宽度 / 条纹数量」估算的条纹间隔渲染到面板。"""
        width = result.get("fringe_width")
        count = result.get("fringe_count", 0)
        span = result.get("span_px", 0.0)
        region = result.get("region")
        period = result.get("period_px")
        kind = result.get("kind", "bright")
        kind_text = {"bright": "亮纹", "dark": "暗纹", "all": "明暗条纹"}.get(
            kind, kind)
        if width is None or count == 0:
            self.fringe_width_status_var.set("未识别到可计数的条纹")
            self.fringe_width_detail_var.set(
                "视场中未识别到可计数的条纹。\n"
                "请确认画面中有清晰、横向展开的干涉条纹，或框选一个效果较好的视场。")
            return
        region_text = ""
        if region and len(region) == 2:
            region_text = f"视场 [{region[0]:.0f}–{region[1]:.0f}]px"
        self.fringe_width_status_var.set(
            f"条纹间隔 = {span:.1f}px ÷ {count} 条{kind_text} = {width:.2f}px")
        self.fringe_width_detail_var.set(
            f"计算方式：视场宽度 ÷ 条纹数量\n"
            f"{region_text}\n"
            f"视场宽度 = {span:.1f} px    条纹数 = {count} 条\n"
            f"条纹间隔 = {width:.2f} px    自相关周期 ≈ {period}px（供核对）")

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

    # ------------------------------------------------------------------
    # 薄膜厚度分布（单帧）
    # ------------------------------------------------------------------
    @property
    def thickness_wavelength_nm(self) -> float | None:
        try:
            value = float(self.thickness_wavelength_var.get())
        except (ValueError, TypeError):
            return None
        return value if value > 0 else None

    @property
    def thickness_refractive_index(self) -> float | None:
        try:
            value = float(self.thickness_refractive_var.get())
        except (ValueError, TypeError):
            return None
        return value if value > 1.0 else None

    @property
    def thickness_calibration_path(self) -> str:
        return self.thickness_calibration_var.get().strip()

    @property
    def thickness_invert(self) -> bool:
        return bool(self.thickness_invert_var.get())

    @property
    def thickness_roi_mode(self) -> bool:
        return bool(self.thickness_roi_mode_var.get())

    @property
    def thickness_initial_mm(self) -> float | None:
        try:
            return float(self.thickness_initial_var.get())
        except (ValueError, TypeError):
            return None

    @property
    def thickness_center_mm(self) -> float | None:
        try:
            return float(self.thickness_center_var.get())
        except (ValueError, TypeError):
            return None

    def set_thickness_initial(self, value_mm: float) -> None:
        self.thickness_initial_var.set(f"{value_mm:.6f}")

    def set_thickness_center(self, value_mm: float) -> None:
        self.thickness_center_var.set(f"{value_mm:.6f}")

    def set_thickness_roi_status(self, text: str) -> None:
        self.thickness_roi_status_var.set(text)

    def set_thickness_calibration(self, path: str) -> None:
        self.thickness_calibration_var.set(path)

    def set_thickness_status(self, text: str) -> None:
        self.thickness_status_var.set(text)

    def set_thickness_result(self, metrics: dict) -> None:
        """把厚度分布指标渲染到面板（含归一化百分比版本）。"""
        mode_text = "标定（颜色→光程差）" if metrics.get("mode") == "calibrated" \
            else "相对（颜色级次插值）"

        def pct(value) -> str:
            return "—" if value is None else f"{value:.3f}%"

        lines = [
            f"模式: {mode_text}",
            f"有效像素: {metrics.get('valid_pixels', 0)}",
            f"稳健最小值(2%): {metrics.get('min_robust_um', 0):.4f} μm",
            f"稳健最大值(98%): {metrics.get('max_robust_um', 0):.4f} μm",
            f"稳健峰谷值 PV: {metrics.get('pv_robust_um', 0):.4f} μm  ({pct(metrics.get('pv_robust_pct'))})",
            f"RMS 不均匀度: {metrics.get('rms_um', 0):.4f} μm  ({pct(metrics.get('rms_pct'))})",
            f"中间90%跨度: {metrics.get('p90_span_um', 0):.4f} μm  ({pct(metrics.get('p90_span_pct'))})",
            f"中位置信度: {metrics.get('median_confidence', 0):.3f}",
        ]
        self.thickness_detail_var.set("\n".join(lines))

    def show_thickness_image(self, bgr) -> None:
        """显示厚度伪彩叠加图缩略图。"""
        try:
            from PIL import Image, ImageTk
        except Exception:
            self.thickness_image_label.configure(image="")
            return
        if bgr is None:
            self.thickness_image_label.configure(image="")
            self._thickness_photo = None
            return
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        pil = Image.fromarray(rgb)
        max_w = 320
        if pil.width > max_w:
            ratio = max_w / pil.width
            pil = pil.resize((max_w, max(1, int(pil.height * ratio))),
                             Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._thickness_photo = photo
        self.thickness_image_label.configure(image=photo)

    def set_thickness_baseline(self, is_set: bool) -> None:
        if is_set:
            self.thickness_baseline_var.set("无膜基准: 已设置（分析时自动扣除）")
        else:
            self.thickness_baseline_var.set("无膜基准: 未设置")

    # ------------------------------------------------------------------
    # 颜色→光程差标定表采集
    # ------------------------------------------------------------------
    @property
    def calibration_opd_um(self) -> float | None:
        try:
            return float(self.calibration_opd_var.get())
        except (ValueError, TypeError):
            return None

    @property
    def calibration_zero_mm(self) -> float | None:
        try:
            return float(self.calibration_zero_var.get())
        except (ValueError, TypeError):
            return None

    @property
    def calibration_auto_opd(self) -> bool:
        return bool(self.calibration_auto_var.get())

    def set_calibration_opd(self, value_um: float) -> None:
        self.calibration_opd_var.set(f"{value_um:.4f}")

    def set_calibration_status(self, text: str) -> None:
        self.calibration_status_var.set(text)

    def append_calibration(self, row: dict) -> None:
        self.calibration_rows.append(row)
        self._render_calibration()

    def clear_calibration(self) -> None:
        self.calibration_rows.clear()
        self._render_calibration()
        self.calibration_status_var.set("")

    def _render_calibration(self) -> None:
        self.calibration_list.configure(state=tk.NORMAL)
        self.calibration_list.delete("1.0", tk.END)
        if not self.calibration_rows:
            self.calibration_list.insert(
                tk.END, "（尚无标定点，点击“标定当前点”采集一条）")
        else:
            for i, row in enumerate(self.calibration_rows, 1):
                self.calibration_list.insert(
                    tk.END,
                    f"#{i:<3} OPD={row['opd_um']:>10.4f} μm   "
                    f"r={row['r']:>3} g={row['g']:>3} b={row['b']:>3}\n")
        self.calibration_list.configure(state=tk.DISABLED)
