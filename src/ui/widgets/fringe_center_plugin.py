"""中心条纹位置分析插件 — 检测 + 多条记录 + 距离计算 + 图上标记"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, simpledialog


COLORS = ["#0066cc", "#cc0000", "#009933", "#cc6600",
          "#6600cc", "#0099cc", "#cc0099", "#666600"]


class FringeCenterPluginPanel(tk.LabelFrame):
    """实时检测零级条纹框内的中心位置，记录多条位置并计算移动距离"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="中心条纹分析", bg="#ffffff", fg="#000000")
        btn = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                   activebackground="#0b0b0b", cursor="hand2")
        sm_btn = dict(relief=tk.FLAT, bd=0, bg="#444444", fg="#ffffff",
                      activebackground="#333333", cursor="hand2")

        self.auto_detect_var = tk.BooleanVar(value=False)
        self.show_line_var = tk.BooleanVar(value=True)
        self.click_record_var = tk.BooleanVar(value=False)
        self.recognition_mode_var = tk.StringVar(value="refined")
        self.result_var = tk.StringVar(value="等待启动...")

        # records: [{name, x_display, y_display, zoom, native, visible}, ...]
        self.records: list[dict] = []
        self._record_counter = 0
        self._ratio = 1.0

        self.dist_var = tk.StringVar(value="")
        self._sel_a = tk.StringVar(value="")
        self._sel_b = tk.StringVar(value="")

        # -- 自动检测 --
        tk.Button(self, text="自动检测中心条纹",
                  command=lambda: self._emit("toggle_auto"), **btn).pack(
            fill=tk.X, padx=8, pady=(8, 2))
        tk.Checkbutton(self, text="显示中心线", variable=self.show_line_var,
                       command=lambda: self._emit("toggle_line"),
                       bg="#fff", fg="#000", activebackground="#fff",
                       selectcolor="#fff").pack(anchor="w", padx=8, pady=2)

        mode_row = tk.Frame(self, bg="#fff")
        mode_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(mode_row, text="识别方式", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="精修", variable=self.recognition_mode_var,
                       value="refined", bg="#fff", fg="#000",
                       activebackground="#fff", selectcolor="#fff").pack(
            side=tk.LEFT, padx=(8, 0))
        tk.Radiobutton(mode_row, text="稳健", variable=self.recognition_mode_var,
                       value="band", bg="#fff", fg="#000",
                       activebackground="#fff", selectcolor="#fff").pack(
            side=tk.LEFT, padx=(4, 0))
        tk.Checkbutton(self, text="手动点击记录（在视频上点击即可记录位置）",
                       variable=self.click_record_var,
                       bg="#fff", fg="#c00", activebackground="#fff",
                       selectcolor="#fff").pack(anchor="w", padx=8, pady=2)

        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)

        # -- 实时检测 --
        tk.Label(self, text="实时检测", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8)
        tk.Label(self, textvariable=self.result_var, bg="#fff", fg="#333",
                 anchor="w", justify="left", wraplength=360,
                 font=("Consolas", 9)).pack(fill=tk.X, padx=8, pady=(0, 4))

        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # -- 记录 --
        tk.Label(self, text="位置记录", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8)

        rec_btn_row = tk.Frame(self, bg="#fff")
        rec_btn_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(rec_btn_row, text="记录当前位置",
                  command=lambda: self._emit("record"), **sm_btn).pack(
            side=tk.LEFT, padx=(0, 4))
        tk.Button(rec_btn_row, text="清空全部",
                  command=lambda: self._emit("clear_record"), **sm_btn).pack(
            side=tk.LEFT)

        # 记录列表（带滚动）
        list_frame = tk.Frame(self, bg="#fff")
        list_frame.pack(fill=tk.X, padx=8, pady=2)
        self._list_canvas = tk.Canvas(list_frame, bg="#fff", height=80,
                                      highlightthickness=1, bd=0,
                                      highlightbackground="#ddd")
        self._list_canvas.pack(fill=tk.X)
        self._list_inner = tk.Frame(self._list_canvas, bg="#fff")
        self._list_win = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor="nw")
        self._list_inner.bind("<Configure>",
            lambda e: self._list_canvas.configure(
                scrollregion=self._list_canvas.bbox("all")))
        def _scroll_list(event):
            self._list_canvas.yview_scroll(int(-event.delta / 60), "units")
        self._list_canvas.bind("<Enter>",
            lambda e: self._list_canvas.bind_all("<MouseWheel>", _scroll_list))
        self._list_canvas.bind("<Leave>",
            lambda e: self._list_canvas.unbind_all("<MouseWheel>"))

        # 距离计算
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        tk.Label(self, text="距离计算", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8)

        dist_row = tk.Frame(self, bg="#fff")
        dist_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(dist_row, text="从", bg="#fff", fg="#000",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self._combo_a = ttk.Combobox(
            dist_row, textvariable=self._sel_a, state="readonly", width=14)
        self._combo_a.pack(side=tk.LEFT, padx=4)
        self._combo_a.bind("<<ComboboxSelected>>", lambda e: self._update_distance())
        tk.Label(dist_row, text="到", bg="#fff", fg="#000",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(8, 0))
        self._combo_b = ttk.Combobox(
            dist_row, textvariable=self._sel_b, state="readonly", width=14)
        self._combo_b.pack(side=tk.LEFT, padx=4)
        self._combo_b.bind("<<ComboboxSelected>>", lambda e: self._update_distance())

        tk.Label(self, textvariable=self.dist_var, bg="#fff", fg="#c00",
                 anchor="w", justify="left", wraplength=360,
                 font=("Consolas", 10, "bold")).pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=4)

        # -- 数轴 --
        tk.Label(self, text="位置数轴", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8)
        self._axis_canvas = tk.Canvas(self, bg="#f5f5f5", height=50,
                                      highlightthickness=1, bd=0,
                                      highlightbackground="#ccc")
        self._axis_canvas.pack(fill=tk.X, padx=8, pady=(2, 6))

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        pass

    def _emit(self, cmd: str):
        self.on_command(cmd)

    # ------------------------------------------------------------------
    # 实时结果
    # ------------------------------------------------------------------
    def update_result(self, center_x: float | None, confidence: float,
                      in_box: bool, msg: str = ""):
        if center_x is not None and in_box:
            self.result_var.set(
                f"中心: {center_x:.1f} px  置信度: {confidence:.2f}")
        elif msg:
            self.result_var.set(f"{msg}")
        else:
            self.result_var.set("未检测到零级条纹")

    def update_auto_state(self, enabled: bool):
        self.configure(text="中心条纹分析 [运行中]" if enabled else "中心条纹分析")

    @property
    def recognition_mode(self) -> str:
        """当前中心识别方式：'refined'（精修，默认）或 'band'（稳健备份）。"""
        return self.recognition_mode_var.get()

    # ------------------------------------------------------------------
    # 记录管理
    # ------------------------------------------------------------------
    def add_record(self, x_display: float, zoom: float):
        self._record_counter += 1
        name = f"记录{self._record_counter}"
        native = x_display / zoom
        self.records.append({
            "name": name, "x_display": x_display,
            "zoom": zoom, "native": native, "visible": True,
        })
        self._refresh_list()
        self._update_combos()
        self._update_distance()
        self._draw_axis()

    def clear_records(self):
        self.records.clear()
        self._record_counter = 0
        self._refresh_list()
        self._update_combos()
        self.dist_var.set("")
        self._draw_axis()

    def set_ratio(self, ratio: float):
        self._ratio = ratio
        self._update_distance()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _refresh_list(self):
        for w in self._list_inner.winfo_children():
            w.destroy()

        if not self.records:
            tk.Label(self._list_inner, text="（无记录）", bg="#fff", fg="#999",
                     font=("Consolas", 9)).pack(anchor="w", padx=4, pady=4)
            return

        for i, rec in enumerate(self.records):
            row = tk.Frame(self._list_inner, bg="#fff")
            row.pack(fill=tk.X, padx=2, pady=1)
            color = COLORS[i % len(COLORS)]
            visible = rec.get("visible", True)

            # 显示/隐藏指示器
            eye = "●" if visible else "○"
            eye_color = color if visible else "#ccc"
            tk.Label(row, text=eye, bg="#fff", fg=eye_color,
                     font=("Consolas", 10), width=2,
                     cursor="hand2").pack(side=tk.LEFT, padx=(2, 0))

            # 色块（点击切换显示）
            block = tk.Label(row, text="  ", bg=color if visible else "#ddd",
                             width=2, cursor="hand2")
            block.pack(side=tk.LEFT, padx=(2, 4))

            # 名称（点击切换显示）
            name_lbl = tk.Label(row, text=rec["name"],
                                bg="#fff" if visible else "#f0f0f0",
                                fg=color if visible else "#ccc",
                                font=("Consolas", 9, "bold"), width=10,
                                anchor="w", cursor="hand2")
            name_lbl.pack(side=tk.LEFT)

            # 数据
            tk.Label(row,
                     text=f"{rec['x_display']:.1f}px  z={rec['zoom']:.1f}",
                     bg="#fff", fg="#333" if visible else "#ccc",
                     font=("Consolas", 9)).pack(side=tk.LEFT, padx=(4, 0))

            # 点击名称/色块切换显示
            for w in (name_lbl, block):
                w.bind("<Button-1>", lambda e, idx=i: self._toggle_visible(idx))

            tk.Button(row, text="改名", font=("Consolas", 7),
                      relief=tk.FLAT, bd=0, bg="#eee", fg="#333",
                      cursor="hand2", width=3,
                      command=lambda idx=i: self._rename(idx)).pack(
                side=tk.RIGHT, padx=2)
            tk.Button(row, text="×", font=("Consolas", 8, "bold"),
                      relief=tk.FLAT, bd=0, bg="#fee", fg="#c00",
                      cursor="hand2", width=2,
                      command=lambda idx=i: self._delete(idx)).pack(
                side=tk.RIGHT, padx=1)

        self._list_inner.update_idletasks()
        h = max(self._list_inner.winfo_reqheight(), 30)
        self._list_canvas.configure(height=min(h, 180))

    def _toggle_visible(self, idx: int):
        self.records[idx]["visible"] = not self.records[idx].get("visible", True)
        self._refresh_list()

    def _rename(self, idx: int):
        new_name = simpledialog.askstring(
            "重命名", f"{self.records[idx]['name']} 的新名称:", parent=self)
        if new_name and new_name.strip():
            self.records[idx]["name"] = new_name.strip()
            self._refresh_list()
            self._update_combos()

    def _delete(self, idx: int):
        del self.records[idx]
        self._refresh_list()
        self._update_combos()
        self._update_distance()
        self._draw_axis()

    def _update_combos(self):
        names = [r["name"] for r in self.records]
        self._combo_a["values"] = names
        self._combo_b["values"] = names

    def _update_distance(self):
        sel_a = self._sel_a.get()
        sel_b = self._sel_b.get()
        rec_a = next((r for r in self.records if r["name"] == sel_a), None)
        rec_b = next((r for r in self.records if r["name"] == sel_b), None)
        if rec_a and rec_b:
            dist_px = abs(rec_b["native"] - rec_a["native"])
            dist_mm = dist_px * self._ratio
            self.dist_var.set(
                f"{rec_a['name']} → {rec_b['name']}: "
                f"{dist_px:.1f} px  |  {dist_mm:.3f} mm")
        else:
            self.dist_var.set("")
        self._draw_axis()

    def _draw_axis(self):
        c = self._axis_canvas
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        margin = 30
        if not self.records:
            return
        c.create_line(margin, h // 2, w - margin, h // 2, fill="#999", width=1)

        natives = [r["native"] for r in self.records]
        lo, hi = min(natives), max(natives)
        if hi - lo < 1:
            hi = lo + 50
        pad = max((hi - lo) * 0.2, 25)
        lo -= pad
        hi += pad

        def to_x(val):
            return margin + (val - lo) / (hi - lo) * (w - 2 * margin)

        for i, rec in enumerate(self.records):
            color = COLORS[i % len(COLORS)]
            x = to_x(rec["native"])
            c.create_line(x, h // 2 - 12, x, h // 2 + 12, fill=color, width=2)
            c.create_text(x, h // 2 - 16, text=rec["name"], fill=color,
                          font=("Consolas", 8, "bold"), anchor="s")
            c.create_text(x, h // 2 + 16, text=f"{rec['native']:.0f}", fill="#666",
                          font=("Consolas", 7), anchor="n")

        sel_a = self._sel_a.get()
        sel_b = self._sel_b.get()
        rec_a = next((r for r in self.records if r["name"] == sel_a), None)
        rec_b = next((r for r in self.records if r["name"] == sel_b), None)
        if rec_a and rec_b:
            x0, x1 = to_x(rec_a["native"]), to_x(rec_b["native"])
            y = h // 2 - 20
            c.create_line(x0, y, x1, y, fill="#666", width=1, dash=(4, 3))
            c.create_line(x0, y - 4, x0, y + 4, fill="#666", width=1)
            c.create_line(x1, y - 4, x1, y + 4, fill="#666", width=1)
            mid = (x0 + x1) / 2
            dist = abs(rec_b["native"] - rec_a["native"])
            c.create_text(mid, y - 8, text=f"{dist:.0f}px", fill="#666",
                          font=("Consolas", 7), anchor="s")
