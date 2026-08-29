"""实验助手面板 — 多次测量玻璃片厚度、求平均值。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.measurement import (
    GLASS_REFRACTIVE_INDEX,
    ExperimentSession,
    MeasurementRound,
    ThicknessReading,
)


class ExperimentAssistantPanel(tk.LabelFrame):
    """管理多次玻璃片厚度测量，支持手动输入、导入读数、求平均值。"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="多轮厚度记录与统计",
                         bg="#ffffff", fg="#10233f")
        self.on_command = lambda _command, _payload=None: None
        self.session = ExperimentSession()
        self._available_readings: list[dict] = []  # 来自厚度测量面板的已记录读数
        self._reading_map: dict[str, dict] = {}     # label -> reading dict
        self._round_row_map: dict[str, int] = {}    # tree item iid -> sequence

        # ---- 状态变量 ----
        self.status_var = tk.StringVar(value="就绪")
        self.manual_d1_var = tk.StringVar(value="")
        self.manual_d2_var = tk.StringVar(value="")
        self.manual_note_var = tk.StringVar(value="")
        self.import_d1_var = tk.StringVar(value="")
        self.import_d2_var = tk.StringVar(value="")
        self.import_note_var = tk.StringVar(value="")
        # 统计
        self.stats_count_var = tk.StringVar(value="0")
        self.stats_mean_var = tk.StringVar(value="--")
        self.stats_std_var = tk.StringVar(value="--")
        self.stats_min_var = tk.StringVar(value="--")
        self.stats_max_var = tk.StringVar(value="--")
        # 实验元信息
        self.exp_name_var = tk.StringVar(value="")
        self.exp_operator_var = tk.StringVar(value="")
        self.exp_sample_var = tk.StringVar(value="")
        # 折射率
        self.n_var = tk.StringVar(value=f"{GLASS_REFRACTIVE_INDEX:.4f}")

        self._build()

    # ==================================================================
    # UI 构建
    # ==================================================================

    def _build(self) -> None:
        # -- 公式说明 --
        tk.Label(
            self,
            text=(f"公式：h = (d2 − d1) / [20 × (n − 1)]，"
                  f"默认 n = {GLASS_REFRACTIVE_INDEX:.4f}"),
            bg="#ffffff", fg="#64748b", anchor="w", justify="left",
            wraplength=360, font=("Microsoft YaHei UI", 8),
        ).pack(fill=tk.X, padx=8, pady=(8, 4))

        # -- 实验信息行（2×2 网格，避免一行挤不下）--
        info_frame = tk.Frame(self, bg="#ffffff")
        info_frame.pack(fill=tk.X, padx=8, pady=3)
        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_columnconfigure(1, weight=1)
        for i, (label, var, width) in enumerate((
            ("实验名称", self.exp_name_var, 14),
            ("操作者", self.exp_operator_var, 14),
            ("样品编号", self.exp_sample_var, 14),
            ("折射率 n", self.n_var, 8),
        )):
            cell = tk.Frame(info_frame, bg="#ffffff")
            cell.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 8), pady=2)
            tk.Label(cell, text=label, bg="#ffffff", fg="#64748b",
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w")
            entry = tk.Entry(
                cell, textvariable=var, width=width,
                bg="#f7f9fc", fg="#10233f", relief=tk.FLAT,
                font=("Consolas", 9), insertbackground="#10233f")
            entry.pack()
            if label == "折射率 n":
                entry.bind("<FocusOut>", lambda e: self._apply_refractive_index())

        # ================================================================
        # 方式一：手动输入
        # ================================================================
        manual_section = tk.LabelFrame(
            self, text="方式一 · 手动输入读数", bg="#ffffff", fg="#10233f",
            font=("Microsoft YaHei UI", 9, "bold"))
        manual_section.pack(fill=tk.X, padx=8, pady=(6, 2))

        input_row = tk.Frame(manual_section, bg="#ffffff")
        input_row.pack(fill=tk.X, padx=6, pady=(4, 2))
        for label, var in (
            ("d1 (mm)", self.manual_d1_var),
            ("d2 (mm)", self.manual_d2_var),
        ):
            col = tk.Frame(input_row, bg="#ffffff")
            col.pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(col, text=label, bg="#ffffff", fg="#10233f",
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w")
            tk.Entry(
                col, textvariable=var, width=18,
                bg="#f7f9fc", fg="#10233f", relief=tk.FLAT,
                font=("Consolas", 10),
                insertbackground="#10233f",
            ).pack()

        note_manual_row = tk.Frame(manual_section, bg="#ffffff")
        note_manual_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(note_manual_row, text="备注", bg="#ffffff", fg="#64748b",
                 font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT)
        tk.Entry(
            note_manual_row, textvariable=self.manual_note_var, width=52,
            bg="#f7f9fc", fg="#10233f", relief=tk.FLAT,
            font=("Consolas", 9),
            insertbackground="#10233f",
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            manual_section, text="手动添加测量",
            command=self._add_manual,
            relief=tk.FLAT, bd=0, bg="#1677ff", fg="#ffffff",
            activebackground="#0f62d6", activeforeground="#ffffff",
            cursor="hand2", padx=12, pady=5,
        ).pack(fill=tk.X, padx=6, pady=(0, 5))

        # ================================================================
        # 方式二：从已记录读数导入
        # ================================================================
        import_section = tk.LabelFrame(
            self, text="方式二 · 从已记录读数导入", bg="#ffffff", fg="#10233f",
            font=("Microsoft YaHei UI", 9, "bold"))
        import_section.pack(fill=tk.X, padx=8, pady=(2, 2))

        sel_row = tk.Frame(import_section, bg="#ffffff")
        sel_row.pack(fill=tk.X, padx=6, pady=(4, 2))
        for label, var in (
            ("选择 d1", self.import_d1_var),
            ("选择 d2", self.import_d2_var),
        ):
            col = tk.Frame(sel_row, bg="#ffffff")
            col.pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(col, text=label, bg="#ffffff", fg="#10233f",
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w")
            box = ttk.Combobox(
                col, textvariable=var, state="readonly", width=22)
            box.pack()
            if label == "选择 d1":
                self.import_d1_box = box
            else:
                self.import_d2_box = box

        note_import_row = tk.Frame(import_section, bg="#ffffff")
        note_import_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(note_import_row, text="备注", bg="#ffffff", fg="#64748b",
                 font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT)
        tk.Entry(
            note_import_row, textvariable=self.import_note_var, width=52,
            bg="#f7f9fc", fg="#10233f", relief=tk.FLAT,
            font=("Consolas", 9),
            insertbackground="#10233f",
        ).pack(side=tk.LEFT, padx=6)

        btn_import_row = tk.Frame(import_section, bg="#ffffff")
        btn_import_row.pack(fill=tk.X, padx=6, pady=(0, 5))
        tk.Button(
            btn_import_row, text="从记录导入",
            command=self._add_from_readings,
            relief=tk.FLAT, bd=0, bg="#1677ff", fg="#ffffff",
            activebackground="#0f62d6", activeforeground="#ffffff",
            cursor="hand2", padx=12, pady=5,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            btn_import_row, text="刷新读数列表",
            command=self._refresh_reading_list,
            relief=tk.FLAT, bd=0, bg="#e8f1ff", fg="#1677ff",
            cursor="hand2", padx=10, pady=5,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        # ================================================================
        # 测量轮次表格
        # ================================================================
        table_header = tk.Frame(self, bg="#ffffff")
        table_header.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(
            table_header, text="测量记录", bg="#ffffff", fg="#10233f",
            font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(
            table_header,
            text=f"共 0 次   |   公式 h=(d2−d1)/[20×(n−1)]",
            bg="#ffffff", fg="#64748b",
            font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT, padx=8)

        # 使用 Treeview 表格（横向滚动条兜底，避免备注列被截断）
        columns = ("seq", "d1", "d2", "thickness", "source", "note")
        table_frame = tk.Frame(self, bg="#ffffff")
        table_frame.pack(fill=tk.X, padx=8, pady=2)
        self.rounds_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings",
            height=6, selectmode="browse")
        self.rounds_tree.heading("seq", text="轮次")
        self.rounds_tree.heading("d1", text="d1 (mm)")
        self.rounds_tree.heading("d2", text="d2 (mm)")
        self.rounds_tree.heading("thickness", text="厚度 h (mm)")
        self.rounds_tree.heading("source", text="来源")
        self.rounds_tree.heading("note", text="备注")
        self.rounds_tree.column("seq", width=40, anchor="center")
        self.rounds_tree.column("d1", width=80, anchor="e")
        self.rounds_tree.column("d2", width=80, anchor="e")
        self.rounds_tree.column("thickness", width=96, anchor="e")
        self.rounds_tree.column("source", width=56, anchor="center")
        self.rounds_tree.column("note", width=90, anchor="w")

        style = ttk.Style()
        style.configure("Treeview",
                        background="#f7f9fc", foreground="#10233f",
                        fieldbackground="#f7f9fc", font=("Consolas", 9))
        style.configure("Treeview.Heading",
                        background="#e8f1ff", foreground="#10233f",
                        font=("Microsoft YaHei UI", 8, "bold"))
        style.map("Treeview",
                  background=[("selected", "#dbeafe")],
                  foreground=[("selected", "#10233f")])

        self.rounds_tree.pack(fill=tk.X)
        h_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.rounds_tree.xview)
        self.rounds_tree.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(fill=tk.X)

        # 表格操作按钮
        table_btn_row = tk.Frame(self, bg="#ffffff")
        table_btn_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        for text, command in (
            ("删除所选", self._delete_selected),
            ("清空全部", self._clear_all),
        ):
            tk.Button(
                table_btn_row, text=text, command=command,
                relief=tk.FLAT, bd=0, bg="#e8f1ff", fg="#1677ff",
                cursor="hand2", padx=8, pady=3,
            ).pack(side=tk.LEFT, padx=(0, 6))

        # ================================================================
        # 统计摘要
        # ================================================================
        stats_frame = tk.LabelFrame(
            self, text="统计摘要", bg="#ffffff", fg="#10233f",
            font=("Microsoft YaHei UI", 9, "bold"))
        stats_frame.pack(fill=tk.X, padx=8, pady=(4, 2))

        stats_grid = tk.Frame(stats_frame, bg="#ffffff")
        stats_grid.pack(fill=tk.X, padx=8, pady=6)
        for c in range(3):
            stats_grid.grid_columnconfigure(c, weight=1)
        stat_items = (
            ("测量次数", self.stats_count_var, ""),
            ("平均值", self.stats_mean_var, "mm"),
            ("标准差", self.stats_std_var, "mm"),
            ("最小值", self.stats_min_var, "mm"),
            ("最大值", self.stats_max_var, "mm"),
        )
        for i, (label, var, unit) in enumerate(stat_items):
            col = tk.Frame(stats_grid, bg="#ffffff")
            col.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 8), pady=3)
            tk.Label(col, text=label, bg="#ffffff", fg="#64748b",
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w")
            val_frame = tk.Frame(col, bg="#eefbf6")
            val_frame.pack(anchor="w")
            tk.Label(
                val_frame, textvariable=var, bg="#eefbf6", fg="#087f5b",
                font=("Consolas", 13, "bold"), padx=6, pady=1,
            ).pack(side=tk.LEFT)
            if unit:
                tk.Label(
                    val_frame, text=f" {unit}", bg="#eefbf6", fg="#087f5b",
                    font=("Consolas", 9)).pack(side=tk.LEFT)

        # ================================================================
        # 底部操作栏
        # ================================================================
        action_row = tk.Frame(self, bg="#ffffff")
        action_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        for text, command in (
            ("保存会话", self._save_session),
            ("加载会话", self._load_session),
        ):
            tk.Button(
                action_row, text=text, command=command,
                relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                activebackground="#0b0b0b", activeforeground="#ffffff",
                cursor="hand2", padx=14, pady=6,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # 状态栏
        tk.Label(
            self, textvariable=self.status_var, bg="#ffffff", fg="#64748b",
            anchor="w", justify="left", wraplength=360,
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

    # ==================================================================
    # 读数列表同步
    # ==================================================================

    def set_available_readings(self, readings: list[dict]) -> None:
        """接收来自 ThicknessMeasurementPanel 的已记录读数列表。

        每个 reading dict 格式：{"id": "R1", "value_mm": 1.234, ...}
        """
        self._available_readings = readings
        self._refresh_reading_list()

    def _refresh_reading_list(self) -> None:
        """刷新导入模式下的 d1/d2 下拉列表。"""
        self._reading_map.clear()
        labels = []
        for r in self._available_readings:
            label = f"{r['id']}  {r['value_mm']:.6f} mm"
            self._reading_map[label] = r
            labels.append(label)
        self.import_d1_box.configure(values=labels)
        self.import_d2_box.configure(values=labels)
        # 保留已选值（如果还在列表中）
        if self.import_d1_var.get() not in labels:
            self.import_d1_var.set("")
        if self.import_d2_var.get() not in labels:
            self.import_d2_var.set("")

    # ==================================================================
    # 折射率
    # ==================================================================

    def _apply_refractive_index(self) -> None:
        try:
            n = float(self.n_var.get())
            if n <= 1.0:
                raise ValueError("折射率必须大于 1")
            self.session.refractive_index = n
            self.status_var.set(f"折射率已更新为 n = {n:.4f}")
        except ValueError:
            self.n_var.set(f"{self.session.refractive_index:.4f}")
            self.status_var.set("折射率无效，已恢复原值")

    # ==================================================================
    # 添加测量
    # ==================================================================

    def _add_manual(self) -> None:
        """手动输入 d1、d2 添加一次测量。"""
        try:
            d1 = float(self.manual_d1_var.get())
            d2 = float(self.manual_d2_var.get())
        except ValueError:
            self.status_var.set("请输入有效的 d1 和 d2 数值")
            return
        try:
            round_ = self.session.add_manual(
                d1, d2, note=self.manual_note_var.get().strip())
        except ValueError as exc:
            self.status_var.set(f"添加失败：{exc}")
            return
        self.manual_d1_var.set("")
        self.manual_d2_var.set("")
        self.manual_note_var.set("")
        self._on_round_added(round_, "手动")

    def _add_from_readings(self) -> None:
        """从已记录的读数中选取 d1、d2 导入。"""
        d1_label = self.import_d1_var.get()
        d2_label = self.import_d2_var.get()
        if d1_label not in self._reading_map or d2_label not in self._reading_map:
            self.status_var.set("请分别选择 d1 和 d2 对应的已记录读数")
            return
        if d1_label == d2_label:
            self.status_var.set("d1 和 d2 必须选择两次不同的读数记录")
            return
        r1 = self._reading_map[d1_label]
        r2 = self._reading_map[d2_label]
        try:
            round_ = self.session.add_from_readings(
                ThicknessReading(
                    sequence=int(r1["sequence"]),
                    value_mm=float(r1["value_mm"]),
                    captured_at=float(r1["captured_at"]),
                ),
                ThicknessReading(
                    sequence=int(r2["sequence"]),
                    value_mm=float(r2["value_mm"]),
                    captured_at=float(r2["captured_at"]),
                ),
                note=self.import_note_var.get().strip(),
            )
        except ValueError as exc:
            self.status_var.set(f"导入失败：{exc}")
            return
        self.import_note_var.set("")
        self._on_round_added(round_, "导入")

    def _on_round_added(self, round_: MeasurementRound, source: str) -> None:
        """测量添加后的统一处理。"""
        self._refresh_rounds_table()
        self._refresh_statistics()
        self.status_var.set(
            f"已{source}添加 {round_.label}："
            f"d1={round_.d1_mm:.6f}，d2={round_.d2_mm:.6f}，"
            f"h={round_.thickness_mm:.6f} mm")
        self.on_command("round_added", round_.as_dict())

    # ==================================================================
    # 表格刷新
    # ==================================================================

    def _refresh_rounds_table(self) -> None:
        """用 session.rounds 刷新 Treeview。"""
        self._round_row_map.clear()
        for item in self.rounds_tree.get_children():
            self.rounds_tree.delete(item)
        for r in self.session.rounds:
            source = f"{r.d1_source},{r.d2_source}" if r.d1_source else "手动"
            iid = self.rounds_tree.insert(
                "", tk.END,
                values=(
                    r.sequence,
                    f"{r.d1_mm:.6f}",
                    f"{r.d2_mm:.6f}",
                    f"{r.thickness_mm:.6f}",
                    source,
                    r.note,
                ),
            )
            self._round_row_map[iid] = r.sequence

    # ==================================================================
    # 统计
    # ==================================================================

    def _refresh_statistics(self) -> None:
        stats = self.session.statistics()
        self.stats_count_var.set(str(stats.count))
        if stats.count == 0:
            self.stats_mean_var.set("--")
            self.stats_std_var.set("--")
            self.stats_min_var.set("--")
            self.stats_max_var.set("--")
        else:
            self.stats_mean_var.set(f"{stats.mean_mm:.6f}")
            self.stats_std_var.set(f"{stats.std_mm:.6f}")
            self.stats_min_var.set(f"{stats.min_mm:.6f}")
            self.stats_max_var.set(f"{stats.max_mm:.6f}")

    # ==================================================================
    # 删除 / 清空
    # ==================================================================

    def _delete_selected(self) -> None:
        selection = self.rounds_tree.selection()
        if not selection:
            self.status_var.set("请先在表格中选择要删除的测量轮次")
            return
        iid = selection[0]
        sequence = self._round_row_map.get(iid)
        if sequence is None:
            return
        removed = self.session.remove(sequence)
        del self._round_row_map[iid]
        self._refresh_rounds_table()
        self._refresh_statistics()
        self.status_var.set(f"已删除 {removed.label}")
        self.on_command("round_deleted", removed.as_dict())

    def _clear_all(self) -> None:
        if not self.session.rounds:
            return
        if not messagebox.askyesno(
                "确认清空", "将清空所有测量记录，此操作不可撤销。\n确认继续？"):
            return
        self.session.clear()
        self._round_row_map.clear()
        self._refresh_rounds_table()
        self._refresh_statistics()
        self.status_var.set("所有测量记录已清空")
        self.on_command("cleared", None)

    # ==================================================================
    # 存档
    # ==================================================================

    def _save_session(self) -> None:
        """保存当前会话到 JSON 文件。"""
        self._sync_metadata()
        path = filedialog.asksaveasfilename(
            title="保存实验会话",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            initialfile=(f"{self.session.sample_id or 'experiment'}"
                         f"_{datetime.now():%Y%m%d_%H%M%S}.json"),
        )
        if not path:
            return
        try:
            self.session.save(path)
            self.status_var.set(f"实验会话已保存至 {Path(path).name}")
            self.on_command("saved", {"path": path})
        except OSError as exc:
            self.status_var.set(f"保存失败：{exc}")
            messagebox.showerror("保存失败", str(exc))

    def _load_session(self) -> None:
        """从 JSON 文件加载实验会话。"""
        path = filedialog.askopenfilename(
            title="加载实验会话",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.session = ExperimentSession.load(path)
        except (OSError, ValueError, KeyError) as exc:
            self.status_var.set(f"加载失败：{exc}")
            messagebox.showerror("加载失败", f"无法读取实验会话：{exc}")
            return
        self._sync_ui_from_session()
        self._refresh_rounds_table()
        self._refresh_statistics()
        self.status_var.set(
            f"已加载实验会话：{Path(path).name}，"
            f"共 {self.session.count} 次测量")
        self.on_command("loaded", {"path": path})

    def _sync_metadata(self) -> None:
        """将 UI 中的元信息同步到 session。"""
        self.session.name = self.exp_name_var.get().strip()
        self.session.operator = self.exp_operator_var.get().strip()
        self.session.sample_id = self.exp_sample_var.get().strip()

    def _sync_ui_from_session(self) -> None:
        """将 session 元信息同步到 UI。"""
        self.exp_name_var.set(self.session.name)
        self.exp_operator_var.set(self.session.operator)
        self.exp_sample_var.set(self.session.sample_id)
        self.n_var.set(f"{self.session.refractive_index:.4f}")

    # ==================================================================
    # 外部接口
    # ==================================================================

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def snapshot(self) -> dict:
        self._sync_metadata()
        return {
            "session": self.session.as_dict(),
            "statistics": self.session.statistics().as_dict(),
        }
