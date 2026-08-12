"""玻璃片厚度测量面板。"""
from __future__ import annotations

from datetime import datetime
import tkinter as tk
from tkinter import ttk

from src.measurement import GLASS_REFRACTIVE_INDEX, ThicknessMeasurement


class ThicknessMeasurementPanel(tk.LabelFrame):
    """人工记录两次中心条纹读数并选择 d1、d2 计算厚度。"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="玻璃片厚度测量", bg="#ffffff", fg="#10233f")
        self.on_command = lambda _command, _payload=None: None
        self.measurement = ThicknessMeasurement()
        self.current_var = tk.StringVar(value="当前可信读数：-- mm")
        self.status_var = tk.StringVar(value="请在中心条纹出现时记录微分表读数")
        self.d1_var = tk.StringVar(value="")
        self.d2_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="厚度结果：-- mm")
        self._current_value_mm: float | None = None
        self._current_captured_at: float | None = None
        self._labels_to_keys: dict[str, str] = {}
        self._last_result: dict | None = None
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text=("公式：h = (d2 - d1) / [10 × (n - 1)]，"
                  f"n = {GLASS_REFRACTIVE_INDEX:.4f}；所有读数和结果单位均为 mm。"),
            bg="#ffffff", fg="#64748b", anchor="w", justify="left",
            wraplength=430,
        ).pack(fill=tk.X, padx=8, pady=(8, 5))

        current_row = tk.Frame(self, bg="#ffffff")
        current_row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(
            current_row, textvariable=self.current_var, bg="#ffffff",
            fg="#10233f", font=("Consolas", 10, "bold"), anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            current_row, text="记录当前微分表读数",
            command=lambda: self.on_command("record", None),
            relief=tk.FLAT, bd=0, bg="#1677ff", fg="#ffffff",
            activebackground="#0f62d6", activeforeground="#ffffff",
            cursor="hand2", padx=10, pady=4,
        ).pack(side=tk.RIGHT)

        list_frame = tk.Frame(self, bg="#ffffff")
        list_frame.pack(fill=tk.X, padx=8, pady=3)
        self.record_list = tk.Listbox(
            list_frame, height=5, exportselection=False,
            bg="#f7f9fc", fg="#10233f", relief=tk.FLAT,
            selectbackground="#dbeafe", selectforeground="#10233f",
            font=("Consolas", 9),
        )
        self.record_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_buttons = tk.Frame(list_frame, bg="#ffffff")
        list_buttons.pack(side=tk.RIGHT, padx=(6, 0), fill=tk.Y)
        for text, command in (
            ("删除所选", self._delete_selected),
            ("清空记录", self._clear_records),
        ):
            tk.Button(
                list_buttons, text=text, command=command,
                relief=tk.FLAT, bd=0, bg="#e8f1ff", fg="#1677ff",
                cursor="hand2", padx=7, pady=3,
            ).pack(fill=tk.X, pady=(0, 4))

        select_row = tk.Frame(self, bg="#ffffff")
        select_row.pack(fill=tk.X, padx=8, pady=(5, 3))
        tk.Label(select_row, text="d1", bg="#ffffff", fg="#10233f").pack(
            side=tk.LEFT)
        self.d1_box = ttk.Combobox(
            select_row, textvariable=self.d1_var, state="readonly", width=18)
        self.d1_box.pack(side=tk.LEFT, padx=(5, 12))
        tk.Label(select_row, text="d2", bg="#ffffff", fg="#10233f").pack(
            side=tk.LEFT)
        self.d2_box = ttk.Combobox(
            select_row, textvariable=self.d2_var, state="readonly", width=18)
        self.d2_box.pack(side=tk.LEFT, padx=5)

        tk.Button(
            self, text="用所选两次读数计算厚度",
            command=self._calculate,
            relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
            activebackground="#0b0b0b", activeforeground="#ffffff",
            cursor="hand2", pady=5,
        ).pack(fill=tk.X, padx=8, pady=(4, 3))
        tk.Label(
            self, textvariable=self.result_var, bg="#eefbf6", fg="#087f5b",
            anchor="w", justify="left", font=("Consolas", 10, "bold"),
            wraplength=430, padx=7, pady=6,
        ).pack(fill=tk.X, padx=8, pady=3)
        tk.Label(
            self, textvariable=self.status_var, bg="#ffffff", fg="#64748b",
            anchor="w", justify="left", wraplength=430,
        ).pack(fill=tk.X, padx=8, pady=(1, 8))

    def set_current_reading(
        self, value_mm: float | None, captured_at: float | None = None,
    ) -> None:
        self._current_value_mm = value_mm
        self._current_captured_at = captured_at
        if value_mm is None:
            self.current_var.set("当前可信读数：-- mm")
            return
        timestamp = (
            datetime.fromtimestamp(captured_at).strftime("%H:%M:%S.%f")[:-3]
            if captured_at is not None else "--")
        self.current_var.set(
            f"当前可信读数：{float(value_mm):.6f} mm  │  采集 {timestamp}")

    def add_record(self, value_mm: float, captured_at: float | None = None):
        record = self.measurement.add(value_mm, captured_at)
        self._last_result = None
        self.result_var.set("厚度结果：-- mm")
        self._refresh_records()
        labels = list(self._labels_to_keys)
        if len(labels) == 1:
            self.d1_var.set(labels[0])
        elif len(labels) >= 2:
            if not self.d1_var.get():
                self.d1_var.set(labels[0])
            self.d2_var.set(labels[-1])
        self.status_var.set(
            f"已记录 {record.key}：{record.value_mm:.6f} mm")
        return record

    def _refresh_records(self) -> None:
        selected_d1 = self.d1_var.get()
        selected_d2 = self.d2_var.get()
        self.record_list.delete(0, tk.END)
        self._labels_to_keys.clear()
        for record in self.measurement.records:
            timestamp = datetime.fromtimestamp(record.captured_at).strftime(
                "%H:%M:%S.%f")[:-3]
            label = f"{record.key}  {record.value_mm:.6f} mm  {timestamp}"
            self._labels_to_keys[label] = record.key
            self.record_list.insert(tk.END, label)
        labels = list(self._labels_to_keys)
        self.d1_box.configure(values=labels)
        self.d2_box.configure(values=labels)
        if selected_d1 not in labels:
            self.d1_var.set("")
        if selected_d2 not in labels:
            self.d2_var.set("")

    def _delete_selected(self) -> None:
        selection = self.record_list.curselection()
        if not selection:
            self.status_var.set("请先在记录列表中选择要删除的读数")
            return
        label = self.record_list.get(selection[0])
        key = self._labels_to_keys[label]
        removed = self.measurement.remove(key)
        self._last_result = None
        self.result_var.set("厚度结果：-- mm")
        self._refresh_records()
        self.status_var.set(f"已删除 {removed.key}")
        self.on_command("delete", removed.as_dict())

    def _clear_records(self) -> None:
        self.measurement.clear()
        self.d1_var.set("")
        self.d2_var.set("")
        self._last_result = None
        self.result_var.set("厚度结果：-- mm")
        self._refresh_records()
        self.status_var.set("厚度测量记录已清空")
        self.on_command("clear", None)

    def _calculate(self) -> None:
        d1_label = self.d1_var.get()
        d2_label = self.d2_var.get()
        if d1_label not in self._labels_to_keys or d2_label not in self._labels_to_keys:
            self.status_var.set("请分别选择 d1 和 d2 两次读数")
            return
        d1_key = self._labels_to_keys[d1_label]
        d2_key = self._labels_to_keys[d2_label]
        try:
            value = self.measurement.calculate(d1_key, d2_key)
        except (KeyError, ValueError) as exc:
            self.status_var.set(str(exc))
            return
        d1 = self.measurement.get(d1_key)
        d2 = self.measurement.get(d2_key)
        self.result_var.set(
            f"厚度 h = ({d2.value_mm:.6f} - {d1.value_mm:.6f}) / "
            f"[10 × ({self.measurement.refractive_index:.4f} - 1)]\n"
            f"= {value:.6f} mm")
        self.status_var.set(
            "计算完成" if value >= 0 else "结果为负值；如方向选反，请交换 d1、d2")
        self._last_result = {
            "d1_id": d1.key,
            "d1_mm": d1.value_mm,
            "d2_id": d2.key,
            "d2_mm": d2.value_mm,
            "refractive_index": self.measurement.refractive_index,
            "thickness_mm": value,
        }
        self.on_command("calculate", dict(self._last_result))

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def snapshot(self) -> dict:
        return {
            "formula": "(d2-d1)/(10*(n-1))",
            "refractive_index": self.measurement.refractive_index,
            "records": [record.as_dict() for record in self.measurement.records],
            "last_result": dict(self._last_result) if self._last_result else None,
        }
