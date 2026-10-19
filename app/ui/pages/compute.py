"""⑤ 计算：由两条读数算厚度 → 多轮统计 → 不确定度 → 导出。"""
from __future__ import annotations

import time

from PySide6.QtWidgets import QDoubleSpinBox, QTableWidget, QTableWidgetItem, QWidget

from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.common import (
    Card,
    StatusPill,
    form_grid,
    hstack,
    label,
    primary_button,
    secondary_button,
)


class ComputePage(Page):
    key = "compute"
    title = "⑤ 计算厚度"
    hint = "h = (d₂ − d₁) / [20 × (n − 1)]。两条读数可以从「④ 读数」直接取。"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        self._rounds: list[dict] = []
        self._build_single()
        self._build_rounds()
        self._build_uncertainty()
        self._build_export()
        self.add_stretch()

    # ------------------------------------------------------------------
    def _build_single(self) -> None:
        card = Card("单次测量", "两个 d 是同一次实验里两条中心条纹对应的微分表读数。")
        self.d1 = self._reading_spin()
        self.d2 = self._reading_spin()
        self.n_index = QDoubleSpinBox()
        self.n_index.setDecimals(4)
        self.n_index.setRange(1.0001, 3.0)
        self.n_index.setSingleStep(0.0001)
        self.n_index.setValue(float(self.ctx.config.refractive_index))
        card.add(form_grid([("读数 d₁ (mm)", self.d1), ("读数 d₂ (mm)", self.d2),
                            ("折射率 n", self.n_index)], columns=2))

        self.take_d1 = secondary_button("当前读数 → d₁", lambda: self._take("d1"))
        self.take_d2 = secondary_button("当前读数 → d₂", lambda: self._take("d2"))
        self.calc_button = primary_button("计算厚度", self._calculate)
        card.add(hstack(self.take_d1, self.take_d2, spacing=8))
        card.add(self.calc_button)

        self.thickness_value = label("待计算", name="BigValue")
        self.opd_label = label("", name="Muted", wrap=True)
        card.add(self.thickness_value)
        card.add(self.opd_label)
        self.single_status = StatusPill("等待输入", "idle")
        card.add(self.single_status)
        self.add(card)

    def _build_rounds(self) -> None:
        card = Card("多轮统计", "把每次测得的厚度加进来，自动算均值 / 标准差 / 极值。")
        self.rounds_table = QTableWidget(0, 4)
        self.rounds_table.setHorizontalHeaderLabels(["序号", "厚度 (mm)", "d₁", "d₂"])
        self.rounds_table.verticalHeader().setVisible(False)
        self.rounds_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rounds_table.setMinimumHeight(200)
        header = self.rounds_table.horizontalHeader()
        header.setStretchLastSection(True)
        card.add(self.rounds_table)
        self.add_round = primary_button("把本次结果加入多轮", self._add_round)
        self.clear_rounds = secondary_button("清空多轮", self._clear_rounds)
        card.add(hstack(self.add_round, self.clear_rounds, spacing=8))
        self.stats_label = label("还没有数据", name="Hint", wrap=True)
        card.add(self.stats_label)
        self.add(card)

    def _build_uncertainty(self) -> None:
        card = Card("不确定度评定", "对多轮厚度序列做 A 类 / B 类评定与 Grubbs 异常值检验。")
        self.unc_button = primary_button("评定不确定度", self._uncertainty)
        card.add(self.unc_button)
        self.unc_label = label("", name="Hint", wrap=True)
        card.add(self.unc_label)
        self.add(card)

    def _build_export(self) -> None:
        card = Card("导出", "文件写到项目下的 exports/ 目录。")
        self.export_csv = secondary_button("导出 CSV", lambda: self._export("csv"))
        self.export_json = secondary_button("导出 JSON", lambda: self._export("json"))
        self.export_md = secondary_button("导出报告 MD", lambda: self._export("md"))
        card.add(hstack(self.export_csv, self.export_json, spacing=8))
        card.add(self.export_md)
        self.add(card)

    # ------------------------------------------------------------------
    @staticmethod
    def _reading_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0, 50.0)
        spin.setSingleStep(0.001)
        spin.setValue(0.0)
        return spin

    def _take(self, which: str) -> None:
        value = self.ctx.state.reading_mm
        if value is None:
            self.ctx.log("没有当前读数，请先到「④ 读数」启动读数")
            return
        (self.d1 if which == "d1" else self.d2).setValue(float(value))
        self.ctx.log(f"已填入 {which} = {value:.4f} mm")

    def _calculate(self) -> None:
        d1, d2 = float(self.d1.value()), float(self.d2.value())
        if d1 == d2:
            self.single_status.set_state("warn", "两个读数相同，无法计算")
            return
        try:
            self.ctx.compute.refractive_index = float(self.n_index.value())
            thickness = self.ctx.compute.thickness_mm(d1, d2)
            opd = self.ctx.compute.opd_um(d1, d2)
        except Exception as exc:
            self.single_status.set_state("error", f"计算失败：{exc}")
            return
        self.thickness_value.setText(f"{thickness:.5f} mm")
        self.opd_label.setText(
            f"光程差 OPD ≈ {opd:.2f} μm   ·   读数差 |Δ| = {abs(d2 - d1):.4f} mm   ·   "
            f"n = {float(self.n_index.value()):.4f}")
        self.single_status.set_state("ok", "计算完成")
        self.ctx.state.update(thickness_mm=thickness)
        self.ctx.log(f"厚度 {thickness:.5f} mm")

    def _add_round(self) -> None:
        text = self.thickness_value.text()
        if text in ("—", ""):
            self.ctx.log("先算一次厚度再添加")
            return
        try:
            value = float(text.replace(" mm", ""))
        except ValueError:
            return
        self._rounds.append({
            "value_mm": value,
            "d1": float(self.d1.value()),
            "d2": float(self.d2.value()),
            "at": time.strftime("%H:%M:%S"),
        })
        self._refresh_rounds()
        self.ctx.log(f"加入第 {len(self._rounds)} 轮：{value:.5f} mm")

    def _clear_rounds(self) -> None:
        self._rounds.clear()
        self._refresh_rounds()

    def _refresh_rounds(self) -> None:
        self.rounds_table.setRowCount(len(self._rounds))
        for row, item in enumerate(self._rounds):
            cells = [str(row + 1), f"{item['value_mm']:.5f}",
                     f"{item['d1']:.4f}", f"{item['d2']:.4f}"]
            for column, text in enumerate(cells):
                self.rounds_table.setItem(row, column, QTableWidgetItem(text))
        if not self._rounds:
            self.stats_label.setText("还没有数据")
            return
        try:
            stats = self.ctx.compute.statistics([r["value_mm"] for r in self._rounds])
        except Exception as exc:
            self.stats_label.setText(f"统计失败：{exc}")
            return
        if stats["count"] == 1:
            self.stats_label.setText(f"1 轮 · 厚度 {stats['mean']:.6f} mm；至少 2 轮才能评估样本标准差")
            return
        parts = [f"轮数 {stats['count']}",
                 f"均值 {stats['mean']:.6f} mm",
                 f"标准差 {stats['std']:.6f} mm",
                 f"范围 {stats['min']:.6f} ~ {stats['max']:.6f} mm"]
        self.stats_label.setText("   ·   ".join(parts))

    def _uncertainty(self) -> None:
        values = [r["value_mm"] for r in self._rounds]
        if not values:
            self.unc_label.setText("先添加至少一轮厚度数据")
            return
        try:
            report = self.ctx.compute.uncertainty(values)
        except Exception as exc:
            self.unc_label.setText(f"评定失败：{exc}")
            return
        lines = []
        for key in ("summary", "type_a", "type_b", "combined", "expanded", "result", "warnings"):
            value = report.get(key)
            if value in (None, "", []):
                continue
            if isinstance(value, (list, tuple)):
                value = "；".join(str(v) for v in value)
            lines.append(f"{key}: {value}")
        self.unc_label.setText("\n".join(lines) or str(report))
        self.ctx.log("不确定度评定完成")

    def _export(self, kind: str) -> None:
        if not self._rounds:
            self.ctx.log("还没有多轮数据可导出")
            return
        rows = [{"序号": index + 1, "厚度mm": item["value_mm"],
                 "d1_mm": item["d1"], "d2_mm": item["d2"]}
                for index, item in enumerate(self._rounds)]
        try:
            if kind == "csv":
                path = self.ctx.record.export_rows(rows, stem="厚度测量")
            elif kind == "json":
                path = self.ctx.record.export_json(
                    {"rounds": rows,
                     "statistics": self.ctx.compute.statistics(
                         [r["value_mm"] for r in self._rounds])},
                    stem="厚度测量")
            else:
                stats = self.ctx.compute.statistics([r["value_mm"] for r in self._rounds])
                body = "\n".join(
                    f"| {i + 1} | {item['value_mm']:.5f} | {item['d1']:.4f} | {item['d2']:.4f} |"
                    for i, item in enumerate(self._rounds))
                table = ("| 序号 | 厚度 (mm) | d₁ (mm) | d₂ (mm) |\n"
                         "| --- | --- | --- | --- |\n" + body)
                std_text = f"{stats['std']:.6f} mm" if stats["std"] is not None else "样本不足（至少 2 轮）"
                summary = (f"- 轮数：{stats['count']}\n"
                           f"- 均值：{stats['mean']:.6f} mm\n"
                           f"- 标准差：{std_text}\n"
                           f"- 范围：{stats['min']:.6f} ~ {stats['max']:.6f} mm")
                path = self.ctx.record.export_report(
                    title="玻璃片厚度测量报告",
                    sections=[("数据", table), ("统计", summary)],
                    stem="厚度测量报告")
        except Exception as exc:
            self.ctx.log(f"导出失败：{exc}")
            return
        self.ctx.log(f"已导出：{path}")

    # ------------------------------------------------------------------
    def on_state(self, state: AppState) -> None:
        has_reading = state.reading_mm is not None
        self.take_d1.setEnabled(has_reading)
        self.take_d2.setEnabled(has_reading)
