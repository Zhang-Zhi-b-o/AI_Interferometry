"""Named positions in analysis-frame pixels; display zoom never changes distances."""
from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QInputDialog, QTableWidget, QTableWidgetItem

from app.ui.widgets.common import Card, hstack, label, secondary_button


def position_distance(a, b, mm_per_px=None):
    distance = abs(float(a["native"]) - float(b["native"]))
    if mm_per_px is not None and (not math.isfinite(mm_per_px) or mm_per_px <= 0):
        raise ValueError("像素标定须为正的有限数")
    return distance, None if mm_per_px is None else distance * mm_per_px


def analysis_overlay(details, *, contours=True, intervals=True):
    """Map ROI-local algorithm geometry into the full analysis-frame coordinates."""
    roi = details.get("roi")
    dx, dy = (max(0, roi[0]), max(0, roi[1])) if roi else (0, 0)
    lines = []
    if contours:
        for band in (details.get("width") or {}).get("bands", []):
            points = band.get("centerline") or []
            if points:
                lines.append({"points": [[float(x)+dx, float(y)+dy] for x, y in points],
                              "kind": band.get("kind"), "width": band.get("width")})
    spacing = details.get("spacing") or {}
    centers = spacing.get("fringe_centers") or []
    valid = spacing.get("interval_valid") or []
    segments = []
    if intervals:
        for index, (a, b) in enumerate(zip(centers, centers[1:])):
            segments.append({"a": [a["x"]+dx, a["y"]+dy], "b": [b["x"]+dx, b["y"]+dy],
                             "valid": bool(valid[index]) if index < len(valid) else True})
    return {"fringe_lines": lines, "fringe_intervals": segments}


class CenterRecords(Card):
    changed = Signal()

    def __init__(self, ctx):
        super().__init__("中心位置记录", "保存全帧分析坐标；显示放大与拖动不会改变两点距离。")
        self.ctx = ctx
        if not hasattr(ctx, "center_records"):
            ctx.center_records = []
        self.scale = None
        self.add(hstack(secondary_button("记录当前中心", self.record),
                        secondary_button("改名", self.rename), secondary_button("删除", self.delete),
                        secondary_button("清空", self.clear)))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["显示", "名称", "位置 px"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.add(self.table)
        self.first, self.second = QComboBox(), QComboBox()
        self.first.currentIndexChanged.connect(self.update_distance)
        self.second.currentIndexChanged.connect(self.update_distance)
        self.add(hstack(self.first, self.second))
        self.distance = label("请选择两个记录", wrap=True)
        self.add(self.distance)
        self.refresh()

    def record(self):
        x = self.ctx.state.center_x
        if x is None or not math.isfinite(x):
            self.ctx.log("未定位到中心条纹，无法记录")
            return
        name, ok = QInputDialog.getText(self, "中心位置名称", "名称", text=f"位置 {len(self.ctx.center_records)+1}")
        if ok and name.strip():
            self.ctx.center_records.append({"name": name.strip(), "native": float(x), "visible": True})
            self.refresh()

    def rename(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.ctx.center_records[row]
        name, ok = QInputDialog.getText(self, "重命名", "名称", text=item["name"])
        if ok and name.strip():
            item["name"] = name.strip()
            self.refresh()

    def delete(self):
        row = self.table.currentRow()
        if row >= 0:
            del self.ctx.center_records[row]
            self.refresh()

    def clear(self):
        self.ctx.center_records.clear()
        self.refresh()

    def refresh(self):
        self.table.setRowCount(len(self.ctx.center_records))
        self.first.blockSignals(True)
        self.second.blockSignals(True)
        self.first.clear()
        self.second.clear()
        for row, record in enumerate(self.ctx.center_records):
            check = QCheckBox()
            check.setChecked(record.get("visible", True))
            check.toggled.connect(lambda visible, item=record: self.toggle(item, visible))
            self.table.setCellWidget(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(record["name"]))
            self.table.setItem(row, 2, QTableWidgetItem(f"{record['native']:.2f}"))
            self.first.addItem(record["name"], row)
            self.second.addItem(record["name"], row)
        if len(self.ctx.center_records) > 1:
            self.second.setCurrentIndex(1)
        self.first.blockSignals(False)
        self.second.blockSignals(False)
        self.update_distance()
        self.changed.emit()

    def toggle(self, record, visible):
        record["visible"] = visible
        self.changed.emit()

    def set_scale(self, value):
        self.scale = value or None
        self.ctx.mm_per_px = self.scale
        self.update_distance()

    def update_distance(self):
        a, b = self.first.currentData(), self.second.currentData()
        if a is None or b is None:
            self.distance.setText("请选择两个记录")
            return
        px, mm = position_distance(self.ctx.center_records[a], self.ctx.center_records[b], self.scale)
        self.distance.setText(f"两点距离：{px:.3f} px" + ("（未标定）" if mm is None else f" / {mm:.6g} mm"))
