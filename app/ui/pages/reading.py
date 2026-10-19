"""④ 读数：第二相机拍微分表，本地 OCR 出读数，一键记进历史。"""
from __future__ import annotations

import time

from PySide6.QtWidgets import QSpinBox, QTableWidget, QTableWidgetItem, QWidget

from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.common import (
    Card,
    MetricCard,
    StatusPill,
    form_grid,
    hstack,
    primary_button,
    secondary_button,
)


class ReadingPage(Page):
    key = "reading"
    title = "④ 读数"
    hint = "用第二台相机读微分表。读数稳定后会自动显示，点「记录」写进历史。"
    MAX_ROWS = 200

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        value_card = Card("当前读数")
        self.value_card = MetricCard("微分表读数", "mm")
        self.stable_pill = StatusPill("未启动", "idle")
        value_card.add(self.value_card)
        value_card.add(self.stable_pill)
        self.add(value_card)

        control = Card("读数相机", "填第二相机的序号，一般是 0。")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        self.camera_index.setValue(int(self.ctx.config.get("reading.camera_index", 0)))
        self.interval = QSpinBox()
        self.interval.setRange(80, 2000)
        self.interval.setSuffix(" ms")
        self.interval.setValue(int(self.ctx.config.get("reading.interval_ms", 200)))
        control.add(form_grid([("相机序号", self.camera_index), ("识别间隔", self.interval)]))
        self.start_button = primary_button("启动读数", self._toggle_reading)
        self.record_button = secondary_button("记录这次读数", self._record)
        self.record_button.setEnabled(False)
        control.add(hstack(self.start_button, self.record_button, spacing=8))
        self.add(control)

        history = Card("读数历史", "最多保留 200 条；「计算」页可以直接导入这些读数。")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["序号", "读数 (mm)", "时间"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(240)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)
        history.add(self.table)
        self.clear_button = secondary_button("清空历史", self._clear)
        history.add(self.clear_button)
        self.add(history)
        self.add_stretch()

    # ------------------------------------------------------------------
    def _toggle_reading(self) -> None:
        service = self.ctx.reading
        if service.connected:
            service.stop()
            self.ctx.log("读数已停止")
            return
        service.start(camera_index=int(self.camera_index.value()),
                      interval_ms=int(self.interval.value()),
                      resolution=self.ctx.config.get("reading.resolution", (1280, 1024)))

    def _record(self) -> None:
        value = self.ctx.state.reading_mm
        if value is None:
            self.ctx.log("还没有有效读数")
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
        self.table.setItem(row, 2, QTableWidgetItem(time.strftime("%H:%M:%S")))
        self.table.scrollToBottom()
        self.ctx.history.append({"kind": "reading", "value_mm": float(value),
                                 "at": time.time()})
        if self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(0)
        self.ctx.log(f"已记录读数 {value:.4f} mm")

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self.ctx.history = [h for h in self.ctx.history if h.get("kind") != "reading"]
        self.ctx.log("读数历史已清空")

    # ------------------------------------------------------------------
    def on_reading(self, payload: dict) -> None:
        self.value_card.set_value(payload.get("value_mm"))
        ready = payload.get("value_mm") is not None
        stable = bool(payload.get("stable"))
        self.record_button.setEnabled(ready)
        if not ready:
            self.stable_pill.set_state("warn", payload.get("message") or "识别中…")
        elif stable:
            self.stable_pill.set_state("ok", "读数稳定")
        else:
            self.stable_pill.set_state("busy", "读数跳动")

    def on_state(self, state: AppState) -> None:
        service = self.ctx.reading
        self.start_button.setText("停止读数" if service.connected else "启动读数")
        self.start_button.setProperty("variant", "danger" if service.connected else "primary")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        if not service.connected:
            self.stable_pill.set_state("idle", "未启动")
            self.record_button.setEnabled(False)
        self.value_card.set_value(state.reading_mm)
