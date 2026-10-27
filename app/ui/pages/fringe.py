"""③ 条纹：一次测量同时给出位置 / 宽度 / 间距 / 倾角。"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QWidget

from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.common import (
    Card,
    MetricCard,
    StatusPill,
    form_grid,
    hstack,
    label,
    primary_button,
    secondary_button,
)


class FringePage(Page):
    key = "fringe"
    title = "③ 确定条纹位置"
    hint = "对当前画面做一次测量，结果同时更新到右侧画面上方的指标卡。"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        card = Card("测量", "参考位置留空则自动定位中心条纹；也可以手动填一个横坐标。")
        self.reference_x = QDoubleSpinBox()
        self.reference_x.setRange(-1.0, 5000.0)
        self.reference_x.setDecimals(1)
        self.reference_x.setSpecialValueText("自动")
        self.reference_x.setValue(-1.0)
        self.reference_x.setSuffix(" px")
        card.add(form_grid([("参考横坐标", self.reference_x)]))
        self.measure_button = primary_button("测量条纹", self._measure)
        self.pin_button = secondary_button("把结果设为参考", self._use_result_as_reference)
        card.add(hstack(self.measure_button, self.pin_button, spacing=8))
        self.state_pill = StatusPill("尚未测量", "idle")
        card.add(self.state_pill)
        self.add(card)

        cards = Card("测量结果")
        self.center_card = MetricCard("中心位置", "px")
        self.width_card = MetricCard("中心条纹宽度", "px")
        self.spacing_card = MetricCard("条纹间距", "px")
        self.tilt_card = MetricCard("倾斜角", "°")
        cards.add(hstack(self.center_card, self.width_card, spacing=10))
        cards.add(hstack(self.spacing_card, self.tilt_card, spacing=10))
        self.detail = label("", name="Muted", wrap=True)
        cards.add(self.detail)
        self.add(cards)

        show_card = Card("显示")
        self.show_center = QCheckBox("在画面上标出中心条纹")
        self.show_center.setChecked(True)
        self.show_center.toggled.connect(self._apply_overlay)
        show_card.add(self.show_center)
        self.add(show_card)
        legacy = Card("连续测量与记录", "中心宽度与同类条纹间距是不同量；mm/px 为当前分析画面的标定。")
        self.method = QComboBox()
        for title, value in (("原综合测量", "default"), ("中心宽度 1D", "width_1d"), ("中心宽度 2D", "width_2d"), ("视场宽度÷条纹数", "count"), ("鲁棒间距", "robust"), ("法向间距 2D", "spacing_2d")):
            self.method.addItem(title, value)
        self.pixel_scale = QDoubleSpinBox()
        self.pixel_scale.setDecimals(8)
        self.pixel_scale.setRange(0, 100)
        self.pixel_scale.setSpecialValueText("未标定（仅 px）")
        self.record_name = QLineEdit()
        legacy.add(form_grid([("算法", self.method), ("标定 (mm/px)", self.pixel_scale), ("记录名称", self.record_name)]))
        self.realtime = QCheckBox("实时测量（每秒，等待上一任务完成）")
        legacy.add(self.realtime)
        legacy.add(hstack(secondary_button("记录当前结果", self._record), secondary_button("导出 CSV", self._export)))
        self.history = QTableWidget(0, 4)
        self.history.setHorizontalHeaderLabels(["名称", "宽度 px", "间距 px", "读数 mm"])
        legacy.add(self.history)
        self.add(legacy)
        from app.ui.widgets.center_records import CenterRecords
        self.position_records = CenterRecords(ctx)
        self.pixel_scale.valueChanged.connect(self.position_records.set_scale)
        self.add(self.position_records)
        self.show_contours = QCheckBox("叠加条纹轮廓（2D 宽度算法）")
        self.show_intervals = QCheckBox("叠加条纹间隔（绿：采用，橙：剔除）")
        self.show_contours.setChecked(True)
        self.show_intervals.setChecked(True)
        self.show_contours.toggled.connect(self._refresh_analysis_overlay)
        self.show_intervals.toggled.connect(self._refresh_analysis_overlay)
        show_card.add(self.show_contours)
        show_card.add(self.show_intervals)
        self.ctx.fringe_overlay = {"fringe_lines": [], "fringe_intervals": []}
        self._busy = False
        self._last_metrics = None
        self._records = []
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(lambda: self._measure() if self.realtime.isChecked() and not self._busy else None)
        self._timer.start()
        self.add_stretch()

    # ------------------------------------------------------------------
    def _measure(self) -> None:
        if self._busy:
            return
        frame = self.ctx.frame()
        if frame is None:
            self.state_pill.set_state("error", "没有画面，请先打开相机")
            return
        reference = self.reference_x.value()
        center_x = None if reference < 0 else float(reference)
        from app.services.diagnostics import contrast_metric

        roi = list(self.ctx.roi) if self.ctx.roi else None
        sample = frame
        if roi:
            x, y, w, h = roi
            sample = frame[y:y+h, x:x+w]
        contrast = contrast_metric(sample)
        self._busy = True
        self.measure_button.setEnabled(False)
        self.state_pill.set_state("busy", "测量中…")

        def done(metrics):
            self._busy = False
            self._last_metrics = metrics
            self._refresh_analysis_overlay()
            self.ctx.timeline.add("fringe", result={**metrics.as_dict(),
                                  "contrast": contrast, "roi": roi})
            self.measure_button.setEnabled(True)
            self.center_card.set_value(metrics.center_x)
            self.width_card.set_value(metrics.width_px)
            self.spacing_card.set_value(metrics.spacing_px)
            self.tilt_card.set_value(metrics.tilt_deg)
            notes = []
            if metrics.band_kind:
                notes.append(f"中心条纹类型：{'亮纹' if metrics.band_kind == 'bright' else '暗纹'}")
            if metrics.num_bands:
                notes.append(f"共 {metrics.num_bands} 条条纹")
            if metrics.confidence is not None:
                notes.append(f"定位置信度 {metrics.confidence:.2f}")
            if metrics.errors:
                notes.append("部分失败：" + "；".join(metrics.errors))
            if metrics.width_mm is not None:
                notes.append(f"宽度 {metrics.width_mm:.6g} mm；间距 {metrics.spacing_mm} mm")
            self.detail.setText("  ·  ".join(notes))
            self.state_pill.set_state(
                "ok" if metrics.center_x is not None else "warn",
                "测量完成" if metrics.center_x is not None else "未定位到中心条纹")
            self.ctx.state.update(
                center_x=metrics.center_x,
                center_conf=metrics.confidence,
                fringe_width_px=metrics.width_px,
                fringe_spacing_px=metrics.spacing_px,
                tilt_deg=metrics.tilt_deg,
            )
            self.ctx.log(f"条纹测量完成：中心 {metrics.center_x} px")

        def failed(exc):
            self._busy = False
            self.measure_button.setEnabled(True)
            self.state_pill.set_state("error", f"测量失败：{exc}")

        method, scale = self.method.currentData(), self.pixel_scale.value() or None
        demo = bool(getattr(self.ctx, "demo_mode", False))
        if demo and (method != "default" or roi or scale):
            failed(RuntimeError("演示模式不支持扩展算法、ROI 与物理标定"))
            return
        self.ctx.run_async(lambda: self.ctx.fringe.measure(frame, center_x) if demo else
                           self.ctx.fringe.measure(frame, center_x, method=method, roi=roi, mm_per_px=scale),
                           on_done=done, on_error=failed, name="fringe")

    def _record(self):
        if self._last_metrics is None:
            return
        import time
        record = {**self._last_metrics.as_dict(), "name": self.record_name.text().strip(),
                  "reading_mm": self.ctx.state.reading_mm, "reading_at": self.ctx.state.reading_at,
                  "at": time.time(), "data_source": "demo" if getattr(self.ctx, "demo_mode", False) else "live"}
        self._records.append(record)
        row = self.history.rowCount()
        self.history.insertRow(row)
        for column, key in enumerate(("name", "width_px", "spacing_px", "reading_mm")):
            self.history.setItem(row, column, QTableWidgetItem(str(record.get(key))))
        self.ctx.timeline.add("fringe_recorded", result=record)

    def _refresh_analysis_overlay(self):
        from app.ui.widgets.center_records import analysis_overlay
        metrics = getattr(self, "_last_metrics", None)
        self.ctx.fringe_overlay = analysis_overlay(
            metrics.details if metrics else {}, contours=self.show_contours.isChecked(),
            intervals=self.show_intervals.isChecked())

    def _export(self):
        if self._records:
            self.ctx.log(f"已导出：{self.ctx.record.export_rows(self._records, stem='fringe')}")

    def _use_result_as_reference(self) -> None:
        center = self.ctx.state.center_x
        if center is None:
            self.ctx.log("还没有中心位置结果")
            return
        self.reference_x.setValue(float(center))

    def _apply_overlay(self) -> None:
        self.ctx.set_overlay_flag("center", self.show_center.isChecked())

    def on_state(self, state: AppState) -> None:
        self.center_card.set_value(None if state.center_x is None else round(state.center_x, 2))
        self.width_card.set_value(None if state.fringe_width_px is None else round(state.fringe_width_px, 2))
        self.spacing_card.set_value(None if state.fringe_spacing_px is None else round(state.fringe_spacing_px, 3))
        self.tilt_card.set_value(None if state.tilt_deg is None else round(state.tilt_deg, 2))
