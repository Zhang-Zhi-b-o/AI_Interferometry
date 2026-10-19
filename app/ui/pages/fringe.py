"""③ 条纹：一次测量同时给出位置 / 宽度 / 间距 / 倾角。"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QWidget

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
        self.add_stretch()

    # ------------------------------------------------------------------
    def _measure(self) -> None:
        frame = self.ctx.frame()
        if frame is None:
            self.state_pill.set_state("error", "没有画面，请先打开相机")
            return
        reference = self.reference_x.value()
        center_x = None if reference < 0 else float(reference)
        self.measure_button.setEnabled(False)
        self.state_pill.set_state("busy", "测量中…")

        def done(metrics):
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
            self.measure_button.setEnabled(True)
            self.state_pill.set_state("error", f"测量失败：{exc}")

        self.ctx.run_async(lambda: self.ctx.fringe.measure(frame, center_x),
                           on_done=done, on_error=failed, name="fringe")

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
