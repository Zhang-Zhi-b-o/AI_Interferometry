"""② 搜索条纹：自动寻中 / 目标读数 两个 Tab。

自动寻中 = 把中心条纹推到画面中央；目标读数 = 把微分表转到指定读数。
两者都要求「电机已连接」，自动寻中还要求「相机已打开」。
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QWidget

from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.motor_connection import MotorConnection
from app.ui.widgets.common import (
    Card,
    Expander,
    MetricCard,
    StatusPill,
    form_grid,
    hstack,
    label,
    primary_button,
)


class SearchPage(Page):
    key = "search"
    title = "② 搜索条纹"
    hint = "自动寻中：让中心（零级）条纹停在画面正中央；目标读数：把微分表转到你要的数字。"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        self.motor = MotorConnection(ctx)
        self.add(self.motor)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_search_tab(), "自动寻中")
        self.tabs.addTab(self._build_target_tab(), "目标读数")
        self.add(self.tabs)
        self.add_stretch()

    # ------------------------------------------------------------------
    def _build_search_tab(self) -> QWidget:
        page = QWidget()
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        status_card = Card("当前状态")
        self.search_state = StatusPill("待机", "idle")
        self.search_error = MetricCard("中心偏差", "px")
        self.search_center = MetricCard("中心位置", "px")
        status_card.add(hstack(self.search_error, self.search_center, spacing=10))
        card = Card("搜索设置")
        self.search_button = primary_button("开始自动寻中", self._toggle_search)
        card.add(hstack(self.search_button, self.search_state))

        self.forward_positive = QComboBox()
        self.forward_positive.addItem("正转：条纹向右", True)
        self.forward_positive.addItem("正转：条纹向左", False)
        self.forward_positive.currentIndexChanged.connect(self._apply_direction)
        self.direction_mode = QComboBox()
        self.direction_mode.addItem("已知方向单向搜索", "single_direction")
        self.direction_mode.addItem("未知方向双向扩展", "bidirectional")
        self.recognition_mode = QComboBox()
        self.recognition_mode.addItem("连续识别", "continuous")
        self.recognition_mode.addItem("转动—停车识别", "stop_and_detect")
        self.search_direction = QComboBox()
        self.search_direction.addItem("正转开始", "forward")
        self.search_direction.addItem("反转开始", "reverse")
        for widget, key in ((self.direction_mode, "direction_mode"),
                            (self.recognition_mode, "recognition_mode"),
                            (self.search_direction, "search_direction")):
            index = widget.findData(self.ctx.config.get("search." + key, widget.itemData(0)))
            widget.setCurrentIndex(max(0, index))
        card.add(form_grid([("方向策略", self.direction_mode),
                            ("识别节拍", self.recognition_mode),
                            ("起始方向", self.search_direction),
                            ("电机方向", self.forward_positive)], columns=2))
        self.progress = label("搜索范围由标称转速积分估计，不是实测位置。", name="Hint", wrap=True)
        status_card.add(self.progress)

        advanced = Expander("高级参数")
        self.tolerance = QSpinBox()
        self.tolerance.setRange(2, 200)
        self.tolerance.setSuffix(" px")
        self.tolerance.setValue(int(self.ctx.config.get("search.tolerance_px", 12)))
        self.slow_zone = QSpinBox()
        self.slow_zone.setRange(20, 800)
        self.slow_zone.setSuffix(" px")
        self.slow_zone.setValue(int(self.ctx.config.get("search.slow_zone_px", 120)))
        self.fast_gear = QSpinBox()
        self.fast_gear.setRange(1, 10)
        self.fast_gear.setValue(int(self.ctx.config.get("search.fast_gear", 9)))
        self.slow_gear = QSpinBox()
        self.slow_gear.setRange(1, 10)
        self.slow_gear.setValue(int(self.ctx.config.get("search.slow_gear", 10)))
        self.max_seconds = QSpinBox()
        self.max_seconds.setRange(10, 900)
        self.max_seconds.setSuffix(" s")
        self.max_seconds.setValue(int(self.ctx.config.get("search.max_seconds", 180)))
        self.search_gear = QSpinBox()
        self.search_gear.setRange(1, 10)
        self.search_gear.setValue(int(self.ctx.config.get("search.search_gear", 9)))
        self.max_span = QDoubleSpinBox()
        self.max_span.setRange(0, 1000000)
        self.max_span.setSuffix(" 圈（估计）")
        self.max_span.setSpecialValueText("不限范围（仍有超时）")
        self.max_span.setValue(float(self.ctx.config.get("search.search_max_span_turns", 0)))
        self.move_seconds = QDoubleSpinBox()
        self.move_seconds.setRange(0.05, 10)
        self.move_seconds.setValue(float(self.ctx.config.get("search.stop_detect_move_seconds", 0.6)))
        self.settle_seconds = QDoubleSpinBox()
        self.settle_seconds.setRange(0.05, 10)
        self.settle_seconds.setValue(float(self.ctx.config.get("search.stop_detect_settle_seconds", 0.3)))
        advanced.add(form_grid([
            ("居中容差", self.tolerance),
            ("减速区", self.slow_zone),
            ("快档", self.fast_gear),
            ("慢档", self.slow_gear),
            ("最长运行", self.max_seconds),
            ("搜索档位", self.search_gear),
            ("最大搜索半程", self.max_span),
            ("转动时间 (s)", self.move_seconds),
            ("停车稳定 (s)", self.settle_seconds),
        ], columns=2))
        card.add(advanced)
        layout.addWidget(card)
        layout.addWidget(status_card)
        layout.addStretch(1)
        return page

    def _build_target_tab(self) -> QWidget:
        page = QWidget()
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        status_card = Card("当前读数")
        self.target_value = MetricCard("微分表读数", "mm")
        self.target_state = StatusPill("待机", "idle")
        status_card.add(self.target_value)
        status_card.add(self.target_state)
        card = Card("目标", "需要先连接电机、启动微分表读数。")
        self.target_input = QDoubleSpinBox()
        self.target_input.setDecimals(4)
        self.target_input.setRange(0.0, 30.0)
        self.target_input.setSingleStep(0.001)
        self.target_input.setValue(1.0000)
        self.target_tolerance = QDoubleSpinBox()
        self.target_tolerance.setDecimals(4)
        self.target_tolerance.setRange(0.0001, 0.1)
        self.target_tolerance.setSingleStep(0.001)
        self.target_tolerance.setValue(float(self.ctx.config.get("motor.tolerance_mm", 0.001) or 0.001))
        card.add(form_grid([("目标读数 (mm)", self.target_input),
                            ("允许误差 (mm)", self.target_tolerance)], columns=2))
        self.use_current = primary_button("填入当前读数", self._fill_current_reading)
        self.target_button = primary_button("开始转动", self._toggle_target)
        card.add(hstack(self.use_current, self.target_button, spacing=8))
        layout.addWidget(card)
        layout.addWidget(status_card)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    def _apply_direction(self) -> None:
        self.ctx.search.forward_positive = bool(self.forward_positive.currentData())

    def _toggle_search(self) -> None:
        if self.motor.busy:
            self.ctx.log("请等待电机连接操作完成")
            return
        if self.ctx.search.running:
            self.ctx.search.stop("手动停止")
            return
        ok, message = self.ctx.search.start(
            tolerance_px=float(self.tolerance.value()),
            stable_frames=int(self.ctx.config.get("search.stable_frames", 5)),
            slow_zone_px=float(self.slow_zone.value()),
            fast_gear=int(self.fast_gear.value()),
            slow_gear=int(self.slow_gear.value()),
            tick_ms=int(self.ctx.config.get("search.tick_ms", 200)),
            max_seconds=float(self.max_seconds.value()),
            direction_mode=self.direction_mode.currentData(),
            recognition_mode=self.recognition_mode.currentData(),
            search_direction=self.search_direction.currentData(),
            search_gear=self.search_gear.value(),
            search_max_span_turns=self.max_span.value(),
            stop_detect_move_seconds=self.move_seconds.value(),
            stop_detect_settle_seconds=self.settle_seconds.value(),
        )
        self.ctx.log(message)
        if ok:
            self.ctx.state.update(search_running=True)

    def _fill_current_reading(self) -> None:
        value = self.ctx.state.reading_mm
        if value is None:
            self.ctx.log("还没有读数，请到「④ 读数」页启动读数")
            return
        self.target_input.setValue(float(value))

    def _toggle_target(self) -> None:
        if self.motor.busy:
            self.ctx.log("请等待电机连接操作完成")
            return
        if self.ctx.target.running:
            self.ctx.target.stop("手动停止")
            return
        ok, message = self.ctx.target.start(
            target_mm=float(self.target_input.value()),
            tolerance_mm=float(self.target_tolerance.value()),
            gear=int(self.ctx.config.get("motor.default_gear", 10)),
        )
        self.ctx.log(message)
        if ok:
            self.ctx.state.update(target_running=True)

    # ------------------------------------------------------------------
    def on_state(self, state: AppState) -> None:
        self.motor.refresh()
        self.search_button.setEnabled(not self.motor.busy)
        self.target_button.setEnabled(not self.motor.busy)
        running = state.search_running or self.ctx.search.running
        self.search_button.setText("停止自动寻中" if running else "开始自动寻中")
        self.search_button.setProperty("variant", "danger" if running else "primary")
        self.search_button.style().unpolish(self.search_button)
        self.search_button.style().polish(self.search_button)
        self.search_state.set_state("busy" if running else "idle",
                                    "运行中" if running else "待机")
        if state.center_x is not None and state.frame_size:
            error = state.center_x - state.frame_size[0] / 2.0
            self.search_error.set_value(round(error, 1))
            self.search_center.set_value(round(state.center_x, 1))
        self.search_center.set_value(None if state.center_x is None else round(state.center_x, 1))
        self.progress.setText(state.search_message + "\n" + state.search_range_text)
        for control in (self.direction_mode, self.recognition_mode, self.search_direction,
                        self.forward_positive):
            control.setEnabled(not self.ctx.search.active)

        target_running = state.target_running or self.ctx.target.running
        self.target_button.setText("停止转动" if target_running else "开始转动")
        self.target_button.setProperty("variant", "danger" if target_running else "primary")
        self.target_button.style().unpolish(self.target_button)
        self.target_button.style().polish(self.target_button)
        self.target_state.set_state("busy" if target_running else "idle",
                                    "运行中" if target_running else "待机")
        self.target_value.set_value(state.reading_mm)

    def on_show(self) -> None:
        self._apply_direction()
