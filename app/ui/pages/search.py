"""② 搜索条纹：自动寻中 / 目标读数 两个 Tab。

自动寻中 = 把中心条纹推到画面中央；目标读数 = 把微分表转到指定读数。
两者都要求「电机已连接」，自动寻中还要求「相机已打开」。
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QWidget

from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.motor_connection import MotorConnection
from app.ui.widgets.manual_motor import ManualMotor
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
        self.manual_motor = ManualMotor(ctx)
        manual = Expander("手动电机控制")
        manual.add(self.manual_motor)
        self.add(manual)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_search_tab(), "自动寻中")
        self.tabs.addTab(self._build_target_tab(), "目标读数")
        self.tabs.addTab(self._build_backlash_tab(), "回程差")
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
        self.search_extra = {}
        specs = [
            ("stable_frames", "居中稳定帧数", 5, 1, 100, True),
            ("search_initial_span_turns", "初始搜索半程 (圈)", 6, .1, 100, False),
            ("search_expansion_factor", "每轮范围扩大倍数", 1.6, 1.1, 3, False),
            ("search_min_gear", "最快搜索档位", 9, 1, 10, True),
            ("search_acceleration_step", "每轮加速档数", 0, 0, 3, True),
            ("blur_slowdown_frames", "模糊降速帧数", 3, 1, 60, True),
            ("blur_safe_gear", "模糊安全档位", 10, 1, 10, True),
            ("blur_recovery_clear_frames", "清晰恢复帧数", 5, 1, 60, True),
            ("stop_detect_frames", "停车识别帧数", 2, 1, 30, True),
        ]
        rows = []
        for key, title, default, low, high, integer in specs:
            control = QSpinBox() if integer else QDoubleSpinBox()
            control.setRange(low, high)
            control.setValue(self.ctx.config.get("search." + key, default))
            self.search_extra[key] = control
            rows.append((title, control))
        self.auto_learn = QComboBox()
        self.auto_learn.addItem("自动学习正反转与画面移动关系", True)
        self.auto_learn.addItem("使用指定方向映射", False)
        self.auto_learn.setCurrentIndex(0 if self.ctx.config.get("search.auto_learn_direction", True) else 1)
        rows.append(("方向学习", self.auto_learn))
        advanced.add(form_grid(rows, columns=2))
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
        advanced = Expander("目标读数高级参数")
        self.target_gear = QSpinBox()
        self.target_gear.setRange(1, 10)
        self.target_gear.setValue(int(self.ctx.config.get("motor.default_gear", 10)))
        self.target_timeout = QDoubleSpinBox()
        self.target_timeout.setRange(1, 900)
        self.target_timeout.setValue(60)
        self.target_tick = QSpinBox()
        self.target_tick.setRange(20, 2000)
        self.target_tick.setValue(250)
        self.target_invert = QComboBox()
        self.target_invert.addItem("正转读数增大", False)
        self.target_invert.addItem("正转读数减小", True)
        advanced.add(form_grid([("接近档位", self.target_gear),
            ("最大运行 (s)", self.target_timeout), ("轮询间隔 (ms)", self.target_tick),
            ("读数方向", self.target_invert)], columns=2))
        card.add(advanced)
        card.add(hstack(self.use_current, self.target_button, spacing=8))
        layout.addWidget(card)
        layout.addWidget(status_card)
        layout.addStretch(1)
        return page

    def _build_backlash_tab(self):
        from PySide6.QtWidgets import QVBoxLayout
        page = QWidget()
        layout = QVBoxLayout(page)
        card = Card("回程差测量", "起点 → 正向对齐中心 → 终点换向一次 → 反向对齐中心 → 返回起点。")
        self.backlash_start = QDoubleSpinBox()
        self.backlash_end = QDoubleSpinBox()
        for control in (self.backlash_start, self.backlash_end):
            control.setDecimals(6)
            control.setRange(-100, 100)
            control.setSingleStep(.001)
        self.backlash_end.setValue(1)
        self.backlash_gear = QSpinBox()
        self.backlash_gear.setRange(1, 10)
        self.backlash_gear.setValue(10)
        self.backlash_timeout = QSpinBox()
        self.backlash_timeout.setRange(1, 900)
        self.backlash_timeout.setValue(60)
        card.add(form_grid([("起点 (mm)", self.backlash_start), ("终点 (mm)", self.backlash_end),
            ("实验档位", self.backlash_gear), ("最大运行 (s)", self.backlash_timeout)], columns=2))
        card.add(hstack(primary_button("当前读数标定起点", lambda: self._set_backlash_endpoint(self.backlash_start)),
                        primary_button("当前读数标定终点", lambda: self._set_backlash_endpoint(self.backlash_end))))
        self.backlash_button = primary_button("开始回程差测量", self._toggle_backlash)
        card.add(self.backlash_button)
        self.backlash_status = label("等待实验；正转须使读数增大。", wrap=True)
        self.backlash_result = label("正向对齐 -- · 反向对齐 -- · 回程差 --", wrap=True)
        card.add(self.backlash_status)
        card.add(self.backlash_result)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _set_backlash_endpoint(self, control):
        import time
        value, at = getattr(self.ctx.reading, "latest_stable", self.ctx.reading.latest)()
        if value is None or not 0 <= time.time() - at <= 1.5:
            self.ctx.log("请等待新的可信微分表读数")
            return
        control.setValue(value)

    def _toggle_backlash(self):
        service = getattr(self.ctx, "backlash", None)
        if service is None:
            self.ctx.log("回程差服务尚未初始化")
            return
        if service.active:
            service.stop("用户停止")
            return
        if self.motor.busy:
            self.ctx.log("请等待电机连接操作完成")
            return
        _, message = service.start(start_mm=self.backlash_start.value(),
            end_mm=self.backlash_end.value(), gear=self.backlash_gear.value(),
            max_seconds=self.backlash_timeout.value())
        self.ctx.log(message)

    def on_backlash(self, payload):
        phases = {"move_to_start": "阶段1/5：接近起点", "forward": "阶段2/5：正向寻找中心",
                  "to_end": "阶段3/5：前往终点", "backward": "阶段4/5：反向寻找中心",
                  "to_start": "阶段5/5：返回起点", "done": "已停车"}
        self.backlash_status.setText(payload.get("reason") or phases.get(payload.get("phase"), "等待实验"))
        def value(key):
            number = payload.get(key)
            return "--" if number is None else f"{number:.6f} mm"
        self.backlash_result.setText(f"正向对齐 {value('forward_mm')} · 反向对齐 {value('backward_mm')} · 回程差 {value('backlash_mm')}")

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
            stable_frames=self.search_extra["stable_frames"].value(),
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
            auto_learn_direction=self.auto_learn.currentData(),
            **{key: control.value() for key, control in self.search_extra.items() if key != "stable_frames"},
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
            gear=self.target_gear.value(),
            max_seconds=self.target_timeout.value(), tick_ms=self.target_tick.value(),
            invert=self.target_invert.currentData(),
        )
        self.ctx.log(message)
        if ok:
            self.ctx.state.update(target_running=True)

    # ------------------------------------------------------------------
    def on_state(self, state: AppState) -> None:
        self.motor.refresh()
        backlash_active = bool(getattr(getattr(self.ctx, "backlash", None), "active", False))
        self.backlash_button.setText("停止回程差测量" if backlash_active else "开始回程差测量")
        self.backlash_button.setEnabled(not self.motor.busy)
        for control in (self.backlash_start, self.backlash_end, self.backlash_gear, self.backlash_timeout):
            control.setEnabled(not backlash_active)
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
