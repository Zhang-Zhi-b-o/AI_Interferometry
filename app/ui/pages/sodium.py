"""钠灯拍频：记录两端微分表读数，自动显示钠双线平均波长。"""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
from app.ui.pages.base import Page
from app.ui.widgets.common import Card, MetricCard, form_grid, hstack, label, primary_button, secondary_button


class SodiumPage(Page):
    key = "sodium"
    title = "⑦ 钠灯拍频"
    hint = "第一端模糊 → 清晰 → 中间模糊（不停）→ 清晰 → 另一端停车并记录。"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        settings = Card("双端测量", "先在实时页打开相机，在搜索页连接电机，在读数页启动微分表。")
        self.direction = QComboBox()
        self.direction.addItem("正转", "forward")
        self.direction.addItem("反转", "reverse")
        self.gear = QSpinBox()
        self.gear.setRange(1, 10)
        self.gear.setValue(10)
        self.gear.setToolTip("1最快，10最慢；初次测量建议最慢档")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.5, .99)
        self.threshold.setDecimals(2)
        self.threshold.setSingleStep(.05)
        self.threshold.setValue(float(ctx.config.get("sodium.threshold", .75)))
        self.max_seconds = QSpinBox()
        self.max_seconds.setRange(10, 1800)
        self.max_seconds.setValue(int(ctx.config.get("sodium.max_seconds", 300)))
        settings.add(form_grid([("移动方向", self.direction), ("电机档位", self.gear),
                                ("端部置信阈值", self.threshold), ("超时上限 (s)", self.max_seconds)], columns=2))
        settings.add(label("以相机实际画面识别，使用独立钠灯模型。可先用 Shift 框选有效光斑视野；运行中更改 ROI 将中止。", name="Hint", wrap=True))
        settings.add(label("需稳定识别中间模糊并再次离开后才会在另一端停车；未识别完整顺序将持续等待至超时。", name="Hint", wrap=True))
        self.start_button = primary_button("开始双端测量（确认后自动转动）", self.start)
        self.stop_button = secondary_button("停止测量", lambda: ctx.sodium.stop("手动停止钠灯测量"))
        settings.add(hstack(self.start_button, self.stop_button))
        self.add(settings)
        self.wavelength = MetricCard("钠双线平均波长", "nm", decimals=3)
        self.add(self.wavelength)
        status = Card("测量状态与读数")
        self.message = label("待机", wrap=True)
        self.scores = label("端部 — · 中间 —", name="Muted", wrap=True)
        self.first = MetricCard("第一端微分表读数", "mm", decimals=3)
        self.second = MetricCard("另一端微分表读数", "mm", decimals=3)
        self.delta = MetricCard("微分表读数差（有符号）", "mm", decimals=3)
        status.add(self.message)
        status.add(self.scores)
        status.add(hstack(self.first, self.second))
        status.add(self.delta)
        status.add(label("停车指令写入后，等待至少0.3秒且新收到的有效读数。两端读数记录完成后自动计算平均波长。", name="Hint", wrap=True))
        self.add(status)
        self.add_stretch()
        self._recorded = None
        self._first_recorded = None
        self._last_message = ""
        ctx.bridge.sodium_update.connect(self._on_update)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def start(self):
        ok, message = self.ctx.sodium.start(
            direction=self.direction.currentData(), gear=self.gear.value(),
            threshold=self.threshold.value(), max_seconds=self.max_seconds.value(),
            roi=self.ctx.roi)
        self.ctx.log(message)
        self.refresh()

    def _on_update(self, payload):
        if payload.get("run_id") != self.ctx.sodium.run_id:
            return
        if payload.get("message") != self._last_message:
            self._last_message = payload.get("message", "")
            self.ctx.log(self._last_message)
        if payload.get("first_mm") is not None and self._first_recorded != payload["run_id"]:
            self._first_recorded = payload["run_id"]
            self.ctx.timeline.add("sodium_first_endpoint", result={k: payload.get(k) for k in (
                "first_mm", "first_reading_at", "first_stop_written_at", "direction", "gear",
                "threshold", "model_path", "roi")})
        if payload.get("result") and self._recorded != payload["run_id"]:
            self._recorded = payload["run_id"]
            self.ctx.timeline.add("sodium_two_end_measurement", result=payload["result"])
        self.refresh()

    def refresh(self):
        payload = self.ctx.sodium.snapshot()
        active = self.ctx.sodium.active
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(self.ctx.sodium.running)
        for field in (self.direction, self.gear, self.threshold, self.max_seconds):
            field.setEnabled(not active)
        self.message.setText(payload.get("message", "待机"))
        self.scores.setText("端部置信度 {:.3f} · 中间置信度 {:.3f}".format(
            payload.get("end_score", 0), payload.get("middle_score", 0)))
        self.first.set_value(payload.get("first_mm"))
        self.second.set_value(payload.get("second_mm"))
        result = payload.get("result") or {}
        self.delta.set_value(result.get("reading_difference_mm"))
        self.wavelength.set_value(result.get("mean_wavelength_nm"))

    def on_state(self, state):
        self.refresh()
