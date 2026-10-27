"""④ 读数：第二相机拍微分表，本地 OCR 出读数，一键记进历史。"""
from __future__ import annotations

import time

from PySide6.QtWidgets import (QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                               QLabel, QToolButton, QTableWidget, QTableWidgetItem, QWidget)

from app.ui.widgets.video import VideoView
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


class ReadingPage(Page):
    key = "reading"
    title = "④ 读数"
    hint = "用第二台相机读微分表。读数稳定后会自动显示，点「记录」写进历史。"
    MAX_ROWS = 200

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        value_card = Card("当前读数")
        self.value_card = MetricCard("微分表读数", "mm",
                                     decimals=int(ctx.config.get("reading.decimal_places", 3)))
        self.stable_pill = StatusPill("未启动", "idle")
        value_card.add(hstack(self.value_card, self.stable_pill, spacing=8))
        self.add(value_card)

        control = Card("读数相机", "填第二相机的序号，一般是 0。")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        self.camera_index.setValue(int(self.ctx.config.get("reading.camera_index", 0)))
        self.camera_index.setKeyboardTracking(False)
        self.camera_index.setMinimumWidth(80)
        self.camera_index.setToolTip("编号从 0 开始；修改后重新启动读数生效。主相机与读数相机应选不同编号。")
        self.interval = QSpinBox()
        self.interval.setRange(80, 2000)
        self.interval.setSuffix(" ms")
        self.interval.setValue(int(self.ctx.config.get("reading.interval_ms", 200)))
        self.start_button = primary_button("启动读数", self._toggle_reading)
        self.record_button = secondary_button("记录这次读数", self._record)
        self.record_button.setEnabled(False)
        control.add(hstack(self.start_button, self.record_button, spacing=8))
        self.detect_camera_button = secondary_button("检测读数相机", self._detect_cameras)
        self.camera_list = QLabel("可用摄像头：未检测")
        self.camera_list.setWordWrap(True)
        self.available_cameras = []
        control.body.insertWidget(0, hstack(label("相机序号"), self.camera_index,
                                            self.detect_camera_button, spacing=8))
        self.raw_label = QLabel("单帧 --  │  置信度 --")
        self.raw_label.setWordWrap(True)
        self.preview = VideoView(roi_enabled=True, empty_text="读数相机未打开\n启动读数后显示预览")
        self.preview.setMinimumSize(240, 180)
        self.preview.roi_changed.connect(self._roi_selected)
        control.add(self.preview)
        control.add(hstack(
            secondary_button("放大", lambda: self.preview.zoom_by(1.2)),
            secondary_button("缩小", lambda: self.preview.zoom_by(1 / 1.2)),
            secondary_button("复位", self.preview.reset_view), spacing=8))
        control.add(self.raw_label)
        self._build_advanced(control)
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
    def _detect_cameras(self) -> None:
        if getattr(self.ctx, "demo_mode", False):
            self.camera_list.setText("演示模式使用模拟相机")
            return
        self.detect_camera_button.setEnabled(False)
        self.camera_list.setText("正在后台检测摄像头…")

        def work():
            from app.services.capture import CameraService

            return CameraService.scan()

        def done(indices):
            self.available_cameras = list(indices)
            self.camera_list.setText("可用摄像头：" + ("、".join(map(str, indices)) or "未检测到"))
            self.detect_camera_button.setEnabled(True)
            self.ctx.log(self.camera_list.text())

        def failed(exc):
            self.camera_list.setText(f"相机检测失败：{exc}")
            self.detect_camera_button.setEnabled(True)
            self.ctx.log(self.camera_list.text())

        self.ctx.run_async(work, on_done=done, on_error=failed, name="reading-camera-detect")

    def _build_advanced(self, control) -> None:
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("高级参数 / LCD 识别区域")
        self.advanced_toggle.setCheckable(True)
        self.advanced = QWidget()
        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self.advanced)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(form_grid([("识别间隔", self.interval)]))
        layout.addWidget(self.camera_list)
        cfg = self.ctx.config
        self.options = {}
        rows = []
        for key, title, default, low, high, decimals in (
            ("fps", "采集帧率", 15, 1, 120, None),
            ("decimal_places", "小数位数", 3, 0, 6, None),
            ("min_score", "最低识别置信度", 0.45, 0, 1, 2),
            ("stable_window", "稳定窗口帧数", 7, 1, 100, None),
            ("stable_required", "稳定所需帧数", 3, 1, 100, None),
            ("max_step_mm", "单步突变阈值 (mm)", 0.05, 0.000001, 100, 6),
            ("jump_required", "突变确认帧数", 6, 1, 100, None),
            ("scale_ratio_tolerance", "数量级误识别容差", 0.03, 0, 1, 3),
        ):
            widget = QSpinBox() if decimals is None else QDoubleSpinBox()
            if decimals is not None:
                widget.setDecimals(decimals)
            widget.setRange(low, high)
            widget.setValue(cfg.get("reading." + key, default))
            self.options[key] = widget
            rows.append((title, widget))
        resolution = cfg.get("reading.resolution", (1280, 1024))
        self.resolution_width, self.resolution_height = QSpinBox(), QSpinBox()
        for widget, value in zip((self.resolution_width, self.resolution_height), resolution):
            widget.setRange(160, 7680)
            widget.setValue(int(value))
        rows.extend([("图像宽度", self.resolution_width), ("图像高度", self.resolution_height)])
        self.model_path = QLineEdit(str(cfg.get(
            "reading.model_path", "models/micrometer/PP-OCRv6_rec_small.onnx")))
        self.auto_roi = QCheckBox("自动定位 LCD")
        self.auto_roi.setChecked(bool(cfg.get("reading.auto_roi", True)))
        self.roi_input = QLineEdit(",".join(str(v) for v in cfg.get("reading.roi", (0, 0, 1, 1))))
        self.roi_input.setPlaceholderText("x, y, 宽, 高（0～1）；Shift＋左键框选")
        rows.extend([("OCR 模型路径", self.model_path), ("识别区域", self.auto_roi),
                     ("手动 ROI（归一化）", self.roi_input)])
        layout.addWidget(form_grid(rows))
        layout.addWidget(secondary_button("应用识别区域", self._apply_roi))
        note = QLabel("框选后立即使用手动区域；右键清除恢复自动定位。其他参数在下次启动时生效。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        control.add(self.advanced_toggle)
        control.add(self.advanced)

    def _roi_settings(self):
        roi = tuple(float(v.strip()) for v in self.roi_input.text().split(","))
        if (len(roi) != 4 or not all(0 <= v <= 1 for v in roi)
                or roi[2] <= 0 or roi[3] <= 0
                or roi[0] + roi[2] > 1.000001 or roi[1] + roi[3] > 1.000001):
            raise ValueError("ROI 应为 x,y,宽,高，均在 0～1 内且不超出图像")
        return roi

    def _apply_roi(self) -> None:
        try:
            roi = self._roi_settings()
        except ValueError as exc:
            self.ctx.log(str(exc))
            return
        service = self.ctx.reading
        if hasattr(service, "set_roi"):
            service.set_roi(auto_roi=self.auto_roi.isChecked(), roi=roi)
        width, height = self.preview.frame_size()
        if width and height:
            self.preview.set_roi(None if self.auto_roi.isChecked() else
                                 tuple(round(v * n) for v, n in zip(roi, (width, height, width, height))))
        self.ctx.log("LCD 识别区域已应用")

    def _roi_selected(self, roi) -> None:
        if roi is None:
            self.auto_roi.setChecked(True)
            self.roi_input.setText("0,0,1,1")
        else:
            width, height = self.preview.frame_size()
            if not width or not height:
                return
            self.auto_roi.setChecked(False)
            self.roi_input.setText(",".join(f"{v / n:.8f}" for v, n in zip(roi, (width, height, width, height))))
        self._apply_roi()

    def _toggle_reading(self) -> None:
        service = self.ctx.reading
        if service.connected or getattr(service, "connecting", False):
            service.stop()
            self.ctx.log("读数已停止")
            self.on_state(self.ctx.state)
            return
        try:
            self.camera_index.interpretText()
            settings = {key: widget.value() for key, widget in self.options.items()}
            if settings["stable_required"] > settings["stable_window"]:
                raise ValueError("稳定所需帧数不能大于稳定窗口")
            settings.update(camera_index=int(self.camera_index.value()),
                            interval_ms=int(self.interval.value()),
                            resolution=(self.resolution_width.value(), self.resolution_height.value()),
                            auto_roi=self.auto_roi.isChecked(), roi=self._roi_settings(),
                            model_path=self.model_path.text().strip())
        except ValueError as exc:
            self.ctx.log(str(exc))
            return
        if (getattr(self.ctx.state, "camera_open", False)
                and not getattr(self.ctx, "demo_mode", False)
                and settings["camera_index"] == self.ctx.state.camera_index):
            self.ctx.log("该序号已用于主相机，请停止主相机或为读数填写不同序号；不会自动更改你的选择")
            return
        self.value_card._decimals = settings["decimal_places"]
        if hasattr(service, "start_async"):
            service.start_async(**settings)
            self.ctx.log("正在后台加载 OCR 并连接读数相机…")
        else:  # demo service has no hardware and completes immediately
            _, message = service.start(**settings)
            self.ctx.log(message)
        self.on_state(self.ctx.state)

    def _record(self) -> None:
        value = self.ctx.state.reading_mm
        if value is None:
            self.ctx.log("还没有有效读数")
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        decimals = self.value_card._decimals
        self.table.setItem(row, 1, QTableWidgetItem(f"{value:.{decimals}f}"))
        self.table.setItem(row, 2, QTableWidgetItem(time.strftime("%H:%M:%S")))
        self.table.scrollToBottom()
        self.ctx.history.append({"kind": "reading", "value_mm": float(value),
                                 "at": time.time()})
        self.ctx.timeline.add("reading_recorded", result={"value_mm": value,
                              "captured_at": self.ctx.state.reading_at,
                              "stable": self.ctx.reading.stable})
        if self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(0)
        self.ctx.log(f"已记录读数 {value:.{decimals}f} mm")

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self.ctx.history = [h for h in self.ctx.history if h.get("kind") != "reading"]
        self.ctx.log("读数历史已清空")

    # ------------------------------------------------------------------
    def on_reading(self, payload: dict) -> None:
        self.preview.set_frame(payload.get("frame"))
        if hasattr(self, "raw_label"):
            score = float(payload.get("score", 0) or 0)
            self.raw_label.setText(
                f"单帧 {payload.get('text') or '--'}  │  置信度 {score:.2f}  │  {payload.get('message') or ''}")
            roi = payload.get("roi_xyxy")
            self.preview.clear_overlay()
            if roi is not None:
                self.preview.set_overlay(boxes=[roi])
        self.value_card.set_value(payload.get("value_mm"))
        ready = payload.get("value_mm") is not None
        stable = bool(payload.get("stable"))
        self.record_button.setEnabled(ready)
        if not ready:
            self.stable_pill.set_state("warn", payload.get("message") or "识别中…")
        elif payload.get("reading_held"):
            self.stable_pill.set_state("warn", "保持上次稳定读数（当前帧未确认）")
        elif stable:
            self.stable_pill.set_state("ok", "读数稳定")
        else:
            self.stable_pill.set_state("busy", "读数跳动")

    def on_state(self, state: AppState) -> None:
        service = self.ctx.reading
        connecting = getattr(service, "connecting", False)
        self.start_button.setText("取消连接" if connecting else "停止读数" if service.connected else "启动读数")
        self.start_button.setProperty("variant", "danger" if service.connected else "primary")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        if not service.connected:
            self.preview.clear()
            self.stable_pill.set_state("busy" if connecting else "idle", "正在连接…" if connecting else "未启动")
            self.record_button.setEnabled(False)
        self.value_card.set_value(state.reading_mm)
