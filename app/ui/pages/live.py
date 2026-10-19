"""① 实时：开相机、装模型、开始识别。整页只有三个按钮。"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QWidget

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


class LivePage(Page):
    key = "live"
    title = "① 实时画面"
    hint = "三步：打开相机 → 加载模型 → 开始识别。识别结果会画在右侧画面上。"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        self._build_camera_card()
        self._build_model_card()
        self._build_overlay_card()
        self.add_stretch()

    # ------------------------------------------------------------------
    def _build_camera_card(self) -> None:
        card = Card("相机", "第一相机看干涉条纹；打不开就换个序号试试。")
        self.camera_index = QComboBox()
        for index in range(6):
            self.camera_index.addItem(f"相机 {index}", index)
        self.camera_index.setCurrentIndex(int(self.ctx.config.camera_index))

        self.camera_button = primary_button("打开相机", self._toggle_camera)
        self.camera_state = StatusPill("未打开", "idle")
        card.add(form_grid([("相机序号", self.camera_index)]))
        card.add(hstack(self.camera_button, self.camera_state, spacing=10))
        self.add(card)

    def _build_model_card(self) -> None:
        card = Card("模型与识别", "模型默认 models/current/best.pt，可改成别的权重。")
        self.model_path = QLineEdit(str(self.ctx.config.model_path))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(float(self.ctx.config.get("vision.confidence", 0.25)))
        card.add(form_grid([("权重路径", self.model_path),
                            ("置信度阈值", self.confidence)]))
        self.load_button = secondary_button("加载模型", self._load_model)
        self.detect_button = secondary_button("开始识别", self._toggle_detect)
        self.detect_button.setEnabled(False)
        self.model_state = StatusPill("未加载", "idle")
        card.add(hstack(self.load_button, self.detect_button, spacing=8))
        card.add(self.model_state)
        self.add(card)

    def _build_overlay_card(self) -> None:
        card = Card("画面叠加层", "叠加层只影响显示，不会进入计算。")
        self.show_center = QCheckBox("显示中心条纹位置")
        self.show_center.setChecked(True)
        self.show_center.toggled.connect(
            lambda value: self.ctx.set_overlay_flag("center", value))
        self.show_boxes = QCheckBox("显示检测框")
        self.show_boxes.setChecked(True)
        self.show_boxes.toggled.connect(
            lambda value: self.ctx.set_overlay_flag("boxes", value))
        card.add(self.show_center)
        card.add(self.show_boxes)
        card.add(label("左键拖拽画面可框选 ROI（黄色虚线框），右键清除。", name="Muted", wrap=True))
        self.add(card)

    # ------------------------------------------------------------------
    def _toggle_camera(self) -> None:
        camera = self.ctx.camera
        if camera.is_open:
            camera.close()
            self.ctx.state.update(camera_open=False, fps=None, frame_size=None)
            self.ctx.bridge.frame.emit(None)
            return
        index = int(self.camera_index.currentData())
        resolution = self.ctx.config.camera_resolution
        ok, message = camera.open(index, resolution=resolution)
        self.ctx.log(message)
        if ok:
            self.ctx.state.update(camera_open=True, camera_index=index)

    def _load_model(self) -> None:
        path = self.model_path.text().strip()
        confidence = float(self.confidence.value())
        self.load_button.setEnabled(False)
        self.model_state.set_state("busy", "加载中…")

        def work():
            return self.ctx.detect.load(
                path, confidence=confidence,
                iou=float(self.ctx.config.get("vision.iou", 0.45)),
                imgsz=int(self.ctx.config.get("vision.imgsz", 640)),
                device=str(self.ctx.config.get("vision.device", "cpu")),
            )

        def done(result):
            ok, message = result
            self.load_button.setEnabled(True)
            self.ctx.log(message)
            self.ctx.state.update(model_loaded=ok)
            if not ok:
                self.model_state.set_state("error", message)
            else:
                self.model_state.set_state("ok", message)

        self.ctx.run_async(work, on_done=done, name="load-model")

    def _toggle_detect(self) -> None:
        detecting = not self.ctx.state.detecting
        self.ctx.state.update(detecting=detecting)
        if detecting and not self.ctx.state.camera_open:
            self.ctx.log("请先打开相机")
            self.ctx.state.update(detecting=False)

    # ------------------------------------------------------------------
    def on_state(self, state: AppState) -> None:
        self.camera_button.setText("关闭相机" if state.camera_open else "打开相机")
        if state.camera_open:
            fps = f"{state.fps:.0f} fps" if state.fps else ""
            self.camera_state.set_state("ok", f"已打开 {fps}".strip())
        else:
            self.camera_state.set_state("idle", "未打开")

        if state.model_loaded:
            self.model_state.set_state("ok", "模型已就绪")
        elif not self.load_button.isEnabled():
            pass
        else:
            self.model_state.set_state("idle", "未加载")

        self.detect_button.setEnabled(state.model_loaded and state.camera_open)
        self.detect_button.setText("停止识别" if state.detecting else "开始识别")
        if state.detecting:
            self.detect_button.setProperty("variant", "danger")
        else:
            self.detect_button.setProperty("variant", "secondary")
        self.detect_button.style().unpolish(self.detect_button)
        self.detect_button.style().polish(self.detect_button)

    # 提供当前叠加层开关给主窗口
    def overlay_options(self) -> dict:
        return {"center": self.show_center.isChecked(),
                "boxes": self.show_boxes.isChecked()}
