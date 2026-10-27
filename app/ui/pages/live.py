"""① 实时：开相机、装模型、开始识别。整页只有三个按钮。"""
from __future__ import annotations

import threading

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QLineEdit,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.core.config import ROOT
from app.core.state import AppState
from app.ui.pages.base import Page
from app.ui.widgets.common import (
    Card,
    Expander,
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
        self._camera_busy = False
        self._camera_cancel = threading.Event()
        self._closing = False
        self._build_camera_card()
        self._build_model_card()
        self._build_overlay_card()
        from app.ui.widgets.recording import RecordingCard

        self.add(RecordingCard(ctx))
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
        advanced = Expander("相机参数与画面校正")
        advanced.add(hstack(secondary_button("扫描相机", self._scan_cameras),
                            secondary_button("保存当前画面", self._save_frame), spacing=8))
        self.resolution = QComboBox()
        for size in ((640, 480), (960, 540), (1280, 720), (1280, 960), (1280, 1024),
                     (1600, 1200), (1920, 1080), (2048, 1536), (2592, 1944)):
            self.resolution.addItem(f"{size[0]} × {size[1]}", size)
        index = self.resolution.findData(self.ctx.config.camera_resolution)
        self.resolution.setCurrentIndex(max(0, index))
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(1, 120)
        self.camera_fps.setValue(int(self.ctx.config.get("camera.fps", 30)))
        self.auto_mode = QCheckBox("自动选择相机支持的分辨率与帧率")
        self.auto_mode.setChecked(bool(self.ctx.config.get("camera.auto_mode", True)))
        self.auto_mode.setToolTip("按所选设备的支持范围，优先满足目标帧率，"
                                 "在最高约 200 万像素内选择预览分辨率。"
                                 "取消勾选可手动选择更高分辨率；不自动提高曝光、增益。")
        self.auto_mode.toggled.connect(lambda enabled: self.resolution.setEnabled(not enabled))
        self.resolution.setEnabled(not self.auto_mode.isChecked())
        self.rotation = QDoubleSpinBox()
        self.rotation.setRange(-180, 180)
        self.rotation.setDecimals(2)
        self.rotation.setSuffix(" °")
        self.rotation.setValue(float(self.ctx.config.get("camera.angle_deg", 0)))
        self.enhance = QCheckBox("运动清晰度增强")
        self.enhance.setChecked(bool(self.ctx.config.get("camera.clarity_assist.motion_enabled_by_default", False)))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-20, 0)
        self.exposure.setValue(float(self.ctx.config.get("camera.clarity_assist.motion_exposure", -6)))
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 255)
        self.gain.setValue(float(self.ctx.config.get("camera.clarity_assist.motion_gain", 0)))
        advanced.add(self.auto_mode)
        advanced.add(form_grid([("手动分辨率（重开生效）", self.resolution),
                                ("目标帧率（重开生效）", self.camera_fps),
                                ("旋转校正", self.rotation),
                                ("运动曝光", self.exposure), ("运动增益", self.gain)]))
        advanced.add(self.enhance)
        advanced.add(label("先停转，Shift＋左键只框选条纹区域，等待约 2 秒建立清晰基线。\n"
                           "勾选增强后点击应用；曝光回读状态显示在画面上方。\n"
                           "若仍有拖影，请缩短曝光、增加照明或降低转速。", name="Muted"))
        advanced.add(hstack(secondary_button("应用校正", self._apply_camera_settings),
                             secondary_button("自动校正倾角", self._auto_rotate), spacing=8))
        advanced.add(secondary_button("旋转复位", self._reset_rotation))
        card.add(advanced)
        self.add(card)

    def _build_model_card(self) -> None:
        card = Card("模型与识别", "启动后自动加载默认模型；切换后点击“加载模型”生效。")
        self.model_path = QLineEdit(str(self.ctx.config.model_path))
        self.model_choice = QComboBox()
        self.model_choice.addItem("YOLOv8n · 全量 2954 张（默认）", str(
            ROOT / "models/releases/yolov8n_full2954_20261027.pt"))
        self.model_choice.addItem("YOLOv8n · 人工确认 600 张", str(
            ROOT / "models/releases/yolov8n_human600_20261027.pt"))
        self.model_choice.addItem("历史模型 · 2026-08-11", str(ROOT / "models/current/best.pt"))
        self.model_choice.addItem("自定义权重（展开下方参数）", None)
        self.model_choice.setCurrentIndex(max(0, self.model_choice.findData(self.model_path.text())))
        if self.model_choice.currentData() != self.model_path.text():
            self.model_choice.setCurrentIndex(self.model_choice.count() - 1)
        self.model_choice.currentIndexChanged.connect(self._select_model)
        self.model_path.textChanged.connect(self._sync_model_choice)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(float(self.ctx.config.get("vision.confidence", 0.25)))
        card.add(form_grid([("模型版本", self.model_choice),
                            ("置信度阈值", self.confidence)], columns=2))
        self.load_button = secondary_button("加载模型", self._load_model)
        self.detect_button = secondary_button("开始识别", self._toggle_detect)
        self.detect_button.setEnabled(False)
        self.model_state = StatusPill("未加载", "idle")
        card.add(hstack(self.load_button, self.detect_button, spacing=8))
        card.add(self.model_state)
        advanced = Expander("推理参数与单帧识别")
        advanced.add(form_grid([("权重路径", self.model_path)]))
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.01, 1)
        self.iou.setValue(float(self.ctx.config.get("vision.iou", 0.45)))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 2048)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(self.ctx.config.get("vision.imgsz", 640)))
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda", "cuda:0"])
        self.device.setCurrentText(str(self.ctx.config.get("vision.device", "cpu")))
        self.standard_strategy = QCheckBox("标准检测策略（分类阈值、时序弱框恢复）")
        self.standard_strategy.setChecked(bool(self.ctx.config.get("vision.standard_strategy", True)))
        advanced.add(form_grid([("IoU", self.iou), ("推理尺寸", self.imgsz), ("设备", self.device)]))
        advanced.add(self.standard_strategy)
        advanced.add(label("标准策略使用旧版分类阈值；取消后使用上方置信度与 IoU。", name="Hint", wrap=True))
        advanced.add(secondary_button("应用推理参数", self._apply_detection_settings))
        self.single_button = secondary_button("单帧识别并查看", self._single_detect)
        advanced.add(self.single_button)
        card.add(advanced)
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
        card.add(label("滚轮缩放，左键拖动画面，双击复位；Shift＋左键框选 ROI，右键清除。", name="Muted", wrap=True))
        self.add(card)

    # ------------------------------------------------------------------
    def _toggle_camera(self) -> None:
        if self._camera_busy or self._closing:
            return
        camera = self.ctx.camera
        self._camera_busy = True
        self.camera_button.setEnabled(False)
        self.camera_index.setEnabled(False)
        self._camera_cancel = threading.Event()
        token = self._camera_cancel
        if camera.is_open:
            self.ctx.stop_motion()
            self.ctx.state.update(camera_open=False, fps=None, frame_size=None,
                                  detecting=False, center_x=None, center_conf=None)
            if hasattr(self.ctx, "invalidate_frame"):
                self.ctx.invalidate_frame()
            self.ctx.run_async(lambda: (camera.close(), (False, "相机已关闭"))[1],
                               on_done=self._camera_done, on_error=self._camera_failed,
                               name="camera-close")
            return
        index = int(self.camera_index.currentData())
        resolution, fps = self.resolution.currentData(), self.camera_fps.value()
        options = ({"auto_mode": True} if self.auto_mode.isChecked()
                   and not getattr(self.ctx, "demo_mode", False) else {})
        self.camera_state.set_state("busy", "正在读取能力并连接…" if options else "连接中…")
        self._apply_camera_settings()
        self.ctx.run_async(lambda: camera.open(index, resolution=resolution, fps=fps,
                                             cancel_event=token, **options),
                           on_done=self._camera_done, on_error=self._camera_failed, name="camera-open")

    def _camera_done(self, result):
        if self._closing:
            return
        self._camera_busy = False
        self.camera_button.setEnabled(True)
        self.camera_index.setEnabled(True)
        ok, message = result
        self.ctx.log(message)
        self.ctx.state.update(camera_open=bool(self.ctx.camera.is_open),
                              camera_index=int(self.camera_index.currentData()))
        if not ok:
            self.ctx.bridge.frame.emit(None)

    def _camera_failed(self, exc):
        self._camera_done((False, f"相机操作失败：{exc}"))

    def _scan_cameras(self):
        if getattr(self.ctx, "demo_mode", False):
            self.ctx.log("演示模式不扫描真实设备")
            return
        from app.services.capture import CameraService

        self.ctx.log("正在扫描空闲相机，已占用相机不会重复打开…")

        def done(indices):
            if self._closing:
                return
            current = self.camera_index.currentData()
            indices = sorted(set(indices) | {int(current)})
            self.camera_index.clear()
            for index in indices:
                self.camera_index.addItem(f"相机 {index}", index)
            self.camera_index.setCurrentIndex(max(0, self.camera_index.findData(current)))
            self.ctx.log("扫描结果：" + ", ".join(map(str, indices)))

        self.ctx.run_async(CameraService.scan, on_done=done, name="camera-scan")

    def _apply_camera_settings(self):
        camera = self.ctx.camera
        if hasattr(self.ctx, "stop_motion"):
            self.ctx.stop_motion()
        if hasattr(camera, "configure_transform"):
            camera.configure_transform(self.rotation.value())
        if hasattr(camera, "configure_motion_clarity"):
            camera.configure_motion_clarity(exposure=self.exposure.value(), gain=self.gain.value(),
                                             enabled=self.enhance.isChecked())
        if hasattr(self.ctx, "invalidate_frame"):
            self.ctx.invalidate_frame()
        elif hasattr(self.ctx, "clear_roi"):
            self.ctx.clear_roi()
        self.ctx.log("已应用画面校正与清晰度参数")

    def _reset_rotation(self):
        self.rotation.setValue(0)
        self._apply_camera_settings()

    def _auto_rotate(self):
        frame = self.ctx.frame()
        if frame is None:
            self.ctx.log("没有画面")
            return
        source_rotation = self.rotation.value()

        def work():
            from app.engine.vision.fringe_angle import estimate_fringe_angle_2d

            return estimate_fringe_angle_2d(frame)

        def done(info):
            if self._closing:
                return
            if self.rotation.value() != source_rotation:
                self.ctx.log("旋转参数已改变，请重新执行自动校正")
                return
            correction = info.get("correction_deg")
            if correction is None or float(info.get("confidence", 0)) < 0.2:
                self.ctx.log("条纹角度置信度不足，未应用旋转")
                return
            if float(info.get("curvature") or 0) > 0.08:
                self.ctx.log("条纹弯曲明显，不适合整体旋转校正")
                return
            self.rotation.setValue(self.rotation.value() + float(correction))
            self._apply_camera_settings()

        self.ctx.run_async(work, on_done=done, name="auto-rotate")

    def _save_frame(self, frame=None):
        from app.services.media import save_image

        frame = self.ctx.frame() if frame is None else frame
        if frame is None:
            self.ctx.log("没有可保存的画面")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存画面", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.ctx.run_async(lambda: save_image(path, frame),
                               on_done=lambda target: self.ctx.log(f"已保存：{target}"), name="save-image")

    def _apply_detection_settings(self):
        if hasattr(self.ctx.detect, "configure"):
            self.ctx.detect.configure(confidence=self.confidence.value(), iou=self.iou.value(),
                                      imgsz=self.imgsz.value(), standard_strategy=self.standard_strategy.isChecked())
        self.ctx.log("推理参数已应用")

    def _single_detect(self):
        frame = self.ctx.frame()
        if frame is None or not self.ctx.detect.is_loaded:
            self.ctx.log("请先打开画面并加载模型")
            return
        self.single_button.setEnabled(False)

        def done(result):
            from app.ui.widgets.video import VideoView

            if self._closing:
                return
            self.single_button.setEnabled(True)
            if result.get("error"):
                self.ctx.log("单帧识别失败：" + result["error"])
                return
            image = result.get("annotated")
            image = frame if image is None else image
            dialog = QDialog(self)
            dialog.setWindowTitle("单帧识别结果")
            dialog.resize(900, 650)
            layout = QVBoxLayout(dialog)
            view = VideoView(roi_enabled=False)
            view.set_frame(image)
            layout.addWidget(view)
            layout.addWidget(secondary_button("保存结果图片", lambda: self._save_frame(image)))
            self._single_dialog = dialog
            dialog.show()

        def failed(exc):
            if self._closing:
                return
            self.single_button.setEnabled(True)
            self.ctx.log(f"单帧识别失败：{exc}")

        self.ctx.run_async(lambda: self.ctx.detect.detect(frame, roi=self.ctx.roi),
                           on_done=done, on_error=failed, name="single-detect", serial=True)

    def shutdown(self):
        self._closing = True
        self._camera_cancel.set()

    def _select_model(self, _index: int) -> None:
        path = self.model_choice.currentData()
        if path:
            self.model_path.setText(path)

    def _sync_model_choice(self, path: str) -> None:
        index = self.model_choice.findData(path.strip())
        self.model_choice.blockSignals(True)
        self.model_choice.setCurrentIndex(index if index >= 0 else self.model_choice.count() - 1)
        self.model_choice.blockSignals(False)

    def _load_model(self) -> None:
        if not self.load_button.isEnabled() or self._closing:
            return
        path = self.model_path.text().strip()
        confidence = float(self.confidence.value())
        iou, imgsz, device = self.iou.value(), self.imgsz.value(), self.device.currentText()
        self.load_button.setEnabled(False)
        self.model_state.set_state("busy", "加载并预热中…")

        def work():
            return self.ctx.detect.load(
                path, confidence=confidence,
                iou=iou, imgsz=imgsz, device=device,
            )

        def done(result):
            if self._closing:
                return
            ok, message = result
            self.load_button.setEnabled(True)
            self.ctx.log(message)
            self.ctx.state.update(model_loaded=ok)
            if not ok:
                self.model_state.set_state("error", message)
            else:
                self.model_state.set_state("ok", message)
                self._apply_detection_settings()

        def failed(exc):
            if not self._closing:
                self.load_button.setEnabled(True)
                self.model_state.set_state("error", str(exc))

        self.ctx.run_async(work, on_done=done, on_error=failed, name="load-model", serial=True)

    def _toggle_detect(self) -> None:
        detecting = not self.ctx.state.detecting
        self.ctx.state.update(detecting=detecting)
        if not detecting:
            self.ctx.detect.reset()
            self.ctx.state.update(center_x=None, center_conf=None, detect_count=0, detect_names=[])
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
