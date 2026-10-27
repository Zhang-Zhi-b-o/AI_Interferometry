"""主窗口：顶栏 + 左侧导航 + 右侧（大画面 + 指标条）+ 页面面板。

装配顺序（改这里之前先看注释）::

    1. 建 core（配置/状态/任务）
    2. 建 services，并把它们的回调统一接到 Qt 信号桥
    3. 建 context，把服务注入
    4. 建界面（导航、画面、页面）
    5. 连定时器（取帧刷新、识别轮询、状态轮询）

画面只在这里推给 ``VideoView``；页面永远不直接持有相机对象。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app import APP_NAME, APP_NAME_EN, __version__
from app.core.config import AppConfig, load_config
from app.core.state import AppState
from app.core.tasks import Background
from app.services.calibration import CalibrationTable
from app.services.backlash import BacklashService
from app.services.media import VideoRecorder
from app.services.capture import CameraService
from app.services.compute import ComputeService
from app.services.detect import DetectService
from app.services.fringe import FringeService
from app.services.motion import MotionService
from app.services.reading import ReadingService
from app.services.record import RecordService
from app.services.search import SearchService, TargetReadingService
from app.services.search_observation import SearchObservation
from app.services.thickness_map import ThicknessMapService
from app.ui import theme as theme_module
from app.ui.bridge import Bridge
from app.ui.context import AppContext
from app.ui.pages.assistant import AssistantPage
from app.ui.pages.compute import ComputePage
from app.ui.pages.fringe import FringePage
from app.ui.pages.live import LivePage
from app.ui.pages.map import MapPage
from app.ui.pages.reading import ReadingPage
from app.ui.pages.search import SearchPage
from app.ui.widgets.common import StatusPill, danger_button, ghost_button, label
from app.ui.widgets.status_bar import MetricsBar, Footer
from app.ui.widgets.video import VideoView

PAGES = (LivePage, SearchPage, FringePage, ReadingPage, ComputePage, MapPage, AssistantPage)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig | None = None, *,
                 demo_image: np.ndarray | None = None, demo_mode: bool = False) -> None:
        super().__init__()
        from app.ui.wheel import install_form_wheel_filter

        install_form_wheel_filter()
        self.config = config or load_config()
        self.state = AppState()
        self.bridge = Bridge()
        self.tasks = Background()
        self.demo_mode = bool(demo_mode)
        self._demo_rig = None
        self._demo_image = demo_image
        self._current_frame: np.ndarray | None = demo_image
        self._detect_busy = False
        self._center_sample = (None, 0, 0.0)
        self._vision_epoch = 0
        self._view_initialized = False
        self._refresh_scheduled = False
        self._current_key: str = PAGES[0].key

        self.ctx = AppContext(
            config=self.config,
            state=self.state,
            bridge=self.bridge,
            tasks=self.tasks,
            frame_provider=self._frame,
            navigate=self._navigate,
            theme_toggle=self._toggle_theme,
            stop_motion=self._emergency_stop,
        )
        self.ctx.demo_mode = self.demo_mode
        self.ctx.clear_roi = lambda: (self.video.set_roi(None), self._on_roi_changed(None))
        self.ctx.invalidate_frame = self._invalidate_frame
        self.ctx.timeline.data_source = "demo" if self.demo_mode else "live"
        self.ctx.timeline.add("session_started")
        self._build_services()
        self._build_ui()
        self._stop_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._stop_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._stop_shortcut.activated.connect(self._emergency_stop)
        if demo_image is not None:
            self._on_frame(demo_image)
        self._connect_signals()
        self._start_timers()
        if self._demo_rig:
            self.state.update(camera_open=True, model_loaded=True, detecting=True,
                              motor_connected=True, message="演示模式 · 所有数据均为模拟")
            self._demo_rig.start()
        self._apply_state()

    # 装配
    def _build_services(self) -> None:
        bridge = self.bridge
        if self.demo_mode:
            from app.services.demo import DemoRig

            self._demo_rig = rig = DemoRig(on_frame=bridge.frame.emit,
                                           on_reading=bridge.reading.emit)
            self.camera, self.detect, self.fringe = rig.camera, rig.detect, rig.fringe
            self.reading, self.motion = rig.reading, rig.motion
            rig.detect.on_detection = bridge.detection.emit
        else:
            self.camera = CameraService(on_frame=bridge.frame.emit, on_status=bridge.message.emit)
            self.detect = DetectService(on_status=bridge.message.emit)
            self.fringe = FringeService()
            self.reading = ReadingService(on_reading=bridge.reading.emit, on_status=bridge.message.emit)
            self.motion = MotionService(on_status=bridge.message.emit)
        self.search = SearchService(
            frame_provider=self.camera.latest_sample,
            motion=self.motion,
            observe=(self._demo_rig.observe if self._demo_rig else
                     SearchObservation(self.detect, roi_provider=lambda: self.ctx.roi,
                                       on_detection=bridge.detection.emit)),
            ready=self._search_problem,
            busy=lambda: self.target.active or self.backlash.active,
            settings=self.config.section("search"),
            on_update=bridge.search_update.emit,
            on_finish=lambda reason: bridge.message.emit(f"自动寻中结束：{reason}"),
        )
        self.target = TargetReadingService(
            reading=self.reading,
            motion=self.motion,
            busy=lambda: self.search.active or self.backlash.active,
            on_update=bridge.target_update.emit,
            on_finish=lambda reason: bridge.message.emit(f"目标读数结束：{reason}"),
        )
        measurement = self.config.section("measurement")
        self.compute = ComputeService(
            refractive_index=float(self.config.refractive_index),
            micrometer_accuracy_mm=float(measurement.get("micrometer_accuracy_mm", 0.0001)),
            refractive_index_tolerance=float(measurement.get("refractive_index_tolerance", 0.001)),
            confidence=float(self.config.get("uncertainty.confidence_level", 0.95)),
        )
        if self.demo_mode:
            from app.services.demo import DemoRecord, DemoThickness

            self.record, self.thickness = DemoRecord(), DemoThickness()
        else:
            self.record, self.thickness = RecordService(), ThicknessMapService()
        self.calibration = CalibrationTable()
        self.backlash = BacklashService(
            reading=self.reading, motion=self.motion,
            center_provider=lambda: self._center_sample,
            busy=lambda: self.search.active or self.target.active,
            ready=self._search_problem,
            on_update=bridge.backlash_update.emit,
            on_finish=bridge.message.emit)
        self.media = VideoRecorder(on_status=bridge.message.emit)
        if not self.demo_mode:
            self.camera.clarity_config = dict(self.config.section("camera.clarity_assist"))
            self.camera.configure_transform(self.config.get("camera.angle_deg", 0))
        self._live_observe = (self._demo_rig.observe if self._demo_rig else
            SearchObservation(self.detect, roi_provider=lambda: self.ctx.roi,
                              on_detection=bridge.detection.emit))

        self.ctx.camera = self.camera
        self.ctx.detect = self.detect
        self.ctx.fringe = self.fringe
        self.ctx.reading = self.reading
        self.ctx.motion = self.motion
        self.ctx.search = self.search
        self.ctx.target = self.target
        self.ctx.compute = self.compute
        self.ctx.record = self.record
        self.ctx.thickness = self.thickness
        self.ctx.calibration = self.calibration
        self.ctx.backlash = self.backlash
        self.ctx.media = self.media

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle(f"{APP_NAME} · v{__version__}" + (" · 演示模式" if self.demo_mode else ""))
        width, height = self.config.window_size
        self.resize(width, height)
        self.setMinimumSize(1180, 640)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())
        root.addLayout(self._build_body(), 1)
        root.addWidget(self._build_footer())

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(72)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        brand = QLabel("AI")
        brand.setFixedSize(38, 38)
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setStyleSheet(
            "background: #4176e6; color: white; border-radius: 10px;"
            "font-size: 17px; font-weight: 700;")

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label(APP_NAME, name="AppTitle"))
        titles.addWidget(label("演示模式 · 合成图像与虚拟设备 · 非真实测量" if self.demo_mode
                               else APP_NAME_EN, name="AppSubtitle"))

        layout.addWidget(brand)
        layout.addLayout(titles)
        layout.addStretch(1)

        self.pill_camera = StatusPill("相机未开", "idle")
        self.pill_motor = StatusPill("电机未连", "idle")
        self.pill_reading = StatusPill("读数未启", "idle")
        for pill in (self.pill_camera, self.pill_motor, self.pill_reading):
            layout.addWidget(pill)

        theme_button = ghost_button("◐ 主题", self._toggle_theme)
        layout.addWidget(theme_button)
        from app.ui.widgets.runtime_log import RuntimeLog

        self.runtime_log = RuntimeLog(self)
        layout.addWidget(ghost_button("运行日志", self.runtime_log.show))
        stop = danger_button("停止运动", self._emergency_stop, minimum_width=112)
        layout.addWidget(stop)
        return bar

    def _build_body(self) -> QHBoxLayout:
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("NavRail")
        nav.setFixedWidth(184)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(12, 16, 12, 12)
        nav_layout.setSpacing(6)
        self._nav_buttons: dict[str, QPushButton] = {}
        self._pages: dict[str, QWidget] = {}
        self.page_host = QStackedWidget()
        self.page_host.setMinimumWidth(460)

        for page_class in PAGES:
            page = page_class(self.ctx)
            self._pages[page.key] = page
            self.page_host.addWidget(page)
            button = QPushButton(f"{page.title}")
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, key=page.key: self._navigate(key))
            nav_layout.addWidget(button)
            self._nav_buttons[page.key] = button
        nav_layout.addStretch(1)
        nav_layout.addWidget(label(f"v{__version__}  ·  重构版", name="Muted"))

        video_column = QWidget()
        video_layout = QVBoxLayout(video_column)
        video_layout.setContentsMargins(16, 16, 16, 12)
        video_layout.setSpacing(12)
        self.video = VideoView()
        self.video.roi_changed.connect(self._on_roi_changed)
        telemetry = QFrame()
        telemetry.setObjectName("VideoTelemetry")
        telemetry_layout = QVBoxLayout(telemetry)
        telemetry_layout.setContentsMargins(10, 6, 10, 6)
        telemetry_layout.setSpacing(3)
        self.video_motor = QLabel()
        self.video_reading = QLabel()
        self.video_detection = QLabel()
        self.video_clarity = QLabel()
        for field in (self.video_motor, self.video_reading, self.video_detection, self.video_clarity):
            field.setWordWrap(True)
            field.setTextFormat(Qt.TextFormat.PlainText)
            telemetry_layout.addWidget(field)
        video_layout.addWidget(telemetry)
        view_controls = QHBoxLayout()
        view_controls.addWidget(label("滚轮缩放 · 拖动画面 · Shift 框选", name="Muted"))
        view_controls.addStretch(1)
        view_controls.addWidget(ghost_button("放大", lambda: self.video.zoom_by(1.2)))
        view_controls.addWidget(ghost_button("缩小", lambda: self.video.zoom_by(1 / 1.2)))
        view_controls.addWidget(ghost_button("复位", self.video.reset_view))
        video_layout.addLayout(view_controls)
        video_layout.addWidget(self.video, 1)
        video_layout.addWidget(self._build_metrics())

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)
        self.splitter.addWidget(video_column)
        self.splitter.addWidget(self.page_host)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([980, 440])

        body.addWidget(nav)
        body.addWidget(self.splitter, 1)
        return body

    def _build_metrics(self) -> QWidget:
        holder = MetricsBar()
        self.metric_center, self.metric_width, self.metric_reading, self.metric_thickness = holder.cards
        return holder

    def _build_footer(self) -> QWidget:
        footer = Footer()
        self.status_label = footer.status
        return footer

    # 信号
    def _connect_signals(self) -> None:
        self.bridge.frame.connect(self._on_frame)
        self.bridge.message.connect(self._on_message)
        self.bridge.reading.connect(self._on_reading)
        self.bridge.detection.connect(self._apply_detection)
        self.bridge.search_update.connect(self._on_search_update)
        self.bridge.target_update.connect(self._on_target_update)
        self.bridge.backlash_update.connect(self._on_backlash_update)
        self.bridge.state.connect(self._schedule_refresh)
        self.state.changed.subscribe(lambda _state: self.bridge.state.emit())

    def _start_timers(self) -> None:
        self._detect_timer = QTimer(self)
        self._detect_timer.setInterval(33)
        self._detect_timer.timeout.connect(self._pump_detection)
        self._detect_timer.start()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start()

        # 只读取内存状态；不在 UI 线程查询串口，也不发送电机命令。
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(250)
        self._telemetry_timer.timeout.connect(self._refresh_video_telemetry)
        self._telemetry_timer.start()

        QTimer.singleShot(0, lambda: self._navigate(PAGES[0].key))

    # 画面与识别
    def _frame(self) -> np.ndarray | None:
        return self._current_frame

    def _on_frame(self, frame) -> None:
        if frame is None:
            self._current_frame = self._demo_image
            self.video.clear()
            return
        if not self.demo_mode and self._demo_image is None and (
                not self.camera.is_open or self._pages["live"]._camera_busy):
            return
        self._current_frame = frame
        self.video.set_frame(frame)
        if not self._view_initialized:
            self.video.zoom_by(float(self.config.get("camera.zoom", 1)))
            self._view_initialized = True
        for source in ("corrected", "annotated"):
            self.media.submit(frame, source=source, overlay=dict(self.video._overlay))
        if self.media.active and self.media.source == "raw":
            self.media.submit(self.camera.raw_frame(), source="raw")
        if self.state.frame_size != (frame.shape[1], frame.shape[0]):
            self.state.update(frame_size=(frame.shape[1], frame.shape[0]))

    def _search_problem(self) -> str:
        if not self.camera.is_open or self.camera.frame_age > 1.5:
            return "请打开干涉相机并等待新画面（画面超过 1.5 秒未更新）"
        if not self.detect.is_loaded:
            return "请先加载条纹模型"
        return ""

    def _pump_detection(self) -> None:
        if self.search.active:
            return  # 寻中线程负责当前帧推理，避免重复检测
        if not ((self.state.detecting or self.backlash.active) and self.detect.is_loaded):
            return
        if self._detect_busy or self._current_frame is None:
            return
        self._detect_busy = True
        frame, captured_at = self.camera.latest_sample()
        if frame is None:
            self._detect_busy = False
            return
        if captured_at == getattr(self, "_last_detect_capture", None):
            self._detect_busy = False
            return
        self._last_detect_capture = captured_at
        epoch = self._vision_epoch

        def work():
            return self._live_observe(frame)

        def done(result):
            self._detect_busy = False
            if epoch != self._vision_epoch or not (self.state.detecting or self.backlash.active):
                return
            center = result.get("center_x")
            self._center_sample = (center, frame.shape[1], captured_at)
            self.state.update(center_x=center, center_conf=result.get("confidence"))

        def failed(_exc):
            self._detect_busy = False
            self._center_sample = (None, 0, 0.0)
            self.state.update(center_x=None, center_conf=None)

        self.ctx.run_async(work, on_done=done, on_error=failed, name="detect", serial=True)

    def _apply_detection(self, result: dict) -> None:
        summary = self.detect.describe(result)
        self.video.set_overlay(boxes=result.get("boxes_xyxy", [])
            if self.ctx.overlay_enabled("boxes") else [],
            class_names=result.get("class_names", []), confs=result.get("confs", []))
        self.state.update(
            detect_count=summary["count"],
            detect_names=summary["names"],
        )
        if summary["error"]:
            self._on_message(f"识别失败：{summary['error']}")

    def _on_roi_changed(self, roi) -> None:
        self._vision_epoch += 1
        self.ctx.roi = tuple(roi) if roi else None
        if hasattr(self.camera, "set_clarity_region"):
            self.camera.set_clarity_region(self.ctx.roi)
        self._center_sample = (None, 0, 0.0)
        self.state.update(center_x=None, center_conf=None)
        self.video.clear_overlay()
        self._on_message("已框选 ROI" if roi else "已清除 ROI")

    def _invalidate_frame(self):
        self._last_detect_capture = None
        self._current_frame = None
        self.video.clear()
        self._on_roi_changed(None)

    # ==================================================================
    # 状态回传
    # ==================================================================
    def _on_message(self, text: str) -> None:
        self.runtime_log.append(str(text))
        self.status_label.setText(str(text))
        self.state.update(message=str(text))

    def _on_reading(self, payload: dict) -> None:
        self.media.submit(payload.get("frame"), source="reading")
        value = payload.get("value_mm")
        self.state.update(reading_mm=value, reading_at=payload.get("at"))
        page = self._pages.get("reading")
        if page is not None:
            page.on_reading(payload)

    def _on_search_update(self, payload: dict) -> None:
        updates = {}
        if "center" in payload:
            updates["center_x"] = payload["center"]
            updates["center_conf"] = payload.get("confidence")
        if payload.get("message"):
            updates["search_message"] = payload["message"]
        low, high = payload.get("searched_min_turns"), payload.get("searched_max_turns")
        if low is not None and high is not None:
            updates["search_range_text"] = f"估算已搜范围 {low:.2f} ～ {high:.2f} 圈（非实测位置）"
        if payload.get("gear") is not None:
            updates["search_gear"] = int(payload["gear"])
        if "running" in payload:
            updates["search_running"] = bool(payload["running"])
        if payload.get("reason"):
            updates["search_message"] = payload["reason"]
        if updates:
            self.state.update(**updates)
        reason = payload.get("reason")
        if reason and payload.get("running") is False:
            self.ctx.timeline.add("search_finished", result={
                "reason": reason, "failed": any(word in reason for word in
                    ("失败", "超时", "异常", "失联", "过期", "未找到"))})
            self._on_message(f"自动寻中结束：{reason}")

    def _on_target_update(self, payload: dict) -> None:
        if payload.get("running") is False:
            self.state.update(target_running=False)
        value = payload.get("value")
        if value is not None:
            self.state.update(reading_mm=float(value))
        reason = payload.get("reason")
        if reason and payload.get("running") is False:
            self.ctx.timeline.add("target_finished", result={"reason": reason})
            self._on_message(f"目标读数结束：{reason}")

    def _on_backlash_update(self, payload):
        self._pages["search"].on_backlash(payload)
        if payload.get("running") is False:
            self.ctx.timeline.add("backlash_finished", result=payload)

    def _schedule_refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        QTimer.singleShot(40, self._apply_state)

    def _apply_state(self) -> None:
        self._refresh_scheduled = False
        state = self.state
        self._refresh_video_telemetry()
        # 顶栏胶囊
        self.pill_camera.set_state("ok" if state.camera_open else "idle",
                                  "相机已开" if state.camera_open else "相机未开")
        self.pill_motor.set_state("ok" if self.motion.connected else "idle",
                                  "电机已连" if self.motion.connected else "电机未连")
        self.pill_reading.set_state("ok" if self.reading.connected else "idle",
                                     "读数已启" if self.reading.connected else "读数未启")
        # 指标条
        self.metric_center.set_value(None if state.center_x is None else round(state.center_x, 1))
        self.metric_width.set_value(
            None if state.fringe_width_px is None else round(state.fringe_width_px, 2))
        self.metric_reading.set_value(state.reading_mm)
        self.metric_thickness.set_value(
            None if state.thickness_mm is None else round(state.thickness_mm, 5))
        # 画面叠加
        if self.ctx.overlay_enabled("center") and state.center_x is not None:
            self.video.set_overlay(center_x=state.center_x, center_conf=state.center_conf)
        else:
            self.video.set_overlay(center_x=None, center_conf=None)
        if not self.ctx.overlay_enabled("boxes"):
            self.video.set_overlay(boxes=[])
        self.video.set_overlay(records=getattr(self.ctx, "center_records", []),
                               **getattr(self.ctx, "fringe_overlay", {}))
        # 各页面
        page = self._pages.get(self._current_key)
        if page is not None:
            page.on_state(state)

    def _refresh_video_telemetry(self) -> None:
        import time
        from app.engine.hardware.motor import MOTOR_GEAR_TABLE
        from app.engine.vision.annotations import STYLES

        state, motion = self.state, self.motion
        connected = motion.connected
        direction = {"forward": "正转指令", "reverse": "反转指令",
                     "stopped": "停止指令"}.get(motion.direction, "状态未知")
        gear = motion.gear if connected else None
        entry = MOTOR_GEAR_TABLE.get(gear)
        speed = "标称转速 —"
        if entry:
            # omega [°/s] × 60 [s/min] / 360 [°/rev] = rpm；不是编码器实测。
            speed = f"标称转速 {entry['omega_deg_s'] / 6:.1f} rpm"
        prefix = "模拟 · " if self.demo_mode else ""
        self.video_motor.setText(prefix + f"电机：{'已连接 · ' + direction if connected else '未连接'}"
                                 f"  |  档位 {gear if gear is not None else '—'}  |  {speed}")
        reading = "—" if state.reading_mm is None else f"{state.reading_mm:.4f} mm"
        age = ""
        if state.reading_at is not None:
            age = f" · {max(0, time.time() - state.reading_at):.1f} 秒前"
        reading_status = "采集中" if self.reading.connected else "未启动 / 已停止"
        self.video_reading.setText(f"当前读数：{reading}  |  {reading_status}{age}")
        detection_status = "检测中" if state.detecting else "已暂停" if state.model_loaded else "模型未加载"
        names = "、".join(STYLES.get(name, (name, ""))[0]
                         for name in dict.fromkeys(state.detect_names)) or "无"
        fps = f"{state.fps:.1f}" if state.camera_open and state.fps is not None else "—"
        self.video_detection.setText(f"检测：{detection_status}  |  最近结果 {state.detect_count} 个框"
                                     f"  |  类别：{names}  |  {fps} FPS")
        clarity = getattr(self.camera, "clarity_status", {})
        self.video_clarity.setVisible(bool(clarity.get("enabled")))
        if clarity.get("enabled"):
            verified = "曝光回读一致" if clarity.get("hardware_verified") else "曝光未核验"
            quality = "低于清晰基线" if clarity.get("blurred") else "未低于清晰基线"
            exposure = clarity.get("exposure")
            exposure_text = "—" if exposure is None else f"{exposure:g}"
            warning = clarity.get("warning", "")
            self.video_clarity.setText(f"运动增强：{verified} · 曝光参数 {exposure_text}"
                f" · {clarity.get('region', '中央区域')} · {quality}"
                + (f"\n{warning}" if warning else ""))

    def _poll_status(self) -> None:
        changes = {}
        if self.camera.is_open:
            fps = round(self.camera.fps, 1)
            if fps and abs((self.state.fps or 0) - fps) > 0.5:
                changes["fps"] = fps
        changes["motor_connected"] = self.motion.connected
        self.state.update(**changes)

    # ==================================================================
    # 导航与主题
    # ==================================================================
    def _navigate(self, key: str) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        self._current_key = key
        self.page_host.setCurrentWidget(page)
        for name, button in self._nav_buttons.items():
            button.setChecked(name == key)
        page.on_show()
        page.on_state(self.state)

    def _toggle_theme(self) -> None:
        from PySide6.QtWidgets import QApplication

        new_name = "dark" if getattr(self, "_theme_name", self.config.theme) != "dark" else "light"
        theme_module.apply_theme(QApplication.instance(), new_name, self.config.font_scale)
        self._theme_name = new_name
        self._on_message(f"主题：{'深色' if new_name == 'dark' else '浅色'}")

    def _emergency_stop(self) -> None:
        self._pages["assistant"].cancel()
        self._pages["search"].motor.cancel_pending()
        try:
            self.search.stop("急停")
            self.target.stop("急停")
            self.backlash.stop("急停")
            self.motion.stop()
        finally:
            self.state.update(search_running=False, target_running=False)
            self.ctx.timeline.add("emergency_stop", result={"requested": True})
            self._on_message("已停止电机")

    # ==================================================================
    # 生命周期
    # ==================================================================
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._pages["assistant"].shutdown()
        self._pages["live"].shutdown()
        self._pages["fringe"]._timer.stop()
        self.media.stop(wait=False)
        self._pages["compute"]._report_cancel.set()
        self._pages["search"].motor.shutdown()
        if self._demo_rig:
            self._demo_rig.close()
        for timer in (getattr(self, "_detect_timer", None),
                      getattr(self, "_telemetry_timer", None),
                      getattr(self, "_status_timer", None)):
            if timer is not None:
                timer.stop()
        try:
            self.search.stop("程序退出")
            self.target.stop("程序退出")
            self.backlash.stop("程序退出")
        except Exception:
            pass
        self.motion.begin_disconnect()
        self.tasks.submit(self.motion.disconnect, name="motor-close")
        self.tasks.submit(self.camera.close, name="camera-close")
        for service, method in ((self.reading, "stop"),):
            try:
                getattr(service, method)()
            except Exception:
                pass
        self.tasks.shutdown()
        self.ctx.timeline.add("session_closed")
        super().closeEvent(event)
