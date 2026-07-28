"""摄像头 YOLO 实时检测 + 电机控制 — Tkinter UI"""
from __future__ import annotations

import time
import queue
from concurrent.futures import Future, ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

from src import PROJECT_ROOT
from src.config import config
from src.logging import logger
from src.camera import CameraManager
from src.vision import (
    CenterTracker,
    FringeMotionTracker,
    MicrometerOCR,
    YOLODetector,
    rotate_expand,
    FrameCorrector,
    find_center_in_region,
)
from src.vision.class_names import get_class_confidences, get_non_center_guide
from src.hardware import MicrometerReader, MotorController, SerialCommandQueue
from src.vision.micrometer_ocr import MicrometerOCRResult
from src.control import CenterControlStateMachine
from src.agent import AgentService, AgentSession
from src.ui.theme import (
    APP_BG,
    BORDER,
    FONT,
    MUTED,
    NAVY,
    PRIMARY,
    PRIMARY_SOFT,
    SURFACE,
    TEXT,
    VIDEO_BG,
    style_legacy_tree,
)
import yaml
from src.ui.widgets import (
    VideoRecorderPanel,
    StatusPanel,
    MotorControlPanel,
    LogPanel,
    CameraPluginPanel,
    ModelPluginPanel,
    FringeCenterPluginPanel,
    AgentPluginPanel,
    AutoCenterControlPanel,
    FloatingAssistantWindow,
    MicrometerPluginPanel,
    RecordingSidebar,
    TemporaryMeasurementPanel,
)
from src.ui.widgets.collapsible import CollapsibleFrame
from src.ui.widgets.plugin_toggles import PluginToggleBar
from src.ui.lifecycle import shutdown_motor_safely
from src.ui.runtime_context import build_runtime_context
from src.ui.manual_modules import MANUAL_MODULES
from src.ui.recording_preset import load_recording_preset


def _decide_motor_command_from_boxes(boxes_xyxy: np.ndarray, confs: np.ndarray,
                                      frame_shape: tuple) -> str:
    if len(boxes_xyxy) == 0:
        return "HOLD"
    h, w = frame_shape[:2]
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes_xyxy[best_idx]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = (cx - w / 2) / w, (cy - h / 2) / h
    if abs(dx) < 0.08 and abs(dy) < 0.08:
        return "HOLD"
    return "TURN_RIGHT" if abs(dx) > abs(dy) and dx > 0 else \
           "TURN_LEFT"  if abs(dx) > abs(dy) else \
           "MOVE_DOWN"  if dy > 0 else "MOVE_UP"


def _decide_measurement_direction(
    current_mm: float | None,
    target_mm: float,
    tolerance_mm: float,
) -> str:
    """按微分表误差决定动作：正转增大、反转减小、无读数等待。"""
    if current_mm is None:
        return "wait"
    error = float(target_mm) - float(current_mm)
    if abs(error) <= max(0.0, float(tolerance_mm)):
        return "stop"
    return "forward" if error > 0 else "reverse"


def _backlash_endpoint_reached(
    current_mm: float | None,
    target_mm: float,
    tolerance_mm: float,
    direction: str,
) -> bool:
    """沿单一方向接近或越过端点即算到达，绝不反向精确修正。"""
    if current_mm is None:
        return False
    tolerance = max(0.0, float(tolerance_mm))
    if direction == "forward":
        return float(current_mm) >= float(target_mm) - tolerance
    if direction == "reverse":
        return float(current_mm) <= float(target_mm) + tolerance
    return abs(float(current_mm) - float(target_mm)) <= tolerance


class YoloCamApp:
    PREDICT_INTERVAL_MS = 90
    PREVIEW_INTERVAL_MS = 30
    MOTOR_POLL_MS = 300

    def __init__(self):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter 不可用")

        self.root = tk.Tk()
        self.root.title("AI Interferometry · 白光干涉实验工作台")
        window_size = config.get("ui", "window_size", default=[1600, 1000])
        self.root.geometry(f"{int(window_size[0])}x{int(window_size[1])}")
        self.root.configure(bg=APP_BG)
        self.root.minsize(1180, 760)
        self.root.option_add("*Font", (FONT, 9))
        self.recording_preset = load_recording_preset()
        yolo_cfg = self.recording_preset["yolo"]

        # ---- 核心模块 ----
        model_path = config.resolve_path(str(yolo_cfg["model_path"]))
        self.cam: CameraManager | None = None
        self.detector = YOLODetector(
            str(model_path),
            confidence=float(yolo_cfg["confidence_threshold"]),
            iou=float(yolo_cfg["iou_threshold"]),
            imgsz=int(yolo_cfg["imgsz"]),
            device=str(yolo_cfg["device"]),
        )
        self.corrector = FrameCorrector()
        self.motor: MotorController | None = None
        self.motor_commands = SerialCommandQueue()
        self.auto_controller = CenterControlStateMachine()
        self.micrometer_reader: MicrometerReader | None = None

        # ---- 状态 ----
        self.camera_running = False
        self.predict_running = False
        self.auto_control_enabled = False
        self.motor_connected = False
        self.micrometer_connected = False
        self.micrometer_reading_mm: float | None = None
        self.micrometer_reading_at: float | None = None
        self._measurement_active = False
        self._measurement_target_mm: float | None = None
        self._measurement_job: str | None = None
        self._measurement_started_at = 0.0
        # 回程差测量
        self._backlash_active = False
        self._backlash_phase = ""          # move_to_start / forward / backward / done
        self._backlash_start_mm: float | None = None
        self._backlash_end_mm: float | None = None
        self._backlash_reading_forward: float | None = None
        self._backlash_reading_backward: float | None = None
        self._backlash_job: str | None = None
        self._backlash_started_at = 0.0
        self._backlash_reading_lost_at = 0.0
        self._backlash_approach_direction = ""
        self._backlash_motor_direction = "stopped"
        self._backlash_generation = 0
        self._backlash_approach_phase = ""  # approaching start or endpoint before checking center
        self._measurement_control_reading_mm: float | None = None
        self._measurement_control_reading_at = 0.0
        self._measurement_direction = "stopped"
        self._measurement_generation = 0
        self.fps = 0.0
        self.last_t = time.time()

        # ---- 定时器 ----
        self._preview_job: str | None = None
        self._predict_job: str | None = None
        self._motor_poll_job: str | None = None
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-scan")
        self._micrometer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="micrometer")
        self._inference_future: Future | None = None
        self._inference_context: tuple | None = None
        self._camera_scan_future: Future | None = None
        self._micrometer_future: Future | None = None
        self._micrometer_task_kind = ""
        self._micrometer_job: str | None = None
        self._agent_context_job: str | None = None
        self._micrometer_results: queue.Queue[MicrometerOCRResult] = queue.Queue(maxsize=2)
        self._model_load_future: Future | None = None
        self._model_load_job: str | None = None
        self._last_logged_micrometer: float | None = None
        self._last_micrometer_snapshot: dict = {}
        self._last_micrometer_log_signature: tuple | None = None
        self._last_yolo_log_at = 0.0
        self._last_yolo_log_signature: tuple | None = None
        self._preview_adjusted = False
        self._closing = False
        self._prediction_generation = 0

        # ---- UI 变量 ----
        self.video_label: tk.Label | None = None
        self.status_var: tk.StringVar | None = None

        # ---- ROI 绘制状态 ----
        self._roi_drawing = False
        self._roi_start = (0, 0)
        self._roi_rect_id: int | None = None
        self._roi_canvas: tk.Canvas | None = None
        self._frame_off_x = 0
        self._frame_off_y = 0
        self._frame_scale = 1.0
        self._img_id = None
        self._panning = False
        self._pan_start = (0, 0)

        # ---- 面板引用 ----
        self.plugin_bar: PluginToggleBar | None = None
        self.camera_plugin: CameraPluginPanel | None = None
        self.model_plugin: ModelPluginPanel | None = None
        self.fringe_center_plugin: FringeCenterPluginPanel | None = None
        self.recorder: VideoRecorderPanel | None = None
        self.status: StatusPanel | None = None
        self.motor_panel: MotorControlPanel | None = None
        self.agent_panel: AgentPluginPanel | None = None
        self.assistant_float: FloatingAssistantWindow | None = None
        # Optional mirrors were removed with the former automatic-experiment page.
        self.auto_dashboard = None
        self.auto_device_debug = None
        self.manual_auto_center_panel: AutoCenterControlPanel | None = None
        self.micrometer_panel: MicrometerPluginPanel | None = None
        self.temporary_measurement_panel: TemporaryMeasurementPanel | None = None
        self.log: LogPanel | None = None
        self._manual_scroll_canvas: tk.Canvas | None = None
        # 可折叠外壳
        self._shells: dict[str, CollapsibleFrame] = {}

        # ---- 中心条纹检测状态 ----
        self._center_line_x: float | None = None  # 全帧坐标下的中心 x
        self._center_line_box: tuple | None = None  # 所属预测框 (x1,y1,x2,y2)
        self._center_tracker = CenterTracker(hold_frames=5, max_jump_px=45.0)
        self._center_yolo_misses = 0
        self._center_confidence = 0.0
        self._prediction_frame_width: int | None = None
        self._last_detection_result: dict | None = None  # 最近一次 YOLO 检测结果
        self._last_non_center_guide = {
            "x": None, "confidence": 0.0, "count": 0, "class_name": ""}
        self._fringe_motion_tracker = FringeMotionTracker(
            window_size=int(yolo_cfg["fringe_motion_window"]),
            movement_threshold_px=float(
                yolo_cfg["fringe_motion_threshold_px"]),
            missing_hold_frames=3,
        )
        self._last_fringe_motion = {
            "has_fringe": False,
            "movement": "unknown",
            "movement_text": "尚未检测",
            "delta_x_px": None,
            "source": "",
        }
        self._last_auto_state = ""
        self._last_auto_mapping = "learning"
        self.agent_service = AgentService(context_provider=self._get_agent_context)
        self.agent_session = AgentSession(self.agent_service)

        # ---- 构建 ----
        self._build_ui()
        self._wire_callbacks()
        self._reload_calibration()
        self.log.write("UI 初始化完成")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_refresh_ports()
        self._refresh_agent_context()

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=APP_BG)
        outer.pack(fill=tk.BOTH, expand=True)

        # 顶部产品栏：对应网页应用的全局导航与运行状态。
        topbar = tk.Frame(outer, bg=SURFACE, height=76, highlightthickness=1,
                          highlightbackground=BORDER)
        topbar.pack(side=tk.TOP, fill=tk.X)
        topbar.pack_propagate(False)

        brand = tk.Label(topbar, text="AI", bg=PRIMARY, fg="#ffffff",
                         font=(FONT, 13, "bold"), width=3, height=1)
        brand.pack(side=tk.LEFT, padx=(20, 12), pady=18, ipady=7)
        title_group = tk.Frame(topbar, bg=SURFACE)
        title_group.pack(side=tk.LEFT, pady=13)
        tk.Label(title_group, text="白光干涉智能实验工作台", bg=SURFACE, fg=TEXT,
                 font=(FONT, 15, "bold"), anchor="w").pack(anchor="w")
        tk.Label(title_group, text="AI INTERFEROMETRY  ·  VISION / CONTROL / LAB ASSISTANT",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(anchor="w")

        self.status_var = tk.StringVar(value="状态: 就绪")
        status_badge = tk.Label(topbar, textvariable=self.status_var,
                                bg=PRIMARY_SOFT, fg=PRIMARY,
                                font=(FONT, 9, "bold"), padx=14, pady=7)
        status_badge.pack(side=tk.RIGHT, padx=(8, 20), pady=20)
        for label in ("实验助手", "手动控制", "视觉识别"):
            tk.Label(topbar, text=label, bg=SURFACE, fg=NAVY,
                     font=(FONT, 9), padx=8).pack(side=tk.RIGHT, pady=24)

        manual_header = tk.Frame(
            outer, bg=PRIMARY_SOFT, highlightthickness=1,
            highlightbackground=BORDER)
        manual_header.pack(fill=tk.X)
        tk.Label(
            manual_header,
            text="手动操作  ·  设备连接、画面识别与人工控制",
            bg=PRIMARY_SOFT, fg=PRIMARY, font=(FONT, 10, "bold"),
            anchor="w", padx=18, pady=10,
        ).pack(fill=tk.X)

        manual_page = tk.Frame(outer, bg=APP_BG)
        manual_page.pack(fill=tk.BOTH, expand=True)

        workspace = tk.Frame(manual_page, bg=APP_BG)
        workspace.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # 左侧控制台，采用网页侧边栏结构。
        left_shell = tk.Frame(workspace, bg=SURFACE, width=470,
                              highlightthickness=1, highlightbackground=BORDER)
        left_shell.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left_shell.pack_propagate(False)

        header = tk.Frame(left_shell, bg=SURFACE)
        header.pack(fill=tk.X, padx=12, pady=(12, 6))
        tk.Label(header, text="视频演示控制台", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=TEXT, anchor="w").pack(fill=tk.X)
        tk.Label(header, text="双相机、位置记录、视觉识别与自动寻中", bg=SURFACE,
                 fg=MUTED, anchor="w", font=(FONT, 8)).pack(fill=tk.X, pady=(1, 6))

        temporary_enabled = bool(config.get(
            "temporary_measurement", "enabled", default=False))
        self.plugin_bar = PluginToggleBar(
            header, show_temporary=temporary_enabled)

        # -- 可滚动插件面板区域 --
        lc = tk.Canvas(left_shell, bg=APP_BG, highlightthickness=0, bd=0)
        ls = tk.Scrollbar(left_shell, orient=tk.VERTICAL, command=lc.yview)
        lc.configure(yscrollcommand=ls.set)
        lc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(2, 10))
        ls.pack(side=tk.RIGHT, fill=tk.Y)

        left = tk.Frame(lc, bg=APP_BG)
        lw = lc.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>", lambda e: lc.itemconfigure(lw, width=e.width))
        # 鼠标滚轮：日志区滚日志，其他区滚 canvas
        def _global_scroll(event):
            w = event.widget
            while w is not None:
                if w == self.log._text:  # 鼠标在日志文本框上
                    return  # 让日志自己处理
                if w == self.assistant_float:
                    return  # 浮动助手内的文本区自行处理滚动
                w = w.master
            active = self._manual_scroll_canvas
            if active is not None:
                active.yview_scroll(int(-event.delta/120), "units")
        self.root.bind_all("<MouseWheel>", _global_scroll)

        # 右侧实验画布卡片。
        right = tk.Frame(workspace, bg=SURFACE, highlightthickness=1,
                         highlightbackground=BORDER)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        viewer_header = tk.Frame(right, bg=SURFACE, height=58)
        viewer_header.pack(fill=tk.X, padx=16)
        viewer_header.pack_propagate(False)
        viewer_title = tk.Frame(viewer_header, bg=SURFACE)
        viewer_title.pack(side=tk.LEFT, pady=10)
        tk.Label(viewer_title, text="第一相机 · 干涉条纹画面", bg=SURFACE, fg=TEXT,
                 font=(FONT, 11, "bold"), anchor="w").pack(anchor="w")
        tk.Label(viewer_title, text="白光竖条纹识别 · 中心定位 · ROI 分析",
                 bg=SURFACE, fg=MUTED, font=(FONT, 8)).pack(anchor="w")
        tk.Label(viewer_header, text="●  LIVE VIEW", bg="#ecfdf3", fg="#07883f",
                 font=("Segoe UI", 8, "bold"), padx=10, pady=5).pack(side=tk.RIGHT, pady=16)

        video_shell = tk.Frame(right, bg=VIDEO_BG, padx=1, pady=1)
        video_shell.pack(fill=tk.BOTH, expand=True, padx=16)
        self._roi_canvas = tk.Canvas(video_shell, bg=VIDEO_BG,
                                     highlightthickness=0, bd=0)
        self._roi_canvas.pack(fill=tk.BOTH, expand=True)
        self._roi_canvas.create_text(400, 400, text="摄像头未打开", fill="#fff",
                                      font=(FONT, 14), tags="placeholder")

        viewer_footer = tk.Frame(right, bg=SURFACE, height=44)
        viewer_footer.pack(fill=tk.X, padx=16)
        viewer_footer.pack_propagate(False)
        tk.Label(viewer_footer, text="提示  ·  左键拖拽框选 ROI   |   启用平移后拖动画面   |   滚轮调整侧栏",
                 bg=SURFACE, fg=MUTED, font=(FONT, 8), anchor="w").pack(
                     side=tk.LEFT, fill=tk.Y)
        tk.Label(viewer_footer, text="1280 × 1024", bg=SURFACE, fg=MUTED,
                 font=("Consolas", 8)).pack(side=tk.RIGHT, fill=tk.Y)

        # ROI 鼠标事件
        self._roi_canvas.bind("<ButtonPress-1>", self._on_roi_press)
        self._roi_canvas.bind("<B1-Motion>", self._on_roi_drag)
        self._roi_canvas.bind("<ButtonRelease-1>", self._on_roi_release)

        # 保存引用用于滚动跳转
        self._left_canvas = lc
        self._manual_scroll_canvas = lc
        self._left_frame = left

        self.recording_sidebar = RecordingSidebar(left)
        self.recording_sidebar.pack(fill=tk.X, pady=(0, 6))

        self.advanced_controls = CollapsibleFrame(
            left, "高级参数与完整控制",
            collapsed=True, show_move_buttons=False)
        self.advanced_controls.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            self.advanced_controls.content,
            text="相机画面、YOLO 阈值与 ROI、电机、自动寻中和测量等完整参数。",
            bg=SURFACE, fg=MUTED, font=(FONT, 8),
            anchor="w", justify=tk.LEFT, wraplength=410,
            padx=6, pady=6,
        ).pack(fill=tk.X)
        legacy_container = tk.Frame(
            self.advanced_controls.content, bg=APP_BG)
        legacy_container.pack(fill=tk.X)

        # 实验助手脱离侧栏，作为页面内非模态浮窗覆盖在工作画布上。
        self.assistant_float = FloatingAssistantWindow(workspace)
        self.agent_panel = AgentPluginPanel(self.assistant_float.content)
        self.agent_panel.configure(text="", relief=tk.FLAT, bd=0)
        self.agent_panel.pack(fill=tk.BOTH, expand=True)

        # 界面保留五个一级模块，临时测量作为独立模块按配置启用。
        # 根据配置过滤模块
        active_modules = list(MANUAL_MODULES)
        if not temporary_enabled:
            active_modules = [m for m in active_modules if m.key != "temporary"]

        self._plugin_order = [module.key for module in active_modules]
        self._shells: dict[str, CollapsibleFrame] = {}
        self._module_frames: dict[str, tk.Frame] = {}

        for module in active_modules:
            module_shell = CollapsibleFrame(
                legacy_container, module.title,
                collapsed=True, show_move_buttons=False)
            module_shell.pack(fill=tk.X, pady=5)
            module_shell.on_move = (
                lambda direction, key=module.key:
                self._move_plugin(key, direction)
            )
            self._shells[module.key] = module_shell
            self._module_frames[module.key] = module_shell.content

            tk.Label(
                module_shell.content, text=module.description,
                bg=SURFACE, fg=MUTED, font=(FONT, 8), anchor="w",
                padx=6, pady=5,
            ).pack(fill=tk.X)

            for spec in module.panels:
                key, cls = spec.key, spec.panel_class
                if key == "agent":
                    launcher = tk.Frame(
                        module_shell.content, bg="#eef5ff",
                        highlightthickness=1, highlightbackground="#c9daf3",
                    )
                    launcher.pack(fill=tk.X, padx=4, pady=4)
                    launcher_text = tk.Frame(launcher, bg="#eef5ff")
                    launcher_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                       padx=(10, 6), pady=9)
                    tk.Label(
                        launcher_text, text="AI 实验助手浮窗",
                        bg="#eef5ff", fg=NAVY, font=(FONT, 9, "bold"),
                        anchor="w",
                    ).pack(fill=tk.X)
                    tk.Label(
                        launcher_text,
                        text="可在实验画面内拖动、缩放和收回，不遮断设备操作",
                        bg="#eef5ff", fg=MUTED, font=(FONT, 8), anchor="w",
                    ).pack(fill=tk.X, pady=(2, 0))
                    tk.Button(
                        launcher, text="打开助手",
                        command=lambda: self.assistant_float.show(expand=True),
                        relief=tk.FLAT, bd=0, bg=PRIMARY, fg="#ffffff",
                        activebackground="#1d4ed8", activeforeground="#ffffff",
                        cursor="hand2", font=(FONT, 8, "bold"),
                        padx=11, pady=5,
                    ).pack(side=tk.RIGHT, padx=9, pady=11)
                    continue
                if key == "camera":
                    camera_preset = self.recording_preset["main_camera"]
                    panel = cls(
                        module_shell.content,
                        default_index=int(camera_preset["index"]),
                        clarity_settings=dict(camera_preset["clarity_assist"]),
                    )
                    panel.angle_var.set(str(camera_preset["angle_deg"]))
                    panel.zoom_var.set(str(camera_preset["zoom"]))
                    self.corrector.set_manual_offset(panel.angle)
                    self.corrector.zoom = panel.zoom
                elif key == "model":
                    yolo_preset = self.recording_preset["yolo"]
                    panel = cls(
                        module_shell.content,
                        confidence=float(yolo_preset["confidence_threshold"]),
                        iou=float(yolo_preset["iou_threshold"]),
                        imgsz=int(yolo_preset["imgsz"]),
                    )
                elif key == "motor":
                    panel = cls(
                        module_shell.content,
                        default_port=str(
                            self.recording_preset["motor"]["port"]),
                    )
                elif key == "micrometer":
                    panel = cls(
                        module_shell.content,
                        dict(self.recording_preset["reading_camera"]),
                    )
                elif key == "auto_control":
                    panel = cls(module_shell.content)
                    panel.load_settings(dict(
                        self.recording_preset["auto_center"]))
                else:
                    panel = cls(module_shell.content)
                panel.configure(text=spec.title)
                panel.pack(fill=tk.X, padx=2, pady=4)
                if key != "agent":
                    style_legacy_tree(panel)
                setattr(self, spec.attribute, panel)

        for key in self._plugin_order:
            self.plugin_bar.bind_toggle(
                key, lambda enabled, k=key: self._toggle_plugin(k, enabled))
            self.plugin_bar.bind_jump(
                key, lambda k=key: self._jump_to_plugin(k))
        self.recording_sidebar.search_direction_var.set(str(
            self.recording_preset["auto_center"]["search_direction"]))
        self.recording_sidebar.main_camera_index_var.set(int(
            self.recording_preset["main_camera"]["index"]))
        self.recording_sidebar.reading_camera_index_var.set(int(
            self.recording_preset["reading_camera"]["camera_index"]))
        self.root.after_idle(lambda: self.assistant_float.show())

    # ==================================================================
    # 模块排序与导航
    # ==================================================================
    def _move_plugin(self, key: str, direction: str):
        """将四个一级模块上移或下移一位。"""
        if key not in self._plugin_order:
            return
        index = self._plugin_order.index(key)
        target = index - 1 if direction == "up" else index + 1
        if target < 0 or target >= len(self._plugin_order):
            return
        self._plugin_order[index], self._plugin_order[target] = (
            self._plugin_order[target], self._plugin_order[index])
        self._reorder_shells()

    def _reorder_shells(self):
        for shell in self._shells.values():
            shell.pack_forget()
        for key in self._plugin_order:
            if self.plugin_bar.is_enabled(key):
                self._shells[key].pack(fill=tk.X, pady=5)

    def _jump_to_plugin(self, key: str):
        if key == "assistant" and self.assistant_float is not None:
            self.assistant_float.show(expand=True)
        shell = self._shells.get(key)
        if shell is None:
            return
        if not self.plugin_bar.is_enabled(key):
            return
        self._left_frame.update_idletasks()
        y = shell.winfo_rooty() - self._left_frame.winfo_rooty()
        height = max(1, self._left_frame.winfo_height())
        self._left_canvas.yview_moveto(max(0, y / height))

    # ==================================================================
    # 回调绑定
    # ==================================================================
    def _wire_callbacks(self):
        self.camera_plugin.on_command = self._on_camera_cmd
        self.model_plugin.on_command = self._on_model_cmd
        self.fringe_center_plugin.on_command = self._on_fringe_center_cmd
        self.agent_panel.on_ask = self._on_agent_ask
        self.agent_panel.on_test = self._on_agent_test
        self.agent_panel.on_cancel = self._on_agent_cancel
        self.manual_auto_center_panel.on_command = (
            lambda command: self._on_auto_center_command(
                command, self.manual_auto_center_panel))
        self.recording_sidebar.on_command = self._on_recording_sidebar_command
        self.micrometer_panel.on_command = self._on_micrometer_command
        self.recorder.on_start = self._on_rec_start
        self.recorder.on_stop = self._on_rec_stop
        mp = self.motor_panel
        mp.on_refresh_ports = lambda: self._on_refresh_ports()
        mp.on_connect = lambda p: self._on_motor_connect(p)
        mp.on_disconnect = lambda: self._on_motor_disconnect()
        mp.on_manual_command = lambda c: self._on_manual_command(c)
        if self.temporary_measurement_panel is not None:
            self.temporary_measurement_panel.on_command = self._on_temporary_measurement_cmd

    def _on_recording_sidebar_command(self, command: str) -> None:
        """把视频侧边栏命令转发给现有设备与视觉控制流程。"""
        sidebar = self.recording_sidebar
        if command == "camera_1":
            self.camera_plugin.index_var.set(str(
                sidebar.main_camera_index))
            self._on_camera_cmd("open")
            sidebar.set_status(
                "第一相机已启动" if self.camera_running else "第一相机启动失败，请查看状态")
        elif command == "camera_2":
            self.micrometer_panel.index_var.set(str(
                sidebar.reading_camera_index))
            self._on_micrometer_command(
                "start", self.micrometer_panel.get_settings())
            sidebar.set_status("正在启动第二相机与读数")
        elif command == "record_position":
            self._on_fringe_center_cmd("record")
            sidebar.set_status("已执行当前位置记录")
        elif command == "load_model":
            self._on_model_cmd("load")
            sidebar.set_status("正在加载 YOLO 模型")
        elif command == "start_prediction":
            self._on_model_cmd("start")
            if self.predict_running:
                auto_detect = bool(
                    self.recording_preset["yolo"]["auto_detect_center"])
                if (auto_detect
                        and not self.fringe_center_plugin.auto_detect_var.get()):
                    self.fringe_center_plugin.auto_detect_var.set(True)
                    self.fringe_center_plugin.update_auto_state(True)
                sidebar.set_status(
                    "预测运行中，中心条纹自动检测已开启"
                    if auto_detect else "预测运行中")
            else:
                sidebar.set_status("预测未启动，请确认第一相机和 YOLO 模型状态")
        elif command == "start_auto_center":
            direction = sidebar.search_direction_var.get()
            if direction not in {"forward", "reverse"}:
                direction = "forward"
                sidebar.search_direction_var.set(direction)
            auto_cfg = self.recording_preset["auto_center"]
            self.manual_auto_center_panel.search_mode_var.set(
                str(auto_cfg["search_mode"]))
            self.manual_auto_center_panel.search_direction_var.set(direction)
            # 搜索阶段严格使用人工方向；稳定识别中心条纹后复用原有
            # 方向学习与闭环居中逻辑，此时允许为居中而变向。
            self.manual_auto_center_panel.auto_learn_direction_var.set(
                bool(auto_cfg["auto_learn_direction"]))
            self._on_auto_center_command(
                "start", self.manual_auto_center_panel)
            sidebar.set_status(
                f"正在按已知方向{'正转' if direction == 'forward' else '反转'}"
                "寻找条纹并寻中")
        elif command == "stop_auto_center":
            self._on_auto_center_command(
                "stop", self.manual_auto_center_panel)
            sidebar.set_status("自动寻找与寻中已停止")

    def _get_agent_context(self) -> dict:
        """生成紧凑的只读状态；不向智能体暴露控制对象。"""
        detections = {}
        result = self._last_detection_result or {}
        for name, conf in zip(result.get("class_names", []), result.get("confs", [])):
            detections[str(name)] = max(detections.get(str(name), 0.0), float(conf))
        detection_details = []
        for name, conf, box in zip(
            result.get("class_names", []), result.get("confs", []),
            result.get("boxes_xyxy", []),
        ):
            detection_details.append({
                "class": str(name),
                "confidence": round(float(conf), 4),
                "bbox_xyxy": [round(float(value), 1) for value in box],
            })
        roi = self.model_plugin.get_roi_xywh() if self.model_plugin else None
        meter_index = (
            self.micrometer_reader.camera_index
            if self.micrometer_reader is not None else None)
        motor_details = {
            "ui_status": (
                self.motor_panel.command_status_var.get()
                if self.motor_panel is not None else ""),
            "direction": self.auto_controller.direction,
            "gear": self.auto_controller.gear,
        }
        clarity_status = (
            self.cam.clarity_status()
            if self.cam is not None and self.camera_running else {})
        return build_runtime_context(
            camera_running=self.camera_running, fps=self.fps,
            model_loaded=self.detector.is_loaded(),
            prediction_running=self.predict_running, detections=detections,
            center_x_px=self._center_line_x, fringe_motion=self._last_fringe_motion,
            motor_connected=self.motor_connected,
            motor_mode=self.motor_panel.mode if self.motor_panel else "unknown",
            auto_enabled=self.auto_control_enabled,
            auto_state=(self.manual_auto_center_panel.status_var.get()
                        if self.manual_auto_center_panel else "unknown"),
            auto_control_state=self._last_auto_state,
            micrometer_connected=self.micrometer_connected,
            micrometer_reading_mm=self.micrometer_reading_mm,
            micrometer_reading_at=self.micrometer_reading_at,
            scale_factor=self.recording_preset[
                "reading_camera"]["scale_factor"],
            record_count=(len(self.fringe_center_plugin.records)
                          if self.fringe_center_plugin else 0),
            interferometer_camera_index=(
                self.camera_plugin.camera_index if self.camera_plugin else None),
            micrometer_camera_index=meter_index,
            preview_adjusted=self._preview_adjusted,
            correction={
                "manual_angle_deg": (
                    self.camera_plugin.angle if self.camera_plugin else 0.0),
                "effective_angle_deg": round(
                    float(self.corrector.effective_angle), 3),
                "zoom": round(float(self.corrector.zoom), 3),
                "pan_x": round(float(self.corrector.pan_x), 1),
                "pan_y": round(float(self.corrector.pan_y), 1),
                "motion_enhancement_requested": bool(
                    self.camera_plugin
                    and self.camera_plugin.motion_enhance_enabled),
                "clarity": clarity_status,
            },
            roi_xywh=roi,
            auto_analysis_enabled=bool(
                self.fringe_center_plugin
                and self.fringe_center_plugin.auto_detect_var.get()),
            detection_details=detection_details[:30],
            center_confidence=self._center_confidence,
            frame_width=self._prediction_frame_width,
            micrometer_ocr=dict(self._last_micrometer_snapshot),
            motor_details=motor_details,
            recent_logs=self.log.recent_entries(100) if self.log else [],
            measurement_records=(
                [dict(record) for record in self.fringe_center_plugin.records]
                if self.fringe_center_plugin else []),
            temporary_measurement={
                "active": self._measurement_active,
                "target_mm": self._measurement_target_mm,
                "current_mm": self._measurement_control_reading_mm,
                "direction": self._measurement_direction,
                "direction_text": {
                    "forward": "正转（读数增大）",
                    "reverse": "反转（读数减小）",
                    "waiting": "停车等待读数",
                    "stopped": "停止",
                }.get(self._measurement_direction, self._measurement_direction),
                "status": (
                    self.temporary_measurement_panel.status_var.get()
                    if self.temporary_measurement_panel is not None else ""),
            },
        )

    def _refresh_agent_context(self) -> None:
        """定时把同一份实时快照同步到助手面板。"""
        self._agent_context_job = None
        if self._closing:
            return
        if self.agent_panel is not None:
            self.agent_panel.set_experiment_context(self._get_agent_context())
        self._agent_context_job = self.root.after(
            500, self._refresh_agent_context)

    def _on_agent_ask(self, question: str, include_status: bool):
        context = self._get_agent_context()
        self.agent_panel.set_experiment_context(context)
        self.log.write(
            f"[实验助手] 提问：{question[:180]}；"
            f"附加实时状态={'是' if include_status else '否'}；"
            f"当前步骤={context.get('experiment_progress', {}).get('step_number', '--')}/5")
        if not self.agent_session.ask(question, include_status, context):
            self.agent_panel.append("系统", "上一条问题仍在处理中。")
            self.agent_panel.set_ai_state("上一任务仍在处理中", "warning")
            return
        self.root.after(50, self._poll_agent_response)

    def _on_agent_test(self):
        if not self.agent_session.test_connection():
            self.agent_panel.set_ai_state("上一任务仍在处理中", "warning")
            return
        self.root.after(50, self._poll_agent_response)

    def _on_agent_cancel(self):
        if self.agent_session.cancel():
            self.agent_panel.thinking_var.set("正在停止生成…")
            self.agent_panel.set_ai_state("正在停止生成…", "warning")

    def _poll_agent_response(self):
        result = self.agent_session.poll()
        if result is None:
            if not self._closing:
                self.root.after(50, self._poll_agent_response)
            return
        if result.cancelled:
            self.log.write("[实验助手] 回答已由用户取消")
            self.agent_panel.append("系统", "本次回答已停止。")
            self.agent_panel.set_busy(False)
            self.agent_panel.set_ai_state("已取消", "warning")
            return
        if result.error:
            self.log.write(f"[错误] 实验助手处理失败：{result.error}")
            self.agent_panel.append("系统", f"助手处理失败：{result.error}")
            self.agent_panel.set_busy(False)
            self.agent_panel.set_ai_state("处理失败", "error")
            return
        response = result.response
        if response is None:
            self.agent_panel.set_busy(False)
            self.agent_panel.set_ai_state("未收到回答", "error")
            return
        try:
            self.agent_panel.set_connection_status(response.online)
            text = response.answer
            if response.warning:
                text += f"\n\n提示：{response.warning}"
            self.agent_panel.append("助手", text)
            self.log.write(
                f"[实验助手] 回答完成；online={response.online}；"
                f"model={self.agent_service.provider.model}；字符数={len(text)}")
        finally:
            self.agent_panel.set_busy(False)
            self.agent_panel.set_ai_state(
                "回答完成" if response.online else "本地回答完成", "success")

    # ==================================================================
    # 插件开关
    # ==================================================================
    def _toggle_plugin(self, key: str, enabled: bool):
        shell = self._shells.get(key)
        if not shell:
            return
        if enabled:
            self._reorder_shells()
            if key == "assistant" and self.assistant_float is not None:
                self.assistant_float.show()
        else:
            shell.pack_forget()
            if key == "vision":
                self._stop_preview()
                self._stop_predict()
                if self.cam: self.cam.stop(); self.cam = None
                self.camera_running = False
                self._set_status("摄像头已关闭")
                self._center_line_x = None
                self._center_line_box = None
                self._center_tracker.reset()
                self._center_yolo_misses = 0
                self.recorder.stop()
            elif key == "motion":
                self._on_auto_stop()
                self._stop_motor_poll()
                self._on_motor_disconnect()
            elif key == "measurement":
                self._stop_micrometer("微分表读数模块已关闭")
            elif key == "assistant" and self.assistant_float is not None:
                self.assistant_float.hide()

    # ==================================================================
    # 摄像头插件
    # ==================================================================
    def _on_camera_cmd(self, cmd: str):
        cp = self.camera_plugin
        if cmd == "detect":
            if self._camera_scan_future is None:
                self._set_status("正在后台检测摄像头...")
                self._camera_scan_future = self._camera_executor.submit(CameraManager.detect_all)
                self.root.after(50, self._poll_camera_scan)
        elif cmd == "open":
            if self.camera_running: return
            try:
                requested_index = cp.camera_index
                if (self.micrometer_reader is not None
                        and self.micrometer_reader.connected
                        and requested_index == self.micrometer_reader.camera_index):
                    raise RuntimeError(
                        f"摄像头 {requested_index} 正由微分表使用；"
                        "请停止微分表读数或为干涉画面选择其他索引")
                camera_preset = self.recording_preset["main_camera"]
                resolution = camera_preset["resolution"]
                self.cam = CameraManager(
                    index=requested_index,
                    resolution=(int(resolution[0]), int(resolution[1])),
                    fps=int(camera_preset["fps"]),
                    clarity_config=dict(camera_preset["clarity_assist"]),
                    owner="interferometer-camera",
                )
                if not self.cam.start(): raise RuntimeError("无法打开摄像头")
                self.camera_running = True
                if self.auto_device_debug is not None:
                    self.auto_device_debug.camera_var.set(str(requested_index))
                    self.auto_device_debug.status_var.set(
                        f"干涉相机 {requested_index} 已连接")
                self._set_status("摄像头已启动"); self._start_preview()
                self._apply_camera_clarity("摄像头启动")
                self.log.write(
                    f"[相机] 干涉画面摄像头 {requested_index} 已打开，"
                    f"分辨率 {resolution[0]}x{resolution[1]}")
            except Exception as e:
                self.camera_running = False; self.cam = None
                self._set_status(f"摄像头失败: {e}"); self.log.write(f"[错误] {e}")
        elif cmd == "close":
            self._stop_preview(); self._stop_predict()
            if self.cam: self.cam.stop(); self.cam = None
            self.camera_running = False; self._set_status("摄像头已关闭")
            cp.set_clarity_status("摄像头未连接")
            self.log.write("[相机] 干涉画面摄像头已关闭，预测与自动寻中已停止")
        elif cmd == "angle_apply":
            self.corrector.set_manual_offset(cp.angle)
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 已应用旋转角度 {cp.angle:+.2f}°")
        elif cmd == "angle_reset":
            angle = float(self.recording_preset["main_camera"]["angle_deg"])
            cp.angle_var.set(str(angle))
            self.corrector.set_manual_offset(angle)
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 旋转角度已恢复预设 {angle:+.2f}°")
        elif cmd == "zoom_apply":
            self.corrector.zoom = cp.zoom
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 已应用缩放倍数 {cp.zoom:.2f}")
        elif cmd == "zoom_reset":
            zoom = float(self.recording_preset["main_camera"]["zoom"])
            cp.zoom_var.set(str(zoom))
            self.corrector.zoom = zoom
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 缩放已恢复预设 {zoom:.2f}")
        elif cmd == "pan_reset":
            self.corrector.pan_x = 0; self.corrector.pan_y = 0
            self._preview_adjusted = True
            self.log.write("[画面矫正] 平移已复位")
        elif cmd in {"clarity_toggle", "clarity_apply"}:
            self._apply_camera_clarity(
                "开关切换" if cmd == "clarity_toggle" else "参数应用")
        elif cmd == "all_reset":
            self.corrector.reset_all()
            camera_preset = self.recording_preset["main_camera"]
            cp.angle_var.set(str(camera_preset["angle_deg"]))
            cp.zoom_var.set(str(camera_preset["zoom"]))
            self.corrector.set_manual_offset(cp.angle)
            self.corrector.zoom = cp.zoom
            cp.reset_clarity()
            self._apply_camera_clarity("全部参数复位")
            self._preview_adjusted = True
            self.log.write("[画面矫正] 旋转、缩放、平移和清晰度配置已全部复位")

    def _apply_camera_clarity(self, reason: str) -> None:
        """合并手动开关与自动寻中的增强请求，统一控制主相机。"""
        cp = self.camera_plugin
        if cp is None:
            return
        requested = bool(cp.motion_enhance_enabled or self.auto_control_enabled)
        exposure = cp.motion_exposure
        gain = cp.motion_gain
        cp.motion_exposure_var.set(f"{exposure:.1f}")
        cp.motion_gain_var.set(f"{gain:.1f}")
        if self.cam is None or not self.camera_running:
            cp.set_clarity_status(
                f"{'待开启' if cp.motion_enhance_enabled else '未开启'} · "
                f"曝光 {exposure:.1f} · 增益 {gain:.1f}")
            if reason in {"开关切换", "参数应用"}:
                self.log.write("[清晰度] 配置已保存，将在摄像头打开后应用")
            return
        self.cam.configure_motion_clarity(
            exposure=exposure, gain=gain, enabled=requested)
        source = "自动寻中" if self.auto_control_enabled else "手动配置"
        state = f"运动增强 · {source}" if requested else "未开启"
        cp.set_clarity_status(
            f"{state} · 曝光 {exposure:.1f} · 增益 {gain:.1f}")
        self.log.write(
            f"[清晰度] {reason}：{state}，曝光 {exposure:.1f}，增益 {gain:.1f}")

    def _poll_camera_scan(self):
        if self._camera_scan_future is None:
            return
        if not self._camera_scan_future.done():
            self.root.after(50, self._poll_camera_scan)
            return
        try:
            cameras = self._camera_scan_future.result()
            messagebox.showinfo("摄像头检测", f"可用摄像头: {cameras}")
            self.log.write(f"摄像头检测: {cameras}")
            self._set_status("摄像头检测完成")
        except Exception as exc:
            self.log.write(f"[错误] 摄像头检测失败: {exc}")
        finally:
            self._camera_scan_future = None

    # ==================================================================
    # 独立摄像头微分表 OCR 插件
    # ==================================================================
    def _on_micrometer_command(self, command: str, settings: dict | None = None):
        settings = settings or self.micrometer_panel.get_settings()
        if command == "detect":
            if self._micrometer_future is not None:
                return
            self.micrometer_panel.set_status("正在后台检测摄像头...")
            self._micrometer_task_kind = "detect"
            self._micrometer_future = self._micrometer_executor.submit(
                MicrometerReader.detect_cameras)
            self.root.after(80, self._poll_micrometer_task)
            return
        if command == "stop":
            self._stop_micrometer("视觉读数已停止")
            return
        if command != "start" or self._micrometer_future is not None:
            return
        index = int(settings.get("camera_index", 0))
        main_index = self.cam.index if self.camera_running and self.cam is not None else None
        if main_index is not None and index == main_index:
            replacement = self.micrometer_panel.select_available_camera({main_index})
            if replacement is None:
                self.micrometer_panel.set_status(
                    f"摄像头 {main_index} 已用于干涉画面，未找到空闲摄像头")
                return
            settings = dict(settings)
            settings["camera_index"] = replacement
            index = replacement
            if self.auto_device_debug is not None:
                self.auto_device_debug.micrometer_var.set(str(replacement))
            self.micrometer_panel.set_status(
                f"索引 {main_index} 已占用，已自动切换到摄像头 {replacement}")
        self._stop_micrometer("正在初始化 OCR 模型...")
        self.micrometer_panel.set_status("正在后台加载 PP-OCR 模型...")
        self._micrometer_task_kind = "start"
        self._micrometer_future = self._micrometer_executor.submit(
            self._create_micrometer_reader, settings)
        self.root.after(80, self._poll_micrometer_task)

    def _create_micrometer_reader(self, settings: dict) -> MicrometerReader:
        resolution = settings.get("resolution", [1280, 1024])
        ocr = MicrometerOCR(
            model_path=config.resolve_path(str(settings.get(
                "model_path", "models/micrometer/PP-OCRv6_rec_small.onnx"))),
            min_score=float(settings.get("min_score", 0.45)),
            decimal_places=int(settings.get("decimal_places", 3)),
            stable_window=int(settings.get("stable_window", 7)),
            stable_required=int(settings.get("stable_required", 3)),
            max_step_mm=float(settings.get("max_step_mm", 0.05)),
            jump_required=int(settings.get("jump_required", 6)),
            scale_ratio_tolerance=float(settings.get(
                "scale_ratio_tolerance", 0.03)),
        )
        ocr.load()
        if self._closing:
            raise RuntimeError("程序正在关闭")
        reader = MicrometerReader(
            camera_index=int(settings.get("camera_index", 0)),
            resolution=(int(resolution[0]), int(resolution[1])),
            fps=int(settings.get("fps", 15)),
            interval_ms=int(settings.get("interval_ms", 200)),
            auto_roi=bool(settings.get("auto_roi", True)),
            manual_roi=tuple(settings.get("roi", (0.0, 0.0, 1.0, 1.0))),
            ocr=ocr,
        )
        if not reader.connect():
            raise RuntimeError(
                f"无法打开微分表摄像头 index={reader.camera_index}")
        return reader

    def _poll_micrometer_task(self) -> None:
        future = self._micrometer_future
        if future is None:
            return
        if not future.done():
            self.root.after(80, self._poll_micrometer_task)
            return
        kind = self._micrometer_task_kind
        self._micrometer_future = None
        self._micrometer_task_kind = ""
        try:
            result = future.result()
            if kind == "detect":
                self.micrometer_panel.set_camera_list(result)
                if self.auto_device_debug is not None:
                    self.auto_device_debug.set_camera_list(result)
                self.micrometer_panel.set_status("摄像头检测完成")
            else:
                reader = result
                if self._closing:
                    reader.close()
                    return
                self.micrometer_reader = reader
                if self.auto_device_debug is not None:
                    self.auto_device_debug.micrometer_var.set(
                        str(reader.camera_index))
                self.micrometer_connected = True
                self.micrometer_reading_mm = None
                self.micrometer_reading_at = None
                reader.start(self._enqueue_micrometer_result)
                self.micrometer_panel.set_status("已连接，正在识别...")
                self.log.write(
                    f"[微分表] 摄像头 {reader.camera_index} 已连接")
                self._schedule_micrometer_results()
        except Exception as exc:
            self.micrometer_connected = False
            self.micrometer_panel.set_status(f"启动失败：{exc}")
            self.log.write(f"[错误] 微分表读数启动失败: {exc}")

    def _enqueue_micrometer_result(self, result: MicrometerOCRResult) -> None:
        try:
            self._micrometer_results.put_nowait(result)
        except queue.Full:
            try:
                self._micrometer_results.get_nowait()
            except queue.Empty:
                pass
            try:
                self._micrometer_results.put_nowait(result)
            except queue.Full:
                pass

    def _schedule_micrometer_results(self) -> None:
        if self._micrometer_job is None and not self._closing:
            self._micrometer_job = self.root.after(
                100, self._poll_micrometer_results)

    def _poll_micrometer_results(self) -> None:
        self._micrometer_job = None
        latest = None
        while True:
            try:
                latest = self._micrometer_results.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.micrometer_panel.update_result(latest)
            self.recording_sidebar.update_meter_preview(
                self.micrometer_panel.preview_photo,
                self.micrometer_panel.stable_var.get(),
            )
            self._last_micrometer_snapshot = {
                "text": latest.text,
                "value_mm": latest.value_mm,
                "confidence": round(float(latest.score), 4),
                "stable": bool(latest.stable),
                "stable_value_mm": latest.stable_value_mm,
                "reading_held": bool(latest.reading_held),
                "rejected": bool(latest.rejected),
                "rejection_reason": latest.rejection_reason,
                "stable_captured_at": latest.stable_captured_at,
                "roi_xyxy": list(latest.roi_xyxy) if latest.roi_xyxy else None,
                "format_hint": latest.format_hint,
                "message": latest.message,
                "captured_at": latest.captured_at,
            }
            meter_signature = (
                latest.text, round(float(latest.score), 2), latest.stable,
                latest.stable_value_mm, latest.rejected,
                latest.rejection_reason, latest.message)
            if meter_signature != self._last_micrometer_log_signature:
                self.log.write(
                    f"[微分表OCR] text={latest.text or '--'}，"
                    f"confidence={latest.score:.2f}，stable={latest.stable}，"
                    f"raw={latest.value_mm} mm，trusted={latest.stable_value_mm} mm，"
                    f"held={latest.reading_held}，rejected={latest.rejected}，"
                    f"状态={latest.message}")
                self._last_micrometer_log_signature = meter_signature
            if self.auto_dashboard is not None:
                self.auto_dashboard.update_micrometer(latest)
            if latest.stable_value_mm is not None:
                self.micrometer_reading_mm = latest.stable_value_mm
                # 只有新的可信确认帧才更新时间戳；保持旧稳定值时沿用原时间，
                # 让下游知道数值可靠但并非刚刚更新。
                if latest.stable:
                    self.micrometer_reading_at = (
                        latest.stable_captured_at or latest.captured_at)
                if (latest.stable
                        and latest.stable_value_mm != self._last_logged_micrometer):
                    self.log.write(
                        f"[微分表] 稳定读数 {latest.stable_value_mm:.6f} mm")
                    self._last_logged_micrometer = latest.stable_value_mm
            else:
                # 尚未建立过稳定读数时才显示为空；一旦建立便持续保持。
                self.micrometer_reading_mm = None
                self.micrometer_reading_at = None
            # 临时测量同样只使用统一稳定层发布的可信值，不直接采用单帧 OCR。
            control_reading = latest.stable_value_mm
            if control_reading is not None and latest.stable:
                self._measurement_control_reading_mm = float(control_reading)
                self._measurement_control_reading_at = float(
                    latest.stable_captured_monotonic
                    or latest.captured_monotonic or time.monotonic())
            if self.temporary_measurement_panel is not None:
                self.temporary_measurement_panel.set_current_reading(
                    control_reading)
        if self.micrometer_reader is not None and self.micrometer_reader.connected:
            self._schedule_micrometer_results()

    def _stop_micrometer(self, message: str = "视觉读数已停止") -> None:
        if self._measurement_active:
            self._stop_measurement("微分表读数已停止")
        if self._micrometer_job is not None:
            self.root.after_cancel(self._micrometer_job)
            self._micrometer_job = None
        reader = self.micrometer_reader
        self.micrometer_reader = None
        if reader is not None:
            reader.close()
        self.micrometer_connected = False
        self.micrometer_reading_mm = None
        self.micrometer_reading_at = None
        self._last_logged_micrometer = None
        self._last_micrometer_snapshot = {}
        self._last_micrometer_log_signature = None
        self._measurement_control_reading_mm = None
        self._measurement_control_reading_at = 0.0
        if self.micrometer_panel is not None:
            self.micrometer_panel.set_status(message)
        if getattr(self, "recording_sidebar", None) is not None:
            self.recording_sidebar.reset_meter_preview(message)

    # ==================================================================
    # 模型插件
    # ==================================================================
    def _on_model_cmd(self, cmd: str):
        if cmd == "load":
            if self.detector.is_loaded(): self.log.write("YOLO 模型已加载"); return
            if self._model_load_future is not None:
                self.log.write("YOLO 模型正在加载，请稍候")
                return
            self._set_status("正在加载YOLO模型...")
            self.log.write("开始加载YOLO模型（后台）...")
            # 后台线程只加载模型，不接触 Tk。加载结果统一由主线程轮询，
            # 避免跨线程调用 Tcl/Tk 导致 Windows 0xC0000409 Fail Fast。
            self._model_load_future = self._inference_executor.submit(
                self.detector.load)
            self._model_load_job = self.root.after(50, self._poll_model_load)
        elif cmd == "roi_toggle":
            self.log.write(f"ROI模式: {'开' if self.model_plugin.roi_mode else '关'}")
        elif cmd == "roi_clear":
            self.model_plugin.roi_pixels = None
            self.model_plugin.roi_mode_var.set(False)
            self.log.write("ROI 已清除")
        elif cmd == "start":
            if self.predict_running: return
            if self.cam is None or not self.camera_running:
                self.log.write("[警告] 请先打开摄像头"); return
            if not self.detector.is_loaded(): self.log.write("[警告] 请先加载 YOLO 模型"); return
            # 相机管理器持续缓存最新帧，预览和推理可以并行；自动页不再因
            # 单次 YOLO 推理耗时而冻结。
            self.predict_running = True; self._set_status("预测运行中"); self._predict_loop()
            self._start_preview()
            self.log.write(
                f"[YOLO] 连续预测已启动，置信度阈值 {self.model_plugin.conf:.2f}，"
                f"IoU {self.model_plugin.iou:.2f}，推理尺寸 {self.model_plugin.imgsz}")
        elif cmd == "single":
            if self.cam is None or not self.detector.is_loaded():
                self.log.write("[警告] 请先打开摄像头并加载模型"); return
            frame = self.cam.read()
            if frame is None: return
            corrected = rotate_expand(frame, self.corrector.effective_angle)
            corrected = self.corrector.apply_zoom_pan(corrected)
            roi = self._get_roi()
            result = self.detector.detect(corrected, roi=roi)
            annotated = result["annotated"] if result["annotated"] is not None else corrected
            if roi: cv2.rectangle(annotated, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,255,0), 2)
            self._show_frame(annotated)
            self.log.write(f"单帧预测: {len(result['boxes_xyxy'])} 个目标")
            # 弹出新窗口
            top = tk.Toplevel(self.root)
            top.title("单帧预测结果")
            top.configure(bg="#000")
            h, w = annotated.shape[:2]
            top.geometry(f"{w}x{h}")
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl = tk.Label(top, image=img, bg="#000")
            lbl.image = img
            lbl.pack()
            # 右键保存
            def _save(event):
                from tkinter import filedialog as fd
                path = fd.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG","*.png"),("JPG","*.jpg")])
                if path:
                    cv2.imwrite(path, annotated)
                    self.log.write(f"图片已保存: {path}")
            lbl.bind("<Button-3>", _save)
            tk.Label(top, text="右键图片保存", bg="#000", fg="#666",
                     font=("Microsoft YaHei UI", 9)).pack(pady=2)
        elif cmd == "stop":
            self._stop_predict()
            self.log.write("[YOLO] 连续预测已停止，检测结果与中心跟踪已清空")

    def _poll_model_load(self) -> None:
        """仅在 Tk 主线程读取后台模型加载结果并更新界面。"""
        self._model_load_job = None
        future = self._model_load_future
        if future is None or self._closing:
            return
        if not future.done():
            self._model_load_job = self.root.after(50, self._poll_model_load)
            return
        self._model_load_future = None
        try:
            loaded = bool(future.result())
        except Exception as exc:
            loaded = False
            logger.exception("YOLO 模型后台加载任务失败: %s", exc)
        if self._closing:
            return
        if loaded:
            self._set_status("YOLO 模型已加载")
            self.log.write("YOLO 模型已加载")
        else:
            self._set_status("YOLO 加载失败")
            self.log.write("[错误] YOLO 加载失败，请查看 logs/app.log")

    def _hold_or_clear_center(self, reason: str, verbose: bool = False):
        """短时沿用上一帧结果；连续丢失后再清空显示。"""
        tracked = self._center_tracker.update(None)
        if tracked["center"] is not None and self._center_line_box is not None:
            self._center_line_x = tracked["center"]
            self._center_confidence = float(tracked["confidence"])
            x1 = self._center_line_box[0]
            self.fringe_center_plugin.update_result(
                self._center_line_x - x1,
                tracked["confidence"],
                True,
                f"短时保持：{reason}",
            )
            if verbose:
                self.log.write(f"[中心条纹] {reason}，暂用上一帧位置")
            return
        self._center_line_x = None
        self._center_confidence = 0.0
        self._center_line_box = None
        self.fringe_center_plugin.update_result(None, 0, False, reason)

    def _track_center_from_previous_roi(self, corrected: np.ndarray) -> bool:
        """YOLO 漏检时，在上一帧零级区域内继续跟踪白光竖条纹。"""
        if self._center_line_x is None or self._center_line_box is None:
            return False
        x1, y1, x2, y2 = self._center_line_box
        height, width = corrected.shape[:2]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        yolo_cfg = self.recording_preset["yolo"]
        expand_ratio = float(yolo_cfg["center_search_expand_ratio"])
        radius_ratio = float(yolo_cfg["center_search_radius_ratio"])
        margin_ratio = float(yolo_cfg["center_search_margin_ratio"])
        expand_w = max(int(box_w * expand_ratio), 80)
        search_margin = max(int(box_w * margin_ratio), 6)
        cx, cy = self._center_line_x, (y1 + y2) / 2.0
        x1c = max(0, int(cx - expand_w / 2))
        x2c = min(width, int(cx + expand_w / 2))
        y1c = max(0, int(cy - max(box_h, 40) / 2))
        y2c = min(height, int(cy + max(box_h, 40) / 2))
        if x2c - x1c < 20 or y2c - y1c < 20:
            return False
        try:
            info = find_center_in_region(
                corrected[y1c:y2c, x1c:x2c],
                expected_center_x=cx - x1c,
                search_radius=max(box_w * radius_ratio, 15.0),
                search_bounds=(
                    max(0, x1 - x1c - search_margin),
                    min(x2c - x1c, x2 - x1c + search_margin),
                ),
            )
        except Exception:
            return False
        if info["orientation"] != "vertical" or info["confidence"] < 0.12:
            return False
        tracked = self._center_tracker.update(x1c + info["center_main"], info["confidence"])
        if tracked["center"] is None:
            return False
        self._center_line_x = tracked["center"]
        self._center_confidence = float(tracked["confidence"])
        self.fringe_center_plugin.update_result(
            self._center_line_x - x1,
            tracked["confidence"],
            True,
            "零级框短时漏检，正在视觉跟踪",
        )
        return True

    def _detect_center_in_result(self, result: dict, corrected: np.ndarray):
        """用零级框先验定位白光干涉竖条纹的相干中心。"""
        class_ids = result["class_ids"]
        class_names = result["class_names"]
        boxes = result["boxes_xyxy"]
        confs = result["confs"]

        # 每 30 帧才输出一次诊断，避免刷屏
        if not hasattr(self, '_center_diag_counter'):
            self._center_diag_counter = 0
        self._center_diag_counter += 1
        verbose = (self._center_diag_counter % 30 == 0)

        if len(boxes) == 0:
            self._center_yolo_misses += 1
            if verbose:
                self.log.write("[中心条纹] YOLO 未检测到任何目标")
            if self._center_yolo_misses <= 15 and self._track_center_from_previous_roi(corrected):
                return
            self._hold_or_clear_center("YOLO 未检测到零级条纹", verbose)
            return

        # 按模型自身的类别名称匹配零级/黑条，禁止假设固定 class_id。
        zero_class_ids = self.detector.find_class_ids("zero", "order", "black", "零级", "黑")
        zero_indices = [i for i, cid in enumerate(class_ids) if int(cid) in zero_class_ids]

        # 不把其他类别的最高置信度框冒充零级条纹，否则会产生大偏移。
        if not zero_indices:
            self._center_yolo_misses += 1
            if self._center_yolo_misses <= 15 and self._track_center_from_previous_roi(corrected):
                return
            self._hold_or_clear_center("未检测到零级条纹框", verbose)
            return

        best_local_idx = int(np.argmax(confs[zero_indices]))
        best_idx = zero_indices[best_local_idx]
        self._center_yolo_misses = 0
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        H, W = corrected.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0

        # 适度扩大零级框周围搜索范围，同时保留 YOLO 框中心先验，
        # 避免搜索范围变大后误抓远处彩色条纹。
        yolo_cfg = self.recording_preset["yolo"]
        expand_ratio = float(yolo_cfg["center_search_expand_ratio"])
        radius_ratio = float(yolo_cfg["center_search_radius_ratio"])
        margin_ratio = float(yolo_cfg["center_search_margin_ratio"])
        expand_w = max(int(box_w * expand_ratio), 80)
        search_margin = max(int(box_w * margin_ratio), 6)
        expand_h = max(box_h, 40)
        x1c = max(0, int(box_cx - expand_w / 2))
        x2c = min(W, int(box_cx + expand_w / 2))
        y1c = max(0, int(box_cy - expand_h / 2))
        y2c = min(H, int(box_cy + expand_h / 2))

        if x2c - x1c < 20 or y2c - y1c < 20:
            if verbose:
                self.log.write(f"[中心条纹] 扩展区域太小 ({x2c-x1c}x{y2c-y1c})")
            self._hold_or_clear_center("扩展区域太小", verbose)
            return

        roi_crop = corrected[y1c:y2c, x1c:x2c]

        try:
            info = find_center_in_region(
                roi_crop,
                expected_center_x=box_cx - x1c,
                search_radius=max(box_w * radius_ratio, 15.0),
                search_bounds=(
                    max(0, x1 - x1c - search_margin),
                    min(x2c - x1c, x2 - x1c + search_margin),
                ),
            )
        except Exception as e:
            if verbose:
                self.log.write(f"[中心条纹] 检测异常: {e}")
            self._hold_or_clear_center(f"检测异常：{e}", verbose)
            return

        if info["orientation"] != "vertical":
            self._hold_or_clear_center("当前区域不是白光竖条纹", verbose)
            return

        measured_x = x1c + info["center_main"]
        # 搜索范围已受零级框约束，这里只防止极端数值越出扩展区域。
        measured_x = float(np.clip(measured_x, x1c + 1, x2c - 1))
        tracked = self._center_tracker.update(measured_x, info["confidence"])
        if tracked["center"] is None:
            self._hold_or_clear_center("中心位置跳变过大", verbose)
            return

        center_x_final = tracked["center"]
        self._center_line_x = center_x_final
        self._center_confidence = float(tracked["confidence"])
        self._center_line_box = (x1, y1, x2, y2)

        if verbose:
            self.log.write(
                f"[中心条纹] x={center_x_final:.1f}px "
                f"conf={tracked['confidence']:.2f} "
                f"period={info['period']:.1f}px "
                f"verticality={info['verticality']:.2f} "
                f"box=({x1},{y1})-({x2},{y2}) "
                f"classes={list(set(class_names))}"
            )

        self.fringe_center_plugin.update_result(
            center_x_final - x1, tracked["confidence"], True)

    def _reload_calibration(self):
        """从 config/calibration.yaml 读取像素→毫米比例"""
        cal_path = PROJECT_ROOT / "config" / "calibration.yaml"
        ratio = 1.0
        if cal_path.exists():
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    ratio = float(data.get("pixel_to_mm_ratio", 1.0))
            except Exception:
                ratio = 1.0
        self.fringe_center_plugin.set_ratio(ratio)

    # ==================================================================
    # 中心条纹分析插件
    # ==================================================================
    def _on_fringe_center_cmd(self, cmd: str):
        if cmd == "toggle_auto":
            enabled = not self.fringe_center_plugin.auto_detect_var.get()
            self.fringe_center_plugin.auto_detect_var.set(enabled)
            self.fringe_center_plugin.update_auto_state(enabled)
            if enabled:
                if self.predict_running:
                    self.log.write("[中心条纹] 自动检测已开启（跟随YOLO预测频率）")
                else:
                    self.log.write("[中心条纹] 自动检测已开启，但预测未运行")
                    self.fringe_center_plugin.update_result(None, 0, False, "请先开始预测")
            else:
                self._center_line_x = None
                self._center_confidence = 0.0
                self._center_line_box = None
                self._center_tracker.reset()
                self._center_yolo_misses = 0
                self.fringe_center_plugin.update_result(None, 0, False)
                self.log.write("[中心条纹] 自动检测已停止")

        elif cmd == "toggle_line":
            pass

        elif cmd == "record":
            if self._center_line_x is None:
                self.log.write("[中心条纹] 记录失败：未检测到中心条纹")
                return
            zoom = self.corrector.zoom
            self.fringe_center_plugin.add_record(self._center_line_x, zoom)
            self._reload_calibration()
            self.log.write(
                f"[中心条纹] 已记录: x={self._center_line_x:.1f}px zoom={zoom:.1f}")

        elif cmd == "clear_record":
            self.fringe_center_plugin.clear_records()
            self.log.write("[中心条纹] 记录已清除")

    # ==================================================================
    # ROI 鼠标绘制
    # ==================================================================
    def _on_roi_press(self, event):
        # 手动点击记录模式
        if (self.fringe_center_plugin is not None
                and self.fringe_center_plugin.click_record_var.get()):
            if self._frame_scale > 0:
                x_frame = (event.x - self._frame_off_x) / self._frame_scale
                zoom = self.corrector.zoom
                self.fringe_center_plugin.add_record(x_frame, zoom)
                self.log.write(
                    f"[中心条纹] 手动记录: x={x_frame:.1f}px zoom={zoom:.1f}")
            return

        if self.model_plugin and self.model_plugin.roi_mode:
            self._roi_drawing = True
            self._roi_start = (event.x, event.y)
            self._roi_rect_id = None
        elif self.camera_plugin and self.camera_plugin.pan_mode:
            self._panning = True
            self._pan_start = (event.x, event.y)
            self._pan_orig = (self.corrector.pan_x, self.corrector.pan_y)

    def _on_roi_drag(self, event):
        if self._roi_drawing:
            c = self._roi_canvas
            if self._roi_rect_id:
                c.coords(self._roi_rect_id, self._roi_start[0], self._roi_start[1], event.x, event.y)
            else:
                self._roi_rect_id = c.create_rectangle(
                    self._roi_start[0], self._roi_start[1], event.x, event.y,
                    outline="#ffff00", width=2, tags="drawing")
        elif self._panning:
            dx = int((event.x - self._pan_start[0]) / self._frame_scale) if self._frame_scale > 0 else 0
            dy = int((event.y - self._pan_start[1]) / self._frame_scale) if self._frame_scale > 0 else 0
            self.corrector.pan_x = self._pan_orig[0] - dx
            self.corrector.pan_y = self._pan_orig[1] - dy

    def _on_roi_release(self, event):
        if self._roi_drawing:
            self._roi_drawing = False
            scale = self._frame_scale  # 使用实际显示缩放比例
            ox, oy = self._frame_off_x, self._frame_off_y
            x1 = int((self._roi_start[0]-ox)/scale) if scale>0 else 0
            y1 = int((self._roi_start[1]-oy)/scale) if scale>0 else 0
            x2 = int((event.x-ox)/scale) if scale>0 else 0
            y2 = int((event.y-oy)/scale) if scale>0 else 0
            self.model_plugin.set_roi(x1, y1, x2, y2)
            # 删除拖拽时的黄色预览框，立即绘制绿色持久 ROI 框
            self._roi_canvas.delete("drawing")
            self._roi_rect_id = None
            if self.model_plugin.roi_pixels:
                rx1, ry1, rx2, ry2 = self.model_plugin.roi_pixels
                self._roi_canvas.create_rectangle(
                    rx1*scale+ox, ry1*scale+oy,
                    rx2*scale+ox, ry2*scale+oy,
                    outline="#00ff00", width=2, tags="roi")
                self._preview_adjusted = True
                self.log.write(
                    f"[ROI] 已标注条纹分析区域 x={rx1}, y={ry1}, "
                    f"width={rx2-rx1}, height={ry2-ry1}")
        elif self._panning:
            self._panning = False

    def _get_display_scale(self) -> float:
        c = self._roi_canvas
        if c is None: return 1.0
        return min(max(1,c.winfo_width())/1280, max(1,c.winfo_height())/1024)

    def _get_roi(self) -> tuple[int,int,int,int] | None:
        """返回当前 ROI（不依赖 roi_mode 复选框，只要设置了 ROI 就生效）"""
        if self.model_plugin is None:
            return None
        return self.model_plugin.get_roi_xywh()

    def _stop_predict(self):
        self.predict_running = False
        self._prediction_generation += 1
        self._on_auto_stop()
        if self._predict_job:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None
        self._last_detection_result = None
        self._center_line_x = None
        self._center_confidence = 0.0
        self._prediction_frame_width = None
        self._center_line_box = None
        self._center_tracker.reset()
        self._fringe_motion_tracker.reset()
        self._last_fringe_motion = {
            "has_fringe": False, "movement": "unknown",
            "movement_text": "尚未检测", "delta_x_px": None, "source": ""}
        self._center_yolo_misses = 0
        self._set_status("预测已停止")
        self._start_preview()  # 恢复预览

    # ==================================================================
    # 预览
    # ==================================================================
    def _start_preview(self):
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None
        self._preview_loop()

    def _stop_preview(self):
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None

    def _preview_loop(self):
        self._preview_job = None
        if not self.camera_running or self.cam is None:
            return
        frame = self.cam.read()
        if frame is not None:
            corrected = rotate_expand(frame, self.corrector.effective_angle)
            corrected = self.corrector.apply_zoom_pan(corrected)
            if self.predict_running:
                # 推理结果继续负责手动页标注画面；自动页始终显示相机最新帧。
                if self.auto_dashboard is not None and self._current_page == "auto":
                    frame_width = corrected.shape[1]
                    self.auto_dashboard.update_interferometer(
                        corrected,
                        center_x=self._center_line_x,
                        target_x=(frame_width / 2.0)
                        if self._auto_center_line_visible() else None,
                    )
            else:
                self._show_frame(corrected)
            if self.recorder and self.recorder.recording:
                src = frame if self.recorder.recording_source == "camera" else corrected
                self._write_rec_frame(src)
        self._preview_job = self.root.after(self.PREVIEW_INTERVAL_MS, self._preview_loop)

    def _show_frame(self, frame_bgr):
        if self.auto_dashboard is not None and self._current_page == "auto":
            frame_width = frame_bgr.shape[1] if frame_bgr is not None else None
            self.auto_dashboard.update_interferometer(
                frame_bgr,
                center_x=self._center_line_x,
                target_x=(frame_width / 2.0)
                if frame_width and self._auto_center_line_visible() else None,
            )
        canvas = self._roi_canvas
        if canvas is None: return
        h, w = frame_bgr.shape[:2]
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        if nw > 0 and nh > 0:
            frame_bgr = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._frame_off_x = (cw - nw)//2
        self._frame_off_y = (ch - nh)//2
        self._frame_scale = scale
        # 更新已有图片（不 delete all 以提高性能）
        if getattr(self, '_img_id', None):
            canvas.itemconfigure(self._img_id, image=self._frame_img)
        else:
            self._img_id = canvas.create_image(cw//2, ch//2, image=self._frame_img, anchor="center")
        # 重画 ROI + 中心线（不删除 "drawing"，它是拖拽时的实时预览框）
        canvas.delete("roi", "center_line", "center_target")
        if self.model_plugin and self.model_plugin.roi_pixels:
            x1, y1, x2, y2 = self.model_plugin.roi_pixels
            canvas.create_rectangle(
                x1*scale+self._frame_off_x, y1*scale+self._frame_off_y,
                x2*scale+self._frame_off_x, y2*scale+self._frame_off_y,
                outline="#00ff00", width=2, tags="roi")

        # 绘制中心条纹线 + 记录位置竖线
        if self.fringe_center_plugin is not None:
            from src.ui.widgets.fringe_center_plugin import COLORS
            # 竖线高度
            if self._center_line_box is not None:
                _, by1, _, by2 = self._center_line_box
                y1s = by1 * scale + self._frame_off_y
                y2s = by2 * scale + self._frame_off_y
            else:
                y1s = 12
                y2s = max(1, canvas.winfo_height()) - 12

            if self._auto_center_line_visible():
                target_x = (w / 2.0) * scale + self._frame_off_x
                target_y1 = self._frame_off_y
                target_y2 = self._frame_off_y + h * scale
                canvas.create_line(
                    target_x, target_y1, target_x, target_y2,
                    fill="#00d4ff", width=2, dash=(6, 5), tags="center_target")
                canvas.create_text(
                    target_x + 5, target_y1 + 6, text="画面中心", fill="#00d4ff",
                    anchor="nw", font=("Microsoft YaHei UI", 8, "bold"),
                    tags="center_target")

            # 中心条纹线（红色虚线）
            if (self._center_line_x is not None
                    and self._center_line_box is not None
                    and self.fringe_center_plugin.show_line_var.get()):
                cx = self._center_line_x * scale + self._frame_off_x
                canvas.create_line(cx, y1s, cx, y2s,
                                   fill="#ff0000", width=2, dash=(10, 10),
                                   tags="center_line")

            # 记录位置竖线（用 dash 参数，每条只需 1 个 canvas 元素）
            for i, rec in enumerate(self.fringe_center_plugin.records):
                if not rec.get("visible", True):
                    continue
                color = COLORS[i % len(COLORS)]
                rx = rec["x_display"] * scale + self._frame_off_x
                canvas.create_line(rx, y1s, rx, y2s,
                                   fill=color, width=2, dash=(10, 10),
                                   tags="center_line")
                # 顶部标签
                canvas.create_text(rx, y1s - 4,
                                   text=rec["name"], fill=color, anchor="s",
                                   font=("Consolas", 8, "bold"),
                                   tags="center_line")

    # ==================================================================
    # 预测循环
    # ==================================================================
    def _predict_loop(self):
        self._predict_job = None
        if not self.predict_running or self.cam is None:
            return
        if self._inference_future is None:
            frame = self.cam.read()
            if frame is not None:
                corrected = rotate_expand(frame, self.corrector.effective_angle)
                corrected = self.corrector.apply_zoom_pan(corrected)
                self.detector.confidence = self.model_plugin.conf
                self.detector.iou = self.model_plugin.iou
                self.detector.imgsz = self.model_plugin.imgsz
                roi = self._get_roi()
                self._inference_context = (self._prediction_generation, frame, corrected, roi)
                self._inference_future = self._inference_executor.submit(
                    self.detector.detect, corrected, roi)
        elif self._inference_future.done():
            generation, frame, corrected, roi = self._inference_context
            try:
                result = self._inference_future.result()
            except Exception as exc:
                result = {"error": str(exc), "annotated": corrected,
                          "boxes_xyxy": np.array([]), "confs": np.array([]),
                          "class_names": [], "class_ids": np.array([])}
            self._inference_future = None
            self._inference_context = None
            if generation == self._prediction_generation and self.predict_running:
                self._consume_prediction(frame, corrected, roi, result)
        self._predict_job = self.root.after(15, self._predict_loop)

    def _consume_prediction(self, frame, corrected, roi, result):
        if result.get("error"):
            self.log.write(f"[错误] 模型推理失败，自动控制已停止: {result['error']}")
            self._on_auto_stop("推理异常")
        annotated = result["annotated"] if result["annotated"] is not None else corrected
        self._prediction_frame_width = int(annotated.shape[1])
        if roi:
            cv2.rectangle(annotated, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,0), 2)

        recommended = _decide_motor_command_from_boxes(
            result["boxes_xyxy"], result["confs"], annotated.shape)

        class_conf = get_class_confidences(result)
        guide = get_non_center_guide(
            result,
            annotated.shape[1],
            previous_x=self._last_non_center_guide.get("x"),
        )
        self._last_non_center_guide = guide

        # ---- 中心条纹自动检测（跟随 YOLO 预测频率）----
        if (self.fringe_center_plugin is not None
                and self.fringe_center_plugin.auto_detect_var.get()):
            self._detect_center_in_result(result, corrected)

        # 优先使用精确中心位置；中心尚未出现时，以非中心条纹框连续位置
        # 判断画面中是否有条纹以及条纹整体的水平移动方向。
        if self._center_line_x is not None:
            fringe_x = self._center_line_x
            fringe_source = "center"
        else:
            fringe_x = guide.get("x")
            fringe_source = "guide"
        has_fringe = bool(
            self._center_line_x is not None
            or guide.get("count", 0)
            or max(class_conf.values(), default=0.0) > 0.0
        )
        self._last_fringe_motion = self._fringe_motion_tracker.update(
            has_fringe=has_fringe,
            position_x=fringe_x,
            source=fringe_source,
        )
        for panel in self._auto_center_panels():
            panel.update_scene_analysis(self._last_fringe_motion)
            panel.update_clarity(
                self.cam.clarity_status() if self.cam is not None else {})

        if self.auto_control_enabled:
            self._auto_motor_control(guide)

        now = time.time()
        dt = max(1e-6, now - self.last_t)
        self.fps = 0.85*self.fps + 0.15*(1.0/dt)
        self.last_t = now

        cv2.putText(annotated, f"fps={self.fps:.1f} angle={self.corrector.effective_angle:+.1f}",
                    (20,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
        cv2.putText(annotated, f"suggest={recommended}",
                    (20,64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        present_text = "YES" if self._last_fringe_motion["has_fringe"] else "NO"
        movement_text = str(self._last_fringe_motion["movement"]).upper()
        cv2.putText(
            annotated, f"fringe={present_text} motion={movement_text}",
            (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2,
        )

        self._show_frame(annotated)
        if self.recorder and self.recorder.recording:
            src = frame if self.recorder.recording_source == "camera" else annotated
            self._write_rec_frame(src)

        self.status.update_fps(self.fps)
        self.model_plugin.update_results(class_conf, len(result["boxes_xyxy"]), recommended)
        self._last_detection_result = result  # 保存供中心条纹分析使用
        log_signature = (
            tuple(sorted((name, round(float(conf), 2))
                         for name, conf in class_conf.items())),
            round(self._center_line_x, 1) if self._center_line_x is not None else None,
            self._last_fringe_motion.get("movement", "unknown"),
        )
        if (log_signature != self._last_yolo_log_signature
                or now - self._last_yolo_log_at >= 5.0):
            detection_text = ", ".join(
                f"{name}={conf:.2f}" for name, conf in sorted(class_conf.items())
            ) or "无目标"
            center_text = (
                f"x={self._center_line_x:.1f}px, confidence={self._center_confidence:.2f}"
                if self._center_line_x is not None else "未定位")
            self.log.write(
                f"[YOLO实时] targets={len(result['boxes_xyxy'])} [{detection_text}]；"
                f"中心={center_text}；条纹移动={self._last_fringe_motion.get('movement_text', '--')}；"
                f"FPS={self.fps:.1f}；ROI={roi or '全画面'}")
            self._last_yolo_log_signature = log_signature
            self._last_yolo_log_at = now

    # ==================================================================
    # 录制
    # ==================================================================
    def _on_rec_start(self, path, fps, source):
        if fps <= 0:
            self.recorder.recording = False
            self.log.write("[错误] 录制 FPS 必须大于 0")
            return
        output = Path(path)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.recorder.recording = False
            self.log.write(f"[错误] 无法创建录制目录: {exc}")
            return
        self.recorder.output_path = str(output)
        self.recorder.output_fps = float(fps)
        self.recorder.video_writer = None
        self.recorder.recorded_frames = 0
        self.log.write(f"[录制] 开始: {path}")

    def _on_rec_stop(self):
        frames = getattr(self.recorder, "recorded_frames", 0)
        if self.recorder and self.recorder.video_writer:
            self.recorder.video_writer.release()
            self.recorder.video_writer = None
        self.log.write(f"[录制] 已停止，共写入 {frames} 帧")

    def _write_rec_frame(self, frame):
        if not self.recorder or not self.recorder.recording:
            return
        wr = self.recorder.video_writer
        if wr is None:
            h, w = frame.shape[:2]
            path = self.recorder.output_path
            fourcc = cv2.VideoWriter_fourcc(
                *("XVID" if path.lower().endswith(".avi") else "mp4v"))
            wr = cv2.VideoWriter(path, fourcc, self.recorder.output_fps, (w, h))
            if not wr.isOpened():
                wr.release()
                self.recorder.recording = False
                self.log.write(f"[错误] 无法打开视频编码器或输出路径: {path}")
                return
            self.recorder.video_writer = wr
            self.recorder.frame_size = (w, h)
        if frame.shape[1::-1] != self.recorder.frame_size:
            frame = cv2.resize(frame, self.recorder.frame_size)
        wr.write(frame)
        self.recorder.recorded_frames += 1

    # ==================================================================
    # 电机
    # ==================================================================
    def _on_refresh_ports(self):
        preferred = str(self.recording_preset["motor"]["port"])

        def scan_ports():
            return MotorController.list_ports(), MotorController.detect_port(preferred)

        self.motor_commands.submit("list_ports", scan_ports, coalesce=True)
        self._start_motor_poll()

    def _on_motor_connect(self, port):
        if self.motor_connected:
            self.log.write("[电机] 已连接，请先断开当前串口")
            return

        def connect():
            selected_port = MotorController.detect_port(port)
            if selected_port is None:
                return None, False, "未检测到可确定的电机串口"
            controller = MotorController(
                port=selected_port,
                baudrate=int(self.recording_preset["motor"]["baudrate"]),
                timeout=float(self.recording_preset["motor"]["timeout"]),
            )
            return controller, controller.connect(), selected_port

        if self.motor_commands.submit("connect", connect, coalesce=True):
            self._set_status(f"正在后台连接电机: {port}")
            self._start_motor_poll()

    def _on_motor_disconnect(self):
        self._on_auto_stop()
        self.motor_connected = False
        self.status.update_motor_connected(False)
        self._set_status("电机已断开")
        controller = self.motor
        self.motor = None
        if controller:
            self.motor_commands.submit("disconnect", controller.close, priority=0, coalesce=True)

    def _on_manual_command(self, cmd):
        if not self.motor_connected or self.motor is None:
            self.log.write("[警告] 电机未连接")
            return
        if cmd != "STATUS" and self.auto_control_enabled:
            self._on_auto_stop("已切换到手动控制")
        controller = self.motor
        if cmd == "FORWARD":
            self.motor_commands.submit(
                "manual_forward", controller.start_forward, coalesce=True)
            self.log.write("[手动] 正转启动 R")
        elif cmd == "REVERSE":
            self.motor_commands.submit(
                "manual_reverse", controller.start_reverse, coalesce=True)
            self.log.write("[手动] 反转启动 r")
        elif cmd == "STOP":
            self.motor_commands.submit("manual_stop", controller.stop, priority=0, coalesce=True)
            self.log.write("[手动] 停止 S")
        elif cmd == "TOGGLE_DIRECTION":
            self.motor_commands.submit(
                "manual_toggle_direction", controller.toggle_direction,
                coalesce=True)
            self.log.write("[手动] 运行中换向 D")
        elif cmd == "SPEED_UP":
            self.motor_commands.submit("speed_up", controller.speed_up, coalesce=True)
            self.log.write("[手动] 加速 +")
        elif cmd == "SPEED_DOWN":
            self.motor_commands.submit("speed_down", controller.speed_down, coalesce=True)
            self.log.write("[手动] 减速 -")
        elif cmd == "STATUS":
            self._on_query_motor_status()

    def _on_query_motor_status(self):
        if not self.motor_connected or self.motor is None:
            self.log.write("[电机] 未连接")
            return
        controller = self.motor
        self.motor_commands.submit("manual_status", controller.query_status, coalesce=True)

    def _auto_center_panels(self) -> tuple[AutoCenterControlPanel, ...]:
        return ((self.manual_auto_center_panel,)
                if self.manual_auto_center_panel is not None else ())

    def _sync_auto_center_settings(
        self, source: AutoCenterControlPanel | None,
    ) -> None:
        if source is None:
            return
        settings = source.get_params()
        for panel in self._auto_center_panels():
            if panel is not source:
                panel.load_settings(settings)

    def _set_auto_center_status(self, text: str) -> None:
        for panel in self._auto_center_panels():
            panel.status_var.set(text)

    def _update_auto_center_panels(self, decision) -> None:
        for panel in self._auto_center_panels():
            panel.update_control(
                decision, self._center_line_x, self._prediction_frame_width)

    def _on_auto_center_command(
        self, command: str, source: AutoCenterControlPanel | None = None,
    ):
        self._sync_auto_center_settings(source)
        if command == "start":
            self._on_auto_start()
        elif command == "stop":
            self._on_auto_stop("用户停止自动寻中")
        elif command == "toggle_center_line":
            # 下一次预览刷新立即生效；该叠加层不会进入相机帧或模型输入。
            return

    def _auto_center_line_visible(self) -> bool:
        panel = self.manual_auto_center_panel
        return bool(
            panel is not None
            and panel.show_center_line_var.get()
        )

    def _on_auto_start(self):
        if self.auto_control_enabled:
            self._on_auto_stop("重新启动自动寻中")
        if not self.motor_connected:
            self.log.write("[警告] 请先连接电机")
            self._set_auto_center_status("无法启动：请先连接电机")
            return
        if (not self.predict_running and self.camera_running
                and self.detector.is_loaded()):
            self._on_model_cmd("start")
        if not self.predict_running:
            self.log.write("[警告] 自动控制要求先启动模型预测")
            self._set_auto_center_status("无法启动：请先启动模型预测")
            return
        if not self.detector.find_class_ids("black", "zero", "order", "黑", "零级"):
            self.log.write(f"[错误] 模型缺少黑条/零级类别: {self.detector.class_names}")
            self._set_auto_center_status("无法启动：模型缺少中心条纹类别")
            return
        if not self.fringe_center_plugin.auto_detect_var.get():
            self.fringe_center_plugin.auto_detect_var.set(True)
            self.fringe_center_plugin.update_auto_state(True)
        decision = self.auto_controller.start(time.monotonic())
        self.auto_control_enabled = True
        self._apply_camera_clarity("自动寻中启动")
        self._last_auto_state = decision.state
        self._last_auto_mapping = decision.direction_mapping
        self._update_auto_center_panels(decision)
        params = self.manual_auto_center_panel.get_params()
        if params.get("search_mode") == "single_direction":
            direction_text = (
                "反转" if params.get("search_direction") == "reverse" else "正转")
            if params.get("invert_direction"):
                direction_text = "正转" if direction_text == "反转" else "反转"
            self.log.write(
                f"[AUTO] 已知方向搜索已启动：找到中心条纹前保持{direction_text}，"
                "找到后使用标准闭环方法移到中心，确认丢失后才恢复搜索")
        else:
            self.log.write("[AUTO] 双向自动寻中已启动")

    def _on_auto_stop(self, reason: str = "用户停止"):
        decision = self.auto_controller.stop(reason)
        self.auto_control_enabled = self.auto_controller.enabled
        self._apply_camera_clarity("自动寻中停止")
        self._dispatch_motor_commands(decision.commands)
        self._update_auto_center_panels(decision)
        if decision.stopped_reason:
            self.log.write(f"[AUTO] 已停止: {reason}")

    def _auto_motor_control(self, guide: dict | None = None):
        if not self.auto_control_enabled or self.motor is None:
            return
        safety = self.recording_preset["motor"]["safety"]
        params = self.manual_auto_center_panel.get_params()
        guide = guide or self._last_non_center_guide
        decision = self.auto_controller.update(
            center_x=self._center_line_x,
            frame_width=self._prediction_frame_width,
            confidence=self._center_confidence,
            guide_x=guide.get("x"),
            guide_confidence=float(guide.get("confidence", 0.0)),
            guide_count=int(guide.get("count", 0)),
            fringe_movement=str(self._last_fringe_motion.get(
                "movement", "unknown")),
            fringe_delta_x_px=self._last_fringe_motion.get("delta_x_px"),
            connected=self.motor_connected and self.motor.is_connected,
            params=params,
            safety=safety,
            now=time.monotonic(),
        )
        self.auto_control_enabled = self.auto_controller.enabled
        if not self.auto_control_enabled:
            self._apply_camera_clarity("自动寻中结束")
        self._dispatch_motor_commands(decision.commands)
        self._update_auto_center_panels(decision)
        if decision.state != self._last_auto_state:
            self._last_auto_state = decision.state
            self.log.write(f"[AUTO] {decision.message}")
        if decision.direction_mapping != self._last_auto_mapping:
            self._last_auto_mapping = decision.direction_mapping
            self.log.write(f"[AUTO] 方向学习完成：{decision.direction_mapping}")
        if decision.stopped_reason:
            self.log.write(f"[AUTO] 已停止: {decision.stopped_reason}")

    def _dispatch_motor_commands(self, commands):
        controller = self.motor
        if controller is None:
            return
        for action, value in commands:
            operation = {
                "start": controller.start,
                "start_forward": controller.start_forward,
                "start_reverse": controller.start_reverse,
                "stop": controller.stop,
                "set_speed": lambda target=value: controller.set_speed(int(target)),
            }[action]
            self.motor_commands.submit(
                f"auto_{action}", operation,
                priority=0 if action == "stop" else 10,
                coalesce=action in ("stop", "start_forward", "start_reverse"),
            )

    # ==================================================================
    # 电机轮询
    # ==================================================================
    def _start_motor_poll(self):
        if self._motor_poll_job is None:
            self._motor_poll_job = self.root.after(50, self._poll_motor)

    def _stop_motor_poll(self):
        if self._motor_poll_job:
            self.root.after_cancel(self._motor_poll_job)
            self._motor_poll_job = None

    def _poll_motor(self):
        self._motor_poll_job = None
        for result in self.motor_commands.drain():
            self._consume_motor_result(result)
        if self.motor and self.motor_connected:
            controller = self.motor
            self.motor_commands.submit("poll_status", controller.query_status, coalesce=True)
        if not self._closing:
            self._motor_poll_job = self.root.after(self.MOTOR_POLL_MS, self._poll_motor)

    def _consume_motor_result(self, result):
        if result.error:
            self.log.write(f"[MOTOR] {result.name}: {result.error}")
            if result.name.startswith("measurement_move_"):
                self._stop_measurement("电机旋转命令异常")
            elif result.name.startswith("backlash_move"):
                self._stop_backlash("电机旋转命令异常")
            elif result.name == "measurement_stop":
                self.log.write("[临时测量] 警告：停车命令执行异常")
            elif result.name == "backlash_stop":
                self.log.write("[回程差] 警告：停车命令执行异常")
            return
        if result.name == "list_ports":
            ports, selected = result.value
            self.motor_panel.update_ports(ports)
            if self.auto_device_debug is not None:
                self.auto_device_debug.set_motor_ports(ports)
            if selected:
                self.motor_panel.port_var.set(selected)
                if self.auto_device_debug is not None:
                    self.auto_device_debug.motor_var.set(selected)
                self.motor_panel.update_command_status(f"已自动检测电机串口：{selected}")
            else:
                self.motor_panel.update_command_status(
                    "未检测到串口" if not ports else "检测到多个串口，请手动选择")
            self.log.write(f"可用串口: {ports}；自动选择: {selected}")
        elif result.name == "connect":
            controller, ok, port = result.value
            if ok:
                self.motor = controller
                self.motor_connected = True
                self.motor_panel.port_var.set(port)
                if self.auto_device_debug is not None:
                    self.auto_device_debug.motor_var.set(port)
                self.status.update_motor_connected(True, port)
                self._set_status(f"电机已连接: {port}")
            else:
                self._set_status(f"电机连接失败: {port}")
                self.motor_panel.update_command_status(f"电机连接失败：{port}")
        elif result.name in ("poll_status", "manual_status"):
            status = result.value
            if not isinstance(status, dict) or not status.get("valid", False):
                if self.motor is not None and not self.motor.is_connected:
                    self.motor_connected = False
                    self.status.update_motor_connected(False)
                    self.motor_panel.update_command_status("电机连接已断开")
                    self._set_status("电机连接已断开")
                else:
                    self.motor_panel.update_command_status("已连接，但未收到有效电机状态")
                if result.name == "manual_status":
                    raw = status.get("response", "") if isinstance(status, dict) else ""
                    self.log.write(f"[电机] 状态读取失败，原始响应={raw!r}")
                return
            self.status.update_motor_speed(status["omega"])
            self.status.update_motor_gear(status["speed"])
            direction = status.get("direction", "unknown")
            direction_text = {
                "forward": "正转",
                "reverse": "反转",
                "stopped": "停止",
                "unknown": "未知",
            }.get(direction, direction)
            self.motor_panel.update_command_status(
                f"{'运行' if status['running'] else '停止'}  │  "
                f"方向 {direction_text}  │  档位 {status['speed']}  │  "
                f"角速度 {status['omega']} deg/s")
            if result.name == "manual_status":
                self.log.write(
                    f"[电机] {'RUN' if status['running'] else 'STOP'} "
                    f"方向={direction_text} 档位={status['speed']} "
                    f"ω={status['omega']}deg/s")
        elif result.name == "auto_set_speed" and result.value is not True:
            self.log.write("[AUTO] 调速状态确认失败，继续发送旋转命令")
        elif (result.name in ("manual_forward", "manual_reverse")
              or result.name in ("auto_start_forward", "auto_start_reverse", "auto_stop")) \
                and result.value is not True:
            self._on_auto_stop(f"电机命令失败: {result.name}")
        elif result.name.startswith("measurement_move_") and result.value is not True:
            self._stop_measurement("电机旋转命令失败")
        elif result.name.startswith("backlash_move_") and result.value is not True:
            self._stop_backlash("电机旋转命令失败")
        elif result.name == "measurement_stop" and result.value is not True:
            self.log.write("[临时测量] 警告：停车命令未确认成功")

    # ==================================================================
    # 临时测量 — 旋转电机使微分表达到目标读数
    # ==================================================================
    def _fresh_micrometer_reading(
        self, max_age_seconds: float | None = None,
    ) -> float | None:
        """返回最近确认的可信读数；旧值可显示，但不能驱动硬件。"""
        if self.micrometer_reading_mm is None or self.micrometer_reading_at is None:
            return None
        if max_age_seconds is None:
            max_age_seconds = float(config.get(
                "temporary_measurement", "reading_timeout_seconds",
                default=1.5))
        if time.time() - self.micrometer_reading_at > max(0.1, max_age_seconds):
            return None
        return self.micrometer_reading_mm

    def _on_temporary_measurement_cmd(self, cmd: str):
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        if cmd == "measurement_start":
            target = panel.target_mm
            current = self._fresh_micrometer_reading()
            if target is None:
                panel.set_status("错误：请输入有效的目标读数")
                return
            if self.motor is None or not self.motor_connected:
                panel.set_status("错误：电机未连接")
                return
            if current is None:
                panel.set_status("错误：微分表无稳定读数，请先开启视觉微分表")
                return
            cfg = config.get("temporary_measurement", default={}) or {}
            tolerance = float(cfg.get("tolerance_mm", 0.001))
            if abs(current - target) <= tolerance:
                panel.set_status(f"已在目标位置：当前 {current:.6f} mm ≈ 目标 {target:.6f} mm")
                return
            self._measurement_active = True
            self._measurement_target_mm = target
            self._measurement_started_at = time.monotonic()
            self._measurement_control_reading_mm = float(current)
            self._measurement_control_reading_at = time.monotonic()
            self._measurement_direction = "stopped"
            panel.set_busy(True)
            direction = _decide_measurement_direction(
                current, target, tolerance)
            direction_text = "正转" if direction == "forward" else "反转"
            panel.set_status(
                f"开始：当前 {current:.6f} → 目标 {target:.6f} mm，{direction_text}")
            self.log.write(
                f"[临时测量] 目标 {target:.6f} mm，当前 {current:.6f}，"
                f"误差 {target-current:+.6f} mm，方向 {direction_text}")
            self._measurement_step()
        elif cmd == "measurement_stop":
            self._stop_measurement("用户停止")

        # ---- 回程差测量命令 ----
        elif cmd == "backlash_set_start":
            current = self._fresh_micrometer_reading()
            if current is not None:
                panel.set_backlash_start(current)
                panel.set_backlash_status(f"起点已标定: {current:.6f} mm")
        elif cmd == "backlash_set_end":
            current = self._fresh_micrometer_reading()
            if current is not None:
                panel.set_backlash_end(current)
                panel.set_backlash_status(f"终点已标定: {current:.6f} mm")
        elif cmd == "backlash_start":
            start_mm = panel.backlash_start_mm
            end_mm = panel.backlash_end_mm
            if start_mm is None or end_mm is None:
                panel.set_backlash_status("错误：请先标定或输入起点和终点读数")
                return
            if self.motor is None or not self.motor_connected:
                panel.set_backlash_status("错误：电机未连接")
                return
            if self._fresh_micrometer_reading() is None:
                panel.set_backlash_status("错误：微分表无稳定读数")
                return
            if abs(start_mm - end_mm) < 0.001:
                panel.set_backlash_status("错误：起点和终点读数相同")
                return
            if end_mm <= start_mm:
                panel.set_backlash_status(
                    "错误：终点读数必须大于起点读数（正转读数增大）")
                return
            if self._measurement_active:
                self._stop_measurement("切换到回程差测量")
            self._backlash_active = True
            self._backlash_start_mm = start_mm
            self._backlash_end_mm = end_mm
            self._backlash_reading_forward = None
            self._backlash_reading_backward = None
            self._backlash_phase = "move_to_start"
            self._backlash_started_at = time.monotonic()
            self._backlash_reading_lost_at = 0.0
            current = self._fresh_micrometer_reading()
            self._backlash_approach_direction = (
                "forward" if current is not None and current < start_mm
                else "reverse")
            self._backlash_motor_direction = "stopped"
            panel.set_backlash_busy(True)
            panel.set_backlash_result(None, None)
            panel.set_backlash_status("阶段 1/5：正在移动到起点...")
            self.log.write(
                f"[回程差] 起点 {start_mm:.6f} mm → 终点 {end_mm:.6f} mm，开始")
            self._backlash_step()
        elif cmd == "backlash_stop":
            self._stop_backlash("用户停止")

    def _measurement_step(self):
        if not self._measurement_active:
            return
        panel = self.temporary_measurement_panel
        if panel is None:
            self._measurement_active = False
            return
        target = self._measurement_target_mm
        if target is None:
            self._stop_measurement("目标读数无效")
            return
        cfg = config.get("temporary_measurement", default={}) or {}
        tolerance = float(cfg.get("tolerance_mm", 0.001))
        max_dur = float(cfg.get("max_duration_seconds", 60))
        poll_ms = int(cfg.get("poll_interval_ms", 250))
        reading_timeout = float(cfg.get("reading_timeout_seconds", 1.5))
        gear = int(cfg.get("approach_gear", 10))
        controller = self.motor
        now = time.monotonic()
        reading_age = now - self._measurement_control_reading_at
        current = (
            self._measurement_control_reading_mm
            if reading_age <= reading_timeout else None)

        # 安全检查
        if now - self._measurement_started_at > max_dur:
            self._stop_measurement("超时：已超过最大运行时间")
            return
        if controller is None or not self.motor_connected:
            self._stop_measurement("电机断开")
            return

        decision = _decide_measurement_direction(current, target, tolerance)
        if decision == "stop":
            self._stop_measurement(
                f"已完成：当前 {current:.6f} mm ≈ 目标 {target:.6f} mm（容差 ±{tolerance:.6f}）")
            return

        if decision == "wait":
            if self._measurement_direction != "waiting":
                self._measurement_generation += 1
                self._measurement_direction = "waiting"
                self.motor_commands.submit(
                    "measurement_stop", controller.stop,
                    priority=0, coalesce=True)
                self.log.write("[临时测量] 当前读数超时，已停车等待新读数")
            panel.set_status("等待：微分表暂无最新有效读数，电机已停车")
            self._measurement_job = self.root.after(
                poll_ms, self._measurement_step)
            return

        if decision != self._measurement_direction:
            if not self._queue_measurement_motion(decision, controller, gear):
                self._stop_measurement("无法提交电机旋转命令")
                return

        # 更新面板状态
        direction_text = "正转（读数增大）" if decision == "forward" else "反转（读数减小）"
        error = target - current
        panel.set_status(
            f"运行中：当前 {current:.6f} → 目标 {target:.6f} mm，"
            f"差值 {error:+.6f}，{direction_text}")

        # 调度下一次检查
        self._measurement_job = self.root.after(poll_ms, self._measurement_step)

    def _queue_measurement_motion(
        self, direction: str, controller: MotorController, gear: int,
    ) -> bool:
        """串行设置档位和方向；代次变化后旧命令自动失效。"""
        self._measurement_generation += 1
        generation = self._measurement_generation
        self._measurement_direction = direction

        def move() -> bool:
            if (not self._measurement_active
                    or generation != self._measurement_generation):
                return True
            if not controller.set_speed(gear):
                return False
            if (not self._measurement_active
                    or generation != self._measurement_generation):
                return True
            operation = (
                controller.start_forward
                if direction == "forward" else controller.start_reverse)
            return operation()

        submitted = self.motor_commands.submit(
            f"measurement_move_{generation}", move, priority=10)
        if submitted:
            direction_text = "正转" if direction == "forward" else "反转"
            self.log.write(
                f"[临时测量] 根据当前差值切换为{direction_text}，档位 {gear}")
        return submitted

    def _stop_measurement(self, reason: str = ""):
        self._measurement_active = False
        self._measurement_generation += 1
        self._measurement_direction = "stopped"
        if self._measurement_job is not None:
            self.root.after_cancel(self._measurement_job)
            self._measurement_job = None
        # 停止电机
        controller = self.motor
        if controller is not None:
            self.motor_commands.submit(
                "measurement_stop", controller.stop, priority=0, coalesce=True)
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.set_busy(False)
            panel.set_status(reason or "已停止")
        if reason:
            self.log.write(f"[临时测量] {reason}")

    # ==================================================================
    # 回程差测量状态机
    # ==================================================================
    def _backlash_step(self):
        """回程差测量主循环：起点→正向找中心→终点→反向找中心→回起点。"""
        if not self._backlash_active:
            return
        panel = self.temporary_measurement_panel
        if panel is None:
            self._backlash_active = False
            return

        controller = self.motor
        if controller is None or not self.motor_connected:
            self._stop_backlash("电机断开")
            return

        cfg = config.get("temporary_measurement", default={}) or {}
        max_dur = float(cfg.get("max_duration_seconds", 60))
        poll_ms = int(cfg.get("poll_interval_ms", 250))
        gear = int(cfg.get("approach_gear", 10))
        tolerance = float(cfg.get("tolerance_mm", 0.001))
        endpoint_tolerance = max(
            tolerance,
            float(cfg.get("backlash_endpoint_tolerance_mm", 0.01)),
        )
        reading_timeout = float(cfg.get("reading_timeout_seconds", 1.5))

        if time.monotonic() - self._backlash_started_at > max_dur:
            self._stop_backlash("超时")
            return

        current_reading = self._fresh_micrometer_reading(reading_timeout)
        phase = self._backlash_phase
        start_mm = self._backlash_start_mm
        end_mm = self._backlash_end_mm

        # ---- 更新中心对齐显示 ----
        self._update_backlash_center_display()

        # 保持值可以继续显示，但过期值绝不能用于位置判断或电机控制。
        if current_reading is None:
            now = time.monotonic()
            if self._backlash_reading_lost_at <= 0:
                self._backlash_reading_lost_at = now
                self.log.write("[回程差] 可信微分表读数过期，已停车等待")
            if self._backlash_motor_direction != "waiting":
                self._backlash_generation += 1
                self._backlash_motor_direction = "waiting"
            self.motor_commands.submit(
                "backlash_stop", controller.stop, priority=0, coalesce=True)
            if now - self._backlash_reading_lost_at > reading_timeout * 3:
                self._stop_backlash("可信微分表读数持续丢失")
                return
            panel.set_backlash_status("等待可信微分表读数，电机已停车")
            self._backlash_job = self.root.after(
                poll_ms, self._backlash_step)
            return
        self._backlash_reading_lost_at = 0.0

        # ---- Phase: move_to_start ----
        if phase == "move_to_start":
            approach_direction = self._backlash_approach_direction
            if self._backlash_at_target(
                    start_mm, current_reading, endpoint_tolerance,
                    reading_timeout, approach_direction):
                # 到达起点，开始正向移动，等待中心条纹对齐
                self._backlash_phase = "forward"
                self._backlash_control_reading_mm = float(current_reading) if current_reading else 0.0
                self._backlash_control_reading_at = time.monotonic()
                panel.set_backlash_status(
                    "阶段 2/5：已到起点附近，开始单向正转，等待中心条纹对齐...")
                self.log.write(
                    f"[回程差] 已到起点附近（±{endpoint_tolerance:.6f} mm），"
                    "开始单向正转")
                self._move_motor(controller, "forward", gear)
            elif current_reading is None:
                if time.monotonic() - self._backlash_started_at > reading_timeout * 3:
                    self._stop_backlash("微分表无读数，无法移动到起点")
                    return
                panel.set_backlash_status("阶段 1/5：等待微分表读数...")
            else:
                panel.set_backlash_status(
                    f"阶段 1/5：沿{('正转' if approach_direction == 'forward' else '反转')}"
                    f"单向接近起点 {start_mm:.6f} mm（当前 {current_reading:.6f}）")
                self._move_motor(controller, approach_direction, gear)
            self._backlash_job = self.root.after(poll_ms, self._backlash_step)
            return

        # ---- Phase: forward (起点→终点，检测中心条纹对齐) ----
        if phase == "forward":
            center_aligned = self._is_center_aligned()
            if center_aligned and current_reading is not None:
                self._backlash_reading_forward = current_reading
                panel.set_backlash_result(current_reading, self._backlash_reading_backward)
                panel.set_backlash_status(
                    f"阶段 3/5：已记录正向对齐读数 {current_reading:.6f} mm，继续向终点移动")
                self.log.write(
                    f"[回程差] 正向对齐读数: {current_reading:.6f} mm")
                # 继续向终点移动
                self._backlash_phase = "to_end"
                self._move_motor(controller, "forward", gear)
                self._backlash_job = self.root.after(poll_ms, self._backlash_step)
                return
            if self._backlash_at_target(
                    end_mm, current_reading, endpoint_tolerance,
                    reading_timeout, "forward"):
                self._backlash_phase = "backward"
                self._backlash_control_reading_mm = float(current_reading) if current_reading else 0.0
                self._backlash_control_reading_at = time.monotonic()
                panel.set_backlash_status(
                    "阶段 4/5：已到终点附近，仅换向一次，反转等待中心条纹对齐...")
                self.log.write(
                    f"[回程差] 已到终点附近（±{endpoint_tolerance:.6f} mm），"
                    "执行唯一实验换向并开始反转")
                self._move_motor(controller, "reverse", gear)
            elif current_reading is None:
                panel.set_backlash_status("阶段 2/5：等待微分表读数...")
            else:
                panel.set_backlash_status(
                    f"阶段 2/5：正向移动中，等待中心对齐 "
                    f"（当前 {current_reading:.6f} mm）")
                self._move_motor(controller, "forward", gear)
            self._backlash_job = self.root.after(poll_ms, self._backlash_step)
            return

        # ---- Phase: to_end (已记录正向读数，继续走到终点) ----
        if phase == "to_end":
            if self._backlash_at_target(
                    end_mm, current_reading, endpoint_tolerance,
                    reading_timeout, "forward"):
                self._backlash_phase = "backward"
                self._backlash_control_reading_mm = float(current_reading) if current_reading else 0.0
                self._backlash_control_reading_at = time.monotonic()
                panel.set_backlash_status(
                    "阶段 4/5：已到终点附近，仅换向一次，反转等待中心条纹对齐...")
                self.log.write(
                    f"[回程差] 已到终点附近（±{endpoint_tolerance:.6f} mm），"
                    "执行唯一实验换向并开始反转")
                self._move_motor(controller, "reverse", gear)
            elif current_reading is None:
                panel.set_backlash_status("阶段 3/5：等待微分表读数...")
            else:
                panel.set_backlash_status(
                    f"阶段 3/5：继续向终点移动 "
                    f"（当前 {current_reading:.6f} mm）")
                self._move_motor(controller, "forward", gear)
            self._backlash_job = self.root.after(poll_ms, self._backlash_step)
            return

        # ---- Phase: backward (终点→起点，检测中心条纹对齐) ----
        if phase == "backward":
            center_aligned = self._is_center_aligned()
            if center_aligned and current_reading is not None:
                self._backlash_reading_backward = current_reading
                panel.set_backlash_result(
                    self._backlash_reading_forward, current_reading)
                diff = abs(
                    (self._backlash_reading_forward or 0) - current_reading)
                panel.set_backlash_status(
                    f"阶段 5/5：已记录反向对齐读数 {current_reading:.6f} mm，"
                    f"回程差 {diff:.6f} mm，继续返回起点")
                self.log.write(
                    f"[回程差] 反向对齐读数: {current_reading:.6f} mm，"
                    f"回程差: {diff:.6f} mm")
                # 继续回到起点
                self._backlash_phase = "to_start"
                self._move_motor(controller, "reverse", gear)
                self._backlash_job = self.root.after(poll_ms, self._backlash_step)
                return
            if self._backlash_at_target(
                    start_mm, current_reading, endpoint_tolerance,
                    reading_timeout, "reverse"):
                self._stop_backlash(self._make_backlash_summary("已完成"))
                return
            if current_reading is None:
                panel.set_backlash_status("阶段 4/5：等待微分表读数...")
            else:
                panel.set_backlash_status(
                    f"阶段 4/5：反向移动中，等待中心对齐 "
                    f"（当前 {current_reading:.6f} mm）")
                self._move_motor(controller, "reverse", gear)
            self._backlash_job = self.root.after(poll_ms, self._backlash_step)
            return

        # ---- Phase: to_start (已记录反向读数，继续回到起点) ----
        if phase == "to_start":
            if self._backlash_at_target(
                    start_mm, current_reading, endpoint_tolerance,
                    reading_timeout, "reverse"):
                self._stop_backlash(self._make_backlash_summary("已完成"))
                return
            if current_reading is None:
                panel.set_backlash_status("阶段 5/5：等待微分表读数...")
            else:
                panel.set_backlash_status(
                    f"阶段 5/5：继续返回起点 "
                    f"（当前 {current_reading:.6f} mm）")
                self._move_motor(controller, "reverse", gear)
            self._backlash_job = self.root.after(poll_ms, self._backlash_step)
            return

    def _make_backlash_summary(self, prefix: str) -> str:
        fwd = self._backlash_reading_forward
        bwd = self._backlash_reading_backward
        if fwd is not None and bwd is not None:
            return (
                f"{prefix} | 正向对齐={fwd:.6f} mm  "
                f"反向对齐={bwd:.6f} mm  "
                f"回程差={abs(fwd - bwd):.6f} mm"
            )
        return prefix

    def _backlash_at_target(self, target_mm: float, current: float | None,
                            tolerance: float, reading_timeout: float,
                            direction: str = "either") -> bool:
        """沿指定方向进入或越过近似端点即判定到达。"""
        del reading_timeout  # 新鲜度已由 _fresh_micrometer_reading 统一检查。
        return _backlash_endpoint_reached(
            current, target_mm, tolerance, direction)

    def _is_center_aligned(self) -> bool:
        """中心条纹横坐标是否接近画面中心线（容差 8 px）。"""
        if self._center_line_x is None:
            return False
        frame_w = self._prediction_frame_width
        if frame_w is None or frame_w <= 0:
            return False
        return abs(self._center_line_x - frame_w / 2) <= 8.0

    def _update_backlash_center_display(self):
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        aligned = self._is_center_aligned()
        panel.set_center_align(
            aligned, self._center_line_x, self._prediction_frame_width)

    def _move_motor(self, controller, direction: str, gear: int):
        """每个阶段只发送一次固定方向命令，禁止端点附近反复换向。"""
        if direction == self._backlash_motor_direction:
            return
        self._backlash_generation += 1
        generation = self._backlash_generation
        self._backlash_motor_direction = direction

        def move() -> bool:
            if (not self._backlash_active
                    or generation != self._backlash_generation):
                return True
            if not controller.set_speed(gear):
                return False
            if (not self._backlash_active
                    or generation != self._backlash_generation):
                return True
            operation = (
                controller.start_forward
                if direction == "forward" else controller.start_reverse)
            return operation()

        self.motor_commands.submit(
            f"backlash_move_{generation}", move, priority=10)

    def _stop_backlash(self, reason: str = ""):
        self._backlash_active = False
        self._backlash_generation += 1
        self._backlash_motor_direction = "stopped"
        if self._backlash_job is not None:
            self.root.after_cancel(self._backlash_job)
            self._backlash_job = None
        controller = self.motor
        if controller is not None:
            self.motor_commands.submit(
                "backlash_stop", controller.stop, priority=0, coalesce=True)
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.set_backlash_busy(False)
            panel.set_backlash_status(reason or "已停止")
        if reason:
            self.log.write(f"[回程差] {reason}")

    # ==================================================================
    # 工具
    # ==================================================================
    def _set_status(self, text):
        self.status_var.set(f"状态: {text}")

    def _on_close(self):
        self._closing = True
        if self._measurement_active:
            self._stop_measurement("程序关闭")
        if self._backlash_active:
            self._stop_backlash("程序关闭")
        if self._agent_context_job is not None:
            self.root.after_cancel(self._agent_context_job)
            self._agent_context_job = None
        if self._model_load_job is not None:
            self.root.after_cancel(self._model_load_job)
            self._model_load_job = None
        if self._model_load_future is not None:
            self._model_load_future.cancel()
        self.root.unbind_all("<MouseWheel>")
        self._stop_preview()
        self.recorder.stop()
        self._stop_predict()
        if self._micrometer_future is not None:
            self._micrometer_future.cancel()
        self._stop_micrometer("程序关闭")
        self._stop_motor_poll()
        if self.cam: self.cam.stop()
        report = shutdown_motor_safely(
            self.motor_commands, self.motor, timeout=3.0)
        if not report.stop_succeeded:
            logger.error("退出停车未确认成功: %s", report.error or "控制器未确认")
        self._inference_executor.shutdown(wait=False, cancel_futures=True)
        self._camera_executor.shutdown(wait=False, cancel_futures=True)
        self._micrometer_executor.shutdown(wait=False, cancel_futures=True)
        self.agent_session.shutdown()
        self.root.destroy()

    def run(self):
        self.log.write("启动 UI")
        self.root.mainloop()


def run_app():
    YoloCamApp().run()
