"""摄像头 YOLO 实时检测 + 电机控制 — Tkinter UI"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time
import queue
import subprocess
import threading
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
from src.agent.conversation_export import export_conversation
from src.config import config
from src.logging import logger
from src.camera import CameraManager
from src.vision import (
    CenterTracker,
    FringeRecognitionTracker,
    FringeMotionTracker,
    MicrometerOCR,
    YOLODetector,
    rotate_expand,
    FrameCorrector,
    find_center_in_region,
    find_center_by_band,
    analyse_fringe_texture,
    analyse_guidance_geometry,
    build_fringe_guidance,
    laser_guidance_signature,
    render_laser_alignment_instruction,
    validate_laser_ai_guidance,
    analyze_thickness_distribution,
    sample_colour_band,
)
from src.vision.class_names import get_class_confidences, get_non_center_guide
from src.vision.fringe_width import (
    measure_center_fringe_width,
    measure_center_fringe_width_2d,
    measure_fringe_width_by_count,
    measure_fringe_spacing_2d,
    measure_fringe_spacing_robust,
)
from src.vision.fringe_angle import estimate_fringe_angle_2d
from src.hardware import MicrometerReader, MotorController, SerialCommandQueue
from src.vision.micrometer_ocr import MicrometerOCRResult
from src.control import AdaptiveResponseLearner, CenterControlStateMachine
from src.agent import AgentService, AgentSession
from src.agent.toolkit import Confirmation
from src.agent.device_tools import ToolContext
from src.agent.tools import build_suggestion
from src.agent.experiment_guidance import (
    INTENT_LABELS,
    render_guidance_decision,
)
from src.agent.proactive import ProactiveCoordinator
from src.agent.laser_guidance_session import (
    LaserGuidanceConfig,
    LaserGuidanceSession,
)
from src.vision.fringe_adjustment import compare_fringe_adjustment
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
    ThicknessMeasurementPanel,
    ExperimentAssistantPanel,
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
    MOTOR_RECONNECT_MAX = 5          # 串口断链后最多自动重连次数
    MOTOR_RECONNECT_DELAY_MS = 1500  # 每次重连尝试之间的退避时间
    LIVE_MEASUREMENT_INTERVAL_MS = 500  # 微分表读数刷新周期
    LIVE_WIDTH_INTERVAL_S = 1.0  # 条纹宽度分析节流：至少间隔 1s，防卡顿
    FRINGE_REALTIME_INTERVAL_MS = 300  # 实时条纹宽度分析刷新周期
    GUIDANCE_GEOMETRY_INTERVAL_S = 1.0  # 角度/间距诊断节流，避免阻塞实时预测
    AGENT_SUGGESTION_CHECK_MS = 2000  # 只检查事件队列，不按周期调用模型

    def __init__(self):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter 不可用")

        self.root = tk.Tk()
        self.root.title("AI Interferometry · 白光干涉实验工作台")
        window_size = config.get("ui", "window_size", default=[1600, 1000])
        self.root.geometry(f"{int(window_size[0])}x{int(window_size[1])}")
        self.root.configure(bg=APP_BG)
        self.root.minsize(1180, 760)
        self.root.option_add("*Font", (FONT, 10))
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
        self._adaptive_response_path = PROJECT_ROOT / "data" / "adaptive_response.json"
        self.adaptive_response = AdaptiveResponseLearner.load(
            self._adaptive_response_path)
        self._last_adaptive_changes: dict = {}
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
        self._motor_reconnecting = False
        self._motor_reconnect_attempts = 0
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-scan")
        self._micrometer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="micrometer")
        self._thickness_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="thickness")
        self._guidance_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fringe-guidance")
        self._inference_future: Future | None = None
        self._thickness_future: Future | None = None
        self._thickness_job: str | None = None
        self._thickness_baseline_frame: np.ndarray | None = None
        self._thickness_anchor_um: float | None = None
        self._thickness_viewers: list = []
        self._inference_context: tuple | None = None
        self._camera_scan_future: Future | None = None
        self._micrometer_future: Future | None = None
        self._micrometer_task_kind = ""
        self._micrometer_job: str | None = None
        self._agent_context_job: str | None = None
        self._live_measurement_job: str | None = None
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
        # ---- 薄膜厚度分析区域（独立于条纹 ROI，单位：矫正后像素）----
        self._thickness_roi: tuple[int, int, int, int] | None = None
        self._thickness_roi_drawing = False
        self._thickness_roi_start = (0, 0)
        self._thickness_roi_rect_id: int | None = None
        self._frame_off_x = 0
        self._frame_off_y = 0
        self._frame_scale = 1.0
        self._frame_w = 0
        self._frame_h = 0
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
        self.manual_auto_center_panel: AutoCenterControlPanel | None = None
        self.micrometer_panel: MicrometerPluginPanel | None = None
        self.thickness_measurement_panel: ThicknessMeasurementPanel | None = None
        self.experiment_assistant_panel: ExperimentAssistantPanel | None = None
        self.temporary_measurement_panel: TemporaryMeasurementPanel | None = None
        self.log: LogPanel | None = None
        self._manual_scroll_canvas: tk.Canvas | None = None
        # 可折叠外壳
        self._shells: dict[str, CollapsibleFrame] = {}

        # ---- 中心条纹检测状态 ----
        self._center_line_x: float | None = None  # 全帧坐标下的中心 x
        self._center_line_box: tuple | None = None  # 所属预测框 (x1,y1,x2,y2)
        self._zero_box_x: float | None = None  # YOLO 零级框中心 x（全帧坐标）
        self._zero_box_confidence: float = 0.0  # YOLO 零级框置信度
        # 零级框稳定性追踪：只有连续稳定的框才用于约束中心搜索
        self._zero_box_stable = False
        self._zero_box_history: deque = deque(maxlen=6)  # (box_cx, box_width)
        self._zero_box_stable_counter = 0
        self._zero_box_unstable_counter = 0
        self._zero_box_missing_counter = 0
        self._center_tracker = CenterTracker(hold_frames=3, max_jump_px=60.0)
        self._center_yolo_misses = 0
        self._center_confidence = 0.0
        self._prediction_frame_width: int | None = None
        self._last_detection_result: dict | None = None  # 最近一次 YOLO 检测结果
        self._latest_corrected_frame: np.ndarray | None = None  # 最近一帧矫正后画面（供条纹宽度分析）
        self._fringe_band_overlay: list[dict] | None = None  # 单次识别所有条纹的边界/宽度标注
        self._fringe_spacing_overlay: dict | None = None  # 沿法向间距标注（绿=采用/橙=剔除）
        # 实时测量缓存：微分表读数 + 中心条纹宽度（供实时刷新与记录复用）
        self._live_measurement: dict = {"reading_mm": None, "width_px": None, "kind": None}
        self._live_measurement_active = False  # 开启后才持续分析
        self._live_last_width_at = 0.0  # 上次宽度分析时刻（monotonic，节流用）
        # 实时条纹宽度分析：持续刷新「视场÷条纹数」的间隔 + 可选画面标注
        self._fringe_realtime_active = False
        self._fringe_realtime_job = None
        self._fringe_count_overlay: dict | None = None  # 视场/条纹/间隔标注
        self._last_non_center_guide = {
            "x": None, "confidence": 0.0, "count": 0, "class_name": ""}
        self._fringe_motion_tracker = FringeMotionTracker(
            window_size=int(yolo_cfg["fringe_motion_window"]),
            movement_threshold_px=float(
                yolo_cfg["fringe_motion_threshold_px"]),
            missing_hold_frames=3,
        )
        self._fringe_recognition_tracker = FringeRecognitionTracker(
            history_size=int(yolo_cfg["fringe_history_size"]),
            missing_hold_frames=int(yolo_cfg["fringe_missing_hold_frames"]),
            visual_threshold=float(yolo_cfg["fringe_visual_threshold"]),
            assisted_threshold=float(yolo_cfg["fringe_assisted_threshold"]),
        )
        self._texture_interval_frames = int(
            yolo_cfg["fringe_texture_interval_frames"])
        self._texture_frame_counter = 0
        self._last_texture_analysis: dict | None = None
        self._last_fringe_motion = {
            "has_fringe": False,
            "movement": "unknown",
            "movement_text": "尚未检测",
            "delta_x_px": None,
            "source": "",
        }
        # 第一阶段实时指导：仅分析和显示，不向电机命令队列写入任何内容。
        self._guidance_future: Future | None = None
        self._guidance_future_generation = -1
        self._guidance_last_submit_at = 0.0
        self._guidance_geometry_completed_at = 0.0
        self._last_guidance_geometry: dict = {}
        self._last_fringe_guidance: dict = {}
        self._laser_alignment_active = False
        self._last_auto_state = ""
        self._last_auto_mapping = "learning"
        self._experiment_intent = {
            "kind": "white_light_centering",
            "objective": INTENT_LABELS["white_light_centering"],
            "required_repeats": 5,
            "response_mode": "standard",
            "confirmed": False,
        }
        agent_cfg = config.agent
        laser_cfg = agent_cfg.get("laser_guidance", {}) or {}
        self._laser_guidance_session = LaserGuidanceSession(
            LaserGuidanceConfig(
                max_tilt_deg=float(laser_cfg.get("max_tilt_deg", 3.0)),
                min_bright_fringes=int(
                    laser_cfg.get("min_bright_fringes", 4)),
                max_bright_fringes=int(
                    laser_cfg.get("max_bright_fringes", 10)),
                consecutive_passes=int(
                    laser_cfg.get("consecutive_passes", 3)),
                settle_seconds=float(laser_cfg.get("settle_seconds", 1.0)),
            ))
        self._last_laser_session: dict = {}
        self._laser_checkpoint: dict | None = None
        self._laser_ai_guidance_enabled = False
        self._laser_ai_guidance_inflight = False
        self._laser_ai_guidance_cancel_event: threading.Event | None = None
        self._laser_ai_guidance_last_signature: tuple | None = None
        self._laser_ai_guidance_last_call_at = 0.0
        self._laser_ai_guidance_generation = 0
        self._laser_ai_min_interval_seconds = float(
            agent_cfg.get("laser_ai_min_interval_seconds", 6))
        self._proactive_coordinator = ProactiveCoordinator(
            min_llm_interval=float(agent_cfg.get(
                "proactive_min_llm_interval_seconds", 60)),
            repeat_suppression=float(agent_cfg.get(
                "proactive_repeat_suppression_seconds", 300)),
            max_calls_per_window=int(agent_cfg.get(
                "proactive_max_calls_per_10_minutes", 3)),
            max_calls_per_session=int(agent_cfg.get(
                "proactive_max_calls_per_session", 12)),
            stalled_stage_seconds=float(agent_cfg.get(
                "proactive_stalled_stage_seconds", 120)),
        )
        self._agent_pending_llm: tuple[str, tuple] | None = None
        self._agent_active_request_key: tuple | None = None
        self._fringe_adjustment_baseline: dict | None = None
        self._vision_review_inflight = False
        self._vision_review_cancel_event: threading.Event | None = None
        self.agent_service = AgentService(context_provider=self._get_agent_context)
        self.agent_session = AgentSession(self.agent_service)
        self._pending_confirmation: Confirmation | None = None
        # 主动建议：本地规则实时响应，只有高价值语义事件才让模型优化表达。
        self._agent_suggestion_job: str | None = None
        self._agent_suggestion_inflight = False

        # ---- 构建 ----
        self._build_ui()
        self._build_agent_tool_context()
        self._wire_callbacks()
        self._reload_calibration()
        self.log.write("UI 初始化完成")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_refresh_ports()
        self._refresh_agent_context()
        self._refresh_live_measurement()
        # 后台模型默认关闭；仅用户提问、手动识图或显式自动 AI 指导会调用模型。

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
            anchor="w", justify=tk.LEFT, wraplength=380,
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
                        launcher_text, text="统一 AI 实验助手",
                        bg="#eef5ff", fg=NAVY, font=(FONT, 9, "bold"),
                        anchor="w",
                    ).pack(fill=tk.X)
                    tk.Label(
                        launcher_text,
                        text="现场质量门、四阶段调节、对话和报告集中在一个浮窗",
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
        self.recording_sidebar.motor_port_var.set(str(
            self.recording_preset["motor"]["port"]))
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
        self.agent_panel.on_confirm_motion = self._on_agent_confirm_motion
        self.agent_panel.on_reject_motion = self._on_agent_reject_motion
        self.agent_panel.on_emergency_stop = self._on_agent_emergency_stop
        self.agent_panel.on_toggle_autonomous = self._on_agent_toggle_autonomous
        self.agent_panel.on_toggle_dry_run = self._on_agent_toggle_dry_run
        self.agent_panel.on_set_guidance_stage = self._on_agent_set_guidance_stage
        self.agent_panel.on_apply_guidance = self._on_agent_apply_guidance
        self.agent_panel.on_auto_center = self._on_agent_auto_center
        self.agent_panel.on_set_intent = self._on_agent_set_intent
        self.agent_panel.on_set_response_mode = self._on_agent_set_response_mode
        self.agent_panel.on_mark_adjustment = self._on_agent_mark_adjustment
        self.agent_panel.on_compare_adjustment = self._on_agent_compare_adjustment
        self.agent_panel.on_review_image = self._on_agent_review_image
        self.agent_panel.on_toggle_laser_alignment = (
            self._on_agent_toggle_laser_alignment)
        self.agent_panel.on_toggle_laser_ai_guidance = (
            self._on_agent_toggle_laser_ai_guidance)
        self.agent_panel.on_laser_recheck = self._on_agent_laser_recheck
        self.agent_panel.on_export_experiment_record = (
            self._on_export_experiment_record)
        self.agent_panel.on_export_chat = self._on_agent_export_chat
        self.manual_auto_center_panel.on_command = self._on_auto_center_command
        self.recording_sidebar.on_command = self._on_recording_sidebar_command
        self.micrometer_panel.on_command = self._on_micrometer_command
        self.thickness_measurement_panel.on_command = (
            self._on_thickness_measurement_command)
        self.experiment_assistant_panel.on_command = (
            self._on_experiment_assistant_command)
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
        elif command == "connect_motor":
            if self.motor_connected:
                sidebar.set_status("电机已经连接")
            else:
                self.motor_panel.port_var.set(sidebar.motor_port)
                self._on_motor_connect(sidebar.motor_port)
                sidebar.set_status(
                    f"正在连接电机：{sidebar.motor_port}")
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
            # 精简侧栏明确选择了已知方向；识别节拍继续使用统一预设。
            self.manual_auto_center_panel.direction_mode_var.set(
                "single_direction")
            self.manual_auto_center_panel.recognition_mode_var.set(
                str(auto_cfg["recognition_mode"]))
            self.manual_auto_center_panel.search_direction_var.set(direction)
            self.manual_auto_center_panel.update_mode_summary()
            # 搜索阶段严格使用人工方向；稳定识别中心条纹后复用原有
            # 方向学习与闭环居中逻辑，此时允许为居中而变向。
            self.manual_auto_center_panel.auto_learn_direction_var.set(
                bool(auto_cfg["auto_learn_direction"]))
            self._on_auto_center_command("start")
            sidebar.set_status(
                f"正在按已知方向{'正转' if direction == 'forward' else '反转'}"
                "寻找条纹并寻中")
        elif command == "stop_auto_center":
            self._on_auto_center_command("stop")
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
                "live_records": [
                    dict(record)
                    for record in (
                        self.temporary_measurement_panel.records
                        if self.temporary_measurement_panel is not None else [])
                ][-20:],
            },
            thickness_measurement=(
                self.thickness_measurement_panel.snapshot()
                if self.thickness_measurement_panel is not None else {}),
            experiment_assistant=(
                self.experiment_assistant_panel.snapshot()
                if self.experiment_assistant_panel is not None else {}),
            fringe_band_overlay=self._fringe_band_overlay,
            fringe_count_overlay=self._fringe_count_overlay,
            fringe_realtime_active=self._fringe_realtime_active,
            texture_analysis=self._last_texture_analysis,
            fringe_guidance=self._last_fringe_guidance,
            laser_alignment_active=self._laser_alignment_active,
            laser_ai_guidance_enabled=self._laser_ai_guidance_enabled,
            laser_guidance_session=self._last_laser_session,
            adaptive_response=self.adaptive_response.snapshot(),
            guidance_execution_stage=(
                self.manual_auto_center_panel.execution_stage
                if self.manual_auto_center_panel is not None else "advisory"),
            auto_direction_mapping=self._last_auto_mapping,
            live_measurement=self._live_measurement,
            live_measurement_active=self._live_measurement_active,
            calibration_rows=(
                list(self.temporary_measurement_panel.calibration_rows)
                if self.temporary_measurement_panel is not None else []),
            experiment_intent=dict(self._experiment_intent),
        )

    def _refresh_agent_context(self) -> None:
        """定时把同一份实时快照同步到助手面板，并记录状态指纹用于空闲检测。"""
        self._agent_context_job = None
        if self._closing:
            return
        context = self._get_agent_context()
        if self._laser_alignment_active:
            session = self._laser_guidance_session.observe(
                context, now=time.monotonic())
            self._last_laser_session = session
            context["vision"]["laser_guidance_session"] = session
        if self.agent_panel is not None:
            self.agent_panel.set_experiment_context(context)
        self._maybe_start_laser_ai_guidance(context)
        update = self._proactive_coordinator.observe(
            context, now=time.monotonic())
        if update.changed and self._agent_active_request_key is not None:
            # 模型分析期间现场已发生实质变化，丢弃即将返回的过期建议。
            self._agent_active_request_key = None
        if update.changed and self.agent_panel is not None:
            self.agent_panel.set_proactive_guidance(
                update.decision.as_dict(),
                llm_calls=self._proactive_coordinator.session_calls,
            )
            self.agent_panel.set_suggestion(
                render_guidance_decision(update.decision),
                source="本地主动指导",
            )
        self._agent_pending_llm = None
        self._agent_context_job = self.root.after(
            500, self._refresh_agent_context)

    def _schedule_agent_suggestion(self) -> None:
        """仅消费关键语义事件；普通实时指导完全由本地规则完成。"""
        self._agent_suggestion_job = None
        if self._closing:
            return
        service = self.agent_service
        pending = self._agent_pending_llm
        now = time.monotonic()
        if pending and service is not None and service.provider.available:
            reason, request_key = pending
            if (not self._agent_suggestion_inflight
                    and self._proactive_coordinator.reserve_llm(
                        request_key, now=now)):
                self._agent_suggestion_inflight = True
                self._agent_pending_llm = None
                self._agent_active_request_key = request_key
                if self.agent_panel is not None:
                    self.agent_panel.set_proactive_budget(
                        self._proactive_coordinator.session_calls)
                context = self._get_agent_context()

                def worker():
                    try:
                        text = service.suggest(context, reason=reason)
                        self._run_on_main(
                            lambda: self._apply_agent_suggestion(
                                "ok", text, request_key))
                    except Exception:
                        self._run_on_main(
                            lambda: self._apply_agent_suggestion(
                                "error", None, request_key))

                threading.Thread(
                    target=worker, name="agent-suggestion", daemon=True).start()
        elif pending and service is not None and not service.provider.available:
            # 离线时本地指导卡已经实时更新，无需重复生成消息。
            self._agent_pending_llm = None
        self._agent_suggestion_job = self.root.after(
            self.AGENT_SUGGESTION_CHECK_MS, self._schedule_agent_suggestion)

    def _apply_agent_suggestion(
        self, kind: str, text: str | None, request_key: tuple,
    ) -> None:
        self._agent_suggestion_inflight = False
        if self._closing or self.agent_panel is None:
            return
        if request_key != self._agent_active_request_key:
            return
        self._agent_active_request_key = None
        if kind == "ok" and text:
            self.agent_panel.set_suggestion(text, source="DeepSeek · 关键事件")
        else:
            fallback = build_suggestion(self._get_agent_context())
            self.agent_panel.set_suggestion(fallback, source="本地提示")

    def _on_agent_ask(self, question: str, include_status: bool):
        context = self._get_agent_context()
        self.agent_panel.set_experiment_context(context)
        self.log.write(
            f"[实验助手] 提问：{question[:180]}；"
            f"附加实时状态={'是' if include_status else '否'}；"
            f"当前步骤={context.get('experiment_progress', {}).get('step_number', '--')}/5")
        if include_status and self._is_image_review_question(question):
            self._start_agent_image_review(question, from_chat=True)
            return
        if not self.agent_session.ask(question, include_status, context):
            self.agent_panel.append("系统", "上一条问题仍在处理中。")
            self.agent_panel.set_ai_state("上一任务仍在处理中", "warning")
            return
        self.root.after(50, self._poll_agent_response)

    @staticmethod
    def _is_image_review_question(question: str) -> bool:
        text = str(question).lower()
        return any(keyword in text for keyword in (
            "看图", "识图", "这张图", "当前画面", "条纹图",
            "效果图", "图像分析", "image", "vision"))

    def _on_agent_review_image(self) -> None:
        self._start_agent_image_review(
            "请根据当前条纹画面和程序指标复核条纹效果，并给出一个小步调整建议。",
            from_chat=False,
        )

    def _on_agent_export_chat(self) -> None:
        """让实验者选择路径，并导出当前助手会话。"""
        from tkinter import filedialog as fd

        default_name = time.strftime("实验助手对话_%Y%m%d_%H%M%S.md")
        path = fd.asksaveasfilename(
            parent=self.root,
            title="导出实验助手对话",
            defaultextension=".md",
            initialfile=default_name,
            filetypes=(
                ("Markdown 文件", "*.md"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*"),
            ),
        )
        if not path:
            return
        try:
            target = export_conversation(
                path, self.agent_panel.conversation_entries())
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "导出失败", f"无法导出实验助手对话：\n{exc}", parent=self.root)
            self.log.write(f"[实验助手] 导出对话失败：{exc}")
            return
        self.log.write(f"[实验助手] 对话已导出：{target}")
        messagebox.showinfo(
            "导出完成", f"实验助手对话已保存到：\n{target}", parent=self.root)

    def _start_agent_image_review(self, prompt: str, *, from_chat: bool) -> None:
        if self._vision_review_inflight:
            if from_chat:
                self.agent_panel.append("系统", "上一项识图复核仍在进行中。")
                self.agent_panel.set_busy(False)
            return
        if not self.agent_service.provider.available:
            self.agent_panel.append("系统", "未配置 DeepSeek API Key，无法进行识图复核。")
            self.agent_panel.set_busy(False)
            return
        frame = self._current_analysis_frame()
        if frame is None:
            self.agent_panel.append("系统", "当前没有可供识图复核的干涉画面。")
            self.agent_panel.set_busy(False)
            return
        context = self._get_agent_context()
        self._vision_review_inflight = True
        cancel_event = threading.Event()
        self._vision_review_cancel_event = cancel_event
        self.agent_panel.set_busy(True)
        self.agent_panel.set_image_review_state(True)
        self.agent_panel.set_ai_state(
            f"正在调用 {self.agent_service.models['vision']} 复核当前条纹图…",
            "working")

        def worker():
            try:
                text = self.agent_service.inspect_fringe_image(
                    frame.copy(), context, cancel_event=cancel_event)
                self._run_on_main(
                    lambda value=text: self._apply_agent_image_review(
                        "ok", value))
            except Exception as exc:
                error = str(exc)
                self._run_on_main(
                    lambda value=error: self._apply_agent_image_review(
                        "error", value))

        threading.Thread(
            target=worker, name="agent-vision-review", daemon=True).start()

    def _apply_agent_image_review(self, kind: str, text: str) -> None:
        self._vision_review_inflight = False
        self._vision_review_cancel_event = None
        if self._closing or self.agent_panel is None:
            return
        self.agent_panel.set_busy(False)
        self.agent_panel.set_image_review_state(False)
        if kind == "ok" and text:
            self.agent_panel.append("助手", text)
            self.agent_panel.set_suggestion(text, source="DeepSeek 识图复核")
            self.agent_panel.set_ai_state("识图复核完成", "success")
            self.log.write(
                f"[实验助手] 识图复核完成：模型={self.agent_service.models['vision']}")
        else:
            self.agent_panel.append("系统", f"识图复核失败：{text}")
            self.agent_panel.set_ai_state("识图复核失败", "warning")

    def _on_agent_test(self):
        if not self.agent_session.test_connection():
            self.agent_panel.set_ai_state("上一任务仍在处理中", "warning")
            return
        self.root.after(50, self._poll_agent_response)

    def _on_agent_cancel(self):
        if (self._vision_review_inflight
                and self._vision_review_cancel_event is not None):
            self._vision_review_cancel_event.set()
            self.agent_panel.thinking_var.set("正在停止识图复核…")
            self.agent_panel.set_ai_state("正在停止识图复核…", "warning")
            return
        if self.agent_session.cancel():
            self.agent_panel.thinking_var.set("正在停止生成…")
            self.agent_panel.set_ai_state("正在停止生成…", "warning")

    def _on_agent_set_intent(self, kind: str) -> None:
        if kind not in INTENT_LABELS:
            return
        self._experiment_intent.update({
            "kind": kind,
            "objective": INTENT_LABELS[kind],
            "confirmed": True,
        })
        self.log.write(f"[实验助手] 实验目的已设为：{INTENT_LABELS[kind]}")

    def _on_agent_set_response_mode(self, mode: str) -> None:
        if mode not in {"quiet", "standard", "teaching"}:
            return
        self._experiment_intent["response_mode"] = mode
        if mode == "quiet":
            self._agent_pending_llm = None
        names = {"quiet": "安静", "standard": "标准", "teaching": "教学"}
        self.log.write(f"[实验助手] 主动响应模式：{names[mode]}")

    def _on_agent_toggle_laser_alignment(self, enabled: bool) -> None:
        """切换人工激光条纹调节模式；只改变指导状态，不执行设备运动。"""
        self._laser_alignment_active = bool(enabled)
        if enabled:
            self._laser_guidance_session.reset()
            self._last_laser_session = {}
            self._laser_checkpoint = None
            self._experiment_intent.update({
                "kind": "fringe_observation",
                "objective": "用激光调出粗细合适的竖直条纹",
                "response_mode": "teaching",
                "confirmed": True,
            })
            self.agent_panel.set_suggestion(
                "激光调节模式已开启。请从动镜背面观察；系统会按实时画面明确指导上方旋钮（左上侧）或下方旋钮（右下侧），每次约 1/16 圈。",
                source="激光条纹指导")
            self.log.write("[实验助手] 激光竖直条纹调节模式已开启（只读人工指导）")
        else:
            self._last_laser_session = {}
            if self._laser_ai_guidance_enabled:
                self._on_agent_toggle_laser_ai_guidance(False)
                self.agent_panel.set_laser_ai_guidance_enabled(False)
            self.log.write("[实验助手] 激光竖直条纹调节模式已结束")

    def _on_agent_laser_recheck(self) -> None:
        """用最新快照重新判断；达到完成门时在内存中保存可导出的检查点。"""
        if not self._laser_alignment_active:
            self.agent_panel.set_laser_alignment_active(True)
            self._on_agent_toggle_laser_alignment(True)
        context = self._get_agent_context()
        session = self._laser_guidance_session.observe(
            context, now=time.monotonic())
        self._last_laser_session = session
        context["vision"]["laser_guidance_session"] = session
        self.agent_panel.set_experiment_context(context)
        if session.get("ready"):
            frame = self._current_analysis_frame()
            self._laser_checkpoint = {
                "captured_at": time.time(),
                "session": dict(session),
                "context": context,
                "frame_bgr": frame.copy() if frame is not None else None,
            }
            self.agent_panel.set_adjustment_result(
                "激光预调检查点已保存，可导出实验记录并准备切换白光。")
            self.log.write("[实验助手] 已保存激光预调检查点")
        else:
            self.log.write("[实验助手] 已使用最新画面重新判断激光预调状态")

    def _on_agent_toggle_laser_ai_guidance(self, enabled: bool) -> None:
        """切换低频视觉模型解释；只读，不向硬件队列写入。"""
        self._laser_ai_guidance_enabled = bool(enabled)
        self._laser_ai_guidance_generation += 1
        if self._laser_ai_guidance_cancel_event is not None:
            self._laser_ai_guidance_cancel_event.set()
        self._laser_ai_guidance_cancel_event = None
        self._laser_ai_guidance_inflight = False
        self._laser_ai_guidance_last_signature = None
        self._laser_ai_guidance_last_call_at = 0.0
        if enabled:
            self._laser_alignment_active = True
            if self.agent_service.provider.available:
                self.agent_panel.set_laser_ai_guidance(
                    "已开启，等待当前条纹状态稳定后进行首次识图…", "working")
            else:
                self.agent_panel.set_laser_ai_guidance(
                    "未配置可用的大模型 API；本地实时指导仍可使用。", "offline")
            self.log.write("[实验助手] 自动 AI 激光指导已开启（只读、状态变化触发）")
        else:
            self.agent_panel.set_laser_ai_guidance_enabled(False)
            self.log.write("[实验助手] 自动 AI 激光指导已关闭")

    def _maybe_start_laser_ai_guidance(self, context: dict) -> None:
        """关键状态变化时将当前 BGR 帧交给视觉模型，避免逐帧调用。"""
        if (self._closing or not self._laser_alignment_active
                or not self._laser_ai_guidance_enabled
                or self._laser_ai_guidance_inflight):
            return
        if not self.agent_service.provider.available:
            self.agent_panel.set_laser_ai_guidance(
                "未配置可用的大模型 API；当前仅显示本地实时指导。", "offline")
            return
        guidance = (context.get("vision") or {}).get("fringe_guidance") or {}
        if not guidance.get("laser_vertical_alignment"):
            self.agent_panel.set_laser_ai_guidance(
                "等待可靠条纹结果；证据不足时不会调用模型猜旋钮方向。", "working")
            return
        signature = laser_guidance_signature(guidance)
        if signature == self._laser_ai_guidance_last_signature:
            return
        now = time.monotonic()
        if now - self._laser_ai_guidance_last_call_at < self._laser_ai_min_interval_seconds:
            return
        frame = self._current_analysis_frame()
        if frame is None:
            self.agent_panel.set_laser_ai_guidance(
                "当前没有有效干涉画面，不调用模型，也不建议转动旋钮。", "error")
            return
        self._laser_ai_guidance_inflight = True
        self._laser_ai_guidance_last_call_at = now
        generation = self._laser_ai_guidance_generation
        cancel_event = threading.Event()
        self._laser_ai_guidance_cancel_event = cancel_event
        self.agent_panel.set_laser_ai_guidance(
            f"{self.agent_service.models['vision']} 正在复核当前画面…", "working")

        def worker():
            try:
                answer = self.agent_service.inspect_fringe_image(
                    frame.copy(), context, cancel_event=cancel_event,
                    guidance_mode="laser_auto")
                kind = "ok"
            except Exception as exc:
                answer = str(exc)
                kind = "error"
            self._run_on_main(
                lambda: self._apply_laser_ai_guidance(
                    kind, answer, signature, generation, guidance))

        threading.Thread(
            target=worker, name="laser-ai-guidance", daemon=True).start()

    def _apply_laser_ai_guidance(
        self, kind: str, text: str, signature: tuple, generation: int,
        guidance: dict,
    ) -> None:
        if generation != self._laser_ai_guidance_generation:
            return
        self._laser_ai_guidance_inflight = False
        self._laser_ai_guidance_cancel_event = None
        if self._closing or not self._laser_ai_guidance_enabled:
            return
        if signature != laser_guidance_signature(self._last_fringe_guidance):
            self.agent_panel.set_laser_ai_guidance(
                "模型返回前画面已经变化，旧建议已丢弃。", "working")
            return
        if kind == "ok" and text and validate_laser_ai_guidance(text, guidance):
            self._laser_ai_guidance_last_signature = signature
            self.agent_panel.set_laser_ai_guidance(text, "ready")
            self.log.write("[实验助手] 自动 AI 激光指导已更新")
            return
        if kind == "ok":
            self._laser_ai_guidance_last_signature = signature
            fallback = render_laser_alignment_instruction(
                guidance.get("laser_vertical_alignment"))
            self.agent_panel.set_laser_ai_guidance(
                "模型建议未通过旋钮/方向校验，已改用本地规则：\n" + fallback,
                "error")
            self.log.write("[实验助手] AI 建议未通过安全校验，已回退本地规则")
        else:
            self.agent_panel.set_laser_ai_guidance(
                f"AI 识图失败：{text}；本地实时指导仍在工作。", "error")

    def _on_export_experiment_record(self) -> None:
        """导出激光任务状态、事件时间线和可复核的当前分析帧。"""
        from tkinter import filedialog as fd

        path = fd.asksaveasfilename(
            parent=self.root, title="导出激光预调实验记录",
            defaultextension=".json",
            initialfile=time.strftime("激光预调记录_%Y%m%d_%H%M%S.json"),
            filetypes=(("JSON 实验记录", "*.json"), ("所有文件", "*.*")))
        if not path:
            return
        target = Path(path)
        checkpoint = self._laser_checkpoint or {}
        frame = checkpoint.get("frame_bgr")
        if frame is None:
            current = self._current_analysis_frame()
            frame = current.copy() if current is not None else None
        frame_file = None
        if isinstance(frame, np.ndarray) and frame.size:
            image_path = target.with_name(target.stem + "_frame.png")
            ok, encoded = cv2.imencode(".png", frame)
            if ok:
                encoded.tofile(str(image_path))
                frame_file = image_path.name
        context = checkpoint.get("context") or self._get_agent_context()
        record = {
            "schema": "ai-interferometry-laser-guidance/v1",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint_captured_at": checkpoint.get("captured_at"),
            "task": "laser_vertical_fringe_prealignment",
            "software": self._software_revision(),
            "session": checkpoint.get("session") or self._last_laser_session,
            "events": self._laser_guidance_session.events(),
            "frame_file": frame_file,
            "camera": context.get("camera"),
            "vision": context.get("vision"),
            "experiment_intent": context.get("experiment_intent"),
            "quantity_notes": {
                "angle_deg": "条纹中心线相对竖直方向的有符号倾角，单位 degree",
                "spacing_px": "ROI 内法向条纹间距，单位 pixel",
                "color": "相机颜色仅作形态描述，未经标定不得解释为光学相位",
            },
        }
        try:
            target.write_text(json.dumps(
                record, ensure_ascii=False, indent=2,
                default=lambda value: (
                    value.item() if isinstance(value, np.generic)
                    else value.tolist() if isinstance(value, np.ndarray)
                    else str(value))), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.log.write(f"[实验助手] 激光预调实验记录已导出：{target}")
        messagebox.showinfo(
            "导出完成",
            f"实验记录：\n{target}"
            + (f"\n检查点画面：{frame_file}" if frame_file else "\n当前无有效画面。"),
            parent=self.root)

    @staticmethod
    def _software_revision() -> dict:
        """返回实验记录所需的软件版本来源；Git 不可用时明确标记未知。"""
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip())
            return {"name": "AI_Interferometry", "git_commit": revision,
                    "working_tree_dirty": dirty}
        except (OSError, subprocess.SubprocessError):
            return {"name": "AI_Interferometry", "git_commit": None,
                    "working_tree_dirty": None}

    def _current_adjustment_metrics(self) -> dict:
        guidance = self._last_fringe_guidance or {}
        metrics = dict(guidance.get("metrics") or {})
        metrics["quality_score"] = guidance.get("quality_score")
        return metrics

    def _on_agent_mark_adjustment(self) -> None:
        metrics = self._current_adjustment_metrics()
        if not any(metrics.get(key) is not None for key in (
                "angle_deg", "curvature", "spacing_cv_percent", "quality_score")):
            self.agent_panel.set_adjustment_result(
                "当前没有可用条纹指标；请先启动预测并获得稳定条纹。")
            return
        self._fringe_adjustment_baseline = metrics
        self.agent_panel.set_adjustment_result(
            "已记录调节前状态。现在只微调一个旋钮，稳定后点击“比较调节后”。")
        self.log.write("[实验助手] 已记录条纹调节前指标")

    def _on_agent_compare_adjustment(self) -> None:
        current = self._current_adjustment_metrics()
        result = compare_fringe_adjustment(
            self._fringe_adjustment_baseline or {},
            current,
        )
        if result["outcome"] != "insufficient":
            self._fringe_adjustment_baseline = current
        text = f"{result['summary']} {result['recommendation']}"
        self.agent_panel.set_adjustment_result(text)
        self.agent_panel.set_suggestion(text, source="调节效果比较")
        self.log.write(f"[实验助手] 条纹调节比较：{result['outcome']}")

    # ==================================================================
    # 智能体工具接线（ToolContext + 确认握手 + 活动流）
    # ==================================================================
    def _run_on_main(self, fn):
        """在 Tk 主线程执行 ``fn`` 并阻塞等待结果，供智能体后台线程安全访问 UI/相机。"""
        if threading.current_thread() is threading.main_thread():
            return fn()
        root = getattr(self, "root", None)
        if root is None:
            return fn()
        box: list = []
        done = threading.Event()

        def _runner():
            try:
                box.append(fn())
            except Exception as exc:  # noqa: BLE001 — 结果回传给调用线程
                box.append(exc)
            finally:
                done.set()

        try:
            root.after(0, _runner)
        except tk.TclError:
            return fn()
        if not done.wait(timeout=15.0):
            return None
        result = box[0] if box else None
        if isinstance(result, Exception):
            raise result
        return result

    def _build_agent_tool_context(self) -> None:
        """构建注入活句柄的 ToolContext，挂到 AgentService 并接好确认 / 活动流回调。"""
        ctx = ToolContext(
            get_snapshot=lambda: self._run_on_main(self._get_agent_context),
            latest_frame=lambda: self._run_on_main(self._current_analysis_frame),
            read_micrometer=self._fresh_micrometer_reading,
            query_motor=self._agent_query_motor,
            center_line_x=lambda: self._center_line_x,
            frame_width=lambda: self._prediction_frame_width,
            start_auto_center=self._agent_start_auto_center,
            stop_auto_center=self._agent_stop_auto_center,
            start_measurement=self._agent_start_measurement,
            stop_measurement=self._agent_stop_measurement,
            start_backlash=self._agent_start_backlash,
            stop_backlash=self._agent_stop_backlash,
            motor_emergency_stop=self._agent_motor_emergency_stop,
            run_on_main=self._run_on_main,
            on_plan=lambda plan: self._run_on_main(
                lambda: self._agent_on_plan(plan)),
            on_note=lambda note: self._run_on_main(
                lambda: self._agent_on_note(note)),
        )
        self.agent_service.set_tool_context(ctx)
        self.agent_service.confirm_handler = self._confirm_agent_motion
        self.agent_service.on_step = self._on_agent_step

    @staticmethod
    def _describe_motion(tool_name: str, arguments: dict) -> str:
        if tool_name == "auto_center_start":
            return "启动自动寻中（把中心黑条纹移到画面中央）"
        if tool_name == "measurement_start":
            target = arguments.get("target_mm")
            target_text = (
                f"目标 {float(target):.6f} mm"
                if target is not None else "面板目标读数")
            return f"启动目标读数测量（{target_text}）"
        if tool_name == "backlash_measure":
            return (f"启动回程差测量（{arguments.get('start_mm')} → "
                    f"{arguments.get('end_mm')} mm）")
        return tool_name

    def _confirm_agent_motion(self, tool_name: str, arguments: dict) -> bool:
        """运动工具确认：在主线程弹确认行，阻塞等待用户点击。"""
        if self.agent_panel is None:
            return False
        confirmation = Confirmation(tool_name=tool_name, arguments=arguments)
        self._pending_confirmation = confirmation
        summary = self._describe_motion(tool_name, arguments)
        try:
            self._run_on_main(
                lambda: self.agent_panel.show_motion_confirmation(
                    tool_name, summary))
            confirmed = confirmation.event.wait(
                timeout=float(self.agent_service.tool_timeout_seconds))
        finally:
            self._pending_confirmation = None
            try:
                self._run_on_main(self.agent_panel.hide_motion_confirmation)
            except Exception:
                pass
        return bool(confirmed and confirmation.approved)

    def _on_agent_confirm_motion(self, tool_name: str) -> None:
        confirmation = self._pending_confirmation
        if confirmation is not None and confirmation.tool_name == tool_name:
            confirmation.approve()

    def _on_agent_reject_motion(self, tool_name: str) -> None:
        confirmation = self._pending_confirmation
        if confirmation is not None and confirmation.tool_name == tool_name:
            confirmation.reject()

    def _on_agent_toggle_autonomous(self, enabled: bool) -> None:
        self.agent_service.autonomous_enabled = bool(enabled)
        self.log.write(f"[实验助手] 自主执行{'开启' if enabled else '关闭'}")

    def _on_agent_toggle_dry_run(self, enabled: bool) -> None:
        self.agent_service.dry_run = bool(enabled)
        self.agent_service.agent_loop.dry_run = bool(enabled)
        self.log.write(f"[实验助手] 仅规划模式{'开启' if enabled else '关闭'}")

    def _on_agent_set_guidance_stage(self, stage: str) -> None:
        panel = self.manual_auto_center_panel
        if panel is None or stage not in {
                "advisory", "confirm", "closed_loop", "adaptive"}:
            return
        panel.set_execution_stage(stage)
        if self._last_fringe_guidance:
            self._last_fringe_guidance["execution_stage"] = stage
        stage_names = {
            "advisory": "阶段 1 只读诊断",
            "confirm": "阶段 2 确认执行",
            "closed_loop": "阶段 3 安全闭环",
            "adaptive": "阶段 4 自适应优化",
        }
        self.log.write(f"[实验助手] 条纹调节切换为{stage_names[stage]}")

    def _on_agent_apply_guidance(self) -> None:
        self._on_auto_center_command("apply_guidance")

    def _on_agent_auto_center(self, command: str) -> None:
        if command not in {"start", "stop"}:
            return
        self._on_auto_center_command(command)

    def _on_agent_emergency_stop(self) -> None:
        self._agent_motor_emergency_stop()
        self.agent_panel.append_tool_activity("急停：已停止电机与所有自动控制")
        self.log.write("[实验助手] 急停")

    def _agent_query_motor(self) -> dict:
        return {
            "connected": self.motor_connected,
            "auto_enabled": self.auto_control_enabled,
            "auto_control_state": self._last_auto_state,
            "direction": self.auto_controller.direction,
            "gear": self.auto_controller.gear,
            "measurement_active": self._measurement_active,
            "backlash_active": self._backlash_active,
            "note": "状态来自运行时快照（不触发串口查询）",
        }

    def _agent_start_auto_center(self) -> dict:
        self._on_auto_start()
        return {
            "started": self.auto_control_enabled,
            "auto_control_state": self._last_auto_state,
            "note": "已请求启动自动寻中，请核对自动寻中面板状态",
        }

    def _agent_stop_auto_center(self) -> dict:
        self._on_auto_stop("智能体停止")
        return {"stopped": True, "note": "已停止自动寻中"}

    def _agent_start_measurement(self, target_mm: float | None) -> dict:
        panel = self.temporary_measurement_panel
        if target_mm is not None and panel is not None:
            panel.target_var.set(f"{target_mm:.6f}")
        self._on_temporary_measurement_cmd("measurement_start")
        return {
            "started": self._measurement_active,
            "target_mm": self._measurement_target_mm,
            "note": "已请求启动目标读数测量，请核对临时测量面板状态",
        }

    def _agent_stop_measurement(self) -> dict:
        self._stop_measurement("智能体停止")
        return {"stopped": True, "note": "已停止目标读数测量"}

    def _agent_start_backlash(self, start_mm: float, end_mm: float) -> dict:
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.set_backlash_start(start_mm)
            panel.set_backlash_end(end_mm)
        self._on_temporary_measurement_cmd("backlash_start")
        return {
            "started": self._backlash_active,
            "start_mm": self._backlash_start_mm,
            "end_mm": self._backlash_end_mm,
            "note": "已请求启动回程差测量，请核对临时测量面板状态",
        }

    def _agent_stop_backlash(self) -> dict:
        self._stop_backlash("智能体停止")
        return {"stopped": True, "note": "已停止回程差测量"}

    def _agent_motor_emergency_stop(self) -> dict:
        if self.auto_control_enabled:
            self._on_auto_stop("智能体急停")
        if self._measurement_active:
            self._stop_measurement("智能体急停")
        if self._backlash_active:
            self._stop_backlash("智能体急停")
        controller = self.motor
        if controller is not None:
            self.motor_commands.submit(
                "agent_emergency_stop", controller.stop,
                priority=0, coalesce=True)
        return {"ok": True, "note": "已急停电机并停止所有自动控制"}

    def _agent_on_plan(self, plan: str) -> None:
        if self.agent_panel is not None:
            self.agent_panel.set_plan(plan)
        self.log.write(f"[实验助手计划] {plan[:200]}")

    def _agent_on_note(self, note: str) -> None:
        if self.agent_panel is not None:
            self.agent_panel.append_tool_activity(f"备注：{note}")
        self.log.write(f"[实验助手备注] {note}")

    def _on_agent_step(self, step) -> None:
        self._run_on_main(lambda: self._render_agent_step(step))

    def _render_agent_step(self, step) -> None:
        if self.agent_panel is None:
            return
        if step.kind == "tool":
            mark = "✓" if step.ok else "✗"
            result = (step.result or "").replace("\n", " ").strip()
            self.agent_panel.append_tool_activity(
                f"{mark} {step.title}：{result[:60]}")
        elif step.kind == "error":
            text = (step.detail or step.result or "").strip()
            self.agent_panel.append_tool_activity(f"✗ {text[:70]}")
        elif step.kind == "final":
            self.agent_panel.append_tool_activity("完成：已生成最终回答")

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
                self._close_interferometer_camera("视觉观察模块已关闭")
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
    def _close_interferometer_camera(self, reason: str = "摄像头已关闭") -> None:
        """统一停止预览、预测、录像并释放干涉相机。"""
        self._stop_preview()
        self._stop_predict()
        cleanup_errors: list[str] = []
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception as exc:
                cleanup_errors.append(f"录像停止失败：{exc}")
        camera = self.cam
        self.cam = None
        self.camera_running = False
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:
                cleanup_errors.append(f"摄像头释放失败：{exc}")

        self._center_line_x = None
        self._center_line_box = None
        self._zero_box_x = None
        self._zero_box_confidence = 0.0
        self._center_tracker.reset()
        self._reset_box_stability()
        self._center_yolo_misses = 0

        self._set_status(reason)
        if self.camera_plugin is not None:
            self.camera_plugin.set_clarity_status("摄像头未连接")
        self.log.write(f"[相机] {reason}，预测、录像与中心跟踪已停止")
        for error in cleanup_errors:
            self.log.write(f"[警告] {error}")

    def _on_camera_cmd(self, cmd: str):
        cp = self.camera_plugin
        if cmd == "detect":
            if self._camera_scan_future is None:
                self._set_status("正在后台检测摄像头...")
                self._camera_scan_future = self._camera_executor.submit(CameraManager.detect_all)
                self.root.after(50, self._poll_camera_scan)
        elif cmd == "open":
            if self.camera_running:
                self._set_status("摄像头已在运行")
                return
            camera = None
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
                camera = CameraManager(
                    index=requested_index,
                    resolution=(int(resolution[0]), int(resolution[1])),
                    fps=int(camera_preset["fps"]),
                    clarity_config=dict(camera_preset["clarity_assist"]),
                    owner="interferometer-camera",
                )
                if not camera.start():
                    raise RuntimeError("无法打开摄像头")
                self.cam = camera
                self.camera_running = True
                self._set_status("摄像头已启动")
                self._start_preview()
                self._apply_camera_clarity("摄像头启动")
                self.log.write(
                    f"[相机] 干涉画面摄像头 {requested_index} 已打开，"
                    f"分辨率 {resolution[0]}x{resolution[1]}")
            except Exception as exc:
                if camera is not None:
                    camera.stop()
                self.camera_running = False
                self.cam = None
                self._set_status(f"摄像头失败: {exc}")
                cp.set_clarity_status("摄像头打开失败")
                self.log.write(f"[错误] 摄像头打开失败：{exc}")
        elif cmd == "close":
            self._close_interferometer_camera("摄像头已关闭")
        elif cmd == "angle_apply":
            angle = cp.angle
            self.corrector.set_manual_offset(angle)
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 已应用旋转角度 {angle:+.2f}°")
        elif cmd == "angle_reset":
            angle = float(self.recording_preset["main_camera"]["angle_deg"])
            cp.angle_var.set(str(angle))
            self.corrector.set_manual_offset(angle)
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 旋转角度已恢复预设 {angle:+.2f}°")
        elif cmd == "angle_auto":
            msg = self._auto_rotate_fringes()
            if msg is not None:
                self._set_status(msg)
        elif cmd == "zoom_apply":
            zoom = cp.zoom
            self.corrector.zoom = zoom
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 已应用缩放倍数 {zoom:.2f}")
        elif cmd == "zoom_reset":
            zoom = float(self.recording_preset["main_camera"]["zoom"])
            cp.zoom_var.set(str(zoom))
            self.corrector.zoom = zoom
            self._preview_adjusted = True
            self.log.write(f"[画面矫正] 缩放已恢复预设 {zoom:.2f}")
        elif cmd == "pan_reset":
            self.corrector.pan_x = 0
            self.corrector.pan_y = 0
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
    def _on_thickness_measurement_command(
        self, command: str, payload: dict | None = None,
    ) -> None:
        panel = self.thickness_measurement_panel
        if panel is None:
            return
        if command == "record":
            reading = self._fresh_micrometer_reading()
            if reading is None:
                panel.set_status("无法记录：请先取得近期稳定的微分表读数")
                self.log.write("[厚度测量] 记录失败：没有近期可信微分表读数")
                return
            try:
                record = panel.add_record(reading, self.micrometer_reading_at)
            except ValueError as exc:
                panel.set_status(f"无法记录：{exc}")
                self.log.write(f"[厚度测量] 记录失败：{exc}")
                return
            self.log.write(
                f"[厚度测量] {record.key}={record.value_mm:.6f} mm")
        elif command == "calculate" and payload:
            self.log.write(
                f"[厚度测量] d1={payload['d1_mm']:.6f} mm，"
                f"d2={payload['d2_mm']:.6f} mm，"
                f"h={payload['thickness_mm']:.6f} mm")
        elif command == "delete" and payload:
            self.log.write(f"[厚度测量] 已删除 {payload['id']}")
        elif command == "clear":
            self.log.write("[厚度测量] 已清空记录")
        # 每次记录变更后同步到实验助手面板
        self._sync_readings_to_assistant()

    def _on_experiment_assistant_command(
        self, command: str, payload: dict | None = None,
    ) -> None:
        panel = self.experiment_assistant_panel
        if panel is None:
            return
        if command == "round_added" and payload:
            self.log.write(
                f"[实验助手] 添加第{payload['sequence']}次测量："
                f"d1={payload['d1_mm']:.6f}，d2={payload['d2_mm']:.6f}，"
                f"h={payload['thickness_mm']:.6f} mm")
        elif command == "round_deleted" and payload:
            self.log.write(
                f"[实验助手] 已删除第{payload['sequence']}次测量")
        elif command == "cleared":
            self.log.write("[实验助手] 所有测量记录已清空")
        elif command == "saved" and payload:
            self.log.write(f"[实验助手] 会话已保存至 {payload['path']}")
        elif command == "loaded" and payload:
            self.log.write(f"[实验助手] 会话已加载自 {payload['path']}")

    def _sync_readings_to_assistant(self) -> None:
        """将厚度测量面板的已记录读数同步到实验助手。"""
        thickness = self.thickness_measurement_panel
        assistant = self.experiment_assistant_panel
        if thickness is None or assistant is None:
            return
        snapshot = thickness.snapshot()
        assistant.set_available_readings(snapshot.get("records", []))

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
                self.micrometer_panel.set_status("摄像头检测完成")
            else:
                reader = result
                if self._closing:
                    reader.close()
                    return
                self.micrometer_reader = reader
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
            if self.thickness_measurement_panel is not None:
                self.thickness_measurement_panel.set_current_reading(
                    control_reading, self.micrometer_reading_at)
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
        if self.thickness_measurement_panel is not None:
            self.thickness_measurement_panel.set_current_reading(None)
        if getattr(self, "recording_sidebar", None) is not None:
            self.recording_sidebar.reset_meter_preview(message)

    # ==================================================================
    # 模型插件
    # ==================================================================
    def _on_model_cmd(self, cmd: str):
        if cmd == "load":
            if self.detector.is_loaded():
                self.log.write("YOLO 模型已加载")
                return
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
            if self.predict_running:
                self._set_status("连续预测已在运行")
                return
            if self.cam is None or not self.camera_running:
                self._set_status("请先打开摄像头")
                self.log.write("[警告] 请先打开摄像头")
                return
            if not self.detector.is_loaded():
                self._set_status("请先加载 YOLO 模型")
                self.log.write("[警告] 请先加载 YOLO 模型")
                return
            # 相机管理器持续缓存最新帧，预览和推理可以并行；自动页不再因
            # 单次 YOLO 推理耗时而冻结。
            self.predict_running = True
            self._set_status("预测运行中")
            self._predict_loop()
            self._start_preview()
            self.log.write(
                f"[YOLO] 连续预测已启动，置信度阈值 {self.model_plugin.conf:.2f}，"
                f"IoU {self.model_plugin.iou:.2f}，推理尺寸 {self.model_plugin.imgsz}")
        elif cmd == "single":
            if self.cam is None or not self.detector.is_loaded():
                self._set_status("请先打开摄像头并加载模型")
                self.log.write("[警告] 请先打开摄像头并加载模型")
                return
            frame = self.cam.read()
            if frame is None:
                self._set_status("未读取到摄像头画面")
                self.log.write("[警告] 单帧预测未读取到摄像头画面")
                return
            corrected = rotate_expand(frame, self.corrector.effective_angle)
            corrected = self.corrector.apply_zoom_pan(corrected)
            roi = self._get_roi()
            result = self.detector.detect(corrected, roi=roi)
            annotated = result["annotated"] if result["annotated"] is not None else corrected
            if roi:
                cv2.rectangle(
                    annotated,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 255, 0),
                    2,
                )
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

    # -----------------------------------------------------------------
    # 零级框稳定性追踪
    # -----------------------------------------------------------------

    def _update_box_stability(self, box_cx: float, box_width: float) -> None:
        """根据最近帧的框位置/宽度方差判断零级框是否稳定。

        只有当框中心抖动 < 15% 框宽且宽度变化 < 20% 均值，
        且连续 3 帧满足条件时，才标记为稳定。连续 3 帧不稳定则取消标记。
        """
        self._zero_box_missing_counter = 0
        self._zero_box_history.append((float(box_cx), float(box_width)))

        if len(self._zero_box_history) >= 5:
            recent = list(self._zero_box_history)[-5:]
            cxs = [h[0] for h in recent]
            widths = [h[1] for h in recent]
            mean_width = float(np.mean(widths))
            if mean_width <= 0:
                return
            cx_std = float(np.std(cxs))
            width_std = float(np.std(widths))

            is_stable = (
                cx_std < mean_width * 0.15
                and width_std < mean_width * 0.20
            )
            if is_stable:
                self._zero_box_stable_counter += 1
                self._zero_box_unstable_counter = 0
            else:
                self._zero_box_stable_counter = 0
                self._zero_box_unstable_counter += 1

            if self._zero_box_stable_counter >= 3:
                self._zero_box_stable = True
            elif self._zero_box_unstable_counter >= 3:
                self._zero_box_stable = False

    def _update_box_missing(self) -> None:
        """当前帧未检测到零级框时调用，超过容限后重置稳定性。"""
        self._zero_box_missing_counter += 1
        if self._zero_box_missing_counter >= 3:
            self._zero_box_stable = False
            self._zero_box_stable_counter = 0
            self._zero_box_unstable_counter = 0
            self._zero_box_history.clear()

    def _reset_box_stability(self) -> None:
        """完全重置零级框稳定性状态（切换模式/关闭相机/重新检测时调用）。"""
        self._zero_box_stable = False
        self._zero_box_stable_counter = 0
        self._zero_box_unstable_counter = 0
        self._zero_box_missing_counter = 0
        self._zero_box_history.clear()

    def _hold_or_clear_center(self, reason: str, verbose: bool = False):
        """短时沿用上一帧结果；连续丢失后回退到 YOLO 框中心。"""
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

        # 精细中心线不可用但 YOLO 零级框仍在 → 用 YOLO 框中心做 fallback
        if (self._zero_box_x is not None
                and self._zero_box_confidence >= 0.40):
            tracked = self._center_tracker.reset_from_yolo(
                self._zero_box_x, self._zero_box_confidence)
            self._center_line_x = tracked["center"]
            self._center_confidence = tracked["confidence"]
            if self._center_line_box is not None:
                x1 = self._center_line_box[0]
                self.fringe_center_plugin.update_result(
                    self._center_line_x - x1,
                    tracked["confidence"],
                    True,
                    f"YOLO回退：{reason}",
                )
            else:
                self.fringe_center_plugin.update_result(
                    None, 0, False,
                    f"YOLO锚定 x={self._center_line_x:.1f}px，{reason}",
                )
            if verbose:
                self.log.write(
                    f"[中心条纹] {reason}，"
                    f"已回退至YOLO零级框中心 x={self._center_line_x:.1f}px")
            return

        self._center_line_x = None
        self._center_confidence = 0.0
        self._center_line_box = None
        self.fringe_center_plugin.update_result(None, 0, False, reason)

    def _track_center_from_previous_roi(self, corrected: np.ndarray) -> bool:
        """YOLO 漏检时，在上一帧零级区域内继续跟踪白光竖条纹。"""
        if self._center_line_x is None or self._center_line_box is None:
            return False
        tracked, _, _ = self._locate_center_in_box(
            corrected,
            self._center_line_box,
            expected_x=self._center_line_x,
            min_confidence=0.08,
        )
        if tracked is None:
            return False
        x1 = self._center_line_box[0]
        self._center_line_x = tracked["center"]
        self._center_confidence = float(tracked["confidence"])
        self.fringe_center_plugin.update_result(
            self._center_line_x - x1,
            tracked["confidence"],
            True,
            "零级框短时漏检，正在视觉跟踪",
        )
        return True

    def _locate_center_in_box(
        self, corrected: np.ndarray, box,
        *, expected_x: float, min_confidence: float = 0.0,
    ) -> tuple[dict | None, dict | None, str]:
        """在检测框扩展区域中定位中心，供 YOLO 命中和短时漏检共用。"""
        x1, y1, x2, y2 = (int(value) for value in box)
        height, width = corrected.shape[:2]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        cfg = self.recording_preset["yolo"]
        expand_w = max(int(box_w * float(cfg["center_search_expand_ratio"])), 80)
        search_margin = max(
            int(box_w * float(cfg["center_search_margin_ratio"])), 6)
        center_y = (y1 + y2) / 2.0
        x1c = max(0, int(expected_x - expand_w / 2))
        x2c = min(width, int(expected_x + expand_w / 2))
        y1c = max(0, int(center_y - max(box_h, 40) / 2))
        y2c = min(height, int(center_y + max(box_h, 40) / 2))
        if x2c - x1c < 20 or y2c - y1c < 20:
            return None, None, f"扩展区域太小 ({x2c-x1c}x{y2c-y1c})"
        # 只有连续稳定的零级框才用于约束搜索范围；不稳定框不传 search_bounds，
        # 避免因框抖动导致搜索区域跳变。
        if self._zero_box_stable:
            search_bounds_param = (
                max(0, x1 - x1c - search_margin),
                min(x2c - x1c, x2 - x1c + search_margin),
            )
        else:
            search_bounds_param = None
        try:
            roi = corrected[y1c:y2c, x1c:x2c]
            expected_in_roi = expected_x - x1c
            search_radius = max(
                box_w * float(cfg["center_search_radius_ratio"]), 15.0)
            mode = getattr(self.fringe_center_plugin, "recognition_mode",
                           "refined")
            if mode == "band":
                info = find_center_by_band(
                    roi,
                    expected_center_x=expected_in_roi,
                    search_radius=search_radius,
                    search_bounds=search_bounds_param,
                )
            else:
                info = find_center_in_region(
                    roi,
                    expected_center_x=expected_in_roi,
                    search_radius=search_radius,
                    search_bounds=search_bounds_param,
                )
        except Exception as exc:
            return None, None, f"检测异常：{exc}"
        if info["orientation"] != "vertical":
            return None, info, "当前区域不是白光竖条纹"
        if float(info["confidence"]) < min_confidence:
            return None, info, "中心定位置信度过低"
        measured_x = float(np.clip(
            x1c + info["center_main"], x1c + 1, x2c - 1))
        tracked = self._center_tracker.update(measured_x, info["confidence"])
        if tracked["center"] is None:
            return None, info, "中心位置跳变过大"
        return tracked, info, ""

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
            self._zero_box_x = None
            self._zero_box_confidence = 0.0
            self._update_box_missing()
            if verbose:
                self.log.write("[中心条纹] YOLO 未检测到任何目标")
            if self._center_yolo_misses <= 8 and self._track_center_from_previous_roi(corrected):
                return
            self._hold_or_clear_center("YOLO 未检测到零级条纹", verbose)
            return

        # 按模型自身的类别名称匹配零级/黑条，禁止假设固定 class_id。
        zero_class_ids = self.detector.find_class_ids("zero", "order", "black", "零级", "黑")
        zero_indices = [i for i, cid in enumerate(class_ids) if int(cid) in zero_class_ids]

        # 不把其他类别的最高置信度框冒充零级条纹，否则会产生大偏移。
        if not zero_indices:
            self._center_yolo_misses += 1
            self._zero_box_x = None
            self._zero_box_confidence = 0.0
            self._update_box_missing()
            if self._center_yolo_misses <= 8 and self._track_center_from_previous_roi(corrected):
                return
            self._hold_or_clear_center("未检测到零级条纹框", verbose)
            return

        best_local_idx = int(np.argmax(confs[zero_indices]))
        best_idx = zero_indices[best_local_idx]
        self._center_yolo_misses = 0
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        box_cx = (x1 + x2) / 2.0
        # 始终记录 YOLO 零级框位置，供自动寻中状态机在阶段二/三使用
        self._zero_box_x = box_cx
        self._zero_box_confidence = float(confs[best_idx])
        self._update_box_stability(box_cx, float(x2 - x1))
        tracked, info, reason = self._locate_center_in_box(
            corrected, (x1, y1, x2, y2), expected_x=box_cx)
        if tracked is None:
            # 精细中心线定位失败 → 有 YOLO 框时直接用框中心作为 fallback
            if (self._zero_box_x is not None
                    and self._zero_box_confidence >= 0.40):
                tracked = self._center_tracker.update(
                    self._zero_box_x, self._zero_box_confidence * 0.85)
                if tracked["center"] is not None:
                    self._center_line_x = tracked["center"]
                    self._center_confidence = float(tracked["confidence"])
                    self._center_line_box = (x1, y1, x2, y2)
                    self.fringe_center_plugin.update_result(
                        self._center_line_x - x1, tracked["confidence"], True)
                    if verbose:
                        self.log.write(
                            f"[中心条纹] {reason}，"
                            f"已回退至YOLO零级框中心 "
                            f"x={self._center_line_x:.1f}px")
                    return
            self._hold_or_clear_center(reason, verbose)
            return

        center_x_final = tracked["center"]

        # 交叉校验：若中心线位置相对稳定零级框漂移过大，向框中心回拉。
        # 仅对稳定框做硬约束，不稳定框不触发回拉以免追逐抖动。
        if (self._zero_box_stable
                and self._zero_box_x is not None
                and self._zero_box_confidence >= 0.55):
            box_half_width = (x2 - x1) / 2.0
            drift = abs(center_x_final - self._zero_box_x)
            # 中心最多偏离框中心 85% 半宽，保证始终在框内
            max_allowed_drift = max(20.0, box_half_width * 0.85)
            if drift > max_allowed_drift:
                yolo_w = self._zero_box_confidence
                line_w = tracked["confidence"]
                total_w = yolo_w + line_w
                if total_w > 0:
                    center_x_final = (
                        yolo_w * self._zero_box_x + line_w * center_x_final
                    ) / total_w
                if verbose:
                    self.log.write(
                        f"[中心条纹] 中心线漂移 {drift:.1f}px > "
                        f"允许 {max_allowed_drift:.1f}px，"
                        f"已校正至 x={center_x_final:.1f}px")

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
                self._zero_box_x = None
                self._zero_box_confidence = 0.0
                self._center_tracker.reset()
                self._reset_box_stability()
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

        panel = self.temporary_measurement_panel
        if panel is not None and panel.thickness_roi_mode:
            self._thickness_roi_drawing = True
            self._thickness_roi_start = (event.x, event.y)
            self._thickness_roi_rect_id = None
        elif self.model_plugin and self.model_plugin.roi_mode:
            self._roi_drawing = True
            self._roi_start = (event.x, event.y)
            self._roi_rect_id = None
        elif self.camera_plugin and self.camera_plugin.pan_mode:
            self._panning = True
            self._pan_start = (event.x, event.y)
            self._pan_orig = (self.corrector.pan_x, self.corrector.pan_y)

    def _on_roi_drag(self, event):
        if self._thickness_roi_drawing:
            c = self._roi_canvas
            if self._thickness_roi_rect_id:
                c.coords(self._thickness_roi_rect_id,
                         self._thickness_roi_start[0], self._thickness_roi_start[1],
                         event.x, event.y)
            else:
                self._thickness_roi_rect_id = c.create_rectangle(
                    self._thickness_roi_start[0], self._thickness_roi_start[1],
                    event.x, event.y, outline="#ff00ff", width=2, tags="drawing")
        elif self._roi_drawing:
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
        if self._thickness_roi_drawing:
            self._thickness_roi_drawing = False
            scale = self._frame_scale
            ox, oy = self._frame_off_x, self._frame_off_y
            x1 = int((self._thickness_roi_start[0]-ox)/scale) if scale > 0 else 0
            y1 = int((self._thickness_roi_start[1]-oy)/scale) if scale > 0 else 0
            x2 = int((event.x-ox)/scale) if scale > 0 else 0
            y2 = int((event.y-oy)/scale) if scale > 0 else 0
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # 夹取到画面内：用户拖到黑边外时框不应越界，也避免存储越界坐标。
            fw, fh = self._frame_w, self._frame_h
            if fw > 0:
                x1 = max(0, min(x1, fw - 1))
                x2 = max(0, min(x2, fw - 1))
            if fh > 0:
                y1 = max(0, min(y1, fh - 1))
                y2 = max(0, min(y2, fh - 1))
            self._roi_canvas.delete("drawing")
            self._thickness_roi_rect_id = None
            if x2 - x1 < 10 or y2 - y1 < 10:
                self._thickness_roi = None
                self._set_thickness_roi_status("分析区域: 全画面")
                self.log.write("[薄膜厚度] 框选过小，已取消（恢复全画面）")
            else:
                self._thickness_roi = (x1, y1, x2, y2)
                self._set_thickness_roi_status(
                    f"分析区域: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
                self.log.write(
                    f"[薄膜厚度] 已框选分析区域 x={x1}, y={y1}, "
                    f"width={x2-x1}, height={y2-y1}")
                # 立即绘制品红持久框，避免冻结画面下 _show_frame 未刷新而不显示
                self._roi_canvas.create_rectangle(
                    x1*scale+ox, y1*scale+oy,
                    x2*scale+ox, y2*scale+oy,
                    outline="#ff00ff", width=2, tags="thickness_roi")
                self._preview_adjusted = True
        elif self._roi_drawing:
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
        self._latest_corrected_frame = None
        self.detector.reset_temporal_history()
        self._center_line_x = None
        self._center_confidence = 0.0
        self._prediction_frame_width = None
        self._center_line_box = None
        self._zero_box_x = None
        self._zero_box_confidence = 0.0
        self._center_tracker.reset()
        self._reset_box_stability()
        self._fringe_motion_tracker.reset()
        self._fringe_recognition_tracker.reset()
        self._texture_frame_counter = 0
        self._last_texture_analysis = None
        self._last_fringe_motion = {
            "has_fringe": False, "movement": "unknown",
            "movement_text": "尚未检测", "delta_x_px": None, "source": ""}
        if self._guidance_future is not None:
            self._guidance_future.cancel()
        self._guidance_future = None
        self._guidance_future_generation = -1
        self._guidance_last_submit_at = 0.0
        self._guidance_geometry_completed_at = 0.0
        self._last_guidance_geometry = {}
        self._last_fringe_guidance = {}
        if self.manual_auto_center_panel is not None:
            self.manual_auto_center_panel.update_guidance({})
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
            if not self.predict_running:
                self._show_frame(corrected)
            if self.recorder and self.recorder.recording:
                src = frame if self.recorder.recording_source == "camera" else corrected
                self._write_rec_frame(src)
        self._preview_job = self.root.after(self.PREVIEW_INTERVAL_MS, self._preview_loop)

    def _show_frame(self, frame_bgr):
        canvas = self._roi_canvas
        if canvas is None:
            return
        h, w = frame_bgr.shape[:2]
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        # 画布尚未完成布局（宽/高仍为 1px）时，nw/nh 可能为 0；此时直接返回，
        # 避免把 _frame_scale 写成近乎 0 的非法值，导致后续 ROI 换算发散。
        if nw < 1 or nh < 1:
            return
        frame_bgr = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        # 用 anchor="nw" + 整数左上角坐标绘制，offset 与图片实际渲染的左上角
        # 严格一致（无中心锚点的 0.5px 舍入歧义）。画布尺寸变化（窗口缩放）后
        # Tk 不会自动重排 image 项，故每次都要重设左上角坐标，否则 ROI 框与
        # 裁剪会整体错位，导致分析的画面与用户画的框不一致。
        img_cx, img_cy = cw // 2, ch // 2
        self._frame_off_x = img_cx - nw // 2
        self._frame_off_y = img_cy - nh // 2
        self._frame_scale = scale
        self._frame_w = w
        self._frame_h = h
        # 更新已有图片（不 delete all 以提高性能）；每次都要重设左上角坐标。
        if getattr(self, '_img_id', None):
            canvas.itemconfigure(self._img_id, image=self._frame_img)
            canvas.coords(self._img_id, self._frame_off_x, self._frame_off_y)
        else:
            self._img_id = canvas.create_image(
                self._frame_off_x, self._frame_off_y,
                image=self._frame_img, anchor="nw")
        # 重画 ROI + 中心线（不删除 "drawing"，它是拖拽时的实时预览框）
        canvas.delete("roi", "thickness_roi", "center_line", "center_target",
                      "fringe_bands", "fringe_count", "fringe_spacing")
        if self.model_plugin and self.model_plugin.roi_pixels:
            x1, y1, x2, y2 = self.model_plugin.roi_pixels
            canvas.create_rectangle(
                x1*scale+self._frame_off_x, y1*scale+self._frame_off_y,
                x2*scale+self._frame_off_x, y2*scale+self._frame_off_y,
                outline="#00ff00", width=2, tags="roi")
        if self._thickness_roi is not None:
            x1, y1, x2, y2 = self._thickness_roi
            canvas.create_rectangle(
                x1*scale+self._frame_off_x, y1*scale+self._frame_off_y,
                x2*scale+self._frame_off_x, y2*scale+self._frame_off_y,
                outline="#ff00ff", width=2, tags="thickness_roi")

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

        # 单次识别的所有条纹轮廓/宽度标注（快照式，随下一次“分析当前画面”刷新）
        if self._fringe_band_overlay:
            y1o = self._frame_off_y + 4
            y2o = self._frame_off_y + h * scale - 4
            for b in self._fringe_band_overlay:
                color = "#ffd24a" if b["kind"] == "bright" else "#9ad0ff"
                centerline = b.get("centerline")
                if centerline and len(centerline) >= 2:
                    # 2D 轮廓：中心线已拟合成平滑曲线（密集采样点），按条纹
                    # 实际走向画曲线，倾斜/弯曲都能如实反映。
                    flat: list[float] = []
                    for x, y in centerline:
                        flat.extend((x * scale + self._frame_off_x,
                                     y * scale + self._frame_off_y))
                    canvas.create_line(*flat, fill=color, width=2,
                                       tags="fringe_bands")
                else:
                    # 无轮廓时退回画左右边界竖线
                    xl = b["left"] * scale + self._frame_off_x
                    xr = b["right"] * scale + self._frame_off_x
                    for xb in (xl, xr):
                        canvas.create_line(
                            xb, y1o, xb, y2o, fill=color, width=1,
                            dash=(3, 3), tags="fringe_bands")
                xl = b["left"] * scale + self._frame_off_x
                xr = b["right"] * scale + self._frame_off_x
                canvas.create_text(
                    (xl + xr) / 2, y1o + 9, text=f"{b['width']:.1f}",
                    fill=color, anchor="n",
                    font=("Consolas", 8, "bold"), tags="fringe_bands")

        # 沿法向间距标注：相邻条纹中心连线，绿=采用、橙=剔除（§七）。
        if self._fringe_spacing_overlay:
            centers = self._fringe_spacing_overlay.get("fringe_centers", [])
            valid = self._fringe_spacing_overlay.get("interval_valid", [])
            for i in range(len(centers) - 1):
                a = centers[i]
                b = centers[i + 1]
                ok = bool(valid[i]) if i < len(valid) else True
                color = "#2ea043" if ok else "#e07b00"
                canvas.create_line(
                    a["x"] * scale + self._frame_off_x,
                    a["y"] * scale + self._frame_off_y,
                    b["x"] * scale + self._frame_off_x,
                    b["y"] * scale + self._frame_off_y,
                    fill=color, width=2, tags="fringe_spacing")

        # 实时条纹宽度分析标注：视场区间 + 亮纹峰 + 间隔（随实时刷新）
        if self._fringe_count_overlay:
            ov = self._fringe_count_overlay
            region = ov.get("region")
            peaks = ov.get("peak_positions", [])
            if region and len(region) == 2:
                rx0 = region[0] * scale + self._frame_off_x
                rx1 = region[1] * scale + self._frame_off_x
                top = self._frame_off_y
                bot = self._frame_off_y + h * scale
                canvas.create_rectangle(
                    rx0, top, rx1, bot, outline="#00c8c8", width=2,
                    tags="fringe_count")
                ay = top + 24
                canvas.create_line(rx0, ay, rx1, ay, fill="#00c8c8", width=2,
                                   tags="fringe_count")
                canvas.create_text(
                    (rx0 + rx1) / 2, ay - 12,
                    text=f"{ov['span_px']:.0f}px ÷ {ov['fringe_count']}",
                    fill="#00c8c8", anchor="n",
                    font=("Consolas", 8, "bold"), tags="fringe_count")
            for i, px in enumerate(peaks, 1):
                gx = px * scale + self._frame_off_x
                canvas.create_line(
                    gx, self._frame_off_y + 40, gx,
                    self._frame_off_y + h * scale - 8,
                    fill="#00e600", width=2, tags="fringe_count")
                canvas.create_text(
                    gx + 4, self._frame_off_y + h * scale - 20, text=str(i),
                    fill="#00e600", anchor="nw",
                    font=("Consolas", 8, "bold"), tags="fringe_count")

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
        self._latest_corrected_frame = corrected
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

        # 条纹场景识别独立于自动寻中控制：YOLO 是主证据，二维局部纹理
        # 负责补偿倾斜、弯曲、变色和偶发漏检，历史轨迹负责短时运动模糊。
        # 自动寻中状态机本身不变；融合层只通过既有的场景识别接口提供
        # 连续的位置和速度信息，零级中心/非中心引导框的控制接口保持原样。
        if self._center_line_x is not None:
            yolo_fringe_x = self._center_line_x
        else:
            yolo_fringe_x = guide.get("x")
        yolo_has_fringe = bool(
            self._center_line_x is not None
            or guide.get("count", 0)
            or max(class_conf.values(), default=0.0) > 0.0
        )
        self._texture_frame_counter += 1
        refresh_texture = (
            self._last_texture_analysis is None
            or not yolo_has_fringe
            or self._texture_frame_counter % self._texture_interval_frames == 0
        )
        if refresh_texture:
            self._last_texture_analysis = analyse_fringe_texture(corrected)
        texture = self._last_texture_analysis
        recognition = self._fringe_recognition_tracker.update(
            yolo_has_fringe=yolo_has_fringe,
            yolo_position_x=yolo_fringe_x,
            yolo_confidence=max(class_conf.values(), default=0.0),
            texture=texture,
        )
        self._last_fringe_motion = self._fringe_motion_tracker.update(
            has_fringe=recognition["has_fringe"],
            position_x=recognition["position_x"],
            # YOLO、视觉回退和历史预测属于同一条融合轨迹，切换证据来源
            # 时不能清空移动窗口。
            source="fused",
        )
        self._last_fringe_motion.update({
            "recognition_confidence": recognition["confidence"],
            "recognition_source": recognition["source"],
            "position_x": recognition["position_x"],
            "velocity_px_s": recognition["velocity_px_s"],
            "blurred": recognition["blurred"],
            "texture_confidence": recognition["texture_confidence"],
            "held": recognition["held"],
        })
        self._update_realtime_guidance(corrected, roi, recognition)
        if self.manual_auto_center_panel is not None:
            self.manual_auto_center_panel.update_scene_analysis(
                self._last_fringe_motion)
            self.manual_auto_center_panel.update_clarity(
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
        source_text = str(
            self._last_fringe_motion.get("recognition_source") or "NONE").upper()
        cv2.putText(
            annotated,
            f"fringe={present_text} motion={movement_text} source={source_text}",
            (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2,
        )

        self._show_frame(annotated)
        if self.recorder and self.recorder.recording:
            src = frame if self.recorder.recording_source == "camera" else annotated
            self._write_rec_frame(src)

        self.status.update_fps(self.fps)
        strategy = result.get("strategy") or {}
        self.model_plugin.update_results(
            class_conf, len(result["boxes_xyxy"]), recommended, strategy)
        self._last_detection_result = result  # 保存供中心条纹分析使用
        log_signature = (
            tuple(sorted((name, round(float(conf), 2))
                         for name, conf in class_conf.items())),
            round(self._center_line_x, 1) if self._center_line_x is not None else None,
            self._last_fringe_motion.get("movement", "unknown"),
            tuple(
                int(strategy.get(key, 0))
                for key in (
                    "raw_count", "removed_low_confidence", "removed_geometry",
                    "removed_overlap", "removed_count_limit",
                )
            ) if strategy else (),
        )
        if (log_signature != self._last_yolo_log_signature
                or now - self._last_yolo_log_at >= 5.0):
            detection_text = ", ".join(
                f"{name}={conf:.2f}" for name, conf in sorted(class_conf.items())
            ) or "无目标"
            center_text = (
                f"x={self._center_line_x:.1f}px, confidence={self._center_confidence:.2f}"
                if self._center_line_x is not None else "未定位")
            recognition_text = (
                f"{self._last_fringe_motion.get('recognition_source') or 'none'}"
                f"/{float(self._last_fringe_motion.get('recognition_confidence') or 0):.2f}"
                f"/{float(self._last_fringe_motion.get('velocity_px_s') or 0):+.1f}px/s")
            filter_text = (
                f"{strategy.get('raw_count', len(result['boxes_xyxy']))}"
                f"→{len(result['boxes_xyxy'])}"
                if strategy else "未启用"
            )
            self.log.write(
                f"[YOLO实时] targets={len(result['boxes_xyxy'])} [{detection_text}]；"
                f"统一筛选={filter_text}；"
                f"中心={center_text}；条纹移动={self._last_fringe_motion.get('movement_text', '--')}；"
                f"融合识别={recognition_text}；"
                f"FPS={self.fps:.1f}；ROI={roi or '全画面'}")
            self._last_yolo_log_signature = log_signature
            self._last_yolo_log_at = now

    def _update_realtime_guidance(
        self,
        corrected: np.ndarray,
        roi: tuple[int, int, int, int] | None,
        recognition: dict,
    ) -> None:
        """低频更新几何分析，高频生成只读诊断与操作建议。"""
        now = time.monotonic()
        if self._guidance_future is not None and self._guidance_future.done():
            generation = self._guidance_future_generation
            try:
                geometry = self._guidance_future.result()
            except Exception as exc:
                logger.debug("实时条纹几何诊断失败: %s", exc)
                geometry = None
            self._guidance_future = None
            self._guidance_future_generation = -1
            if geometry is not None and generation == self._prediction_generation:
                self._last_guidance_geometry = geometry
                self._guidance_geometry_completed_at = now

        has_fringe = bool(recognition.get("has_fringe", False))
        if not has_fringe:
            self._last_guidance_geometry = {}
            self._guidance_geometry_completed_at = 0.0
        elif (self._guidance_future is None
              and now - self._guidance_last_submit_at
              >= self.GUIDANCE_GEOMETRY_INTERVAL_S):
            # 后台线程只读副本；不会修改摄像头缓冲区或执行设备动作。
            self._guidance_future_generation = self._prediction_generation
            self._guidance_future = self._guidance_executor.submit(
                analyse_guidance_geometry, corrected.copy(), roi)
            self._guidance_last_submit_at = now

        geometry = self._last_guidance_geometry
        if (self._guidance_geometry_completed_at > 0
                and now - self._guidance_geometry_completed_at > 4.0):
            geometry = {}
        clarity = self.cam.clarity_status() if self.cam is not None else {}
        guidance = build_fringe_guidance(
            recognition=recognition,
            motion=self._last_fringe_motion,
            texture=self._last_texture_analysis,
            geometry=geometry,
            clarity=clarity,
            center_x=self._center_line_x,
            frame_width=self._prediction_frame_width,
            motor_connected=self.motor_connected,
            auto_enabled=self.auto_control_enabled,
            current_correction_deg=(
                self.camera_plugin.angle if self.camera_plugin else 0.0),
            motion_enhancement_enabled=bool(
                self.camera_plugin
                and self.camera_plugin.motion_enhance_enabled),
        )
        guidance["execution_stage"] = (
            self.manual_auto_center_panel.execution_stage
            if self.manual_auto_center_panel is not None else "advisory")
        guidance["quality_gate_passed"] = bool(
            guidance.get("measurement_ready", False))
        guidance["adaptive_response"] = self.adaptive_response.snapshot()
        self._last_fringe_guidance = guidance
        if self.manual_auto_center_panel is not None:
            self.manual_auto_center_panel.update_guidance(guidance)

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
        self._motor_reconnecting = False
        self._on_auto_stop()
        self.motor_connected = False
        self.status.update_motor_connected(False)
        self._set_status("电机已断开")
        controller = self.motor
        self.motor = None
        if controller:
            # 使用带停车保护的关闭，即使 _connected 已被异常置 False 也会尽力发送停车命令
            self.motor_commands.submit(
                "disconnect", controller.try_stop_on_close, priority=0, coalesce=True)

    def _handle_motor_dropped(self) -> None:
        """电机串口异常断开：停止自动控制并进入后台自动重连流程。

        与手动断开（``_on_motor_disconnect``）不同，这里保留 ``self.motor``
        句柄，以便在原端口重新出现时重开串口恢复控制。
        """
        if self._motor_reconnecting:
            return
        if self.auto_control_enabled:
            self._on_auto_stop("串口失联")
        self.motor_connected = False
        self.status.update_motor_connected(False)
        self._set_status("电机串口已断开，正在尝试重连…")
        self.motor_panel.update_command_status("电机串口已断开，尝试重连…")
        self._motor_reconnect_attempts = 0
        self._motor_reconnecting = True
        self._schedule_motor_reconnect()

    def _schedule_motor_reconnect(self) -> None:
        """后台提交一次串口重连操作；失败则退避后重试，超限则放弃。"""
        if self._closing or self.motor is None or self.motor_connected:
            self._motor_reconnecting = False
            return
        if self._motor_reconnect_attempts >= self.MOTOR_RECONNECT_MAX:
            self._motor_reconnecting = False
            self.motor_panel.update_command_status("电机串口重连失败，请手动重新连接")
            self.log.write("[电机] 串口重连失败，请检查接线后手动重新连接")
            return
        self._motor_reconnect_attempts += 1
        controller = self.motor
        self.motor_commands.submit(
            "reconnect", lambda c=controller: (c.reconnect(), c.port), coalesce=True)

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

    def _set_auto_center_status(self, text: str) -> None:
        if self.manual_auto_center_panel is not None:
            self.manual_auto_center_panel.status_var.set(text)

    def _update_auto_center_panel(self, decision) -> None:
        if self.manual_auto_center_panel is not None:
            self.manual_auto_center_panel.update_control(
                decision, self._center_line_x, self._prediction_frame_width)

    def _on_auto_center_command(self, command: str):
        if command == "start":
            self._on_auto_start()
        elif command == "stop":
            self._on_auto_stop("用户停止自动寻中")
        elif command == "toggle_center_line":
            # 下一次预览刷新立即生效；该叠加层不会进入相机帧或模型输入。
            return
        elif command == "apply_guidance":
            self._apply_guidance_action()

    def _apply_guidance_action(self) -> None:
        """只执行视觉指导器生成的固定白名单动作。"""
        panel = self.manual_auto_center_panel
        action = panel.primary_action if panel is not None else None
        if panel is None or action is None:
            return
        if panel.execution_stage == "advisory":
            self.log.write("[AI指导] 当前为只读模式，未执行设备动作")
            return
        code = str(action.get("code") or "")
        allowed = {
            "apply_angle_correction",
            "enable_motion_enhancement",
            "start_auto_search",
            "start_auto_center",
            "stop_auto_center",
        }
        if code not in allowed:
            self.log.write(f"[AI指导] 拒绝未知动作: {code or '--'}")
            return
        description = str(action.get("description") or action.get("label") or code)
        if not messagebox.askyesno(
                "确认执行 AI 建议",
                f"建议：{action.get('label', code)}\n\n{description}\n\n确认执行吗？"):
            self.log.write(f"[AI指导] 用户取消: {action.get('label', code)}")
            return

        if code == "apply_angle_correction":
            if self.camera_plugin is None:
                return
            try:
                delta = float((action.get("params") or {}).get("delta_deg"))
            except (TypeError, ValueError):
                self.log.write("[AI指导] 角度校正参数无效，已拒绝")
                return
            if not np.isfinite(delta) or abs(delta) > 30.0:
                self.log.write("[AI指导] 角度校正超出单次 ±30° 安全范围，已拒绝")
                return
            target = max(-180.0, min(180.0, self.camera_plugin.angle + delta))
            self.camera_plugin.angle_var.set(f"{target:.3f}")
            self.corrector.set_manual_offset(target)
            self._preview_adjusted = True
            self.log.write(f"[AI指导] 已应用画面角度校正: {delta:+.2f}°，当前 {target:+.2f}°")
        elif code == "enable_motion_enhancement":
            if self.camera_plugin is None:
                return
            self.camera_plugin.motion_enhance_var.set(True)
            self._apply_camera_clarity("AI 建议确认执行")
        elif code in {"start_auto_search", "start_auto_center"}:
            self._on_auto_start()
        elif code == "stop_auto_center":
            self._on_auto_stop("AI 建议经用户确认停车")

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
        self._update_auto_center_panel(decision)
        params = self.manual_auto_center_panel.get_params()
        if params.get("direction_mode") == "single_direction":
            direction_text = (
                "反转" if params.get("search_direction") == "reverse" else "正转")
            if params.get("invert_direction"):
                direction_text = "正转" if direction_text == "反转" else "反转"
            direction_mode_text = f"已知方向单向寻找（{direction_text}）"
        else:
            direction_mode_text = "未知方向双向扩展寻找"
        recognition_mode_text = (
            "转停识别" if params.get("recognition_mode") == "stop_and_detect"
            else "边转边识别"
        )
        self.log.write(
            f"[AUTO] 自动寻中已启动：{direction_mode_text} + {recognition_mode_text}；"
            "找到中心后切换标准闭环居中")

    def _on_auto_stop(self, reason: str = "用户停止"):
        decision = self.auto_controller.stop(reason)
        self.auto_control_enabled = self.auto_controller.enabled
        self._apply_camera_clarity("自动寻中停止")
        self._dispatch_motor_commands(decision.commands)
        self._update_auto_center_panel(decision)
        if decision.stopped_reason:
            self.log.write(f"[AUTO] 已停止: {reason}")

    def _auto_motor_control(self, guide: dict | None = None):
        if not self.auto_control_enabled or self.motor is None:
            return
        safety = self.recording_preset["motor"]["safety"]
        panel = self.manual_auto_center_panel
        params = panel.get_params()
        now = time.monotonic()
        movement = str(self._last_fringe_motion.get("movement", "unknown"))
        blurred = bool(self._last_fringe_motion.get("blurred", False))
        held = bool(self._last_fringe_motion.get("held", False))
        self.adaptive_response.observe(
            now=now,
            direction=self.auto_controller.direction,
            gear=self.auto_controller.gear,
            velocity_px_s=self._last_fringe_motion.get("velocity_px_s"),
            stable=movement == "stable",
            blurred=blurred,
            held=held,
            profile_key=(
                f"width={self._prediction_frame_width or 0};"
                f"zoom={float(self.corrector.zoom):.2f}"),
        )
        self._last_adaptive_changes = {}
        if panel.execution_stage == "adaptive":
            spacing_px = (
                (self._last_fringe_guidance.get("metrics") or {}).get("spacing_px"))
            params, self._last_adaptive_changes = (
                self.adaptive_response.optimized_params(
                    params, spacing_px=spacing_px))
        panel.update_adaptive(
            self.adaptive_response.snapshot(), self._last_adaptive_changes)
        guide = guide or self._last_non_center_guide
        decision = self.auto_controller.update(
            center_x=self._center_line_x,
            frame_width=self._prediction_frame_width,
            confidence=self._center_confidence,
            guide_x=guide.get("x"),
            guide_confidence=float(guide.get("confidence", 0.0)),
            guide_count=int(guide.get("count", 0)),
            fringe_movement=movement,
            fringe_delta_x_px=self._last_fringe_motion.get("delta_x_px"),
            fringe_velocity_px_s=self._last_fringe_motion.get("velocity_px_s"),
            scene_has_fringe=bool(self._last_fringe_motion.get("has_fringe")),
            scene_position_x=self._last_fringe_motion.get("position_x"),
            scene_confidence=float(self._last_fringe_motion.get(
                "recognition_confidence", 0.0)),
            scene_source=str(self._last_fringe_motion.get(
                "recognition_source", "")),
            scene_blurred=blurred,
            scene_held=held,
            zero_box_x=self._zero_box_x,
            zero_box_confidence=self._zero_box_confidence,
            zero_box_half_width=(
                (self._center_line_box[2] - self._center_line_box[0]) / 2.0
                if (self._center_line_box is not None and self._zero_box_stable)
                else 0.0
            ),
            connected=self.motor_connected and self.motor.is_connected,
            params=params,
            safety=safety,
            now=now,
        )
        self.auto_control_enabled = self.auto_controller.enabled
        if not self.auto_control_enabled:
            self._apply_camera_clarity("自动寻中结束")
        self._dispatch_motor_commands(decision.commands)
        self._update_auto_center_panel(decision)
        if (decision.state == "centered"
                and not self._last_fringe_guidance.get("measurement_ready", False)):
            self._set_auto_center_status(
                f"{decision.message}；中心已到位，但测量质量门未通过")
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
            if selected:
                self.motor_panel.port_var.set(selected)
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
                self.status.update_motor_connected(True, port)
                self._set_status(f"电机已连接: {port}")
            else:
                self._set_status(f"电机连接失败: {port}")
                self.motor_panel.update_command_status(f"电机连接失败：{port}")
        elif result.name == "reconnect":
            ok, port = result.value
            if ok:
                self.motor_connected = True
                self.motor_panel.port_var.set(port)
                self.status.update_motor_connected(True, port)
                self._set_status(f"电机已重连: {port}")
                self.motor_panel.update_command_status("电机串口已重连")
                self.log.write(f"[电机] 串口已重连: {port}")
                # 重连后先停车，防止掉线期间电机仍在运行
                controller = self.motor
                if controller is not None:
                    self.motor_commands.submit(
                        "post_reconnect_stop", controller.stop, priority=0, coalesce=True)
                self._motor_reconnecting = False
                self._motor_reconnect_attempts = 0
            else:
                # 退避后重试，由仍在运行的 _poll_motor 驱动结果回执
                self._motor_reconnecting = False
                self.root.after(
                    self.MOTOR_RECONNECT_DELAY_MS, self._schedule_motor_reconnect)
        elif result.name in ("poll_status", "manual_status"):
            status = result.value
            if not isinstance(status, dict) or not status.get("valid", False):
                if self.motor is not None and not self.motor.is_connected:
                    self._handle_motor_dropped()
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

        # ---- 中心条纹宽度测量 ----
        elif cmd == "fringe_width_analyze":
            self._analyze_center_fringe_width()
        elif cmd == "fringe_realtime_toggle":
            self._toggle_fringe_realtime()
        elif cmd == "fringe_auto_angle":
            msg = self._auto_rotate_fringes()
            if msg is None:
                panel.set_fringe_width_status("自动旋转条纹：无画面或未识别到条纹")
            else:
                panel.set_fringe_width_status(msg)

        # ---- 实时测量与记录 ----
        elif cmd == "live_toggle":
            self._toggle_live_measurement()
        elif cmd == "live_record":
            self._record_live_measurement()
        elif cmd == "live_clear":
            panel.clear_records()
            self.log.write("[实时测量] 已清空记录")

        # ---- 薄膜厚度分布（单帧）----
        elif cmd == "thickness_analyze":
            self._analyze_thickness_distribution()
        elif cmd == "thickness_browse":
            self._browse_thickness_calibration()
        elif cmd == "thickness_capture_baseline":
            self._capture_thickness_baseline()
        elif cmd == "thickness_clear_baseline":
            self._clear_thickness_baseline()
        elif cmd == "thickness_roi_mode":
            self._toggle_thickness_roi_mode()
        elif cmd == "thickness_roi_clear":
            self._clear_thickness_roi()
        elif cmd == "thickness_set_initial":
            current = self._fresh_micrometer_reading()
            if current is not None:
                panel.set_thickness_initial(current)
                panel.set_thickness_status(f"初始读数已记录: {current:.6f} mm")
            else:
                panel.set_thickness_status("错误：微分表无稳定读数，无法记录初始读数")
        elif cmd == "thickness_set_center":
            current = self._fresh_micrometer_reading()
            if current is not None:
                panel.set_thickness_center(current)
                panel.set_thickness_status(f"中心条纹读数已记录: {current:.6f} mm")
            else:
                panel.set_thickness_status("错误：微分表无稳定读数，无法记录中心条纹读数")

        # ---- 颜色→光程差标定表采集 ----
        elif cmd == "calibration_capture":
            self._calibration_capture()
        elif cmd == "calibration_save":
            self._calibration_save()
        elif cmd == "calibration_clear":
            self._calibration_clear()

    def _current_analysis_frame(self) -> np.ndarray | None:
        """返回用于条纹分析/实时测量的当前矫正后画面（可能为 None）。"""
        frame = self._latest_corrected_frame
        if frame is None and self.camera_running and self.cam is not None:
            # 预测未运行时退回到当前相机画面，并套用同一套矫正，保证与
            # 中心条纹位置（矫正后坐标）一致。
            raw = self.cam.read()
            if raw is not None:
                try:
                    frame = rotate_expand(
                        raw, self.corrector.effective_angle)
                    frame = self.corrector.apply_zoom_pan(frame)
                except Exception:
                    frame = None
        return frame

    def _measure_center_fringe_width_result(self, use_2d: bool = False) -> dict | None:
        """分析当前画面并返回中心条纹宽度结果；无画面或失败时返回 None。

        ``use_2d=True`` 用 2D 轮廓版算法，可处理倾斜 / 弯曲条纹并给出每段
        条纹的 2D 轮廓（供画面上绘制轮廓而非直线）。
        """
        frame = self._current_analysis_frame()
        if frame is None:
            return None
        try:
            if use_2d:
                return measure_center_fringe_width_2d(
                    frame, center_x=self._center_line_x)
            return measure_center_fringe_width(
                frame, center_x=self._center_line_x)
        except Exception:
            return None

    def _fringe_count_region(self) -> tuple[float, float] | None:
        """返回「视场宽度/条纹数量」计数用的横向视场区间 (x0, x1)。

        优先取用户框选的 ROI 横向范围，即「效果较好的视场」；无 ROI 时返回
        None，由算法自动选择覆盖所有识别条纹的区间。
        """
        if not self.model_plugin:
            return None
        roi = self.model_plugin.get_roi_xywh()
        if not roi:
            return None
        x, _y, w, _h = roi
        if w is None or w <= 0:
            return None
        return (float(x), float(x + w))

    def _measure_fringe_width_by_count_result(self) -> dict | None:
        """分析当前画面并用「视场宽度 / 条纹数量」估算条纹间隔；失败返回 None。"""
        frame = self._current_analysis_frame()
        if frame is None:
            return None
        try:
            return measure_fringe_width_by_count(
                frame, x_range=self._fringe_count_region(),
                mm_per_px=self._fringe_mm_per_px())
        except Exception:
            return None

    def _measure_fringe_spacing_robust_result(self) -> dict | None:
        """分析当前画面并用「相邻条纹中心距离中位数 + MAD 剔除」估算间距。"""
        frame = self._current_analysis_frame()
        if frame is None:
            return None
        try:
            return measure_fringe_spacing_robust(
                frame, x_range=self._fringe_count_region(),
                mm_per_px=self._fringe_mm_per_px())
        except Exception:
            return None

    def _measure_fringe_spacing_2d_result(self) -> dict | None:
        """分析当前画面并用「沿法向的相邻条纹中心间距」主算法估算间距。"""
        frame = self._current_analysis_frame()
        if frame is None:
            return None
        try:
            return measure_fringe_spacing_2d(
                frame, pixel_scale_mm=self._fringe_mm_per_px())
        except Exception:
            return None

    def _fringe_mm_per_px(self) -> float | None:
        """读取面板上的 mm/px 标定系数，留空则返回 None（按像素）。"""
        panel = self.temporary_measurement_panel
        if panel is None:
            return None
        return panel.fringe_mm_per_px

    def _estimate_fringe_correction(self) -> tuple[float, dict] | None:
        """在原始相机画面上估计条纹校正角，返回 ``(correction_deg, est)`` 或 None。

        校正角可直接作为 ``rotate_expand`` 的旋转角把条纹转到竖直方向。无画面、
        无法估计或未识别到条纹时返回 None。
        """
        if not (self.camera_running and self.cam is not None):
            return None
        raw = self.cam.read()
        if raw is None:
            return None
        try:
            est = estimate_fringe_angle_2d(raw)
        except Exception:
            return None
        corr = est.get("correction_deg")
        if corr is None or not np.isfinite(corr):
            return None
        return float(corr), est

    def _auto_rotate_fringes(self) -> str | None:
        """自动估计条纹倾角，把校正角写入手动偏置并同步角度输入框。

        返回一句中文状态描述；失败时写日志并返回 None。
        """
        res = self._estimate_fringe_correction()
        if res is None:
            self.log.write("[画面矫正] 自动旋转条纹：无画面或未识别到条纹")
            return None
        corr, est = res
        self.corrector.set_manual_offset(corr)
        if self.camera_plugin is not None:
            self.camera_plugin.angle_var.set(f"{corr:+.2f}")
        # 角度变了，作废缓存的矫正帧，让下一次分析/显示立即用新角度重读画面，
        # 避免测量仍用旧角度的画面。
        self._latest_corrected_frame = None
        self._preview_adjusted = True
        msg = (
            f"自动旋转条纹：倾角 {est['tilt_deg']:+.2f}° → 校正 {corr:+.2f}°"
            f"（置信度 {est['confidence']:.2f}）")
        if est.get("curvature") and est["curvature"] > 0.05:
            msg += "，条纹弯曲明显"
        self.log.write(f"[画面矫正] {msg}")
        return msg

    def _toggle_fringe_realtime(self) -> None:
        """开始/停止实时条纹宽度分析（视场÷条纹数 + 可选画面标注）。"""
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        self._fringe_realtime_active = not self._fringe_realtime_active
        if self._fringe_realtime_active:
            panel.set_fringe_realtime_running(True)
            self.log.write("[条纹宽度] 开始实时分析（视场÷条纹数）")
            self._refresh_fringe_realtime()
        else:
            self._fringe_count_overlay = None
            panel.set_fringe_realtime_running(False)
            panel.set_fringe_realtime_text("")
            self.log.write("[条纹宽度] 停止实时分析")

    def _refresh_fringe_realtime(self) -> None:
        """定时刷新实时条纹宽度分析与画面标注。"""
        self._fringe_realtime_job = None
        if self._closing:
            return
        panel = self.temporary_measurement_panel
        if panel is not None and self._fringe_realtime_active:
            result = self._measure_fringe_width_by_count_result()
            if result is not None and result.get("fringe_width") is not None:
                panel.set_fringe_realtime_text(
                    f"实时间隔 = {result['span_px']:.1f}px ÷ "
                    f"{result['fringe_count']} 条 = "
                    f"{result['fringe_width']:.2f}px")
                self._fringe_count_overlay = (
                    result if panel.annotate_fringe else None)
            else:
                panel.set_fringe_realtime_text("未识别到可计数的条纹")
                self._fringe_count_overlay = None
        else:
            self._fringe_count_overlay = None
        if self._fringe_realtime_active:
            self._fringe_realtime_job = self.root.after(
                self.FRINGE_REALTIME_INTERVAL_MS, self._refresh_fringe_realtime)

    def _analyze_center_fringe_width(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        # 勾选「分析时自动转正条纹」时，先自动估计条纹倾角并把校正角写入旋转
        # 角度，使随后测量在竖直条纹上进行（间距更接近真实周期）。估计失败时
        # 静默继续用当前画面与角度，不影响后续测量。
        if panel.auto_straighten_fringe:
            self._auto_rotate_fringes()
        spacing_result = self._measure_fringe_spacing_2d_result()
        result = self._measure_center_fringe_width_result(use_2d=True)
        count_result = self._measure_fringe_width_by_count_result()
        if result is None and spacing_result is None and count_result is None:
            panel.set_fringe_width_status(
                "错误：当前没有可分析画面，请先打开干涉摄像头")
            panel.fringe_width_detail_var.set("")
            self._fringe_band_overlay = None
            self._fringe_spacing_overlay = None
            return
        # 主显示优先用「沿法向的相邻条纹中心间距」主算法（PDF 报告口径：两端距离
        # ÷ 周期数，用中位数 + MAD 剔除异常）；退化为「视场宽度 ÷ 条纹数」，再
        # 退化为中心条纹宽度。2D 轮廓仅用于画面标注。
        if spacing_result is not None and spacing_result.get("spacing_px") is not None:
            panel.show_fringe_spacing_2d_result(spacing_result)
        elif count_result is not None:
            panel.show_fringe_width_by_count_result(count_result)
        else:
            panel.show_fringe_width_result(result)
        # 勾选「标注所有条纹宽度」时，把每段条纹的边界/宽度画到画面上；
        # 未勾选则清除上次的标注。
        self._fringe_band_overlay = (
            result.get("bands") if (result and panel.show_all_bands) else None)
        # 沿法向间距标注：相邻中心连线，绿=采用、橙=剔除。
        self._fringe_spacing_overlay = spacing_result if (
            spacing_result is not None
            and spacing_result.get("fringe_centers")) else None
        if spacing_result is not None and spacing_result.get("spacing_px") is not None:
            s = spacing_result
            self.log.write(
                f"[条纹宽度] 间距 {s['spacing_px']:.2f}px"
                f"（法向，σ={s['std_spacing_px']:.2f}px，CV={s['cv_percent']:.2f}%，"
                f"有效 {s['num_valid_intervals']}/{s['num_intervals']}，"
                f"剔除 {s['num_rejected']}，倾角 {s['angle_deg']:.2f}°，"
                f"置信度 {s['confidence']:.2f}）")
            return
        if count_result is not None:
            self.log.write(
                f"[条纹宽度] 视场 {count_result['span_px']:.1f}px ÷ "
                f"{count_result['fringe_count']} 条 = "
                f"{count_result['fringe_width']:.2f}px"
                f"（周期≈{count_result['period_px']}px）")
            return
        band = result.get("center_band") if result else None
        if band is not None:
            self.log.write(
                f"[条纹宽度] {band['kind']} x={band['center_x']:.1f}px "
                f"宽度={band['width']:.1f}px 周期≈{result['period_px']}px")
        else:
            self.log.write("[条纹宽度] 未识别到条纹")

    # ==================================================================
    # 实时测量 — 记录微分表读数 + 中心条纹宽度（可命名）
    # ==================================================================
    def _refresh_live_measurement(self) -> None:
        """定时刷新实时测量的微分表读数与中心条纹宽度。

        仅在开启（_live_measurement_active）时工作；微分表读数每拍刷新，
        条纹宽度分析按 LIVE_WIDTH_INTERVAL_S 节流，避免频繁全帧分析卡顿。
        """
        self._live_measurement_job = None
        if self._closing:
            return
        panel = self.temporary_measurement_panel
        if panel is not None and self._live_measurement_active:
            reading_mm = self._fresh_micrometer_reading()
            now = time.monotonic()
            if now - self._live_last_width_at >= self.LIVE_WIDTH_INTERVAL_S:
                result = self._measure_fringe_width_by_count_result()
                self._live_measurement["width_px"] = (
                    result.get("fringe_width") if result else None)
                self._live_measurement["kind"] = (
                    result.get("kind") if result else None)
                self._live_last_width_at = now
            self._live_measurement["reading_mm"] = reading_mm
            panel.set_live_measurement(
                reading_mm,
                self._live_measurement.get("width_px"),
                self._live_measurement.get("kind"))
        self._live_measurement_job = self.root.after(
            self.LIVE_MEASUREMENT_INTERVAL_MS, self._refresh_live_measurement)

    def _toggle_live_measurement(self) -> None:
        """开启/停止实时测量持续分析。"""
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        self._live_measurement_active = not self._live_measurement_active
        if self._live_measurement_active:
            self._live_last_width_at = 0.0  # 下一拍立即分析一次
            panel.set_live_running(True)
            self.log.write("[实时测量] 开始持续分析")
        else:
            self._live_measurement = {"reading_mm": None, "width_px": None, "kind": None}
            panel.set_live_running(False)
            self.log.write("[实时测量] 停止持续分析")

    def _record_live_measurement(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        if not self._live_measurement_active:
            panel.set_live_status("请先点击“开始实时测量”再记录数据")
            return
        reading_mm = self._fresh_micrometer_reading()
        width_px = self._live_measurement.get("width_px")
        kind = self._live_measurement.get("kind")
        if width_px is None:
            # 实时缓存为空时现场分析一次，尽量取到同步的宽度值
            result = self._measure_fringe_width_by_count_result()
            if result is not None and result.get("fringe_width") is not None:
                width_px = result["fringe_width"]
                kind = result.get("kind")
        if reading_mm is None and width_px is None:
            panel.set_live_status("错误：微分表与中心条纹宽度均无有效读数")
            return
        record = {
            "name": panel.record_name,
            "reading_mm": reading_mm,
            "width_px": width_px,
            "kind": kind,
        }
        panel.append_record(record)
        reading_text = "--" if reading_mm is None else f"{reading_mm:.6f} mm"
        width_text = "--" if width_px is None else f"{width_px:.1f} px"
        panel.set_live_status(
            f"已记录“{record['name']}”：微分表 {reading_text}，宽度 {width_text}")
        self.log.write(
            f"[实时测量] 记录“{record['name']}” 微分表={reading_text} "
            f"宽度={width_text}")

    # ==================================================================
    # 薄膜厚度分布（单帧）
    # ==================================================================
    def _browse_thickness_calibration(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        try:
            from tkinter import filedialog as fd
            path = fd.askopenfilename(
                title="选择颜色标定 CSV（列：opd_um,r,g,b）",
                filetypes=[("CSV", "*.csv"), ("所有文件", "*.*")])
        except Exception:
            path = ""
        if path:
            panel.set_thickness_calibration(path)

    def _analyze_thickness_distribution(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        if self._thickness_future is not None and not self._thickness_future.done():
            panel.set_thickness_status("上一次分析仍在进行…")
            return
        frame = self._current_analysis_frame()
        if frame is None:
            panel.set_thickness_status("错误：当前没有可分析画面，请先打开干涉摄像头")
            panel.show_thickness_image(None)
            return
        # 只分析框选区域（未框选则整幅）；基准图用同一区域裁剪，保证相减对齐。
        frame = self._crop_to_thickness_roi(frame)
        baseline = self._crop_to_thickness_roi(self._thickness_baseline_frame)
        wavelength = panel.thickness_wavelength_nm
        refractive = panel.thickness_refractive_index
        if refractive is None:
            panel.set_thickness_status("错误：折射率须大于 1")
            return
        # 绝对厚度锚定：中心条纹读数 − 初始读数 → 基准厚度 μm。
        self._thickness_anchor_um = self._thickness_anchor_um_value(refractive)
        params = dict(
            wavelength_nm=wavelength if wavelength is not None else 589.3,
            refractive_index=refractive,
            calibration=panel.thickness_calibration_path or None,
            invert=panel.thickness_invert,
            reference_image=baseline,
            reference_thickness_um=self._thickness_anchor_um,
            # 用户框选后整幅框内都是待分析区域，跳过亮膜自动分割，避免
            # Otsu 在纯亮区把掩膜收缩到一小块导致热力图只覆盖局部。
            whole_region=self._thickness_roi is not None,
        )
        panel.set_thickness_status("分析中…（解包彩色条纹相位）")
        self._thickness_future = self._thickness_executor.submit(
            self._run_thickness_analysis, frame.copy(), params)
        self._poll_thickness_analysis()

    def _run_thickness_analysis(self, frame: np.ndarray, params: dict) -> dict:
        try:
            result = analyze_thickness_distribution(frame, **params)
            return {"ok": True, "result": result}
        except Exception as exc:
            logger.exception("单帧厚度分布分析失败: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _poll_thickness_analysis(self) -> None:
        self._thickness_job = None
        future = self._thickness_future
        if future is None or self._closing:
            return
        if not future.done():
            self._thickness_job = self.root.after(
                120, self._poll_thickness_analysis)
            return
        self._thickness_future = None
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        try:
            outcome = future.result()
        except Exception as exc:
            outcome = {"ok": False, "error": str(exc)}
        if not outcome["ok"]:
            panel.set_thickness_status(f"错误：{outcome['error']}")
            panel.show_thickness_image(None)
            return
        result = outcome["result"]
        metrics = result["metrics"]
        panel.set_thickness_result(metrics)
        panel.show_thickness_image(result["overlay"])
        out_dir = self._save_thickness_result(result, metrics)
        self._show_thickness_viewer(result, self._thickness_anchor_um)
        mode_text = "标定" if result["mode"] == "calibrated" else "相对"
        ref_text = "，已扣基准" if metrics.get("has_reference") else ""
        anchor = self._thickness_anchor_um
        anchor_text = f"，锚定 {anchor:.3f} μm" if anchor is not None else ""
        panel.set_thickness_status(
            f"完成（{mode_text}{ref_text}{anchor_text}）：PV {metrics['pv_robust_um']:.3f} μm，"
            f"RMS {metrics['rms_um']:.3f} μm；已保存到 {out_dir}")
        self.log.write(
            f"[薄膜厚度] {mode_text}{ref_text}{anchor_text} "
            f"PV={metrics['pv_robust_um']:.3f}μm "
            f"RMS={metrics['rms_um']:.3f}μm "
            f"有效像素={metrics['valid_pixels']}")

    def _save_thickness_result(self, result: dict, metrics: dict) -> Path:
        import json

        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.monotonic() * 1000) % 1000:03d}"
        out_dir = PROJECT_ROOT / "outputs" / "thickness" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_image_cv(out_dir / "thickness_overlay.png", result["overlay"])
        self._save_image_cv(out_dir / "thickness_map.png", result["heatmap"])
        self._save_image_cv(
            out_dir / "sample_mask.png", np.uint8(result["mask"]) * 255)
        np.savetxt(out_dir / "thickness_map_um.csv", result["thickness"],
                   delimiter=",", fmt="%.6f")
        (out_dir / "summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_dir

    @staticmethod
    def _save_image_cv(path: Path, bgr: np.ndarray) -> None:
        ok, encoded = cv2.imencode(path.suffix or ".png", bgr)
        if not ok:
            raise ValueError(f"Cannot encode image: {path}")
        encoded.tofile(str(path))

    def _capture_thickness_baseline(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        frame = self._current_analysis_frame()
        if frame is None:
            panel.set_thickness_status("错误：当前没有可分析画面，无法捕获基准")
            return
        self._thickness_baseline_frame = frame.copy()
        panel.set_thickness_baseline(True)
        # 捕获无膜初始画面时一并记录当前读数，作为绝对厚度锚定的初始读数。
        reading = self._fresh_micrometer_reading()
        if reading is not None:
            panel.set_thickness_initial(reading)
            panel.set_thickness_status(
                f"无膜基准图已捕获，初始读数 {reading:.6f} mm 已记录，分析时自动扣除")
            self.log.write(
                f"[薄膜厚度] 已捕获无膜基准图（初始读数 {reading:.6f} mm，分析时扣除系统光程差）")
        else:
            panel.set_thickness_status(
                "无膜基准图已捕获（无稳定读数，请手动填初始读数）")
            self.log.write("[薄膜厚度] 已捕获无膜基准图（分析时扣除系统光程差）")

    def _clear_thickness_baseline(self) -> None:
        self._thickness_baseline_frame = None
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.set_thickness_baseline(False)
            panel.set_thickness_status("无膜基准图已清除")

    def _set_thickness_roi_status(self, text: str) -> None:
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.set_thickness_roi_status(text)

    def _toggle_thickness_roi_mode(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        if panel.thickness_roi_mode:
            # 进入厚度框选模式时关掉条纹 ROI 框选，避免两种拖拽冲突。
            if self.model_plugin is not None:
                self.model_plugin.roi_mode_var.set(False)
            self.log.write("[薄膜厚度] 已进入框选模式：在视频上拖拽选择分析区域")
        else:
            self.log.write("[薄膜厚度] 已退出框选模式")

    def _clear_thickness_roi(self) -> None:
        self._thickness_roi = None
        self._set_thickness_roi_status("分析区域: 全画面")
        canvas = self._roi_canvas
        if canvas is not None:
            canvas.delete("thickness_roi")
        self.log.write("[薄膜厚度] 已清除分析区域（恢复全画面）")

    def _crop_to_thickness_roi(self, frame: np.ndarray) -> np.ndarray:
        """把矫正后画面裁剪到框选的分析区域；未框选时原样返回。"""
        if frame is None or self._thickness_roi is None:
            return frame
        x1, y1, x2, y2 = self._thickness_roi
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    def _thickness_anchor_um_value(self, refractive: float) -> float | None:
        """由「中心条纹读数 − 初始读数」计算绝对厚度锚定基准（μm）。

        换算关系与标定表自动算 OPD 一致：|Δ|(mm) ÷ 20 × 1000 得到光程差 μm，
        再除以 (n-1) 得到薄膜厚度 μm。两读数任一缺失时返回 None（保持相对分布）。
        """
        panel = self.temporary_measurement_panel
        if panel is None:
            return None
        initial = panel.thickness_initial_mm
        center = panel.thickness_center_mm
        if initial is None or center is None:
            return None
        return abs(center - initial) / 20.0 * 1000.0 / (refractive - 1.0)

    def _show_thickness_viewer(self, result: dict, anchor_um: float | None) -> None:
        """弹出单帧厚度分布结果窗口（热力图 + 详细数据 + 可旋转 3D）。"""
        try:
            from src.ui.widgets.thickness_viewer import ThicknessViewer
        except Exception as exc:
            logger.exception("无法打开厚度分布结果窗口: %s", exc)
            return
        viewer = ThicknessViewer(self.root, result, anchor_um=anchor_um)
        self._thickness_viewers.append(viewer)

    def _calibration_capture(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        frame = self._current_analysis_frame()
        if frame is None:
            panel.set_calibration_status("错误：当前没有可分析画面，无法采集颜色")
            return
        opd = self._calibration_opd_um(panel)
        if opd is None:
            panel.set_calibration_status(
                "错误：请填写 OPD 值，或开启微分表并勾选自动计算")
            return
        center_x = frame.shape[1] / 2.0
        try:
            r, g, b = sample_colour_band(frame, center_x)
        except Exception as exc:
            panel.set_calibration_status(f"错误：取色失败 {exc}")
            return
        row = {"opd_um": opd, "r": r, "g": g, "b": b}
        panel.append_calibration(row)
        panel.set_calibration_status(
            f"已采集第 {len(panel.calibration_rows)} 点：OPD={opd:.4f} μm "
            f"r={r} g={g} b={b}")
        self.log.write(
            f"[标定表] 采集 OPD={opd:.4f}μm r={r} g={g} b={b} "
            f"(画面中心线 x={center_x:.1f})")

    def _calibration_opd_um(self, panel) -> float | None:
        if panel.calibration_auto_opd:
            zero = panel.calibration_zero_mm
            current = self._fresh_micrometer_reading()
            if zero is None or current is None:
                return None
            # 微分表读数变化 ÷20 才是 OPD（mm），再 ×1000 转 μm。
            return abs(current - zero) / 20.0 * 1000.0
        return panel.calibration_opd_um

    def _calibration_save(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is None:
            return
        rows = panel.calibration_rows
        if not rows:
            panel.set_calibration_status("错误：尚无标定点，请先采集")
            return
        try:
            from tkinter import filedialog as fd
            path = fd.asksaveasfilename(
                title="保存颜色标定 CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("所有文件", "*.*")])
        except Exception:
            path = ""
        if not path:
            return
        import csv
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["opd_um", "r", "g", "b"])
            writer.writeheader()
            writer.writerows(rows)
        panel.set_calibration_status(f"已保存 {len(rows)} 点到 {path}")
        self.log.write(f"[标定表] 已保存 {len(rows)} 点到 {path}")

    def _calibration_clear(self) -> None:
        panel = self.temporary_measurement_panel
        if panel is not None:
            panel.clear_calibration()
            panel.set_calibration_status("已清空标定点")

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
        if self._laser_ai_guidance_cancel_event is not None:
            self._laser_ai_guidance_cancel_event.set()
        if self._measurement_active:
            self._stop_measurement("程序关闭")
        if self._backlash_active:
            self._stop_backlash("程序关闭")
        if self._agent_context_job is not None:
            self.root.after_cancel(self._agent_context_job)
            self._agent_context_job = None
        if self._agent_suggestion_job is not None:
            self.root.after_cancel(self._agent_suggestion_job)
            self._agent_suggestion_job = None
        if self._live_measurement_job is not None:
            self.root.after_cancel(self._live_measurement_job)
            self._live_measurement_job = None
        if self._thickness_job is not None:
            self.root.after_cancel(self._thickness_job)
            self._thickness_job = None
        if self._thickness_future is not None:
            self._thickness_future.cancel()
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
        try:
            self.adaptive_response.save(self._adaptive_response_path)
        except OSError as exc:
            logger.warning("自适应响应参数保存失败: %s", exc)
        # 先安全停车再关闭摄像头，避免摄像头 USB 复位干扰电机串口
        report = shutdown_motor_safely(
            self.motor_commands, self.motor, timeout=3.0)
        if not report.stop_succeeded:
            # 仅当电机正常连接但停车失败时才报错；此前已断开则静默处理
            if self.motor is not None and self.motor.is_connected:
                logger.error("退出停车未确认成功: %s", report.error or "控制器未确认")
            else:
                logger.info("退出时电机已断开，无需停车")
        if self.cam:
            self.cam.stop()
        self._inference_executor.shutdown(wait=False, cancel_futures=True)
        self._camera_executor.shutdown(wait=False, cancel_futures=True)
        self._micrometer_executor.shutdown(wait=False, cancel_futures=True)
        self._thickness_executor.shutdown(wait=False, cancel_futures=True)
        self._guidance_executor.shutdown(wait=False, cancel_futures=True)
        self.agent_session.shutdown()
        self.root.destroy()

    def run(self):
        self.log.write("启动 UI")
        self.root.mainloop()


def run_app():
    YoloCamApp().run()
