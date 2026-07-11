"""摄像头 YOLO 实时检测 + 电机控制 — Tkinter UI"""
from __future__ import annotations

import time
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
from src.config import config
from src.logging import logger
from src.camera import CameraManager
from src.vision import YOLODetector, rotate_expand, FrameCorrector, find_center_in_region
from src.vision.class_names import get_class_confidences
from src.hardware import MotorController
import yaml
from src.ui.widgets import (
    VideoRecorderPanel,
    StatusPanel,
    MotorControlPanel,
    LogPanel,
    CameraPluginPanel,
    ModelPluginPanel,
    FringeCenterPluginPanel,
)
from src.ui.widgets.collapsible import CollapsibleFrame
from src.ui.widgets.plugin_toggles import PluginToggleBar


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


class YoloCamApp:
    PREDICT_INTERVAL_MS = 90
    PREVIEW_INTERVAL_MS = 30
    MOTOR_POLL_MS = 300

    def __init__(self):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter 不可用")

        self.root = tk.Tk()
        self.root.title("摄像头 YOLO 实时预测 + 电机控制")
        window_size = config.get("ui", "window_size", default=[1600, 1000])
        self.root.geometry(f"{int(window_size[0])}x{int(window_size[1])}")
        self.root.configure(bg="#ffffff")

        # ---- 核心模块 ----
        # 自动选取 models/current/ 下第一个 .pt 文件
        current_dir = PROJECT_ROOT / "models" / "current"
        pt_files = sorted(current_dir.glob("*.pt"))
        if pt_files:
            model_path = pt_files[0]
        else:
            model_path = config.resolve_path(
                config.get("vision", "model_path", default="models/yolov8_interference.pt"))
        self.cam: CameraManager | None = None
        self.detector = YOLODetector(
            str(model_path),
            confidence=float(config.get("vision", "confidence_threshold", default=0.5)),
            iou=float(config.get("vision", "iou_threshold", default=0.45)),
            imgsz=int(config.get("vision", "imgsz", default=640)),
            device=config.get("vision", "device", default="cuda"),
        )
        self.corrector = FrameCorrector()
        self.motor: MotorController | None = None

        # ---- 状态 ----
        self.camera_running = False
        self.predict_running = False
        self.auto_control_enabled = False
        self.auto_speed_stage = "idle"
        self.auto_cycle_phase = "idle"
        self.auto_cycle_ts = 0.0
        self.auto_best_black_conf = 0.0
        self.auto_started_at = 0.0
        self.auto_black_frames = 0
        self.auto_missing_frames = 0
        self._cycle_phase_ms = 1000
        self.motor_connected = False
        self.fps = 0.0
        self.last_t = time.time()

        # ---- 定时器 ----
        self._preview_job: str | None = None
        self._predict_job: str | None = None
        self._motor_poll_job: str | None = None
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self._motor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motor")
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-scan")
        self._inference_future: Future | None = None
        self._inference_context: tuple | None = None
        self._motor_poll_future: Future | None = None
        self._camera_scan_future: Future | None = None
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
        self.log: LogPanel | None = None
        # 可折叠外壳
        self._shells: dict[str, CollapsibleFrame] = {}

        # ---- 中心条纹检测状态 ----
        self._center_line_x: float | None = None  # 全帧坐标下的中心 x
        self._center_line_box: tuple | None = None  # 所属预测框 (x1,y1,x2,y2)
        self._last_detection_result: dict | None = None  # 最近一次 YOLO 检测结果

        # ---- 构建 ----
        self._build_ui()
        self._wire_callbacks()
        self._reload_calibration()
        self.log.write("UI 初始化完成")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self):
        B = "#ffffff"
        outer = tk.Frame(self.root, bg=B)
        outer.pack(fill=tk.BOTH, expand=True)

        # 左侧容器
        left_shell = tk.Frame(outer, bg=B, width=500)
        left_shell.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        left_shell.pack_propagate(False)

        # -- 置顶区域（不随滚动的固定头部） --
        header = tk.Frame(left_shell, bg=B)
        header.pack(fill=tk.X)

        tk.Label(header, text="摄像头YOLO实时预测+电机控制",
                 font=("Microsoft YaHei UI", 16, "bold"), bg=B, fg="#000").pack(anchor="w")
        self.status_var = tk.StringVar(value="状态: 就绪")
        tk.Label(header, textvariable=self.status_var, bg=B, fg="#666",
                 anchor="w").pack(fill=tk.X, pady=(2, 4))

        self.plugin_bar = PluginToggleBar(header)
        self.plugin_bar.pack(fill=tk.X, pady=(0, 4))

        # -- 可滚动插件面板区域 --
        lc = tk.Canvas(left_shell, bg=B, highlightthickness=0, bd=0)
        ls = tk.Scrollbar(left_shell, orient=tk.VERTICAL, command=lc.yview)
        lc.configure(yscrollcommand=ls.set)
        lc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ls.pack(side=tk.RIGHT, fill=tk.Y)

        left = tk.Frame(lc, bg=B)
        lw = lc.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>", lambda e: lc.itemconfigure(lw, width=e.width))
        # 鼠标滚轮：日志区滚日志，其他区滚 canvas
        def _global_scroll(event):
            w = event.widget
            while w is not None:
                if w == self.log._text:  # 鼠标在日志文本框上
                    return  # 让日志自己处理
                w = w.master
            lc.yview_scroll(int(-event.delta/120), "units")
        self.root.bind_all("<MouseWheel>", _global_scroll)

        # 右侧视频（Canvas 用于 ROI 绘制）
        right = tk.Frame(outer, bg=B)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=12, pady=12)
        self._roi_canvas = tk.Canvas(right, bg="#000", highlightthickness=0, bd=0)
        self._roi_canvas.pack(fill=tk.BOTH, expand=True)
        self._roi_canvas.create_text(400, 400, text="摄像头未打开", fill="#fff",
                                      font=("Microsoft YaHei UI", 14), tags="placeholder")

        # ROI 鼠标事件
        self._roi_canvas.bind("<ButtonPress-1>", self._on_roi_press)
        self._roi_canvas.bind("<B1-Motion>", self._on_roi_drag)
        self._roi_canvas.bind("<ButtonRelease-1>", self._on_roi_release)

        # 保存引用用于滚动跳转
        self._left_canvas = lc
        self._left_frame = left

        # 可折叠插件面板
        self._plugin_order: list[str] = []  # 当前排列顺序
        shells: dict[str, CollapsibleFrame] = {}
        attr_map = {"camera": "camera_plugin", "model": "model_plugin",
                    "fringe_center": "fringe_center_plugin",
                    "recorder": "recorder", "status": "status", "motor": "motor_panel"}

        for key, title, cls in [
            ("status",   "实时状态",   StatusPanel),
            ("camera",   "摄像头控制", CameraPluginPanel),
            ("model",    "模型与预测", ModelPluginPanel),
            ("fringe_center", "中心条纹分析", FringeCenterPluginPanel),
            ("recorder", "视频录制",   VideoRecorderPanel),
            ("motor",    "电机控制",   MotorControlPanel),
        ]:
            shell = CollapsibleFrame(left, title)
            shell.pack(fill=tk.X, pady=4)
            if key == "camera":
                panel = cls(shell.content, default_index=int(
                    config.get("camera", "index", default=0)))
            elif key == "model":
                panel = cls(
                    shell.content,
                    confidence=float(config.get("vision", "confidence_threshold", default=0.5)),
                    iou=float(config.get("vision", "iou_threshold", default=0.45)),
                    imgsz=int(config.get("vision", "imgsz", default=640)),
                )
            else:
                panel = cls(shell.content)
            panel.pack(fill=tk.X)
            shells[key] = shell
            self._plugin_order.append(key)
            setattr(self, attr_map[key], panel)
            # ▲▼ 移动回调
            shell.on_move = lambda d, k=key: self._move_plugin(k, d)

        # 运行日志
        log_shell = CollapsibleFrame(left, "运行日志")
        log_shell.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log = LogPanel(log_shell.content)
        self.log.pack(fill=tk.BOTH, expand=True)
        shells["log"] = log_shell
        self._plugin_order.append("log")
        log_shell.on_move = lambda d: self._move_plugin("log", d)

        self._shells = shells

        # 插件开关 + 跳转绑定
        for key in self._plugin_order:
            self.plugin_bar.bind_toggle(key, lambda e, k=key: self._toggle_plugin(k, e))
            self.plugin_bar.bind_jump(key, lambda k=key: self._jump_to_plugin(k))

    # ==================================================================
    # 插件排序
    # ==================================================================
    def _move_plugin(self, key: str, direction: str):
        """将插件面板上移/下移一位"""
        order = self._plugin_order
        if key not in order: return
        idx = order.index(key)
        if direction == "up" and idx > 0:
            order[idx], order[idx-1] = order[idx-1], order[idx]
        elif direction == "down" and idx < len(order) - 1:
            order[idx], order[idx+1] = order[idx+1], order[idx]
        else:
            return
        self._reorder_shells()

    def _reorder_shells(self):
        """按 _plugin_order 重新排列面板"""
        # 先全部移除
        for shell in self._shells.values():
            shell.pack_forget()
        # 按新顺序重新 pack
        for key in self._plugin_order:
            shell = self._shells[key]
            fill = tk.BOTH if key == "log" else tk.X
            shell.pack(fill=fill, pady=4)

    def _jump_to_plugin(self, key: str):
        """滚动到指定插件"""
        shell = self._shells.get(key)
        if not shell: return
        # 确保可见
        if self.plugin_bar.is_enabled(key):
            self._toggle_plugin(key, True)
        # 计算位置并滚动
        self._left_frame.update_idletasks()
        y = shell.winfo_y()
        self._left_canvas.yview_moveto(max(0, y / self._left_frame.winfo_height()))

    # ==================================================================
    # 回调绑定
    # ==================================================================
    def _wire_callbacks(self):
        self.camera_plugin.on_command = self._on_camera_cmd
        self.model_plugin.on_command = self._on_model_cmd
        self.fringe_center_plugin.on_command = self._on_fringe_center_cmd
        self.recorder.on_start = self._on_rec_start
        self.recorder.on_stop = self._on_rec_stop
        mp = self.motor_panel
        mp.on_refresh_ports = lambda: self._on_refresh_ports()
        mp.on_connect = lambda p: self._on_motor_connect(p)
        mp.on_disconnect = lambda: self._on_motor_disconnect()
        mp.on_mode_change = lambda m: self._on_motor_mode_change(m)
        mp.on_manual_command = lambda c: self._on_manual_command(c)
        mp.on_manual_calibrate = lambda s: self._on_manual_calibrate(s)
        mp.on_apply_continuous = lambda *a: self._on_apply_continuous_params(*a)
        mp.on_apply_step = lambda *a: self._on_apply_step_params(*a)
        mp.on_auto_start = lambda: self._on_auto_start()
        mp.on_auto_stop = lambda: self._on_auto_stop()
        mp.on_query_status = lambda: self._on_query_motor_status()

    # ==================================================================
    # 插件开关
    # ==================================================================
    def _toggle_plugin(self, key: str, enabled: bool):
        shell = self._shells.get(key)
        if not shell:
            return
        if enabled:
            shell.pack(fill=tk.BOTH if key == "log" else tk.X, pady=4)
        else:
            shell.pack_forget()
            if key == "camera":
                self._stop_preview()
                self._stop_predict()
                if self.cam: self.cam.stop(); self.cam = None
                self.camera_running = False
                self._set_status("摄像头已关闭")
            elif key == "model":
                self._stop_predict()
            elif key == "fringe_center":
                self._center_line_x = None
                self._center_line_box = None
            elif key == "recorder":
                self.recorder.stop()
            elif key == "motor":
                self._on_auto_stop()
                self._stop_motor_poll()
                if self.motor: self.motor.close(); self.motor = None
                self.motor_connected = False
                self.status.update_motor_connected(False)
                self._set_status("电机已断开")

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
                resolution = config.get("camera", "resolution", default=[1280, 1024])
                self.cam = CameraManager(
                    index=cp.camera_index,
                    resolution=(int(resolution[0]), int(resolution[1])),
                    fps=int(config.get("camera", "fps", default=60)),
                )
                if not self.cam.start(): raise RuntimeError("无法打开摄像头")
                self.camera_running = True
                self._set_status("摄像头已启动"); self._start_preview()
            except Exception as e:
                self.camera_running = False; self.cam = None
                self._set_status(f"摄像头失败: {e}"); self.log.write(f"[错误] {e}")
        elif cmd == "close":
            self._stop_preview(); self._stop_predict()
            if self.cam: self.cam.stop(); self.cam = None
            self.camera_running = False; self._set_status("摄像头已关闭")
        elif cmd == "angle_apply":
            self.corrector.set_manual_offset(cp.angle)
        elif cmd == "angle_reset":
            cp.angle_var.set("0"); self.corrector.set_manual_offset(0)
        elif cmd == "zoom_apply":
            self.corrector.zoom = cp.zoom
        elif cmd == "zoom_reset":
            cp.zoom_var.set("2.0"); self.corrector.zoom = 2.0
        elif cmd == "pan_reset":
            self.corrector.pan_x = 0; self.corrector.pan_y = 0
        elif cmd == "all_reset":
            self.corrector.reset_all()
            cp.angle_var.set("0"); cp.zoom_var.set("1.0")

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
    # 模型插件
    # ==================================================================
    def _on_model_cmd(self, cmd: str):
        if cmd == "load":
            if self.detector.is_loaded(): self.log.write("YOLO 模型已加载"); return
            self._set_status("正在加载YOLO模型...")
            self.log.write("开始加载YOLO模型（后台）...")
            def _load():
                ok = self.detector.load()
                self.root.after(0, lambda: self._set_status("YOLO 模型已加载" if ok else "YOLO 加载失败"))
                self.root.after(0, lambda: self.log.write("YOLO 模型已加载" if ok else "[错误] YOLO 加载失败"))
            threading.Thread(target=_load, daemon=True).start()
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
            self._stop_preview()  # 停预览，避免画面覆盖
            self.predict_running = True; self._set_status("预测运行中"); self._predict_loop()
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

    def _detect_center_in_result(self, result: dict, corrected: np.ndarray):
        """扩大区域检测条纹中心，约束到零级条纹框内并精修。"""
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
            if verbose:
                self.log.write("[中心条纹] YOLO 未检测到任何目标")
            self._center_line_x = None
            self._center_line_box = None
            self.fringe_center_plugin.update_result(None, 0, False, "YOLO 未检测到目标")
            return

        # 按模型自身的类别名称匹配零级/黑条，禁止假设固定 class_id。
        zero_class_ids = self.detector.find_class_ids("zero", "order", "black", "零级", "黑")
        zero_indices = [i for i, cid in enumerate(class_ids) if int(cid) in zero_class_ids]

        # 再回退：最高置信度框
        if not zero_indices:
            if len(confs) > 0:
                zero_indices = [int(np.argmax(confs))]
            else:
                self._center_line_x = None
                self._center_line_box = None
                self.fringe_center_plugin.update_result(None, 0, False, "无检测框")
                return

        best_local_idx = int(np.argmax(confs[zero_indices]))
        best_idx = zero_indices[best_local_idx]
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        H, W = corrected.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0

        # 阶段 1：扩展区域检测（宽度 1.5 倍框宽，高度与框等高）
        expand_w = int(box_w * 1.5)
        expand_h = box_h
        x1c = max(0, int(box_cx - expand_w / 2))
        x2c = min(W, int(box_cx + expand_w / 2))
        y1c = max(0, int(box_cy - expand_h / 2))
        y2c = min(H, int(box_cy + expand_h / 2))

        if x2c - x1c < 20 or y2c - y1c < 20:
            if verbose:
                self.log.write(f"[中心条纹] 扩展区域太小 ({x2c-x1c}x{y2c-y1c})")
            self._center_line_x = None
            self._center_line_box = None
            self.fringe_center_plugin.update_result(None, 0, False, "扩展区域太小")
            return

        roi_crop = corrected[y1c:y2c, x1c:x2c]

        try:
            info = find_center_in_region(roi_crop)
        except Exception as e:
            if verbose:
                self.log.write(f"[中心条纹] 检测异常: {e}")
            self._center_line_x = None
            self._center_line_box = None
            self.fringe_center_plugin.update_result(None, 0, False, f"检测异常: {e}")
            return

        if info["orientation"] != "vertical":
            self._center_line_x = None
            self._center_line_box = None
            self.fringe_center_plugin.update_result(None, 0, False,
                f"条纹方向={info['orientation']}（暂只支持竖直）")
            return

        # 阶段 2：转换坐标 + 约束到框内
        center_x_final = x1c + info["center_main"]
        center_x_final = max(x1 + 1, min(x2 - 1, center_x_final))

        self._center_line_x = center_x_final
        self._center_line_box = (x1, y1, x2, y2)

        if verbose:
            self.log.write(
                f"[中心条纹] x={center_x_final:.1f}px "
                f"conf={info['confidence']:.2f} "
                f"box=({x1},{y1})-({x2},{y2}) "
                f"classes={list(set(class_names))}"
            )

        self.fringe_center_plugin.update_result(
            center_x_final - x1, info["confidence"], True)

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
                self._center_line_box = None
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
        self._center_line_box = None
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
            self._show_frame(corrected)
            if self.recorder and self.recorder.recording:
                src = frame if self.recorder.recording_source == "camera" else corrected
                self._write_rec_frame(src)
        self._preview_job = self.root.after(self.PREVIEW_INTERVAL_MS, self._preview_loop)

    def _show_frame(self, frame_bgr):
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
        canvas.delete("roi", "center_line")
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
        if roi:
            cv2.rectangle(annotated, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,0), 2)

        recommended = _decide_motor_command_from_boxes(
            result["boxes_xyxy"], result["confs"], annotated.shape)

        class_conf = get_class_confidences(result)

        # ---- 中心条纹自动检测（跟随 YOLO 预测频率）----
        if (self.fringe_center_plugin is not None
                and self.fringe_center_plugin.auto_detect_var.get()):
            self._detect_center_in_result(result, corrected)

        if self.motor_panel.mode in ("continuous", "step") and self.auto_control_enabled:
            self._auto_motor_control(class_conf)

        now = time.time()
        dt = max(1e-6, now - self.last_t)
        self.fps = 0.85*self.fps + 0.15*(1.0/dt)
        self.last_t = now

        cv2.putText(annotated, f"fps={self.fps:.1f} angle={self.corrector.effective_angle:+.1f}",
                    (20,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
        cv2.putText(annotated, f"suggest={recommended}",
                    (20,64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        self._show_frame(annotated)
        if self.recorder and self.recorder.recording:
            src = frame if self.recorder.recording_source == "camera" else annotated
            self._write_rec_frame(src)

        self.status.update_fps(self.fps)
        self.model_plugin.update_results(class_conf, len(result["boxes_xyxy"]), recommended)
        self._last_detection_result = result  # 保存供中心条纹分析使用

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
        ports = MotorController.list_ports()
        self.motor_panel.update_ports(ports)
        self.log.write(f"可用串口: {ports}")

    def _on_motor_connect(self, port):
        try:
            self.motor = MotorController(
                port=port,
                baudrate=int(config.get("motor", "baudrate", default=9600)),
                timeout=float(config.get("motor", "timeout", default=1.0)),
            )
            if self.motor.connect():
                self.motor_connected = True
                self._set_status(f"电机已连接: {port}")
                self.status.update_motor_connected(True, port)
                self._start_motor_poll()
            else:
                self._set_status(f"电机连接失败: {port}")
        except Exception as e:
            self.log.write(f"[错误] 电机连接失败: {e}")

    def _on_motor_disconnect(self):
        self._stop_motor_poll()
        self._on_auto_stop()
        if self.motor:
            self.motor.close()
            self.motor = None
        self.motor_connected = False
        self.status.update_motor_connected(False)
        self._set_status("电机已断开")

    def _on_motor_mode_change(self, mode):
        self._on_auto_stop()

    def _on_manual_command(self, cmd):
        if not self.motor_connected or self.motor is None:
            self.log.write("[警告] 电机未连接")
            return
        if cmd == "START":
            t = self.motor_panel.get_manual_speed()
            if self.motor_panel.get_manual_auto_fix():
                self.motor.set_speed(t)
            self.motor.start()
            self.log.write(f"[手动] 启动 (速度={t})")
        elif cmd == "STOP":
            self.motor.stop()
            self.log.write("[手动] 停止")
        elif cmd == "SPEED_UP":
            self.motor.speed_up()
            self.log.write("[手动] 加速")
        elif cmd == "SPEED_DOWN":
            self.motor.speed_down()
            self.log.write("[手动] 减速")
        elif cmd == "STATUS":
            self._on_query_motor_status()

    def _on_manual_calibrate(self, target):
        if self.motor_connected and self.motor:
            ok = self.motor.set_speed(target)
            self.log.write(f"[手动] 校准到{target}: {'OK' if ok else 'FAIL'}")

    def _on_apply_continuous_params(self, s1, s2, s3, th):
        self.log.write(f"[连续] 搜索={s1} 彩条={s2} 黑条={s3} 阈值={th}")

    def _on_apply_step_params(self, fms, cms, pms, spd, th):
        self.log.write(f"[步进] 首轮={fms}ms 循环={cms}ms 暂停={pms}ms 速度={spd} 阈值={th}")

    def _on_query_motor_status(self):
        if not self.motor_connected or self.motor is None:
            self.log.write("[电机] 未连接")
            return
        s = self.motor.query_status()
        self.log.write(f"[电机] {'RUN' if s['running'] else 'STOP'} 档位={s['speed']} ω={s['omega']}deg/s")

    def _on_auto_start(self):
        if not self.motor_connected:
            self.log.write("[警告] 请先连接电机")
            return
        if not self.predict_running:
            self.log.write("[警告] 自动控制要求先启动模型预测")
            return
        if not self.detector.find_class_ids("black", "zero", "order", "黑", "零级"):
            self.log.write(f"[错误] 模型缺少黑条/零级类别: {self.detector.class_names}")
            return
        self.auto_control_enabled = True
        self.auto_speed_stage = "idle"
        self.auto_cycle_phase = "idle"
        self.auto_best_black_conf = 0.0
        self.auto_started_at = time.monotonic()
        self.auto_black_frames = 0
        self.auto_missing_frames = 0
        self.log.write("[AUTO] 自动控制已启动")

    def _on_auto_stop(self, reason: str = "用户停止"):
        was_enabled = self.auto_control_enabled
        self.auto_control_enabled = False
        self.auto_speed_stage = "idle"
        self.auto_cycle_phase = "idle"
        if self.motor:
            self.motor.stop()
        if was_enabled:
            self.motor_panel.update_auto_status(f"自动控制: 已停止（{reason}）")
            self.log.write(f"[AUTO] 已停止: {reason}")

    def _auto_motor_control(self, class_conf):
        if not self.auto_control_enabled or self.motor is None:
            return
        cc = class_conf.get("color", 0)
        bc = class_conf.get("black", 0)
        now = time.time()
        safety = config.get("motor", "safety", default={}) or {}
        max_run_s = float(safety.get("max_run_seconds", 60))
        confirm_frames = max(1, int(safety.get("black_confirm_frames", 3)))
        max_missing = max(1, int(safety.get("max_missing_frames", 30)))
        if time.monotonic() - self.auto_started_at > max_run_s:
            self._on_auto_stop("达到最大运行时间")
            return
        if not self.motor.is_connected:
            self._on_auto_stop("串口失联")
            return
        if cc <= 0 and bc <= 0:
            self.auto_missing_frames += 1
            if self.auto_missing_frames >= max_missing:
                self._on_auto_stop("连续未检测到条纹")
                return
        else:
            self.auto_missing_frames = 0
        if self.motor_panel.mode == "step":
            p = self.motor_panel.get_step_params()
            if self.auto_cycle_phase == "idle":
                self.motor.set_speed(p["speed"]); self.motor.start()
                self.auto_cycle_phase = "move"; self.auto_cycle_ts = now
                self._cycle_phase_ms = p["first_ms"] if self.auto_best_black_conf == 0 else p["cycle_ms"]
            elif self.auto_cycle_phase == "move":
                if now - self.auto_cycle_ts > self._cycle_phase_ms / 1000.0:
                    self.motor.stop(); self.auto_cycle_phase = "pause"; self.auto_cycle_ts = now
            elif self.auto_cycle_phase == "pause":
                if now - self.auto_cycle_ts > p["pause_ms"] / 1000.0:
                    if bc > self.auto_best_black_conf: self.auto_best_black_conf = bc
                    if bc > p["black_threshold"]:
                        self.auto_black_frames += 1
                    else:
                        self.auto_black_frames = 0
                    if self.auto_black_frames >= confirm_frames:
                        self.motor.stop(); self.auto_cycle_phase = "locked"
                        self.motor_panel.update_auto_status("自动控制: 已锁定")
                        self.log.write("[AUTO] 步进锁定")
                    else:
                        self.auto_cycle_phase = "idle"
        elif self.motor_panel.mode == "continuous":
            p = self.motor_panel.get_continuous_params()
            if self.auto_speed_stage == "idle":
                self.motor.set_speed(p["search_speed"])
                if not self.motor.start():
                    self._on_auto_stop("电机启动失败")
                    return
                self.auto_speed_stage = "searching"
            elif self.auto_speed_stage == "searching":
                if cc > 0.3:
                    self.auto_speed_stage = "color"; self.motor.set_speed(p["color_speed"])
            elif self.auto_speed_stage == "color":
                self.motor.set_speed(p["color_speed"])
                if bc > p["black_threshold"]:
                    self.auto_black_frames += 1
                    if self.auto_black_frames >= confirm_frames:
                        self.auto_speed_stage = "black"; self.motor.set_speed(p["black_speed"])
                        self.log.write("[AUTO] 连续检测到黑条")
                else:
                    self.auto_black_frames = 0
                    if cc <= 0.3:
                        self.auto_speed_stage = "searching"
                        self.motor.set_speed(p["search_speed"])
            elif self.auto_speed_stage == "black":
                if bc > p["black_threshold"]:
                    self.auto_black_frames += 1
                    if self.auto_black_frames >= confirm_frames * 2:
                        self.motor.stop()
                        self.auto_speed_stage = "locked"
                        self.motor_panel.update_auto_status("自动控制: 已锁定")
                        self.log.write("[AUTO] 黑条锁定")
                else:
                    self.auto_black_frames = 0
                    self.auto_speed_stage = "color" if cc > 0.3 else "searching"

    # ==================================================================
    # 电机轮询
    # ==================================================================
    def _start_motor_poll(self):
        self._poll_motor()

    def _stop_motor_poll(self):
        if self._motor_poll_job:
            self.root.after_cancel(self._motor_poll_job)
            self._motor_poll_job = None

    def _poll_motor(self):
        self._motor_poll_job = None
        if self.motor and self.motor_connected:
            if self._motor_poll_future is None:
                self._motor_poll_future = self._motor_executor.submit(self.motor.query_status)
            elif self._motor_poll_future.done():
                try:
                    s = self._motor_poll_future.result()
                    self.status.update_motor_speed(s['omega'])
                    self.status.update_motor_gear(s['speed'])
                except Exception as e:
                    self.log.write(f"[MOTOR] {e}")
                finally:
                    self._motor_poll_future = None
        self._motor_poll_job = self.root.after(50, self._poll_motor)

    # ==================================================================
    # 工具
    # ==================================================================
    def _set_status(self, text):
        self.status_var.set(f"状态: {text}")

    def _on_close(self):
        self._closing = True
        self.root.unbind_all("<MouseWheel>")
        self._stop_preview()
        self.recorder.stop()
        self._stop_predict()
        self._stop_motor_poll()
        if self.cam: self.cam.stop()
        if self.motor: self.motor.close()
        self._inference_executor.shutdown(wait=False, cancel_futures=True)
        self._motor_executor.shutdown(wait=False, cancel_futures=True)
        self._camera_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self):
        self.log.write("启动 UI")
        self.root.mainloop()


def run_app():
    YoloCamApp().run()
