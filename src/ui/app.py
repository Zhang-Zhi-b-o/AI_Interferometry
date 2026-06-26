"""摄像头 YOLO 实时检测 + 电机控制 — Tkinter UI"""
from __future__ import annotations

import time
import threading
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
from src.vision import YOLODetector, rotate_expand, FrameCorrector
from src.vision.class_names import get_class_confidences
from src.hardware import MotorController
from src.ui.widgets import (
    VideoRecorderPanel,
    StatusPanel,
    MotorControlPanel,
    LogPanel,
    CameraPluginPanel,
    ModelPluginPanel,
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
        self.root.geometry("1600x1000")
        self.root.configure(bg="#ffffff")

        # ---- 核心模块 ----
        model_path = config.resolve_path(
            config.get("vision", "model_path", default="models/yolov8_interference.pt"))
        self.cam: CameraManager | None = None
        self.detector = YOLODetector(
            str(model_path),
            confidence=0.5, iou=0.45, imgsz=640,
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
        self._cycle_phase_ms = 1000
        self.motor_connected = False
        self.fps = 0.0
        self.last_t = time.time()

        # ---- 定时器 ----
        self._preview_job: str | None = None
        self._predict_job: str | None = None
        self._motor_poll_job: str | None = None

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
        self.recorder: VideoRecorderPanel | None = None
        self.status: StatusPanel | None = None
        self.motor_panel: MotorControlPanel | None = None
        self.log: LogPanel | None = None
        # 可折叠外壳
        self._shells: dict[str, CollapsibleFrame] = {}

        # ---- 构建 ----
        self._build_ui()
        self._wire_callbacks()
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
                    "recorder": "recorder", "status": "status", "motor": "motor_panel"}

        for key, title, cls in [
            ("status",   "实时状态",   StatusPanel),
            ("camera",   "摄像头控制", CameraPluginPanel),
            ("model",    "模型与预测", ModelPluginPanel),
            ("recorder", "视频录制",   VideoRecorderPanel),
            ("motor",    "电机控制",   MotorControlPanel),
        ]:
            shell = CollapsibleFrame(left, title)
            shell.pack(fill=tk.X, pady=4)
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
            cameras = CameraManager.detect_all()
            messagebox.showinfo("摄像头检测", f"可用摄像头: {cameras}")
            self.log.write(f"摄像头检测: {cameras}")
        elif cmd == "open":
            if self.camera_running: return
            try:
                self.cam = CameraManager(index=cp.camera_index)
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
            cp.zoom_var.set("1.0"); self.corrector.zoom = 1.0
        elif cmd == "pan_reset":
            self.corrector.pan_x = 0; self.corrector.pan_y = 0
        elif cmd == "all_reset":
            self.corrector.reset_all()
            cp.angle_var.set("0"); cp.zoom_var.set("1.0")

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

    # ==================================================================
    # ROI 鼠标绘制
    # ==================================================================
    def _on_roi_press(self, event):
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
            scale = self._get_display_scale()
            ox, oy = self._frame_off_x, self._frame_off_y
            x1 = int((self._roi_start[0]-ox)/scale) if scale>0 else 0
            y1 = int((self._roi_start[1]-oy)/scale) if scale>0 else 0
            x2 = int((event.x-ox)/scale) if scale>0 else 0
            y2 = int((event.y-oy)/scale) if scale>0 else 0
            self.model_plugin.set_roi(x1, y1, x2, y2)
            self._roi_canvas.delete("drawing")
            self._roi_rect_id = None
        elif self._panning:
            self._panning = False

    def _get_display_scale(self) -> float:
        c = self._roi_canvas
        if c is None: return 1.0
        return min(max(1,c.winfo_width())/1280, max(1,c.winfo_height())/1024)

    def _get_roi(self) -> tuple[int,int,int,int] | None:
        if not self.model_plugin.roi_mode: return None
        return self.model_plugin.get_roi_xywh()

    def _stop_predict(self):
        self.predict_running = False
        self._on_auto_stop()
        if self._predict_job:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None
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
        # 重画 ROI + 绘制框
        canvas.delete("roi", "drawing")
        if self.model_plugin and self.model_plugin.roi_pixels:
            x1, y1, x2, y2 = self.model_plugin.roi_pixels
            canvas.create_rectangle(
                x1*scale+self._frame_off_x, y1*scale+self._frame_off_y,
                x2*scale+self._frame_off_x, y2*scale+self._frame_off_y,
                outline="#00ff00", width=2, tags="roi")

    # ==================================================================
    # 预测循环
    # ==================================================================
    def _predict_loop(self):
        self._predict_job = None
        if not self.predict_running or self.cam is None:
            return
        frame = self.cam.read()
        if frame is None:
            self._predict_job = self.root.after(self.PREDICT_INTERVAL_MS, self._predict_loop)
            return

        conf = self.model_plugin.conf
        iou = self.model_plugin.iou
        imgsz = self.model_plugin.imgsz

        corrected = rotate_expand(frame, self.corrector.effective_angle)
        corrected = self.corrector.apply_zoom_pan(corrected)
        self.detector.confidence = conf
        self.detector.iou = iou
        self.detector.imgsz = imgsz

        roi = self._get_roi()
        result = self.detector.detect(corrected, roi=roi)
        annotated = result["annotated"] if result["annotated"] is not None else corrected
        if roi:
            cv2.rectangle(annotated, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,0), 2)

        recommended = _decide_motor_command_from_boxes(
            result["boxes_xyxy"], result["confs"], annotated.shape)

        class_conf = get_class_confidences(result)
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
        self._predict_job = self.root.after(self.PREDICT_INTERVAL_MS, self._predict_loop)

    # ==================================================================
    # 录制
    # ==================================================================
    def _on_rec_start(self, path, fps, source):
        fourcc = cv2.VideoWriter_fourcc(*"XVID") if path.endswith(".avi") else cv2.VideoWriter_fourcc(*"mp4v")
        self.recorder.video_writer = cv2.VideoWriter(path, fourcc, fps, (1280, 1024))
        self.log.write(f"[录制] 开始: {path}")

    def _on_rec_stop(self):
        if self.recorder and self.recorder.video_writer:
            self.recorder.video_writer.release()
            self.recorder.video_writer = None
        self.log.write("[录制] 已停止")

    def _write_rec_frame(self, frame):
        wr = self.recorder.video_writer if self.recorder else None
        if wr is None:
            return
        if frame.shape[1::-1] != (1280, 1024):
            frame = cv2.resize(frame, (1280, 1024))
        wr.write(frame)

    # ==================================================================
    # 电机
    # ==================================================================
    def _on_refresh_ports(self):
        ports = MotorController.list_ports()
        self.motor_panel.update_ports(ports)
        self.log.write(f"可用串口: {ports}")

    def _on_motor_connect(self, port):
        try:
            self.motor = MotorController(port=port)
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
        self.auto_control_enabled = True
        self.auto_speed_stage = "idle"
        self.auto_cycle_phase = "idle"
        self.auto_best_black_conf = 0.0
        self.log.write("[AUTO] 自动控制已启动")

    def _on_auto_stop(self):
        self.auto_control_enabled = False
        self.auto_speed_stage = "idle"
        self.auto_cycle_phase = "idle"
        if self.motor:
            self.motor.stop()

    def _auto_motor_control(self, class_conf):
        if not self.auto_control_enabled or self.motor is None:
            return
        cc = class_conf.get("color", 0)
        bc = class_conf.get("black", 0)
        now = time.time()
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
                        self.motor.stop(); self.auto_cycle_phase = "locked"
                        self.motor_panel.update_auto_status("自动控制: 已锁定")
                        self.log.write("[AUTO] 步进锁定")
                    else:
                        self.auto_cycle_phase = "idle"
        elif self.motor_panel.mode == "continuous":
            p = self.motor_panel.get_continuous_params()
            if self.auto_speed_stage == "idle":
                if cc > 0.3:
                    self.auto_speed_stage = "color"; self.motor.set_speed(p["color_speed"])
                else:
                    self.motor.set_speed(p["search_speed"])
            elif self.auto_speed_stage == "color":
                self.motor.set_speed(p["color_speed"])
                if bc > p["black_threshold"]:
                    self.auto_speed_stage = "black"; self.motor.set_speed(p["black_speed"])
                    self.log.write("[AUTO] 检测到黑条")
            elif self.auto_speed_stage == "black":
                if bc > p["black_threshold"]:
                    self.motor.stop()
                    self.motor_panel.update_auto_status("自动控制: 已锁定")
                    self.log.write("[AUTO] 黑条锁定")

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
            try:
                s = self.motor.query_status()
                self.status.update_motor_speed(s['omega'])
                self.status.update_motor_gear(s['speed'])
            except Exception as e:
                self.log.write(f"[MOTOR] {e}")
        self._motor_poll_job = self.root.after(self.MOTOR_POLL_MS, self._poll_motor)

    # ==================================================================
    # 工具
    # ==================================================================
    def _set_status(self, text):
        self.status_var.set(f"状态: {text}")

    def _on_close(self):
        self.root.unbind_all("<MouseWheel>")
        self._stop_preview()
        self.recorder.stop()
        self._stop_predict()
        self._stop_motor_poll()
        if self.cam: self.cam.stop()
        if self.motor: self.motor.close()
        self.root.destroy()

    def run(self):
        self.log.write("启动 UI")
        self.root.mainloop()


def run_app():
    YoloCamApp().run()
