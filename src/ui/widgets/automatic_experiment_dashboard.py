"""自动实验页的双画面预览与实时数据仪表板。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class AutomaticExperimentDashboard(tk.Frame):
    BG = "#f4f7fb"
    SURFACE = "#ffffff"
    TEXT = "#10233f"
    MUTED = "#64748b"
    BLUE = "#1677ff"
    GREEN = "#07883f"
    VIDEO_BG = "#111827"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, bg=self.BG)
        self.on_prepare = lambda: None
        self.on_emergency_stop = lambda: None
        self.on_roi_changed = lambda _roi: None
        self.on_pan_changed = lambda _dx, _dy: None
        self.roi_enabled = False
        self.pan_enabled = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_mode = ""
        self._source_shape: tuple[int, int] | None = None
        self._display_transform = (1.0, 0.0, 0.0)
        self._roi_rect_id: int | None = None
        self.auto_record_var = tk.BooleanVar(value=True)
        self.stage_var = tk.StringVar(value="等待开始")
        self.elapsed_var = tk.StringVar(value="00:00 / 10:00")
        self.progress_var = tk.IntVar(value=0)
        self.device_var = tk.StringVar(value="相机 --  │  模型 --  │  电机 --  │  微分表 --")
        self.center_var = tk.StringVar(value="中心 --  │  目标 --  │  偏差 --  │  置信度 --")
        self.motion_var = tk.StringVar(value="条纹 --  │  移动方向 --  │  搜索状态 --")
        self.meter_var = tk.StringVar(value="参考 -- mm  │  当前 -- mm  │  位移 -- mm")
        self.record_var = tk.StringVar(value="录像：未启动")
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._build()

    def _build(self) -> None:
        controls = tk.Frame(self, bg=self.SURFACE, highlightthickness=1,
                            highlightbackground="#dbe3ef")
        controls.pack(fill=tk.X, pady=(0, 10))
        tk.Button(
            controls, text="一键启动自动实验", command=lambda: self.on_prepare(),
            bg=self.BLUE, fg="#ffffff", relief=tk.FLAT, cursor="hand2",
            padx=16, pady=7, font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=12, pady=10)
        tk.Button(
            controls, text="紧急停车", command=lambda: self.on_emergency_stop(),
            bg="#fee4e2", fg="#b42318", relief=tk.FLAT, cursor="hand2",
            padx=18, pady=7, font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, pady=10)
        tk.Checkbutton(
            controls, text="自动实验期间录制干涉画面",
            variable=self.auto_record_var, bg=self.SURFACE,
            activebackground=self.SURFACE, selectcolor=self.SURFACE,
        ).pack(side=tk.LEFT, padx=16)
        tk.Label(controls, textvariable=self.record_var, bg=self.SURFACE,
                 fg=self.MUTED).pack(side=tk.RIGHT, padx=12)

        status = tk.Frame(self, bg=self.SURFACE, highlightthickness=1,
                          highlightbackground="#dbe3ef")
        status.pack(fill=tk.X, pady=(0, 10))
        row = tk.Frame(status, bg=self.SURFACE)
        row.pack(fill=tk.X, padx=14, pady=(10, 3))
        tk.Label(row, textvariable=self.stage_var, bg=self.SURFACE, fg=self.BLUE,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(row, textvariable=self.elapsed_var, bg=self.SURFACE, fg=self.MUTED,
                 font=("Consolas", 10)).pack(side=tk.RIGHT)
        ttk.Progressbar(status, maximum=100, variable=self.progress_var).pack(
            fill=tk.X, padx=14, pady=(2, 10))

        previews = tk.Frame(self, bg=self.BG)
        previews.pack(fill=tk.X, pady=(0, 10))
        previews.columnconfigure(0, weight=1, uniform="preview")
        previews.columnconfigure(1, weight=1, uniform="preview")
        self.interferometer_preview = self._preview_card(
            previews, 0, "干涉条纹实时画面", "等待主摄像头画面")
        self.micrometer_preview = self._preview_card(
            previews, 1, "微分表实时画面", "等待微分表摄像头画面")

        data = tk.Frame(self, bg=self.SURFACE, highlightthickness=1,
                        highlightbackground="#dbe3ef")
        data.pack(fill=tk.X, pady=(0, 10))
        for variable, color in (
            (self.device_var, self.TEXT),
            (self.center_var, self.TEXT),
            (self.motion_var, self.MUTED),
            (self.meter_var, self.GREEN),
        ):
            tk.Label(data, textvariable=variable, bg=self.SURFACE, fg=color,
                     anchor="w", justify="left", font=("Consolas", 9)).pack(
                         fill=tk.X, padx=14, pady=3)

    def _preview_card(
        self, parent: tk.Widget, column: int, title: str, placeholder: str,
    ) -> tk.Canvas:
        card = tk.Frame(parent, bg=self.SURFACE, highlightthickness=1,
                        highlightbackground="#dbe3ef")
        card.grid(row=0, column=column, sticky="nsew",
                  padx=(0, 5) if column == 0 else (5, 0))
        tk.Label(card, text=title, bg=self.SURFACE, fg=self.TEXT,
                 anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(
                     fill=tk.X, padx=10, pady=(8, 5))
        canvas = tk.Canvas(card, bg=self.VIDEO_BG, width=500, height=280,
                           highlightthickness=0, bd=0, cursor="crosshair")
        canvas.pack(padx=10, pady=(0, 10))
        canvas.create_text(250, 140, text=placeholder, fill="#ffffff",
                           font=("Microsoft YaHei UI", 10), tags="placeholder")
        if column == 0:
            canvas.bind("<ButtonPress-1>", self._on_preview_press)
            canvas.bind("<B1-Motion>", self._on_preview_drag)
            canvas.bind("<ButtonRelease-1>", self._on_preview_release)
        return canvas

    @staticmethod
    def _letterbox(frame: np.ndarray, width: int = 500, height: int = 280) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        source = frame
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        h, w = source.shape[:2]
        scale = min(width / max(1, w), height / max(1, h))
        resized = cv2.resize(
            source, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    def _show(self, key: str, canvas: tk.Canvas, frame: np.ndarray) -> None:
        boxed = self._letterbox(frame)
        rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photos[key] = photo
        canvas.delete("placeholder")
        image_id = getattr(self, f"_{key}_image_id", None)
        if image_id is None:
            image_id = canvas.create_image(
                250, 140, image=photo, anchor="center", tags="preview_image")
            setattr(self, f"_{key}_image_id", image_id)
        else:
            canvas.itemconfigure(image_id, image=photo)
        canvas.tag_lower("preview_image")

    def update_interferometer(
        self, frame: np.ndarray, center_x: float | None = None,
        target_x: float | None = None,
    ) -> None:
        preview = frame.copy()
        source_h, source_w = preview.shape[:2]
        scale = min(500 / max(1, source_w), 280 / max(1, source_h))
        self._source_shape = (source_h, source_w)
        self._display_transform = (
            scale,
            (500 - source_w * scale) / 2.0,
            (280 - source_h * scale) / 2.0,
        )
        h = preview.shape[0]
        if target_x is not None:
            cv2.line(preview, (int(target_x), 0), (int(target_x), h - 1),
                     (255, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(
                preview, "FRAME CENTER", (int(target_x) + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2,
                cv2.LINE_AA,
            )
        if center_x is not None:
            cv2.line(preview, (int(center_x), 0), (int(center_x), h - 1),
                     (0, 0, 255), 2, cv2.LINE_AA)
        self._show("interferometer", self.interferometer_preview, preview)

    def update_micrometer(self, result) -> None:
        frame = getattr(result, "frame", None)
        if frame is None or frame.size == 0:
            return
        preview = frame.copy()
        box = getattr(result, "roi_xyxy", None)
        if box is not None:
            x1, y1, x2, y2 = (int(value) for value in box)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (46, 204, 113), 3)
        self._show("micrometer", self.micrometer_preview, preview)

    def set_interaction_modes(self, *, roi: bool, pan: bool) -> None:
        self.roi_enabled = bool(roi)
        self.pan_enabled = bool(pan) and not self.roi_enabled
        self.interferometer_preview.configure(
            cursor="crosshair" if self.roi_enabled else
            ("fleur" if self.pan_enabled else "arrow"))

    def clear_roi_overlay(self) -> None:
        if self._roi_rect_id is not None:
            self.interferometer_preview.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _on_preview_press(self, event) -> None:
        if not (self.roi_enabled or self.pan_enabled):
            return
        self._drag_start = (event.x, event.y)
        self._drag_mode = "roi" if self.roi_enabled else "pan"
        if self._drag_mode == "roi":
            self.clear_roi_overlay()
            self._roi_rect_id = self.interferometer_preview.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#22c55e", width=2, dash=(5, 3), tags="roi_overlay")

    def _on_preview_drag(self, event) -> None:
        if self._drag_start is None:
            return
        if self._drag_mode == "roi" and self._roi_rect_id is not None:
            self.interferometer_preview.coords(
                self._roi_rect_id, self._drag_start[0], self._drag_start[1],
                event.x, event.y)

    def _on_preview_release(self, event) -> None:
        if self._drag_start is None:
            return
        start_x, start_y = self._drag_start
        self._drag_start = None
        scale, off_x, off_y = self._display_transform
        if scale <= 0 or self._source_shape is None:
            return
        if self._drag_mode == "pan":
            self.on_pan_changed(
                int(round((event.x - start_x) / scale)),
                int(round((event.y - start_y) / scale)),
            )
            return
        height, width = self._source_shape
        x1 = int(round((start_x - off_x) / scale))
        y1 = int(round((start_y - off_y) / scale))
        x2 = int(round((event.x - off_x) / scale))
        y2 = int(round((event.y - off_y) / scale))
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            self.on_roi_changed((x1, y1, x2, y2))

    def update_data(self, data: dict) -> None:
        self.stage_var.set(str(data.get("stage", "等待开始")))
        self.progress_var.set(int(data.get("progress", 0)))
        elapsed = max(0.0, float(data.get("elapsed_seconds", 0.0)))
        limit = max(1.0, float(data.get("limit_seconds", 600.0)))
        self.elapsed_var.set(
            f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d} / "
            f"{int(limit)//60:02d}:{int(limit)%60:02d}")
        yes = lambda value: "已连接" if value else "未连接"
        self.device_var.set(
            f"主相机 {yes(data.get('camera'))}  │  "
            f"模型 {'运行中' if data.get('prediction') else '未运行'}  │  "
            f"电机 {yes(data.get('motor'))}  │  微分表 {yes(data.get('micrometer'))}")
        center = data.get("center_x")
        target = data.get("target_x")
        error = data.get("error_px")
        confidence = float(data.get("confidence", 0.0))
        fmt = lambda value: "--" if value is None else f"{float(value):.1f}"
        self.center_var.set(
            f"中心 {fmt(center)} px  │  目标 {fmt(target)} px  │  "
            f"偏差 {fmt(error)} px  │  置信度 {confidence:.2f}")
        self.motion_var.set(
            f"条纹 {'存在' if data.get('has_fringe') else '未检测'}  │  "
            f"移动 {data.get('movement', '--')}  │  "
            f"搜索 {data.get('search_state', '--')}")
        reference = data.get("reference_mm")
        reading = data.get("reading_mm")
        displacement = data.get("displacement_mm")
        mm = lambda value: "--" if value is None else f"{float(value):.6f}"
        self.meter_var.set(
            f"参考 {mm(reference)} mm  │  当前 {mm(reading)} mm  │  "
            f"动镜位移 {mm(displacement)} mm")
        self.record_var.set("录像：进行中" if data.get("recording") else "录像：未启动")
