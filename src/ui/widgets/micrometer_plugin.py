"""第二摄像头数显微分表 OCR 插件。"""
from __future__ import annotations

from datetime import datetime
import tkinter as tk

import cv2
from PIL import Image, ImageTk


class MicrometerPluginPanel(tk.LabelFrame):
    BG = "#ffffff"
    TEXT = "#10233f"
    MUTED = "#64748b"
    BLUE = "#1677ff"

    def __init__(self, parent: tk.Widget, settings: dict | None = None):
        super().__init__(parent, text="视觉微分表读数", bg=self.BG, fg=self.TEXT)
        settings = settings or {}
        resolution = settings.get("resolution", [1280, 1024])
        roi = settings.get("roi", [0.0, 0.0, 1.0, 1.0])
        self.on_command = lambda _command, _payload=None: None
        self.index_var = tk.StringVar(value=str(settings.get("camera_index", 0)))
        self.auto_roi_var = tk.BooleanVar(value=bool(settings.get("auto_roi", True)))
        self.roi_var = tk.StringVar(value=",".join(str(v) for v in roi))
        self.status_var = tk.StringVar(value="尚未启动")
        self.raw_var = tk.StringVar(value="单帧 --  │  置信度 --")
        self.stable_var = tk.StringVar(value="稳定读数 -- mm")
        self._decimal_places = max(0, int(settings.get("decimal_places", 3)))
        self.camera_list_var = tk.StringVar(value="可用摄像头：未检测")
        self.available_cameras: list[int] = []
        self._settings = {
            "model_path": settings.get(
                "model_path", "models/micrometer/PP-OCRv6_rec_small.onnx"),
            "resolution": resolution,
            "fps": settings.get("fps", 15),
            "interval_ms": settings.get("interval_ms", 200),
            "min_score": settings.get("min_score", 0.45),
            "decimal_places": settings.get("decimal_places", 3),
            "stable_window": settings.get("stable_window", 7),
            "stable_required": settings.get("stable_required", 3),
            "max_step_mm": settings.get("max_step_mm", 0.05),
            "jump_required": settings.get("jump_required", 6),
            "scale_ratio_tolerance": settings.get(
                "scale_ratio_tolerance", 0.03),
        }
        self._preview_photo = None
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text="使用独立摄像头识别数显微分表；稳定读数会自动提供给实验流程。",
            bg=self.BG, fg=self.MUTED, justify="left", anchor="w", wraplength=360,
        ).pack(fill=tk.X, padx=8, pady=(8, 5))

        camera_row = tk.Frame(self, bg=self.BG)
        camera_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(camera_row, text="摄像头索引", bg=self.BG, fg=self.TEXT).pack(side=tk.LEFT)
        tk.Spinbox(
            camera_row, from_=0, to=20, textvariable=self.index_var,
            width=5, justify="center").pack(side=tk.LEFT, padx=6)
        tk.Button(
            camera_row, text="检测", command=lambda: self._emit("detect"),
            bg="#e8f1ff", fg=self.BLUE, relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=2)
        tk.Label(
            self, textvariable=self.camera_list_var, bg=self.BG, fg=self.MUTED,
            anchor="w", justify="left", wraplength=360,
        ).pack(fill=tk.X, padx=8, pady=(0, 3))

        roi_row = tk.Frame(self, bg=self.BG)
        roi_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Checkbutton(
            roi_row, text="自动定位 LCD", variable=self.auto_roi_var,
            bg=self.BG, activebackground=self.BG, selectcolor=self.BG,
        ).pack(side=tk.LEFT)
        tk.Label(roi_row, text="手动 ROI", bg=self.BG, fg=self.MUTED).pack(
            side=tk.LEFT, padx=(10, 3))
        tk.Entry(roi_row, textvariable=self.roi_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        buttons = tk.Frame(self, bg=self.BG)
        buttons.pack(fill=tk.X, padx=8, pady=5)
        tk.Button(
            buttons, text="启动读数", command=lambda: self._emit("start"),
            bg=self.BLUE, fg="#ffffff", activebackground="#0f62d6",
            activeforeground="#ffffff", relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3), ipady=4)
        tk.Button(
            buttons, text="停止", command=lambda: self._emit("stop"),
            bg="#fee4e2", fg="#b42318", relief=tk.FLAT, cursor="hand2", width=8,
        ).pack(side=tk.RIGHT, ipady=4)

        tk.Label(
            self, text="摄像头实时预览（绿色框为当前 LCD 识别区域）",
            bg=self.BG, fg=self.MUTED, anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(4, 1))
        self.preview_label = tk.Label(
            self, text="启动读数后显示第二摄像头画面",
            bg="#172033", fg="#ffffff", height=14,
            anchor="center", compound=tk.CENTER)
        self.preview_label.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(
            self, textvariable=self.status_var, bg=self.BG, fg=self.BLUE,
            font=("Microsoft YaHei UI", 9, "bold"), anchor="w",
            justify="left", wraplength=360,
        ).pack(fill=tk.X, padx=8, pady=(3, 1))
        tk.Label(
            self, textvariable=self.raw_var, bg=self.BG, fg=self.MUTED,
            font=("Consolas", 9), anchor="w", justify="left", wraplength=360,
        ).pack(fill=tk.X, padx=8)
        tk.Label(
            self, textvariable=self.stable_var, bg=self.BG, fg=self.TEXT,
            font=("Consolas", 10, "bold"), anchor="w",
            justify="left", wraplength=360,
        ).pack(fill=tk.X, padx=8, pady=(1, 8))

    def _emit(self, command: str) -> None:
        self.on_command(command, self.get_settings())

    def get_settings(self) -> dict:
        try:
            index = max(0, int(self.index_var.get()))
        except ValueError:
            index = 1
        try:
            roi = tuple(float(v.strip()) for v in self.roi_var.get().split(","))
            if len(roi) != 4:
                raise ValueError
        except ValueError:
            roi = (0.0, 0.0, 1.0, 1.0)
        return {
            **self._settings,
            "camera_index": index,
            "auto_roi": self.auto_roi_var.get(),
            "roi": roi,
        }

    def set_camera_list(self, cameras: list[int]) -> None:
        self.available_cameras = [int(value) for value in cameras]
        text = "、".join(str(v) for v in cameras) if cameras else "未检测到"
        self.camera_list_var.set(f"可用摄像头：{text}")

    def select_available_camera(self, excluded: set[int]) -> int | None:
        """选择未被其他模块占用的摄像头，并同步更新输入框。"""
        preferred = [int(self._settings.get("camera_index", 0))]
        candidates = self.available_cameras + preferred
        for index in candidates:
            if index >= 0 and index not in excluded:
                self.index_var.set(str(index))
                return index
        return None

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    @property
    def preview_photo(self):
        return self._preview_photo

    def update_result(self, result) -> None:
        self.status_var.set(result.message)
        score = "--" if result.score <= 0 else f"{result.score:.2f}"
        self.raw_var.set(f"单帧 {result.text or '--'}  │  置信度 {score}")
        if result.stable_value_mm is not None:
            timestamp = ""
            stable_at = (
                result.stable_captured_at
                if result.stable_captured_at is not None
                else result.captured_at if result.stable else None)
            if stable_at is not None:
                timestamp = datetime.fromtimestamp(stable_at).strftime("%H:%M:%S.%f")[:-3]
            suffix = f"  │  采集 {timestamp}" if timestamp else ""
            self.stable_var.set(
                f"稳定读数 {result.stable_value_mm:.{self._decimal_places}f} mm{suffix}")
        frame = result.frame
        if frame is None or frame.size == 0:
            frame = result.crop
        if frame is None or frame.size == 0:
            return
        preview = frame.copy()
        if result.frame is not None and result.roi_xyxy is not None:
            h, w = preview.shape[:2]
            x1, y1, x2, y2 = result.roi_xyxy
            x1, x2 = sorted((max(0, min(w - 1, int(x1))),
                             max(0, min(w - 1, int(x2)))))
            y1, y2 = sorted((max(0, min(h - 1, int(y1))),
                             max(0, min(h - 1, int(y2)))))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (46, 204, 113), 3)
            label = result.text or "LCD"
            cv2.putText(
                preview, label, (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (46, 204, 113), 2,
                cv2.LINE_AA,
            )
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB) \
            if preview.ndim == 3 else preview
        image = Image.fromarray(rgb)
        image.thumbnail((420, 280), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(image)
        # Label 的 height 在文字模式下按“行”计算，在图片模式下却按像素
        # 计算。保留初始化时的 height=14 会把整幅图片裁成一条横带。
        # 图片模式设为 0，让控件采用 PhotoImage 的自然高度完整显示。
        self.preview_label.configure(
            image=self._preview_photo, text="", height=0)

    def reset(self) -> None:
        self.status_var.set("尚未启动")
        self.raw_var.set("单帧 --  │  置信度 --")
        self.stable_var.set("稳定读数 -- mm")
        self._preview_photo = None
        self.preview_label.configure(
            image="", text="启动读数后显示第二摄像头画面", height=14)
