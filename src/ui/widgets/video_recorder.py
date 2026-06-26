"""视频录制面板"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from src.logging import logger


class VideoRecorderPanel(tk.LabelFrame):
    """视频录制控制：路径选择、录制源、开始/停止、帧率"""

    UI_BG = "#ffffff"
    UI_TEXT = "#000000"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="视频录制", bg=self.UI_BG, fg=self.UI_TEXT)
        self.configure(bg=self.UI_BG, fg=self.UI_TEXT)

        # 状态 — 由外部设置
        self.recording = False
        self.recording_source = "preview"
        self.video_writer = None

        # Tk 变量
        self.path_var = tk.StringVar(value="recorded_videos/output.avi")
        self.source_var = tk.StringVar(value="preview")
        self.fps_var = tk.StringVar(value="20")

        self._build()

    def _build(self):
        btn_cfg = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                       activebackground="#0b0b0b", cursor="hand2")

        # 保存路径
        rp = tk.Frame(self, bg=self.UI_BG)
        rp.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(rp, text="保存路径", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(rp, textvariable=self.path_var, width=34).pack(
            side=tk.LEFT, padx=(8, 6), fill=tk.X, expand=True)
        tk.Button(rp, text="选择", command=self._choose_path, **btn_cfg).pack(side=tk.LEFT)

        # 录制源
        rs = tk.Frame(self, bg=self.UI_BG)
        rs.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(rs, text="录制内容", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Radiobutton(rs, text="预览画面", variable=self.source_var, value="preview",
                       bg=self.UI_BG, fg=self.UI_TEXT, activebackground=self.UI_BG,
                       selectcolor=self.UI_BG).pack(side=tk.LEFT, padx=(8, 6))
        tk.Radiobutton(rs, text="相机实际画面", variable=self.source_var, value="camera",
                       bg=self.UI_BG, fg=self.UI_TEXT, activebackground=self.UI_BG,
                       selectcolor=self.UI_BG).pack(side=tk.LEFT)

        # 控制
        rc = tk.Frame(self, bg=self.UI_BG)
        rc.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(rc, text="fps", bg=self.UI_BG, fg=self.UI_TEXT).pack(side=tk.LEFT)
        tk.Entry(rc, textvariable=self.fps_var, width=6).pack(side=tk.LEFT, padx=(8, 10))
        tk.Button(rc, text="开始录制", command=self.start, **btn_cfg).pack(side=tk.LEFT)
        tk.Button(rc, text="停止录制", command=self.stop, **btn_cfg).pack(
            side=tk.LEFT, padx=(6, 0))

    # ------------------------------------------------------------------
    # 回调（由外部注入）
    # ------------------------------------------------------------------
    def on_start(self, path: str, fps: float, source: str):
        """外部注入：开始录制回调"""
        pass

    def on_stop(self):
        """外部注入：停止录制回调"""
        pass

    def on_write_frame(self, frame):
        """外部注入：写帧回调"""
        pass

    # ------------------------------------------------------------------
    # 按钮事件
    # ------------------------------------------------------------------
    def _choose_path(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".avi", filetypes=[("AVI", "*.avi"), ("MP4", "*.mp4")])
        if path:
            self.path_var.set(path)

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.recording_source = self.source_var.get()
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            fps = 20
        self.on_start(self.path_var.get(), fps, self.recording_source)

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self.on_stop()
