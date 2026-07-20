"""摄像头控制 + 画面控制插件"""
from __future__ import annotations
import tkinter as tk


class CameraPluginPanel(tk.LabelFrame):
    """摄像头控制 + 画面旋转/缩放/平移（鼠标拖拽）"""

    def __init__(
        self,
        parent: tk.Widget,
        default_index: int = 1,
        clarity_settings: dict | None = None,
    ):
        super().__init__(parent, text="摄像头与画面控制", bg="#ffffff", fg="#000000")
        clarity_settings = clarity_settings or {}
        btn = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                   activebackground="#0b0b0b", cursor="hand2")
        sm_btn = dict(relief=tk.FLAT, bd=0, bg="#444444", fg="#ffffff",
                      activebackground="#333333", cursor="hand2")

        # -- 摄像头索引 --
        self.index_var = tk.StringVar(value=str(default_index))
        r = tk.Frame(self, bg="#fff")
        r.pack(fill=tk.X, padx=8, pady=(8,4))
        tk.Label(r, text="摄像头索引", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=self.index_var, width=6).pack(side=tk.LEFT, padx=(8,0))

        for text, cmd in [("检测所有摄像头","detect"),("打开摄像头","open"),("关闭摄像头","close")]:
            tk.Button(self, text=text, command=lambda c=cmd: self._emit(c), **btn).pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)

        # -- 画面旋转角度 --
        self.angle_var = tk.StringVar(value="0")
        ar = tk.Frame(self, bg="#fff")
        ar.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(ar, text="旋转角度", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Entry(ar, textvariable=self.angle_var, width=6).pack(side=tk.LEFT, padx=(8,6))
        tk.Button(ar, text="应用", command=lambda: self._emit("angle_apply"), **sm_btn).pack(side=tk.LEFT)
        tk.Button(ar, text="归零", command=lambda: self._emit("angle_reset"), **sm_btn).pack(side=tk.LEFT, padx=(4,0))

        # -- 缩放 --
        self.zoom_var = tk.StringVar(value="2.0")
        zr = tk.Frame(self, bg="#fff")
        zr.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(zr, text="缩放倍数", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Spinbox(zr, from_=1.0, to=8.0, increment=0.1, textvariable=self.zoom_var,
                   width=5, format="%.1f").pack(side=tk.LEFT, padx=(8,0))
        tk.Button(zr, text="应用", command=lambda: self._emit("zoom_apply"), **sm_btn).pack(side=tk.LEFT, padx=(8,0))
        tk.Button(zr, text="复位", command=lambda: self._emit("zoom_reset"), **sm_btn).pack(side=tk.LEFT, padx=(4,0))

        # -- 平移（鼠标拖拽） --
        self.pan_mode_var = tk.BooleanVar(value=False)
        pr = tk.Frame(self, bg="#fff")
        pr.pack(fill=tk.X, padx=8, pady=(2,6))
        tk.Checkbutton(pr, text="鼠标拖拽平移画面", variable=self.pan_mode_var,
                       bg="#fff", fg="#000", activebackground="#fff", selectcolor="#fff").pack(side=tk.LEFT)
        tk.Button(pr, text="复位平移", command=lambda: self._emit("pan_reset"), **sm_btn).pack(side=tk.LEFT, padx=(8,0))

        # -- 清晰度配置 --
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)
        clarity_header = tk.Frame(self, bg="#fff")
        clarity_header.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            clarity_header, text="清晰度配置", bg="#fff", fg="#000",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side=tk.LEFT)
        self.motion_enhance_var = tk.BooleanVar(value=bool(
            clarity_settings.get("motion_enabled_by_default", False)))
        tk.Checkbutton(
            clarity_header, text="运动增强", variable=self.motion_enhance_var,
            command=lambda: self._emit("clarity_toggle"), bg="#fff", fg="#000",
            activebackground="#fff", selectcolor="#fff",
        ).pack(side=tk.RIGHT)

        clarity_row = tk.Frame(self, bg="#fff")
        clarity_row.pack(fill=tk.X, padx=8, pady=2)
        self.motion_exposure_var = tk.StringVar(value=str(float(
            clarity_settings.get("motion_exposure", -7.0))))
        self.motion_gain_var = tk.StringVar(value=str(float(
            clarity_settings.get("motion_gain", 80.0))))
        tk.Label(clarity_row, text="曝光", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Spinbox(
            clarity_row, from_=-20.0, to=0.0, increment=0.1,
            textvariable=self.motion_exposure_var, width=6, format="%.1f",
        ).pack(side=tk.LEFT, padx=(5, 10))
        tk.Label(clarity_row, text="增益", bg="#fff", fg="#000").pack(side=tk.LEFT)
        tk.Spinbox(
            clarity_row, from_=0.0, to=255.0, increment=1.0,
            textvariable=self.motion_gain_var, width=6, format="%.1f",
        ).pack(side=tk.LEFT, padx=(5, 8))
        tk.Button(
            clarity_row, text="应用", command=lambda: self._emit("clarity_apply"),
            **sm_btn,
        ).pack(side=tk.LEFT)
        self.clarity_status_var = tk.StringVar(
            value=(
                f"未开启 · 曝光 {float(clarity_settings.get('motion_exposure', -7.0)):.1f}"
                f" · 增益 {float(clarity_settings.get('motion_gain', 80.0)):.1f}"
            )
        )
        tk.Label(
            self, textvariable=self.clarity_status_var, bg="#fff", fg="#666",
            anchor="w",
        ).pack(fill=tk.X, padx=9, pady=(0, 5))

        # -- 全部复位 --
        tk.Button(self, text="全部画面参数复位", command=lambda: self._emit("all_reset"), **btn).pack(fill=tk.X, padx=8, pady=2)

    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """处理摄像头、画面矫正和清晰度配置命令。"""
        pass

    def _emit(self, cmd: str):
        self.on_command(cmd)

    @property
    def camera_index(self) -> int:
        try: return int(self.index_var.get())
        except ValueError: return 0

    @property
    def angle(self) -> float:
        try: return float(self.angle_var.get())
        except ValueError: return 0.0

    @property
    def zoom(self) -> float:
        try: return float(self.zoom_var.get())
        except ValueError: return 1.0

    @property
    def pan_mode(self) -> bool:
        return self.pan_mode_var.get()

    @property
    def motion_enhance_enabled(self) -> bool:
        return bool(self.motion_enhance_var.get())

    @property
    def motion_exposure(self) -> float:
        try:
            return min(0.0, max(-20.0, float(self.motion_exposure_var.get())))
        except ValueError:
            return -7.0

    @property
    def motion_gain(self) -> float:
        try:
            return min(255.0, max(0.0, float(self.motion_gain_var.get())))
        except ValueError:
            return 80.0

    def reset_clarity(self) -> None:
        self.motion_enhance_var.set(False)
        self.motion_exposure_var.set("-7.0")
        self.motion_gain_var.set("80.0")

    def set_clarity_status(self, text: str) -> None:
        self.clarity_status_var.set(text)
