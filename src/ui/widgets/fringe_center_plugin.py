"""中心条纹位置分析插件 — 在零级条纹预测框内定位条纹中心"""
from __future__ import annotations
import tkinter as tk


class FringeCenterPluginPanel(tk.LabelFrame):
    """实时检测零级条纹框内的中心位置，并在画面上画线"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="中心条纹分析", bg="#ffffff", fg="#000000")
        btn = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                   activebackground="#0b0b0b", cursor="hand2")

        # 自动检测开关
        self.auto_detect_var = tk.BooleanVar(value=False)

        # 中心线显示开关
        self.show_line_var = tk.BooleanVar(value=True)

        # 检测结果展示
        self.result_var = tk.StringVar(value="等待启动...")

        # -- 自动检测按钮 --
        tk.Button(self, text="自动检测中心条纹",
                  command=lambda: self._emit("toggle_auto"),
                  **btn).pack(fill=tk.X, padx=8, pady=(8, 2))

        # -- 显示/隐藏中心线 --
        tk.Checkbutton(self, text="显示中心线",
                       variable=self.show_line_var,
                       command=lambda: self._emit("toggle_line"),
                       bg="#fff", fg="#000",
                       activebackground="#fff",
                       selectcolor="#fff").pack(anchor="w", padx=8, pady=2)

        # -- 分隔 --
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)

        # -- 结果展示 --
        tk.Label(self, text="检测结果", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8)
        tk.Label(self, textvariable=self.result_var, bg="#fff", fg="#333",
                 anchor="w", justify="left",
                 font=("Consolas", 9)).pack(fill=tk.X, padx=8, pady=(0, 6))

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """cmd: toggle_auto / toggle_line"""
        pass

    def _emit(self, cmd: str):
        self.on_command(cmd)

    # ------------------------------------------------------------------
    # 更新结果
    # ------------------------------------------------------------------
    def update_result(self, center_x: float | None, confidence: float,
                      in_box: bool, msg: str = ""):
        """更新检测结果显示"""
        if center_x is not None and in_box:
            self.result_var.set(
                f"中心位置: {center_x:.1f} px\n"
                f"置信度: {confidence:.2f}\n"
                f"状态: 已锁定"
            )
        elif msg:
            self.result_var.set(f"状态: {msg}")
        else:
            self.result_var.set("状态: 未检测到零级条纹")

    def update_auto_state(self, enabled: bool):
        """更新按钮文字"""
        if enabled:
            self.configure(text="中心条纹分析 [运行中]")
        else:
            self.configure(text="中心条纹分析")
