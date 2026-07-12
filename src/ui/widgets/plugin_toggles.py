"""插件开关栏 — 全开/全关 + 右键跳转"""
from __future__ import annotations
import tkinter as tk


class PluginToggleBar(tk.LabelFrame):
    """插件管理：复选框列表 + 快捷操作"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="插件管理", bg="#ffffff", fg="#000000")
        self._vars: dict[str, tk.BooleanVar] = {}
        self._callbacks: dict[str, callable] = {}
        self._jump_callbacks: dict[str, callable] = {}

        self._plugins = [
            ("camera", "摄像头", True),
            ("model", "模型预测", True),
            ("fringe_center", "中心条纹分析", True),
            ("recorder", "视频录制", True),
            ("status", "实时状态", True),
            ("motor", "电机控制", True),
            ("agent", "实验助手", True),
            ("log", "运行日志", True),
        ]
        self._build()

    def _build(self):
        # 第一行：按钮 + 提示
        top = tk.Frame(self, bg="#fff")
        top.pack(fill=tk.X, padx=4, pady=(4, 2))
        for text, cmd in [("全开", self.enable_all), ("全关", self.disable_all)]:
            tk.Button(top, text=text, command=cmd,
                      relief=tk.FLAT, bd=0, bg="#e6e6e6", fg="#000",
                      activebackground="#d8d8d8", cursor="hand2",
                      font=("Microsoft YaHei UI", 9)).pack(side=tk.LEFT, padx=2)
        tk.Label(top, text="右键=跳转 | ▲▼=排序", bg="#fff", fg="#999",
                 font=("Microsoft YaHei UI", 8)).pack(side=tk.RIGHT)

        # 第二行：可横向滚动的复选框，避免插件过多时被截断。
        scroll_shell = tk.Frame(self, bg="#fff")
        scroll_shell.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(scroll_shell, text="◀", command=lambda: self._scroll(-4),
                  relief=tk.FLAT, bd=0, bg="#eee", cursor="hand2", width=2).pack(side=tk.LEFT)
        tk.Button(scroll_shell, text="▶", command=lambda: self._scroll(4),
                  relief=tk.FLAT, bd=0, bg="#eee", cursor="hand2", width=2).pack(side=tk.RIGHT)

        canvas_shell = tk.Frame(scroll_shell, bg="#fff")
        canvas_shell.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self._canvas = tk.Canvas(
            canvas_shell, bg="#fff", height=28, highlightthickness=0, bd=0)
        self._canvas.pack(fill=tk.X, expand=True)
        self._scrollbar = tk.Scrollbar(
            canvas_shell, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._scrollbar.pack(fill=tk.X)
        self._canvas.configure(xscrollcommand=self._scrollbar.set)

        cb_row = tk.Frame(self._canvas, bg="#fff")
        self._checkbox_row = cb_row
        self._window_id = self._canvas.create_window((0, 0), window=cb_row, anchor="nw")
        cb_row.bind("<Configure>", self._update_scroll_region)
        self._canvas.bind("<Configure>", self._update_viewport)
        for widget in (self._canvas, cb_row):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Shift-MouseWheel>", self._on_mousewheel)

        for key, label, default in self._plugins:
            var = tk.BooleanVar(value=default)
            self._vars[key] = var
            cb = tk.Checkbutton(
                cb_row, text=label, variable=var,
                command=lambda k=key: self._on_toggle(k),
                bg="#fff", fg="#000", activebackground="#fff", selectcolor="#fff",
                cursor="hand2")
            cb.pack(side=tk.LEFT, padx=2)
            cb.bind("<Button-3>", lambda e, k=key: self._on_jump(k))
            cb.bind("<MouseWheel>", self._on_mousewheel)
            cb.bind("<Shift-MouseWheel>", self._on_mousewheel)

    def _update_scroll_region(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _update_viewport(self, event):
        self._canvas.itemconfigure(
            self._window_id,
            width=max(event.width, self._checkbox_row.winfo_reqwidth()),
        )

    def _scroll(self, units: int):
        self._canvas.xview_scroll(units, "units")

    def _on_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self._scroll(direction * 3)
        return "break"

    # ------------------------------------------------------------------
    def _on_toggle(self, key: str):
        cb = self._callbacks.get(key)
        if cb: cb(self._vars[key].get())

    def _on_jump(self, key: str):
        cb = self._jump_callbacks.get(key)
        if cb: cb()

    def bind_toggle(self, key: str, callback: callable):
        self._callbacks[key] = callback

    def bind_jump(self, key: str, callback: callable):
        self._jump_callbacks[key] = callback

    def is_enabled(self, key: str) -> bool:
        v = self._vars.get(key)
        return v.get() if v else False

    def enable_all(self):
        for key, var in self._vars.items():
            var.set(True); self._on_toggle(key)

    def disable_all(self):
        for key, var in self._vars.items():
            var.set(False); self._on_toggle(key)
