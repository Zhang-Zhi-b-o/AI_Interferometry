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
            ("recorder", "视频录制", True),
            ("status", "实时状态", True),
            ("motor", "电机控制", True),
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

        # 第二行：复选框
        cb_row = tk.Frame(self, bg="#fff")
        cb_row.pack(fill=tk.X, padx=4, pady=(0, 4))

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
