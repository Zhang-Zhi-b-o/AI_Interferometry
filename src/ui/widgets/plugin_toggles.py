"""插件开关栏 — 全开/全关 + 右键跳转"""
from __future__ import annotations
import tkinter as tk
from collections.abc import Callable

from src.ui.theme import BORDER, FONT, MUTED, PRIMARY, PRIMARY_SOFT, SURFACE, TEXT


def plugin_definitions(show_temporary: bool = True) -> list[tuple[str, str, bool]]:
    plugins = [
        ("vision", "视觉观察", True),
        ("motion", "运动控制", True),
        ("measurement", "测量记录", True),
        ("assistant", "实验助手", True),
    ]
    if show_temporary:
        plugins.append(("temporary", "临时测量", True))
    return plugins


class PluginToggleBar(tk.LabelFrame):
    """插件管理：复选框列表 + 快捷操作"""

    def __init__(self, parent: tk.Widget, *, show_temporary: bool = True):
        super().__init__(parent, text="模块导航", bg=SURFACE, fg=TEXT,
                         relief=tk.FLAT, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, font=(FONT, 9, "bold"))
        self._vars: dict[str, tk.BooleanVar] = {}
        self._callbacks: dict[str, Callable[[bool], None]] = {}
        self._jump_callbacks: dict[str, Callable[[], None]] = {}

        self._plugins = plugin_definitions(show_temporary)
        self._build()

    def _build(self):
        # 第一行：按钮 + 提示
        top = tk.Frame(self, bg=SURFACE)
        top.pack(fill=tk.X, padx=8, pady=(7, 4))
        for text, cmd in [("全开", self.enable_all), ("全关", self.disable_all)]:
            tk.Button(top, text=text, command=cmd,
                      relief=tk.FLAT, bd=0, bg=PRIMARY_SOFT, fg=PRIMARY,
                      activebackground="#dce8ff", cursor="hand2",
                      font=(FONT, 8), padx=8, pady=3).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(top, text="右键跳转 · 箭头排序", bg=SURFACE, fg=MUTED,
                 font=(FONT, 8)).pack(side=tk.RIGHT)

        # 第二行：可横向滚动的复选框，避免插件过多时被截断。
        scroll_shell = tk.Frame(self, bg=SURFACE)
        scroll_shell.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(scroll_shell, text="◀", command=lambda: self._scroll(-4),
                  relief=tk.FLAT, bd=0, bg=PRIMARY_SOFT, fg=PRIMARY,
                  cursor="hand2", width=2).pack(side=tk.LEFT)
        tk.Button(scroll_shell, text="▶", command=lambda: self._scroll(4),
                  relief=tk.FLAT, bd=0, bg=PRIMARY_SOFT, fg=PRIMARY,
                  cursor="hand2", width=2).pack(side=tk.RIGHT)

        canvas_shell = tk.Frame(scroll_shell, bg=SURFACE)
        canvas_shell.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self._canvas = tk.Canvas(
            canvas_shell, bg=SURFACE, height=28, highlightthickness=0, bd=0)
        self._canvas.pack(fill=tk.X, expand=True)
        self._scrollbar = tk.Scrollbar(
            canvas_shell, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._scrollbar.pack(fill=tk.X)
        self._canvas.configure(xscrollcommand=self._scrollbar.set)

        cb_row = tk.Frame(self._canvas, bg=SURFACE)
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
                bg=SURFACE, fg=TEXT, activebackground=SURFACE,
                selectcolor=PRIMARY_SOFT, activeforeground=PRIMARY,
                cursor="hand2", font=(FONT, 8))
            cb.pack(side=tk.LEFT, padx=4)
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
        if cb:
            cb(self._vars[key].get())

    def _on_jump(self, key: str):
        cb = self._jump_callbacks.get(key)
        if cb:
            cb()

    def bind_toggle(self, key: str, callback: Callable[[bool], None]):
        self._callbacks[key] = callback

    def bind_jump(self, key: str, callback: Callable[[], None]):
        self._jump_callbacks[key] = callback

    def is_enabled(self, key: str) -> bool:
        v = self._vars.get(key)
        return v.get() if v else False

    def enable_all(self):
        for key, var in self._vars.items():
            if not var.get():
                var.set(True)
                self._on_toggle(key)

    def disable_all(self):
        for key, var in self._vars.items():
            if var.get():
                var.set(False)
                self._on_toggle(key)
