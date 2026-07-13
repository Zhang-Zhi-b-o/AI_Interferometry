"""可折叠面板基类 — 支持折叠/展开 + ▲▼移动排序"""
from __future__ import annotations

import tkinter as tk

from src.ui.theme import BORDER, FONT, MUTED, PRIMARY, SURFACE, TEXT


class CollapsibleFrame(tk.Frame):
    """带折叠和排序功能的面板容器"""

    COLLAPSED_SYMBOL = "▸"
    EXPANDED_SYMBOL = "▾"

    def __init__(self, parent: tk.Widget, title: str, bg: str = SURFACE,
                 fg: str = TEXT, collapsed: bool = False):
        super().__init__(parent, bg=bg, bd=0, relief=tk.FLAT,
                         highlightthickness=1, highlightbackground=BORDER)

        self._bg = bg
        self._fg = fg
        self._collapsed = collapsed

        # 标题栏
        self._title_bar = tk.Frame(self, bg=bg, height=42)
        self._title_bar.pack(fill=tk.X, padx=8)
        self._title_bar.pack_propagate(False)

        # 折叠按钮
        self._toggle_btn = tk.Label(
            self._title_bar, text=self.EXPANDED_SYMBOL if not collapsed else self.COLLAPSED_SYMBOL,
            bg=bg, fg=PRIMARY, font=("Consolas", 12, "bold"), cursor="hand2")
        self._toggle_btn.pack(side=tk.LEFT, padx=(2, 7), fill=tk.Y)
        self._toggle_btn.bind("<Button-1>", lambda e: self.toggle())

        # 标题
        self._title_label = tk.Label(
            self._title_bar, text=title, bg=bg, fg=fg,
            font=(FONT, 10, "bold"), cursor="hand2")
        self._title_label.pack(side=tk.LEFT, fill=tk.Y)
        self._title_label.bind("<Button-1>", lambda e: self.toggle())

        # ▲▼ 排序按钮（右侧）
        self._up_btn = tk.Label(self._title_bar, text="↑", bg=bg, fg=MUTED,
                                 font=("Segoe UI", 9), cursor="hand2")
        self._up_btn.pack(side=tk.RIGHT, padx=(2, 4), fill=tk.Y)
        self._up_btn.bind("<Button-1>", lambda e: self._emit_move("up"))

        self._down_btn = tk.Label(self._title_bar, text="↓", bg=bg, fg=MUTED,
                                   font=("Segoe UI", 9), cursor="hand2")
        self._down_btn.pack(side=tk.RIGHT, padx=2, fill=tk.Y)
        self._down_btn.bind("<Button-1>", lambda e: self._emit_move("down"))

        # 内容区
        self._content = tk.Frame(self, bg=bg, highlightthickness=0)
        if not collapsed:
            self._content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def on_move(self, direction: str):
        """外部注入：direction = 'up' / 'down'"""
        pass

    def _emit_move(self, direction: str):
        self.on_move(direction)

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------
    def toggle(self):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._content.pack_forget()
            self._toggle_btn.configure(text=self.COLLAPSED_SYMBOL)
        else:
            self._content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            self._toggle_btn.configure(text=self.EXPANDED_SYMBOL)

    @property
    def content(self) -> tk.Frame:
        return self._content

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    @property
    def title(self) -> str:
        return self._title_label.cget("text")
