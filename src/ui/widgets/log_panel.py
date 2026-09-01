"""运行日志面板"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext
from collections import deque
from datetime import datetime

from src.ui.theme import (
    BORDER,
    DANGER,
    FONT,
    MUTED,
    NAVY,
    PRIMARY,
    PRIMARY_SOFT,
    SURFACE,
    SURFACE_ALT,
    WARNING,
)


class LogPanel(tk.LabelFrame):
    """可滚动的运行日志"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="运行日志", bg=SURFACE, fg=NAVY,
                         relief=tk.FLAT, bd=0)

        toolbar = tk.Frame(self, bg=SURFACE)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))
        self.summary_var = tk.StringVar(value="0 条 · 错误 0 · 警告 0")
        tk.Label(
            toolbar, textvariable=self.summary_var, bg=SURFACE, fg=MUTED,
            font=(FONT, 8),
        ).pack(side=tk.LEFT)
        tk.Button(
            toolbar, text="清空", command=self.clear, relief=tk.FLAT, bd=0,
            bg=PRIMARY_SOFT, fg=PRIMARY, activebackground="#dce8ff",
            cursor="hand2", font=(FONT, 8), padx=8, pady=2,
        ).pack(side=tk.RIGHT)

        self._text = scrolledtext.ScrolledText(
            self, height=10, bg=SURFACE_ALT, fg=NAVY,
            insertbackground=NAVY, relief=tk.FLAT, bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            font=("Consolas", 9), padx=8, pady=7, state=tk.DISABLED)
        self._text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._text.tag_configure("info", foreground=NAVY)
        self._text.tag_configure("warning", foreground=WARNING)
        self._text.tag_configure("error", foreground=DANGER)
        self._entries: deque[dict] = deque(maxlen=500)

    @staticmethod
    def classify_level(message: str) -> str:
        """从统一日志前缀推断显示级别。"""
        if "[错误]" in message:
            return "error"
        if "[警告]" in message:
            return "warning"
        return "info"

    def _update_summary(self) -> None:
        errors = sum(item["level"] == "error" for item in self._entries)
        warnings = sum(item["level"] == "warning" for item in self._entries)
        self.summary_var.set(
            f"{len(self._entries)} 条 · 错误 {errors} · 警告 {warnings}")

    def write(self, msg: str):
        """追加一条日志"""
        timestamp = time.strftime("%H:%M:%S")
        message = str(msg)
        level = self.classify_level(message)
        self._entries.append({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        })
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, f"{timestamp} {message}\n", level)
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)
        self._update_summary()

    def recent_entries(self, limit: int = 80) -> list[dict]:
        """返回供实验助手读取的近期结构化日志副本。"""
        count = max(1, min(int(limit), self._entries.maxlen or 500))
        return [dict(item) for item in list(self._entries)[-count:]]

    def clear(self):
        self._entries.clear()
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
        self._update_summary()
