"""运行日志面板"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext
from collections import deque
from datetime import datetime

from src.ui.theme import BORDER, NAVY, SURFACE, SURFACE_ALT


class LogPanel(tk.LabelFrame):
    """可滚动的运行日志"""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="运行日志", bg=SURFACE, fg=NAVY,
                         relief=tk.FLAT, bd=0)

        self._text = scrolledtext.ScrolledText(
            self, height=10, bg=SURFACE_ALT, fg=NAVY,
            insertbackground=NAVY, relief=tk.FLAT, bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            font=("Consolas", 9), padx=8, pady=7)
        self._text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._entries: deque[dict] = deque(maxlen=500)

    def write(self, msg: str):
        """追加一条日志"""
        timestamp = time.strftime("%H:%M:%S")
        message = str(msg)
        level = (
            "error" if "[错误]" in message else
            "warning" if "[警告]" in message else
            "info")
        self._entries.append({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        })
        self._text.insert(tk.END, f"{timestamp} {message}\n")
        self._text.see(tk.END)

    def recent_entries(self, limit: int = 80) -> list[dict]:
        """返回供实验助手读取的近期结构化日志副本。"""
        count = max(1, min(int(limit), self._entries.maxlen or 500))
        return [dict(item) for item in list(self._entries)[-count:]]

    def clear(self):
        self._text.delete("1.0", tk.END)
