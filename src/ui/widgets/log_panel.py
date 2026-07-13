"""运行日志面板"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext

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

    def write(self, msg: str):
        """追加一条日志"""
        timestamp = time.strftime("%H:%M:%S")
        self._text.insert(tk.END, f"{timestamp} {msg}\n")
        self._text.see(tk.END)

    def clear(self):
        self._text.delete("1.0", tk.END)
