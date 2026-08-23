"""Tkinter 网页仪表盘风格主题。"""
from __future__ import annotations

import tkinter as tk


APP_BG = "#f3f6fb"
SURFACE = "#ffffff"
SURFACE_ALT = "#f8fafc"
BORDER = "#dfe7f1"
TEXT = "#182230"
MUTED = "#667085"
PRIMARY = "#2563eb"
PRIMARY_HOVER = "#1d4ed8"
PRIMARY_SOFT = "#eaf1ff"
SUCCESS = "#12a150"
NAVY = "#102a43"
VIDEO_BG = "#08111f"
FONT = "Microsoft YaHei UI"


_LEGACY_SURFACES = {"#fff", "#ffffff", "white", "#fafafa", "#f4f7fb"}
_LEGACY_PRIMARY = {"#111111", "#0b0b0b", "#444444", "#444", "#333333", "#333"}
_LEGACY_SECONDARY = {"#e6e6e6", "#eeeeee", "#eee", "#dce3ec"}


def primary_button(**overrides) -> dict:
    config = dict(
        relief=tk.FLAT,
        bd=0,
        bg=PRIMARY,
        fg="#ffffff",
        activebackground=PRIMARY_HOVER,
        activeforeground="#ffffff",
        cursor="hand2",
        font=(FONT, 10, "bold"),
        padx=12,
        pady=6,
    )
    config.update(overrides)
    return config


def secondary_button(**overrides) -> dict:
    config = dict(
        relief=tk.FLAT,
        bd=0,
        bg=PRIMARY_SOFT,
        fg=PRIMARY,
        activebackground="#dce8ff",
        activeforeground=PRIMARY_HOVER,
        cursor="hand2",
        font=(FONT, 10),
        padx=10,
        pady=5,
    )
    config.update(overrides)
    return config


def _colour(widget: tk.Widget, option: str, fallback: str = "") -> str:
    try:
        return str(widget.cget(option)).lower()
    except (tk.TclError, AttributeError):
        return fallback


def style_legacy_tree(widget: tk.Widget) -> None:
    """把旧插件的黑白控件映射到统一网页主题，保留语义状态色。"""
    if isinstance(widget, tk.LabelFrame):
        widget.configure(
            bg=SURFACE,
            fg=TEXT,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            font=(FONT, 10, "bold"),
            padx=2,
            pady=2,
        )
    for child in widget.winfo_children():
        bg = _colour(child, "background")
        parent_bg = _colour(child.master, "background", SURFACE)

        if isinstance(child, tk.LabelFrame):
            child.configure(
                bg=SURFACE,
                fg=TEXT,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
                font=(FONT, 10, "bold"),
                padx=2,
                pady=2,
            )
        elif isinstance(child, tk.Frame) and bg in _LEGACY_SURFACES:
            child.configure(bg=SURFACE if parent_bg != APP_BG else APP_BG)
        elif isinstance(child, tk.Button):
            if bg in _LEGACY_PRIMARY:
                child.configure(**primary_button())
            elif bg in _LEGACY_SECONDARY:
                child.configure(**secondary_button())
        elif isinstance(child, (tk.Checkbutton, tk.Radiobutton)):
            if bg in _LEGACY_SURFACES:
                surface = _colour(child.master, "background", SURFACE)
                child.configure(
                    bg=surface,
                    fg=TEXT,
                    activebackground=surface,
                    activeforeground=TEXT,
                    selectcolor=SURFACE_ALT,
                    font=(FONT, 10),
                )
        elif isinstance(child, tk.Label) and bg in _LEGACY_SURFACES:
            child.configure(
                bg=_colour(child.master, "background", SURFACE),
                fg=MUTED if _colour(child, "foreground") in {"#666", "#666666", "#999999", "#999"} else TEXT,
                font=(FONT, 10),
            )
        elif isinstance(child, (tk.Entry, tk.Spinbox)):
            entry_style = {
                "relief": tk.FLAT,
                "bd": 0,
                "bg": SURFACE_ALT,
                "fg": TEXT,
                "insertbackground": TEXT,
                "highlightthickness": 1,
                "highlightbackground": BORDER,
                "highlightcolor": PRIMARY,
                "font": (FONT, 10),
            }
            # ttk.Combobox 在 Python 侧继承 Entry，但 Tcl 端不支持全部 Tk 参数。
            supported = set(child.keys())
            child.configure(**{key: value for key, value in entry_style.items()
                               if key in supported})
        elif isinstance(child, tk.OptionMenu):
            child.configure(
                relief=tk.FLAT,
                bd=0,
                bg=SURFACE_ALT,
                fg=TEXT,
                activebackground=PRIMARY_SOFT,
                highlightthickness=1,
                highlightbackground=BORDER,
                font=(FONT, 10),
            )
            child["menu"].configure(bg=SURFACE, fg=TEXT, font=(FONT, 10))

        style_legacy_tree(child)
