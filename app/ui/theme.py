"""主题：仿 DeepSeek Harness 的中性偏蓝配色 + 大字号。

取色来源：DSH 前端的 ``--dsw-static-*`` 设计令牌（neutral-bluish / deepseek /
red / green / amber 五组），本文件把它们映射成 Qt 的调色板与 QSS。

三件事想改外观时只改这里：
- ``PALETTES``：颜色；
- ``FONT_STACK`` / ``MONO_STACK``：字体；
- ``SIZES``：字号阶梯（正文 15px 起，保证不刺眼又看得清）。
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
from PySide6.QtWidgets import QApplication

FONT_STACK = ['"Microsoft YaHei UI"', '"Microsoft YaHei"', '"Segoe UI"',
              '"PingFang SC"', "sans-serif"]
MONO_STACK = ['"JetBrains Mono"', "Consolas", '"SF Mono"', "monospace"]

#: 基准字号（px）。``ui.font_scale`` 会整体乘上去。
SIZES = {
    "display": 34,
    "big": 26,
    "title": 18,
    "subtitle": 16,
    "body": 15,
    "label": 14,
    "small": 13,
}

LIGHT = {
    "bg": "#f9fafb",
    "surface": "#ffffff",
    "surface_alt": "#f5f6f7",
    "surface_sunken": "#f1f3f5",
    "border": "#e1e5ee",
    "border_strong": "#cfd3d6",
    "text": "#1b1b1c",
    "text_secondary": "#61666b",
    "text_muted": "#979da6",
    "primary": "#4176e6",
    "primary_hover": "#5686fe",
    "primary_pressed": "#3462c4",
    "primary_soft": "#edf3fe",
    "on_primary": "#ffffff",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "danger_hover": "#f25a5a",
    "video_bg": "#11161f",
    "shadow": "rgba(15, 17, 21, 0.06)",
}

DARK = {
    "bg": "#151517",
    "surface": "#1b1b1c",
    "surface_alt": "#232324",
    "surface_sunken": "#232324",
    "border": "#2c2c2e",
    "border_strong": "#43454a",
    "text": "#ebeef2",
    "text_secondary": "#adb2b8",
    "text_muted": "#81858c",
    "primary": "#5686fe",
    "primary_hover": "#679efe",
    "primary_pressed": "#4176e6",
    "primary_soft": "#1d2a44",
    "on_primary": "#ffffff",
    "success": "#4ed17e",
    "warning": "#f7ad31",
    "danger": "#f25a5a",
    "danger_hover": "#ff7b7b",
    "video_bg": "#0c0f14",
    "shadow": "rgba(0, 0, 0, 0.4)",
}


@dataclass(frozen=True)
class Theme:
    name: str
    colors: dict
    scale: float

    def size(self, key: str) -> int:
        return max(11, int(round(SIZES[key] * self.scale)))

    def color(self, key: str) -> str:
        return self.colors.get(key, "#ff00ff")


def build_theme(name: str = "light", scale: float = 1.0) -> Theme:
    return Theme(name="dark" if name == "dark" else "light",
                 colors=DARK if name == "dark" else LIGHT,
                 scale=float(scale))


def _font_stack() -> str:
    return ", ".join(FONT_STACK)


def _mono_stack() -> str:
    return ", ".join(MONO_STACK)


def stylesheet(theme: Theme) -> str:
    c = theme.colors
    s = theme.size
    return f"""
/* ---------- 基础 ---------- */
QWidget {{
    background: transparent;
    color: {c['text']};
    font-family: {_font_stack()};
    font-size: {s('body')}px;
}}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QToolTip {{
    background-color: {c['surface']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 6px 8px;
    font-size: {s('label')}px;
    border-radius: 3px;
}}
#VideoTelemetry {{ background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 8px; }}

/* ---------- 顶栏 / 导航 ---------- */
#TopBar {{
    background: {c['surface']};
    border-bottom: 1px solid {c['border']};
}}
#NavRail {{
    background: {c['surface_alt']};
    border-right: 1px solid {c['border']};
}}
#AppTitle {{
    font-size: {s('title')}px;
    font-weight: 700;
    color: {c['text']};
}}
#AppSubtitle {{
    font-size: {s('small')}px;
    color: {c['text_muted']};
    letter-spacing: 1px;
}}
#Footer {{
    background: {c['surface']};
    border-top: 1px solid {c['border']};
    color: {c['text_secondary']};
    font-size: {s('small')}px;
}}

QPushButton#NavButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: left;
    font-size: {s('subtitle')}px;
    color: {c['text_secondary']};
}}
QPushButton#NavButton:hover {{
    background: {c['surface']};
    color: {c['text']};
}}
QPushButton#NavButton:checked {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    color: {c['primary']};
    font-weight: 600;
}}

/* ---------- 卡片 ---------- */
#Card {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 14px;
}}
#CardTitle {{
    font-size: {s('subtitle')}px;
    font-weight: 600;
    color: {c['text']};
}}
#CardHint {{
    font-size: {s('small')}px;
    color: {c['text_muted']};
}}
#PageTitle {{
    font-size: {s('title')}px;
    font-weight: 700;
}}
#PageHint {{
    font-size: {s('label')}px;
    color: {c['text_secondary']};
}}
#SectionLabel {{
    font-size: {s('label')}px;
    font-weight: 600;
    color: {c['text_secondary']};
}}
#Muted {{ color: {c['text_muted']}; font-size: {s('small')}px; }}
#Hint {{ color: {c['text_secondary']}; font-size: {s('label')}px; }}

/* ---------- 指标 ---------- */
#MetricLabel {{
    font-size: {s('small')}px;
    color: {c['text_muted']};
}}
#MetricValue {{
    font-family: {_mono_stack()};
    font-size: {s('big')}px;
    font-weight: 600;
    color: {c['text']};
}}
#MetricUnit {{
    font-size: {s('small')}px;
    color: {c['text_muted']};
}}
#BigValue {{
    font-family: {_mono_stack()};
    font-size: {s('display')}px;
    font-weight: 700;
    color: {c['primary']};
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {c['surface']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
    border-radius: 10px;
    padding: 9px 16px;
    font-size: {s('body')}px;
}}
QPushButton:hover {{ background: {c['surface_alt']}; }}
QPushButton:disabled {{
    color: {c['text_muted']};
    border-color: {c['border']};
    background: {c['surface_alt']};
}}
QPushButton[variant="primary"] {{
    background: {c['primary']};
    color: {c['on_primary']};
    border: 1px solid {c['primary']};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {c['primary_hover']}; }}
QPushButton[variant="primary"]:pressed {{ background: {c['primary_pressed']}; }}
QPushButton[variant="primary"]:disabled {{
    background: {c['surface_alt']};
    color: {c['text_muted']};
    border: 1px solid {c['border']};
}}
QPushButton[variant="secondary"]:disabled {{
    color: {c['text_muted']};
    background: {c['surface_alt']};
    border: 1px solid {c['border']};
}}
QPushButton[variant="danger"] {{
    background: {c['danger']};
    color: #ffffff;
    border: 1px solid {c['danger']};
    font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{ background: {c['danger_hover']}; }}
QPushButton[variant="danger"]:disabled {{
    background: {c['surface_alt']};
    color: {c['text_muted']};
    border: 1px solid {c['border']};
}}
QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid transparent;
    color: {c['primary']};
    padding: 6px 10px;
}}
QPushButton[variant="ghost"]:hover {{ background: {c['primary_soft']}; }}
QPushButton[variant="chip"] {{
    background: {c['surface_alt']};
    border: 1px solid {c['border']};
    border-radius: 14px;
    padding: 5px 12px;
    font-size: {s('small')}px;
    color: {c['text_secondary']};
}}
QPushButton[variant="chip"]:hover {{ background: {c['primary_soft']}; color: {c['primary']}; }}

/* ---------- 输入 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {c['surface']};
    border: 1px solid {c['border_strong']};
    border-radius: 9px;
    padding: 8px 10px;
    font-size: {s('body')}px;
    min-height: 22px;
    selection-background-color: {c['primary']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {c['primary']};
}}
/* 数字输入框隐藏上下箭头：挡字又难看，滚轮和键盘照样能改 */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0; height: 0; border: none;
}}
QComboBox QAbstractItemView {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    selection-background-color: {c['primary_soft']};
    selection-color: {c['primary']};
    outline: none;
    padding: 4px;
}}
QCheckBox, QRadioButton {{    font-size: {s('label')}px;
    spacing: 8px;
    color: {c['text']};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c['border_strong']};
    border-radius: 4px;
    background: {c['surface']};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {c['primary']};
    border-color: {c['primary']};
}}
QSlider::groove:horizontal {{
    height: 4px; border-radius: 2px; background: {c['surface_sunken']};
}}
QSlider::handle:horizontal {{
    width: 14px; margin: -6px 0; border-radius: 7px;
    background: {c['primary']};
}}

/* ---------- 页签 ---------- */
QTabWidget::pane {{
    border: 1px solid {c['border']};
    border-radius: 12px;
    background: {c['surface']};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {c['text_secondary']};
    border: none;
    padding: 9px 18px;
    margin-right: 4px;
    font-size: {s('body')}px;
    border-radius: 9px;
}}
QTabBar::tab:selected {{
    background: {c['primary_soft']};
    color: {c['primary']};
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {c['text']}; }}

/* ---------- 表格 ---------- */
QTableWidget, QTableView {{
    background: {c['surface']};
    alternate-background-color: {c['surface_alt']};
    border: 1px solid {c['border']};
    border-radius: 10px;
    gridline-color: {c['border']};
    font-size: {s('label')}px;
    selection-background-color: {c['primary_soft']};
    selection-color: {c['primary']};
}}
QHeaderView::section {{
    background: {c['surface_alt']};
    color: {c['text_secondary']};
    border: none;
    border-bottom: 1px solid {c['border']};
    padding: 8px 6px;
    font-size: {s('small')}px;
    font-weight: 600;
}}
QTableWidget::item {{ padding: 6px 4px; }}

/* ---------- 文本 / 滚动 ---------- */
QTextBrowser, QTextEdit, QPlainTextEdit {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 10px;
    padding: 8px;
    font-size: {s('label')}px;
    selection-background-color: {c['primary']};
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {c['border_strong']}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {c['text_muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {c['border_strong']}; border-radius: 5px; min-width: 28px;
}}

/* ---------- 其它 ---------- */
QProgressBar {{
    background: {c['surface_sunken']};
    border: none; border-radius: 5px; height: 8px;
    text-align: center; color: {c['text_muted']}; font-size: {s('small')}px;
}}
QProgressBar::chunk {{ background: {c['primary']}; border-radius: 5px; }}
QFrame#Divider {{ background: {c['border']}; max-height: 1px; min-height: 1px; }}
QFrame#VDivider {{ background: {c['border']}; max-width: 1px; min-width: 1px; }}
QToolButton#Expander {{
    background: transparent;
    border: none;
    color: {c['text_secondary']};
    font-size: {s('label')}px;
    padding: 4px 0;
    text-align: left;
}}
QToolButton#Expander:hover {{ color: {c['primary']}; }}
QStatusBar {{
    background: {c['surface']};
    border-top: 1px solid {c['border']};
    color: {c['text_secondary']};
    font-size: {s('small')}px;
}}
"""


def apply_theme(app: QApplication, name: str = "light", scale: float = 1.0) -> Theme:
    import os
    from pathlib import Path

    # Windows 离屏插件有时不枚举系统字体；直接登记已有中文字体，避免截图缺字。
    candidate = QFont(_first_available(FONT_STACK))
    if not QFontMetrics(candidate).inFontUcs4(ord("厚")):
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    theme = build_theme(name, scale)
    app.setStyle("Fusion")
    font = QFont(_first_available(FONT_STACK), theme.size("body"))
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(stylesheet(theme))
    return theme


def _first_available(stack: list[str]) -> str:
    families = set(QFontDatabase.families())
    for item in stack:
        name = item.strip('"')
        if name in families:
            return name
    return "Segoe UI"
