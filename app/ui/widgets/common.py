"""界面通用组件。所有页面都从这里拿零件，保证观感一致。

命名约定：``primary_button`` / ``secondary_button`` / ``ghost_button`` /
``danger_button`` 是四种按钮语义；``Card`` 是白色圆角卡片；``MetricCard`` 是
指标卡（大号等宽数字）。
"""
from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ----------------------------------------------------------------------
# 布局小工具
# ----------------------------------------------------------------------
def vbox(*children: QWidget, spacing: int = 10, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for child in children:
        if child is not None:
            layout.addWidget(child)
    return layout


def hbox(*children: QWidget, spacing: int = 10, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for child in children:
        if child is not None:
            layout.addWidget(child)
    return layout


def hstack(*children: QWidget, spacing: int = 8, stretch: bool = False) -> QWidget:
    """把几个控件横排成一个整体（返回 QWidget，可直接 add 到卡片里）。"""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for child in children:
        if child is not None:
            layout.addWidget(child)
    if stretch:
        layout.addStretch(1)
    return holder


def vstack(*children: QWidget, spacing: int = 8) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for child in children:
        if child is not None:
            layout.addWidget(child)
    return holder


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    line.setFixedHeight(1)
    return line


def label(text: str, *, name: str | None = None, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


# ----------------------------------------------------------------------
# 按钮
# ----------------------------------------------------------------------
def _button(text: str, variant: str, on_click: Callable[[], None] | None,
            minimum_width: int = 0) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", variant)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if minimum_width:
        button.setMinimumWidth(minimum_width)
    if on_click is not None:
        button.clicked.connect(lambda: on_click())
    return button


def primary_button(text: str, on_click: Callable[[], None] | None = None,
                   *, minimum_width: int = 0) -> QPushButton:
    return _button(text, "primary", on_click, minimum_width)


def secondary_button(text: str, on_click: Callable[[], None] | None = None,
                     *, minimum_width: int = 0) -> QPushButton:
    return _button(text, "secondary", on_click, minimum_width)


def ghost_button(text: str, on_click: Callable[[], None] | None = None) -> QPushButton:
    return _button(text, "ghost", on_click)


def danger_button(text: str, on_click: Callable[[], None] | None = None,
                  *, minimum_width: int = 0) -> QPushButton:
    return _button(text, "danger", on_click, minimum_width)


def chip_button(text: str, on_click: Callable[[], None] | None = None) -> QPushButton:
    return _button(text, "chip", on_click)


# ----------------------------------------------------------------------
# 卡片
# ----------------------------------------------------------------------
class Card(QFrame):
    """白色圆角卡片，带可选的标题与说明。内容往 ``body`` 里加。"""

    def __init__(self, title: str = "", hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)
        if title:
            outer.addWidget(label(title, name="CardTitle"))
        if hint:
            hint_label = label(hint, name="CardHint", wrap=True)
            outer.addWidget(hint_label)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body)
        self._outer = outer

    def add(self, widget: QWidget | None) -> None:
        if widget is not None:
            self.body.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def add_stretch(self) -> None:
        self.body.addStretch(1)


class MetricCard(QFrame):
    """指标卡：上小标题，下大号数字 + 单位。"""

    def __init__(self, title: str, unit: str = "", parent: QWidget | None = None,
                 *, decimals: int = 2) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._decimals = int(decimals)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        self._title = label(title, name="MetricLabel")
        row = QHBoxLayout()
        row.setSpacing(6)
        self._value = label("—", name="MetricValue")
        self._unit = label(unit, name="MetricUnit")
        row.addWidget(self._value, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        layout.addWidget(self._title)
        layout.addLayout(row)

    def set_value(self, value, unit: str | None = None) -> None:
        if value is None or value == "":
            self._value.setText("—")
        elif isinstance(value, float):
            self._value.setText(f"{value:.{self._decimals}f}")
        else:
            self._value.setText(str(value))
        if unit is not None:
            self._unit.setText(unit)


class StatusPill(QLabel):
    """状态胶囊：圆点 + 文本，颜色随状态变。"""

    COLORS = {
        "ok": "#22c55e",
        "warn": "#f59e0b",
        "error": "#ef4444",
        "idle": "#979da6",
        "busy": "#4176e6",
    }

    def __init__(self, text: str = "未连接", state: str = "idle",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        color = self.COLORS.get(state, self.COLORS["idle"])
        if text is not None:
            self._text = text
        self.setText(f"● {getattr(self, '_text', '')}")
        self.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: 600; padding: 2px 0;")


class Expander(QWidget):
    """「高级参数」折叠区：默认收起，避免主界面被参数淹没。"""

    def __init__(self, title: str = "高级参数", *, expanded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._toggle = QToolButton()
        self._toggle.setObjectName("Expander")
        self._toggle.setCheckable(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title = title
        self._toggle.clicked.connect(self._sync)
        layout.addWidget(self._toggle)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 0, 0, 0)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.content)
        self.set_expanded(expanded)

    def _sync(self) -> None:
        self.set_expanded(self._toggle.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)
        self._toggle.setText(("▾ " if expanded else "▸ ") + self._title)
        self.content.setVisible(expanded)

    def add(self, widget: QWidget | None) -> None:
        if widget is not None:
            self.content_layout.addWidget(widget)


def field_row(title: str, widget: QWidget, *, hint: str = "") -> QWidget:
    """一行「左侧标签 + 右侧控件」，参数区统一用它。"""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(label(title, name="SectionLabel"))
    layout.addWidget(widget)
    if hint:
        layout.addWidget(label(hint, name="Muted", wrap=True))
    return holder


def form_grid(rows: Iterable[tuple[str, QWidget]], *, columns: int = 1) -> QWidget:
    """把多组 ``(标题, 控件)`` 排成网格，窄屏自动换行。"""
    from PySide6.QtWidgets import QGridLayout

    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(10)
    for index, (title, widget) in enumerate(rows):
        row, column = divmod(index, columns)
        grid.addWidget(field_row(title, widget), row, column)
    for column in range(columns):
        grid.setColumnStretch(column, 1)
    return holder


def mono_label(text: str, *, size: int = 26, color: str | None = None) -> QLabel:
    widget = QLabel(text)
    font = QFont("Consolas", size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    widget.setFont(font)
    if color:
        widget.setStyleSheet(f"color: {color};")
    return widget


def spacer(height: int = 8) -> QWidget:
    widget = QWidget()
    widget.setFixedHeight(height)
    return widget


def icon_text(symbol: str, text: str) -> str:
    return f"{symbol}  {text}"
