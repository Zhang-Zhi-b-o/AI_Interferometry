"""页面基类：统一的标题区 + 可滚动内容区 + 两个钩子。

新增一个页面只要：
1. 继承 ``Page``，设 ``key`` / ``title`` / ``hint``；
2. 在 ``__init__`` 里往 ``self.add(...)`` 塞卡片；
3. 需要跟着状态刷新就覆盖 ``on_state``，需要进入页面时做事就覆盖 ``on_show``；
4. 在 ``app/ui/main_window.py`` 的 ``PAGES`` 里登记一行。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from app.core.state import AppState
from app.ui.context import AppContext
from app.ui.widgets.common import label


class Page(QWidget):
    key: str = ""
    title: str = ""
    hint: str = ""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("Page")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(12)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        if self.title:
            self.body.addWidget(label(self.title, name="PageTitle"))
        if self.hint:
            self.body.addWidget(label(self.hint, name="PageHint", wrap=True))

    # ------------------------------------------------------------------
    def add(self, widget: QWidget | None) -> None:
        if widget is not None:
            self.body.addWidget(widget)

    def add_stretch(self) -> None:
        self.body.addStretch(1)

    def on_show(self) -> None:
        """页面被切到时调用。"""

    def on_state(self, state: AppState) -> None:
        """任意状态变化时调用（页面只读，不要在这里改状态）。"""
