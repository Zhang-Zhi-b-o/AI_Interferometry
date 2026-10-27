"""主画面指标与底部状态栏。"""
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from app.ui.widgets.common import MetricCard


class MetricsBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.cards = (
            MetricCard("中心位置", "px", decimals=1),
            MetricCard("中心条纹宽度", "px", decimals=2),
            MetricCard("微分表读数", "mm", decimals=4),
            MetricCard("玻璃片厚度", "mm", decimals=5),
        )
        for card in self.cards:
            layout.addWidget(card)


class Footer(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Footer")
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.status)
        layout.addStretch(1)
        layout.addWidget(QLabel("h = (d₂−d₁) / [20(n−1)]    Shift＋左键框选 ROI · 右键清除"))
