"""有界运行日志与导出，替代旧 Tk 日志插件。"""
from datetime import datetime

from PySide6.QtWidgets import QDialog, QFileDialog, QPlainTextEdit, QVBoxLayout

from app.ui.widgets.common import hstack, secondary_button


class RuntimeLog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("运行日志")
        self.resize(800, 480)
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        layout.addWidget(self.text)
        layout.addWidget(hstack(secondary_button("导出日志", self._export),
                               secondary_button("清空显示", self.text.clear)))

    def append(self, text):
        self.text.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def _export(self):
        from pathlib import Path

        path, _ = QFileDialog.getSaveFileName(self, "导出运行日志", "", "文本 (*.txt)")
        if path:
            Path(path).write_text(self.text.toPlainText(), encoding="utf-8")
