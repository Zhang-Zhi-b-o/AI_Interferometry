"""搜索页共用的电机连接栏；串口连接/关闭均在后台执行。"""
from __future__ import annotations

import threading

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app.ui.widgets.common import label


class MotorConnection(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.busy = False
        self._closing = False
        self._last_connected = None
        self._cancel = threading.Event()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(label("电机"))
        self.port = QLineEdit(str(ctx.config.get("motor.port", "auto")))
        self.port.setPlaceholderText("auto 或 COM3")
        self.port.setMinimumWidth(80)
        self.port.setToolTip("填 auto 自动检测，或指定 COM3 等串口；演示模式只操作虚拟设备。")
        row.addWidget(self.port, 1)
        self.baudrate = QComboBox()
        for baud in (9600, 19200, 38400, 57600, 115200):
            self.baudrate.addItem(str(baud), baud)
        value = int(ctx.config.get("motor.baudrate", 9600))
        index = self.baudrate.findData(value)
        if index < 0:
            self.baudrate.addItem(str(value), value)
            index = self.baudrate.count() - 1
        self.baudrate.setCurrentIndex(index)
        row.addWidget(self.baudrate)
        self.button = QPushButton()
        self.button.clicked.connect(self._toggle)
        row.addWidget(self.button)
        layout.addLayout(row)
        self.status = label("", name="Hint", wrap=True)
        layout.addWidget(self.status)
        self.refresh()

    def _toggle(self):
        if self.busy or self._closing:
            return
        connected = self.ctx.motion.connected
        port = self.port.text().strip()
        if not connected and not port:
            self.status.setText("请填写 auto 或串口号")
            return
        self.busy = True
        self.refresh()
        if connected:
            self.ctx.stop_motion()
            self.ctx.motion.begin_disconnect()
            self.status.setText("正在停车并断开…")

            def work():
                self.ctx.motion.disconnect()
                return False, "电机已断开"
        else:
            if self.ctx.search.active or self.ctx.target.active:
                self.busy = False
                self.status.setText("请等待闭环完全停止后再连接")
                self.refresh()
                return
            self._cancel = threading.Event()
            token = self._cancel
            baud = self.baudrate.currentData()
            self.status.setText("正在连接…")

            def work():
                return self.ctx.motion.connect(port, baudrate=baud, cancel_event=token)

        self.ctx.run_async(work, on_done=self._done, on_error=self._failed, name="motor-connect")

    def _done(self, result):
        if self._closing:
            return
        self.busy = False
        _, message = result
        self.ctx.state.update(motor_connected=self.ctx.motion.connected)
        self.refresh()
        self.status.setText(message)
        self.ctx.log(message)

    def _failed(self, exc):
        self._done((False, "连接操作失败：" + str(exc)))

    def cancel_pending(self):
        self._cancel.set()

    def shutdown(self):
        self._closing = True
        self.cancel_pending()

    def refresh(self):
        connected = self.ctx.motion.connected
        self.button.setText("处理中…" if self.busy else ("断开" if connected else "连接"))
        self.button.setEnabled(not self.busy)
        self.port.setEnabled(not self.busy and not connected)
        self.baudrate.setEnabled(not self.busy and not connected)
        if not self.busy and connected != self._last_connected:
            self.status.setText("电机已连接" if connected else "电机未连接")
            self._last_connected = connected
