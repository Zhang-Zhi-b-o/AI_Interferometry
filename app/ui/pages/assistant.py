"""实验助手：后台会话、工具活动与主线程运动确认。"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMessageBox, QPlainTextEdit, QWidget

from app.agent.runtime import AgentRuntime
from app.ui.pages.base import Page
from app.ui.widgets.common import Card, hstack, primary_button, secondary_button


class AssistantPage(Page):
    key = "assistant"
    title = "⑦ 智能体助手"
    hint = "读取状态、分析条纹、计算厚度；运动操作需要逐次确认。"

    activity = Signal(str, object)
    confirmation = Signal(object)

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        self.runtime = AgentRuntime(
            ctx, provider=SimpleNamespace(available=False) if getattr(ctx, "demo_mode", False) else None,
            on_event=self.activity.emit, confirm_motion=self._request_confirmation)
        self.activity.connect(self._activity)
        self.confirmation.connect(self._confirm)
        self._running = False
        self._closing = False
        self._waiting = None
        self._streaming = False
        self._streamed: list[str] = []
        card = Card("实验助手", "无密钥时提供本地流程建议与知识库摘录。")
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(180)
        card.add(self.transcript)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("例如：现在该做什么？测量当前条纹宽度。")
        self.input.setFixedHeight(80)
        card.add(self.input)
        self.send = primary_button("发送", self._send)
        self.cancel_button = secondary_button("取消本轮", self.cancel)
        self.cancel_button.setEnabled(False)
        card.add(hstack(self.send, self.cancel_button))
        self.add(card)
        activity = Card("工具活动", "可查看计划、工具参数和执行结果。")
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setMaximumBlockCount(200)
        activity.add(self.events)
        self.add(activity)
        self.add_stretch()

    def _send(self):
        question = self.input.toPlainText().strip()
        if self._closing or self._running or not question:
            return
        self._running = True
        self.send.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.transcript.appendPlainText("你：" + question)
        self.input.clear()
        self._streaming = False
        self._streamed = []
        turn_cancel = threading.Event()
        self.runtime.cancel_event = turn_cancel
        self.ctx.run_async(
            lambda: self.runtime.ask(question, cancel_event=turn_cancel), on_done=self._done,
            on_error=lambda exc: self._done("助手请求失败：" + str(exc)), name="assistant")

    def _done(self, answer):
        self._running = False
        self.send.setEnabled(True)
        self.cancel_button.setEnabled(False)
        text = str(answer)
        streamed = "".join(self._streamed)
        self._streamed = []
        if self._streaming:
            self._streaming = False
            # 流式内容已显示；只有取消/回退等未流入的结尾才补一行。
            if text.strip() and not streamed.strip().endswith(text.strip()):
                self.transcript.appendPlainText("助手：" + text)
        else:
            self.transcript.appendPlainText("助手：" + text)

    def _activity(self, kind, value):
        if self._closing:
            return
        if kind == "delta":
            if not self._streaming:
                self._streaming = True
                self.transcript.appendPlainText("助手：")
            cursor = self.transcript.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(str(value))
            self.transcript.setTextCursor(cursor)
            self._streamed.append(str(value))
            return
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        self.events.appendPlainText(f"{kind}：{text}")

    def _request_confirmation(self, tool, arguments, cancel):
        request = {"tool": tool, "arguments": arguments,
                   "ready": threading.Event(), "approved": False}
        self.confirmation.emit(request)
        while not request["ready"].wait(0.05):
            if cancel.is_set():
                return False
        return request["approved"] and not cancel.is_set()

    def _confirm(self, request):
        if self._closing or self.runtime.cancel_event.is_set():
            request["ready"].set()
            return
        self._waiting = request
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认电机运动")
        dialog.setText("助手请求按默认正转方向连续搜索并寻中。请确认电机方向映射、起始方向与行程安全。")
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        self._dialog = dialog

        def finished(result):
            request["approved"] = result == QMessageBox.StandardButton.Yes
            request["ready"].set()
            self._waiting = None
            self._dialog = None

        dialog.finished.connect(finished)
        dialog.open()

    def cancel(self):
        self.runtime.cancel()
        if self._waiting is not None:
            self._waiting["ready"].set()
        if getattr(self, "_dialog", None) is not None:
            self._dialog.reject()

    def shutdown(self):
        self._closing = True
        self.cancel()
