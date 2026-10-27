"""实验助手：后台会话、工具活动与主线程运动确认。"""
from __future__ import annotations

import json
import html
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Signal, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QLabel, QMessageBox, QPlainTextEdit,
    QSizePolicy, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from app.agent.runtime import AgentRuntime
from app.ui.pages.base import Page
from app.ui.widgets.common import Card, hstack, primary_button, secondary_button
from app.ui.widgets.math_chat import MathChat


class AssistantPage(Page):
    key = "assistant"
    title = "⑦ 智能体助手"
    hint = "读取状态、分析条纹、计算厚度；运动操作需要逐次确认。"

    activity = Signal(str, object)
    confirmation = Signal(object)

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent, scrollable=False)
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
        self._performance_dialog = None
        self._conversation_entries = []
        self._review_cancel = None
        self._review_running = False
        self._adjustment_before = None
        self._last_question = ""
        self._stream_prefix = ""
        self._chat_dialog = None
        self.workspace = QSplitter(Qt.Orientation.Vertical)
        self.workspace.setHandleWidth(8)
        self.workspace.setChildrenCollapsible(False)
        card = self.chat_card = Card("实验助手", "支持 Markdown 与数学公式；拖动分隔条调整对话和输入区，或打开独立窗口。")
        self.message_splitter = QSplitter(Qt.Orientation.Vertical)
        self.message_splitter.setHandleWidth(8)
        self.message_splitter.setChildrenCollapsible(False)
        self.transcript = MathChat()
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(120)
        self.message_splitter.addWidget(self.transcript)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("例如：现在该做什么？测量当前条纹宽度。")
        self.input.setMinimumHeight(56)
        self.message_splitter.addWidget(self.input)
        self.message_splitter.setStretchFactor(0, 4)
        self.message_splitter.setStretchFactor(1, 1)
        self.message_splitter.setSizes([280, 80])
        card.add(self.message_splitter)
        self.send = primary_button("发送", self._send)
        self.cancel_button = secondary_button("取消本轮", self.cancel)
        self.cancel_button.setEnabled(False)
        self.performance_button = secondary_button("性能评测", self._show_performance)
        self.retry_button = secondary_button("重试上一问", self._retry)
        self.retry_button.setEnabled(False)
        self.popout_button = secondary_button("独立对话窗口", self._popout_chat)
        card.add(hstack(self.send, self.cancel_button, self.retry_button))
        card.add(hstack(self.popout_button, self.performance_button))
        self.workspace.addWidget(card)
        self.detail_tabs = QTabWidget()
        activity = Card("工具活动", "可查看计划、工具参数和执行结果。")
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setMaximumBlockCount(200)
        activity.add(self.events)
        self.detail_tabs.addTab(activity, "工具活动")
        sources = Card("建议依据", "点击资料标题打开本地原文；PDF 页码见出处。")
        self.sources = QTextBrowser()
        self.sources.setOpenLinks(False)
        self.sources.anchorClicked.connect(self._open_source)
        self.sources.setMinimumHeight(120)
        self._sources = {}
        sources.add(self.sources)
        self.detail_tabs.addTab(sources, "建议依据")
        from app.services.assistant_legacy import STAGES
        legacy = Card("实验观察与复核", "四阶段仅切换本地教学建议；点击图像复核才发送当前画面。")
        self.guidance_stage = QComboBox()
        for key, title, *_ in STAGES:
            self.guidance_stage.addItem(title, key)
        legacy.add(self.guidance_stage)
        legacy.add(secondary_button("查看本阶段建议", self._local_guidance))
        legacy.add(hstack(secondary_button("标记调节前", self._mark_adjustment),
                          secondary_button("比较调节后", self._compare_adjustment)))
        self.review_button = secondary_button("发送单帧图像复核", self._review_image)
        self.review_cancel_button = secondary_button("取消图像复核", self._cancel_review)
        self.review_cancel_button.setEnabled(False)
        legacy.add(hstack(self.review_button, self.review_cancel_button))
        legacy.add(secondary_button("导出会话 Markdown / JSON", self._export_conversation))
        self.detail_tabs.addTab(legacy, "观察与复核")
        # 附加功能滚动显示，避免长卡片挤占对话；聊天本身使用固定可拖动分区。
        from PySide6.QtWidgets import QScrollArea

        legacy_scroll = QScrollArea()
        legacy_scroll.setWidgetResizable(True)
        legacy_scroll.setWidget(legacy)
        self.detail_tabs.removeTab(2)
        self.detail_tabs.addTab(legacy_scroll, "观察与复核")
        self.detail_tabs.setMinimumHeight(100)
        self.detail_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.workspace.addWidget(self.detail_tabs)
        self.workspace.setStretchFactor(0, 5)
        self.workspace.setStretchFactor(1, 1)
        self.workspace.setSizes([500, 150])
        self.body.addWidget(self.workspace, 1)

    def _popout_chat(self):
        if self._chat_dialog is not None:
            self._chat_dialog.showNormal()
            self._chat_dialog.raise_()
            self._chat_dialog.activateWindow()
            return
        self._workspace_sizes = self.workspace.sizes()
        self._chat_placeholder = QLabel("对话已在独立窗口打开；关闭该窗口后回到此处。")
        self._chat_placeholder.setWordWrap(True)
        self.workspace.insertWidget(0, self._chat_placeholder)
        dialog = QDialog(self)
        dialog.setWindowTitle("实验助手 · 对话")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        dialog.setMinimumSize(540, 400)
        dialog.resize(960, 760)
        dialog._stop_shortcut = QShortcut(QKeySequence("Esc"), dialog)
        dialog._stop_shortcut.activated.connect(self.ctx.stop_motion)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.chat_card)
        self._chat_dialog = dialog
        dialog.finished.connect(self._dock_chat)
        dialog.show()

    def _dock_chat(self, _result=0):
        dialog, self._chat_dialog = self._chat_dialog, None
        if dialog is None:
            return
        self.workspace.insertWidget(0, self.chat_card)
        self.chat_card.show()
        self._chat_placeholder.deleteLater()
        self.workspace.setSizes(self._workspace_sizes)
        dialog.deleteLater()

    def _retry(self):
        if self._last_question and not self._running:
            self.input.setPlainText(self._last_question)
            self._send()

    def _send(self):
        question = self.input.toPlainText().strip()
        if self._closing or self._running or not question:
            return
        self._running = True
        self._last_question = question
        self.retry_button.setEnabled(False)
        self.send.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.transcript.appendPlainText("你：" + question)
        self._conversation_entries.append({"role": "你", "text": question})
        self.input.clear()
        self._streaming = False
        self._streamed = []
        turn_cancel = threading.Event()
        self.runtime.cancel_event = turn_cancel
        self.ctx.run_async(
            lambda: self.runtime.ask(question, cancel_event=turn_cancel), on_done=self._done,
            on_error=self._failed, name="assistant")

    def _failed(self, exc):
        self._done("请求未完成：" + str(exc) + "\n可点击「重试上一问」。")

    def _done(self, answer):
        self._running = False
        self.send.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.retry_button.setEnabled(bool(self._last_question))
        text = str(answer)
        self._conversation_entries.append({"role": "助手", "text": text})
        streamed = "".join(self._streamed)
        self._streamed = []
        if self._streaming:
            self._streaming = False
            # 流式内容已显示；只有取消/回退等未流入的结尾才补一行。
            if text.strip() and not streamed.strip().endswith(text.strip()):
                self.transcript.appendPlainText("助手：" + text)
        else:
            self.transcript.appendPlainText("助手：" + text)
        self.transcript.render_now()

    def _activity(self, kind, value):
        if self._closing:
            return
        if kind == "turn_finished":
            if self._performance_dialog is not None:
                self._performance_dialog.refresh()
            return
        if kind == "sources":
            self._sources = {item["id"]: item for item in value}
            self.sources.setHtml("".join(
                '<p><a href="source:' + html.escape(item["id"], quote=True) + '">'
                + html.escape(f"[{item['id']}] {item['title']} · "
                              + (f"第{item['page']}页" if item.get("kind") not in ("kb", "guide") else "文本块"))
                + "</a><br>" + html.escape(item["text"][:260]) + "</p>" for item in value))
            return
        if kind == "delta":
            if not self._streaming:
                self._stream_prefix = self.transcript.toPlainText()
                self._streaming = True
                self.transcript.appendPlainText("助手：")
            self.transcript.append_delta(str(value))
            self._streamed.append(str(value))
            return
        if kind == "stream_reset":
            if self._streaming:
                self.transcript.setPlainText(self._stream_prefix)
            self._streamed = []
            self._streaming = False
            return
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        self.events.appendPlainText(f"{kind}：{text}")

    def _append_legacy(self, text):
        if self._closing:
            return
        self._conversation_entries.append({"role": "助手", "text": str(text)})
        self.transcript.appendPlainText("助手：" + str(text))
        self.transcript.render_now()

    def _local_guidance(self):
        from app.services.assistant_legacy import local_guidance
        result = local_guidance(self.ctx.state.snapshot(), self.guidance_stage.currentData())
        self._append_legacy(f"{result['stage']}\n{result['diagnosis']}\n下一步：{result['action']}\n观察确认：{result['observation_question']}")

    def _mark_adjustment(self):
        from app.services.assistant_legacy import adjustment_metrics
        metrics = adjustment_metrics(self.ctx.state.snapshot())
        if not any(value is not None for value in metrics.values()):
            self._append_legacy("当前没有可比较条纹指标，请先测量条纹。")
            return
        self._adjustment_before = metrics
        self._append_legacy("已记录调节前指标。只微调一个旋钮，重新测量后比较。")

    def _compare_adjustment(self):
        from app.services.assistant_legacy import adjustment_metrics, compare_adjustment
        current = adjustment_metrics(self.ctx.state.snapshot())
        result = compare_adjustment(self._adjustment_before, current)
        if result["outcome"] != "insufficient":
            self._adjustment_before = current
        self._append_legacy(result["summary"] + " " + result["recommendation"])

    def _review_image(self):
        if self._review_running or self._closing:
            return
        if getattr(self.ctx, "demo_mode", False):
            self._append_legacy("演示模式不发送图像复核请求。")
            return
        frame = self.ctx.frame()
        if frame is None:
            self._append_legacy("当前没有可供复核画面。")
            return
        from app.services.assistant_legacy import review_image
        state = self.ctx.state.snapshot()
        cancel_event = threading.Event()
        self._review_cancel = cancel_event
        self._review_running = True
        self.review_button.setEnabled(False)
        self.review_cancel_button.setEnabled(True)
        def finished(value):
            self._review_running = False
            self.review_button.setEnabled(True)
            self.review_cancel_button.setEnabled(False)
            self._append_legacy("图像复核已取消。" if cancel_event.is_set() else str(value))
        self.ctx.run_async(lambda: review_image(frame.copy(), state, self.ctx.config,
                            cancel_event=cancel_event), on_done=finished,
                           on_error=lambda exc: finished("图像复核失败：" + str(exc)), name="assistant-image-review")

    def _cancel_review(self):
        if self._review_cancel is not None:
            self._review_cancel.set()

    def _export_conversation(self):
        from app.services.assistant_legacy import export_conversation
        path, _ = QFileDialog.getSaveFileName(self, "导出助手会话", "实验助手会话.md", "Markdown (*.md);;JSON (*.json)")
        if not path:
            return
        try:
            target = export_conversation(path, self._conversation_entries,
                    data_source="demo" if getattr(self.ctx, "demo_mode", False) else "live")
            self.ctx.log(f"助手会话已导出：{target}")
        except (OSError, ValueError) as exc:
            self.ctx.log(f"会话导出失败：{exc}")

    def _show_performance(self):
        from app.ui.widgets.performance import PerformanceDialog

        if self._performance_dialog is None:
            self._performance_dialog = PerformanceDialog(self.ctx, self)
        self._performance_dialog.refresh()
        self._performance_dialog.show()
        self._performance_dialog.raise_()

    def _open_source(self, url):
        from app.core.config import ROOT

        item = self._sources.get(url.toString().removeprefix("source:"))
        if not item or not item.get("path"):
            return
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            self.ctx.log("原文文件不存在或不在项目内")
            return
        target = QUrl.fromLocalFile(str(Path(path)))
        if path.suffix.lower() == ".pdf" and item.get("page"):
            target.setFragment(f"page={item['page']}")
        QDesktopServices.openUrl(target)

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
        dialog = QMessageBox(self._chat_dialog or self)
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
        self._cancel_review()
        self.runtime.cancel()
        if self._waiting is not None:
            self._waiting["ready"].set()
        if getattr(self, "_dialog", None) is not None:
            self._dialog.reject()

    def shutdown(self):
        self._closing = True
        if self._chat_dialog is not None:
            self._chat_dialog.close()
        if self._performance_dialog is not None:
            self._performance_dialog.close()
        self.cancel()
