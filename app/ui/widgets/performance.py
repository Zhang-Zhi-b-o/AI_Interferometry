"""性能评测窗口：固定题集、逐轮验收与 DOCX 导出，后台执行检索/排版。"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from app.services.performance import MODES, evaluate_retrieval
from app.services.performance_report import export_performance, summary_rows, seconds_text


class PerformanceDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("AI 性能评测")
        self.resize(1000, 720)
        self.setMinimumSize(800, 540)
        self._cancel = threading.Event()
        self._running = False
        self._turn_ids = []
        layout = QVBoxLayout(self)
        description = QLabel("Hit@3 使用预标注的固定题集；任务完成率按逐条验收计算；在线、离线和降级分别统计。")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(145)
        layout.addWidget(self.summary)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("运行检索评测")
        self.run_button.clicked.connect(self._evaluate)
        self.cancel_button = QPushButton("取消评测")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel.set)
        self.export_button = QPushButton("导出 DOCX 评测报告")
        self.export_button.clicked.connect(self._export)
        for widget in (self.run_button, self.cancel_button, self.export_button):
            buttons.addWidget(widget)
        layout.addLayout(buttons)
        self.tabs = QTabWidget()
        tasks = QWidget()
        task_layout = QVBoxLayout(tasks)
        task_layout.addWidget(QLabel("验收标准：原始请求的目标是否全部达到，结果是否可用。未验收轮次不进入完成率分母。"))
        self.tasks_table = QTableWidget(0, 5)
        self.tasks_table.setHorizontalHeaderLabels(["问题", "运行模式 / 数据", "结果", "耗时 (s)", "验收依据"])
        self.tasks_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tasks_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.tasks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.setColumnWidth(0, 290)
        self.tasks_table.setColumnWidth(1, 170)
        self.tasks_table.itemSelectionChanged.connect(self._selection_changed)
        task_layout.addWidget(self.tasks_table)
        self.note = QLineEdit()
        self.note.setPlaceholderText("填写验收依据或未完成原因（可选）")
        task_layout.addWidget(self.note)
        review = QHBoxLayout()
        self.complete_button = QPushButton("标记为完成")
        self.fail_button = QPushButton("标记为未完成")
        self.complete_button.clicked.connect(lambda: self._review(True))
        self.fail_button.clicked.connect(lambda: self._review(False))
        review.addWidget(self.complete_button)
        review.addWidget(self.fail_button)
        task_layout.addLayout(review)
        self.tabs.addTab(tasks, "助手任务验收")
        self.retrieval = QPlainTextEdit()
        self.retrieval.setReadOnly(True)
        self.tabs.addTab(self.retrieval, "检索逐题结果")
        layout.addWidget(self.tabs, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.refresh()

    def refresh(self):
        snapshot = self.ctx.performance.snapshot()
        lines = [" | ".join(row) for row in summary_rows(snapshot)[1:]]
        if snapshot.get("persistence_error"):
            lines.append("记录写盘失败：" + snapshot["persistence_error"])
        self.summary.setPlainText("\n".join(lines))
        selected = self._selected_id()
        turns = snapshot["turns"]
        self._turn_ids = [t["id"] for t in turns]
        self.tasks_table.setRowCount(len(turns))
        for row, turn in enumerate(turns):
            verdict = "取消（不计分）" if turn["status"] == "cancelled" else (
                "待验收" if turn.get("completed") is None else "完成" if turn["completed"] else "未完成")
            cells = [turn["question"], MODES[turn["mode"]] + " / " + turn["data_source"],
                     verdict, f"{turn['elapsed_s']:.3f}", turn.get("review_note", "")]
            for column, value in enumerate(cells):
                self.tasks_table.setItem(row, column, QTableWidgetItem(value))
        if selected in self._turn_ids:
            self.tasks_table.selectRow(self._turn_ids.index(selected))
        run = snapshot.get("retrieval")
        if run:
            details = [f"{run['version']} · {run['count']} 题；{run['timing_note']}"]
            for item in run["rows"]:
                verdict = "标注失效" if item["hit"] is None else "命中" if item["hit"] else "未命中"
                details += [f"\n{item['id']} {item['query']}\n{verdict} · {seconds_text(item['elapsed_s'])}",
                            "预标注：" + ", ".join(item["expected_ids"]),
                            "实际前三条：" + ", ".join(r["id"] for r in item["retrieved"])]
                if item["error"]:
                    details.append("查询异常：" + item["error"])
            self.retrieval.setPlainText("\n".join(details))
        self._selection_changed()

    def _selected_id(self):
        row = self.tasks_table.currentRow()
        return self._turn_ids[row] if 0 <= row < len(self._turn_ids) else None

    def _selection_changed(self):
        key = self._selected_id()
        turn = next((t for t in self.ctx.performance.snapshot()["turns"] if t["id"] == key), None)
        enabled = turn is not None and turn["status"] != "cancelled"
        self.complete_button.setEnabled(enabled)
        self.fail_button.setEnabled(enabled)

    def _review(self, completed):
        turn_id = self._selected_id()
        if turn_id is None:
            return
        self.ctx.performance.review(turn_id, completed, self.note.text().strip())
        self.note.clear()
        self.refresh()

    def _evaluate(self):
        if self._running:
            return
        self._running = True
        self._cancel = threading.Event()
        cancel = self._cancel
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("正在运行固定检索题集，不调用模型或设备…")

        def done(result):
            self._running = False
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.ctx.performance.set_retrieval(result)
            self.refresh()
            self.tabs.setCurrentWidget(self.retrieval)
            self.status.setText(f"检索评测完成：{result['hits']}/{result['count']}；无效标注 {result['invalid_labels']}")

        def failed(exc):
            self._running = False
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.status.setText("评测已取消" if cancel.is_set() else f"评测失败：{exc}")

        self.ctx.run_async(lambda: evaluate_retrieval(cancel_event=cancel),
                           on_done=done, on_error=failed, name="retrieval-evaluation")

    def _export(self):
        if self._running:
            self.status.setText("请等待当前检索评测完成后导出。")
            return
        snapshot = self.ctx.performance.snapshot()
        directory = Path(self.ctx.record.out_dir) / "performance"
        self.export_button.setEnabled(False)

        def done(paths):
            self.export_button.setEnabled(True)
            self.status.setText("已导出：" + paths["docx"])
            self.ctx.log("已导出 AI 性能评测报告：" + paths["docx"])

        def failed(exc):
            self.export_button.setEnabled(True)
            self.status.setText(f"导出失败：{exc}")

        self.ctx.run_async(lambda: export_performance(snapshot, directory),
                           on_done=done, on_error=failed, name="performance-report")

    def closeEvent(self, event):  # noqa: N802
        self._cancel.set()
        super().closeEvent(event)
