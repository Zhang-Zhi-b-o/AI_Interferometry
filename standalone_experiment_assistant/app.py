"""独立的迈克尔逊干涉实验助手界面。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from src.ui.theme import APP_BG, FONT, MUTED, NAVY, PRIMARY, SURFACE, TEXT
from src.ui.widgets.agent_plugin import AgentPluginPanel
from src.ui.widgets.collapsible import CollapsibleFrame
from src.agent import AgentService, AgentSession

from standalone_experiment_assistant.data import STEPS


class StandaloneExperimentAssistant:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Interferometry · 白光干涉实验工作台")
        self.root.geometry("760x920")
        self.root.minsize(620, 700)
        self.root.configure(bg=APP_BG)
        self.root.option_add("*Font", (FONT, 9))
        self.step_index = 0
        self.agent_service = AgentService(context_provider=self._context)
        self.agent_session = AgentSession(self.agent_service)
        self._agent_operation = ""
        self._agent_poll_job = None
        self._build()
        self._set_step(0)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(250, self._connection)

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=APP_BG)
        shell.pack(fill=tk.BOTH, expand=True)
        self.progress_control = CollapsibleFrame(
            shell, "实验进度控制", collapsed=True,
            show_move_buttons=False,
        )
        self.progress_control.pack(fill=tk.X, padx=8, pady=(8, 0))
        self._build_control(self.progress_control.content)

        self.assistant = AgentPluginPanel(shell)
        self.assistant.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.assistant.on_ask = self._ask
        self.assistant.on_test = self._connection
        self.assistant.on_cancel = self._cancel
        self.assistant.set_connection_status(True)

    def _build_control(self, parent: tk.Widget) -> None:
        tk.Label(
            parent, text="选择实验阶段后，助手状态和对话上下文将同步更新。",
            bg=SURFACE, fg=MUTED, font=(FONT, 9), anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(8, 6))

        selector = tk.Frame(parent, bg=SURFACE)
        selector.pack(fill=tk.X, padx=10)
        self.step_var = tk.StringVar()
        self.step_box = ttk.Combobox(
            selector, textvariable=self.step_var, state="readonly",
            values=[f"{i + 1}. {step['title']}" for i, step in enumerate(STEPS)],
        )
        self.step_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.step_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._set_step(self.step_box.current()))
        tk.Button(
            selector, text="上一步", command=lambda: self._set_step(self.step_index - 1),
            bg="#eef2f7", fg=TEXT, relief=tk.FLAT, padx=10, pady=5,
        ).pack(side=tk.LEFT, padx=(8, 3))
        tk.Button(
            selector, text="下一步", command=lambda: self._set_step(self.step_index + 1),
            bg=PRIMARY, fg="#ffffff", relief=tk.FLAT, padx=10, pady=5,
        ).pack(side=tk.LEFT, padx=(3, 0))

        self.stage_var = tk.StringVar()
        self.next_var = tk.StringVar()
        status = tk.Frame(parent, bg="#f5f8fc")
        status.pack(fill=tk.X, padx=10, pady=(7, 9))
        tk.Label(
            status, textvariable=self.stage_var, bg="#f5f8fc", fg=NAVY,
            font=(FONT, 9, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=9, pady=(7, 2))
        tk.Label(
            status, textvariable=self.next_var, bg="#f5f8fc", fg="#475569",
            font=(FONT, 8), justify=tk.LEFT, anchor="w", wraplength=690,
        ).pack(fill=tk.X, padx=9, pady=(0, 7))

    def _context(self) -> dict:
        step = STEPS[self.step_index]
        camera_a, camera_b, model, motor = step["devices"]
        return {
            "camera": {
                "interferometer_running": camera_a,
                "micrometer_running": camera_b,
            },
            "vision": {
                "model_loaded": model,
                "prediction_running": model,
                "detections": {"zero_fringe": 0.91} if model else {},
            },
            "motor": {"connected": motor},
            "experiment_progress": {
                "step_number": self.step_index + 1,
                "progress_percent": step["progress"],
                "stage": step["title"],
                "next_action": step["next_action"],
                "completion_criterion": step["criterion"],
            },
        }

    def _set_step(self, index: int) -> None:
        self.step_index = max(0, min(len(STEPS) - 1, int(index)))
        step = STEPS[self.step_index]
        self.step_box.current(self.step_index)
        self.stage_var.set(
            f"{self.step_index + 1}/{len(STEPS)}  {step['title']} · {step['progress']}%")
        self.next_var.set(
            f"下一步：{step['next_action']}\n完成标志：{step['criterion']}")
        self.assistant.set_experiment_context(self._context())

    def _ask(self, question: str, _include_status: bool) -> None:
        if not self.agent_session.ask(
            question, _include_status, self._context()
        ):
            self.assistant.set_busy(False)
            self.assistant.append("系统", "上一项请求尚未结束，请稍后再试。")
            return
        self._agent_operation = "ask"
        self._schedule_agent_poll()

    def _schedule_agent_poll(self) -> None:
        if self._agent_poll_job is None:
            self._agent_poll_job = self.root.after(80, self._poll_agent)

    def _connection(self) -> None:
        if self.agent_session.busy:
            return
        self.assistant._active_task = "connection"
        self.assistant.set_busy(True)
        if self.agent_session.test_connection():
            self._agent_operation = "connection"
            self._schedule_agent_poll()

    def _cancel(self) -> None:
        if self.agent_session.cancel():
            self.assistant.set_ai_state("正在停止", "warning")
        else:
            self.assistant.set_busy(False)

    def _poll_agent(self) -> None:
        self._agent_poll_job = None
        result = self.agent_session.poll()
        if result is None:
            if self.agent_session.busy:
                self._schedule_agent_poll()
            return
        operation = self._agent_operation
        self._agent_operation = ""
        self.assistant.set_busy(False)
        if result.cancelled:
            self.assistant.set_ai_state("已停止", "warning")
            return
        if result.error is not None:
            self.assistant.set_connection_status(False)
            self.assistant.set_ai_state("请求失败", "error")
            self.assistant.append("系统", f"DeepSeek 请求失败：{result.error}")
            return
        response = result.response
        if response is None:
            self.assistant.set_ai_state("未收到响应", "error")
            return
        self.assistant.set_connection_status(response.online)
        if operation == "connection":
            self.assistant.append("系统", response.answer)
        else:
            self.assistant.append("助手", response.answer)
            if response.warning:
                self.assistant.append("系统", response.warning)
        self.assistant.set_ai_state(
            "就绪" if response.online else "连接异常",
            "success" if response.online else "warning",
        )

    def _close(self) -> None:
        if self._agent_poll_job is not None:
            self.root.after_cancel(self._agent_poll_job)
            self._agent_poll_job = None
        self.agent_session.shutdown()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    StandaloneExperimentAssistant().run()
