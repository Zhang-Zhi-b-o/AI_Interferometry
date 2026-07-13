"""沉浸式实验辅助智能体面板。"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext


class AgentPluginPanel(tk.LabelFrame):
    BG = "#f4f7fb"
    NAVY = "#10233f"
    BLUE = "#1677ff"
    CYAN = "#00a6a6"

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="实验助手 · MICHELSON AI LAB",
                         bg=self.BG, fg=self.NAVY)
        self.on_ask = lambda question, include_status: None
        self.on_test = lambda: None
        self.on_cancel = lambda: None
        self.include_status_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="●  尚未测试 DeepSeek")
        self.context_var = tk.StringVar(value="实验状态：等待连接仪器")
        self.thinking_var = tk.StringVar(value="")
        self._thinking_job = None
        self._thinking_step = 0

        self._build_header()
        self._build_chat()
        self._build_actions()
        self._build_input()
        self.append(
            "系统",
            "欢迎进入迈克尔逊实验工作台。你可以询问实验原理、条纹异常、操作步骤和不确定度。"
            "助手只读取实验状态，不会直接控制电机。",
        )

    def _build_header(self):
        header = tk.Frame(self, bg=self.NAVY)
        header.pack(fill=tk.X, padx=6, pady=(6, 0))
        tk.Label(header, text="◉  MICHELSON AI LAB", bg=self.NAVY, fg="#ffffff",
                 font=("Microsoft YaHei UI", 11, "bold"), anchor="w").pack(
            fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(header, text="实时实验辅助 · 证据约束 · 只读安全模式",
                 bg=self.NAVY, fg="#9fc5ff", anchor="w",
                 font=("Microsoft YaHei UI", 8)).pack(fill=tk.X, padx=10)
        status_row = tk.Frame(header, bg=self.NAVY)
        status_row.pack(fill=tk.X, padx=10, pady=(4, 8))
        self.status_label = tk.Label(
            status_row, textvariable=self.status_var, bg=self.NAVY, fg="#f0b429",
            font=("Microsoft YaHei UI", 8), anchor="w")
        self.status_label.pack(side=tk.LEFT)
        tk.Button(status_row, text="测试连接", command=self.test_connection,
                  relief=tk.FLAT, bd=0, bg="#24486f", fg="#ffffff",
                  activebackground="#315c8b", activeforeground="#ffffff",
                  cursor="hand2", font=("Microsoft YaHei UI", 8)).pack(side=tk.RIGHT)

        context = tk.Label(self, textvariable=self.context_var, bg="#e7f0ff",
                           fg="#24558c", anchor="w", justify=tk.LEFT,
                           font=("Microsoft YaHei UI", 8))
        context.pack(fill=tk.X, padx=6, pady=(4, 0), ipady=5)

    def _build_chat(self):
        self.output = scrolledtext.ScrolledText(
            self, height=15, wrap=tk.WORD, bg="#ffffff", fg="#1d2b3a",
            insertbackground=self.NAVY, relief=tk.FLAT, bd=0,
            font=("Microsoft YaHei UI", 9), state=tk.DISABLED,
            padx=10, pady=8, spacing1=2, spacing3=5)
        self.output.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 3))
        self.output.tag_configure("user_role", foreground=self.BLUE,
                                  font=("Microsoft YaHei UI", 9, "bold"))
        self.output.tag_configure("assistant_role", foreground=self.CYAN,
                                  font=("Microsoft YaHei UI", 9, "bold"))
        self.output.tag_configure("system_role", foreground="#7a5b00",
                                  font=("Microsoft YaHei UI", 9, "bold"))
        self.output.tag_configure("timestamp", foreground="#8995a3",
                                  font=("Consolas", 8))
        self.output.tag_configure("message", foreground="#1d2b3a",
                                  lmargin1=8, lmargin2=8, rmargin=8)

    def _build_actions(self):
        tk.Label(self, text="快捷任务", bg=self.BG, fg="#637083", anchor="w",
                 font=("Microsoft YaHei UI", 8, "bold")).pack(fill=tk.X, padx=9)
        quick = tk.Frame(self, bg=self.BG)
        quick.pack(fill=tk.X, padx=6, pady=(2, 3))
        for label, question in [
            ("◎ 原理解析", "结合实验说明迈克尔逊干涉的基本原理。"),
            ("◇ 异常诊断", "结合当前状态，看不到干涉条纹应该按什么顺序排查？"),
            ("△ 误差分析", "结合当前实验说明主要不确定度来源。"),
        ]:
            tk.Button(
                quick, text=label, command=lambda q=question: self.ask(q),
                relief=tk.FLAT, bd=0, bg="#ddeaff", fg="#174f8f",
                activebackground="#c9ddff", cursor="hand2",
                font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT, padx=(0, 4), ipady=2)

    def _build_input(self):
        self.input = tk.Text(
            self, height=3, wrap=tk.WORD, font=("Microsoft YaHei UI", 9),
            relief=tk.SOLID, bd=1, highlightthickness=1,
            highlightbackground="#c7d4e5", highlightcolor=self.BLUE)
        self.input.pack(fill=tk.X, padx=6, pady=(2, 3))
        self.input.insert("1.0", "描述你观察到的现象，或询问下一步实验操作……")
        self.input.configure(fg="#8a96a5")
        self.input.bind("<FocusIn>", self._clear_placeholder)
        self.input.bind("<Control-Return>", lambda event: self.ask())

        controls = tk.Frame(self, bg=self.BG)
        controls.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Checkbutton(
            controls, text="附加实时实验状态", variable=self.include_status_var,
            bg=self.BG, fg="#34495e", activebackground=self.BG,
            selectcolor="#ffffff", font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT)
        tk.Label(controls, textvariable=self.thinking_var, bg=self.BG,
                 fg=self.CYAN, font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT, padx=5)
        self.ask_button = tk.Button(
            controls, text="发送  Ctrl+Enter", command=self.ask,
            relief=tk.FLAT, bd=0, bg=self.BLUE, fg="#fff",
            activebackground="#0c61d6", activeforeground="#fff",
            cursor="hand2", font=("Microsoft YaHei UI", 8, "bold"))
        self.ask_button.pack(side=tk.RIGHT, ipadx=5, ipady=2)
        self.cancel_button = tk.Button(
            controls, text="停止", command=lambda: self.on_cancel(),
            relief=tk.FLAT, bd=0, bg="#dce3ec", fg="#52606d",
            activebackground="#cbd5e1", cursor="hand2",
            font=("Microsoft YaHei UI", 8), state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 5), ipadx=4, ipady=2)

    def _clear_placeholder(self, _event=None):
        if self.input.get("1.0", tk.END).strip().startswith("描述你观察到的现象"):
            self.input.delete("1.0", tk.END)
            self.input.configure(fg="#1d2b3a")

    def set_experiment_context(self, context: dict):
        camera = context.get("camera", {})
        vision = context.get("vision", {})
        motor = context.get("motor", {})
        detected = len(vision.get("detections", {}))
        self.context_var.set(
            f"实验状态  │  相机 {'运行' if camera.get('running') else '未开'} "
            f"{camera.get('fps', 0):.1f} FPS  │  模型 "
            f"{'就绪' if vision.get('model_loaded') else '未加载'}  │  "
            f"目标 {detected}  │  电机 {'已连接' if motor.get('connected') else '未连接'}"
        )

    def ask(self, preset: str | None = None):
        question = preset or self.input.get("1.0", tk.END).strip()
        if not question or question.startswith("描述你观察到的现象"):
            return
        if not preset:
            self.input.delete("1.0", tk.END)
        self.append("你", question)
        self.set_busy(True)
        self.on_ask(question, self.include_status_var.get())

    def append(self, role: str, text: str):
        self.output.configure(state=tk.NORMAL)
        start = self.output.index(tk.END)
        role_tag = {"你": "user_role", "助手": "assistant_role"}.get(role, "system_role")
        self.output.insert(tk.END, f"{role}  ", role_tag)
        self.output.insert(tk.END, time.strftime("%H:%M"), "timestamp")
        self.output.insert(tk.END, f"\n{text.strip()}\n\n", "message")
        self.output.configure(state=tk.DISABLED)
        if role == "你":
            self.output.see(tk.END)
        else:
            self.output.yview(start)

    def set_busy(self, busy: bool):
        self.ask_button.configure(state=tk.DISABLED if busy else tk.NORMAL,
                                  text="分析中…" if busy else "发送  Ctrl+Enter")
        self.cancel_button.configure(state=tk.NORMAL if busy else tk.DISABLED)
        if busy:
            self._thinking_step = 0
            self._animate_thinking()
        else:
            if self._thinking_job:
                self.after_cancel(self._thinking_job)
                self._thinking_job = None
            self.thinking_var.set("")

    def _animate_thinking(self):
        dots = "." * (self._thinking_step % 3 + 1)
        self.thinking_var.set(f"正在结合实验状态分析{dots}")
        self._thinking_step += 1
        self._thinking_job = self.after(420, self._animate_thinking)

    def test_connection(self):
        self.set_busy(True)
        self.status_var.set("●  正在连接 DeepSeek")
        self.status_label.configure(fg="#f0b429")
        self.on_test()

    def set_connection_status(self, online: bool):
        if online:
            self.status_var.set("●  DeepSeek 在线")
            self.status_label.configure(fg="#37d67a")
        else:
            self.status_var.set("●  本地模式 / 连接失败")
            self.status_label.configure(fg="#ff6b6b")
