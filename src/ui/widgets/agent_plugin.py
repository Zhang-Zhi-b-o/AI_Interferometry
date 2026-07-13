"""沉浸式实验辅助智能体面板。"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext

from src.ui.markdown_renderer import insert_markdown


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
        self.ai_state_var = tk.StringVar(value="●  AI 状态 · 就绪")
        self.thinking_var = tk.StringVar(value="")
        self._thinking_job = None
        self._thinking_step = 0
        self._active_task = "general"
        self._busy = False

        self._build_header()
        self._build_chat()
        self._build_actions()
        self._build_input()
        self.append(
            "系统",
            "欢迎进入迈克尔逊实验工作台。我可以陪你预习实验、指导当前步骤、分析白光条纹、"
            "计算误差，并按固定格式整理实验报告。硬件操作仍由你确认执行。",
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
        self.ai_state_label = tk.Label(
            self, textvariable=self.ai_state_var, bg="#edf8f7", fg=self.CYAN,
            anchor="w", font=("Microsoft YaHei UI", 8, "bold"))
        self.ai_state_label.pack(fill=tk.X, padx=6, pady=(3, 0), ipady=4)

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
        self.output.tag_configure("heading1", foreground=self.NAVY,
                                  font=("Microsoft YaHei UI", 14, "bold"),
                                  spacing1=9, spacing3=5)
        self.output.tag_configure("heading2", foreground="#173f6b",
                                  font=("Microsoft YaHei UI", 12, "bold"),
                                  spacing1=8, spacing3=4)
        self.output.tag_configure("heading3", foreground="#24558c",
                                  font=("Microsoft YaHei UI", 10, "bold"),
                                  spacing1=6, spacing3=3)
        self.output.tag_configure("bold", font=("Microsoft YaHei UI", 9, "bold"))
        self.output.tag_configure("bullet", lmargin1=18, lmargin2=34, spacing3=2)
        self.output.tag_configure("code", font=("Consolas", 9),
                                  background="#eef2f6", foreground="#9b2c2c")
        self.output.tag_configure("code_block", font=("Consolas", 9),
                                  background="#eef2f6", lmargin1=18, lmargin2=18)
        self.output.tag_configure("math", font=("Cambria Math", 10),
                                  foreground="#53389e")
        self.output.tag_configure("math_display", font=("Cambria Math", 11),
                                  foreground="#53389e", justify=tk.CENTER,
                                  spacing1=5, spacing3=5)
        self.output.tag_configure("quote", foreground="#52606d",
                                  lmargin1=20, lmargin2=20, background="#f3f6f9")
        self.output.tag_configure("table", font=("Microsoft YaHei UI", 9),
                                  background="#f7f9fc", lmargin1=12, lmargin2=12)
        self.output.tag_configure("divider", foreground="#c7d4e5")

    def _build_actions(self):
        tk.Label(self, text="快捷任务", bg=self.BG, fg="#637083", anchor="w",
                 font=("Microsoft YaHei UI", 8, "bold")).pack(fill=tk.X, padx=9)
        quick = tk.Frame(self, bg=self.BG)
        quick.pack(fill=tk.X, padx=6, pady=(2, 3))
        quick.columnconfigure(0, weight=1)
        quick.columnconfigure(1, weight=1)
        for index, (label, question) in enumerate([
            ("◎ 实验预习", "带我预习迈克尔逊干涉实验，包括目的、原理、关键公式、安全事项和预期现象。"),
            ("◇ 过程指导", "结合当前实验状态判断进展，并告诉我接下来应该做什么、观察什么。"),
            ("△ 误差计算", "请根据我提供的实验数据进行误差和不确定度计算；缺少数据时列出需要补充的项目。"),
            ("▣ 生成报告", "请按固定格式生成迈克尔逊干涉实验报告，缺少的内容明确标记为待补充。"),
        ]):
            button = tk.Button(
                quick, text=label, command=lambda q=question: self.ask(q),
                relief=tk.FLAT, bd=0, bg="#ddeaff", fg="#174f8f",
                activebackground="#c9ddff", cursor="hand2",
                font=("Microsoft YaHei UI", 8))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2, ipady=2)

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
        if self._busy:
            return
        question = preset or self.input.get("1.0", tk.END).strip()
        if not question or question.startswith("描述你观察到的现象"):
            return
        if not preset:
            self.input.delete("1.0", tk.END)
        lowered = question.lower()
        if any(word in lowered for word in ("报告", "report")):
            self._active_task = "report"
        elif any(word in lowered for word in ("误差", "不确定度", "计算", "数据处理")):
            self._active_task = "calculation"
        elif any(word in lowered for word in ("预习", "原理", "目的", "公式")):
            self._active_task = "preview"
        else:
            self._active_task = "general"
        self.append("你", question)
        self.set_busy(True)
        self.on_ask(question, self.include_status_var.get())

    def append(self, role: str, text: str):
        self.output.configure(state=tk.NORMAL)
        start = self.output.index(tk.END)
        role_tag = {"你": "user_role", "助手": "assistant_role"}.get(role, "system_role")
        self.output.insert(tk.END, f"{role}  ", role_tag)
        self.output.insert(tk.END, time.strftime("%H:%M"), "timestamp")
        self.output.insert(tk.END, "\n", "message")
        if role == "助手":
            insert_markdown(self.output, text.strip(), "message")
            self.output.insert(tk.END, "\n", "message")
        else:
            self.output.insert(tk.END, f"{text.strip()}\n\n", "message")
        self.output.configure(state=tk.DISABLED)
        if role == "你":
            self.output.see(tk.END)
        else:
            self.output.yview(start)

    def set_busy(self, busy: bool):
        self._busy = busy
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
            self.set_ai_state("就绪", "idle")

    def _animate_thinking(self):
        messages = {
            "preview": ("检索预习资料", "梳理实验原理", "整理关键公式", "准备预习指导"),
            "calculation": ("读取实验数据", "选择计算公式", "核对单位与有效数字", "计算误差与不确定度"),
            "report": ("读取实验资料", "整理报告结构", "核对数据缺口", "生成实验报告"),
            "connection": ("连接 DeepSeek", "验证模型响应"),
            "general": ("读取实验状态", "检索实验资料", "分析与推理", "组织回答"),
        }[self._active_task]
        message = messages[self._thinking_step % len(messages)]
        dots = "." * (self._thinking_step % 3 + 1)
        self.thinking_var.set(message + dots)
        self.set_ai_state(message + dots, "working")
        self._thinking_step += 1
        self._thinking_job = self.after(850, self._animate_thinking)

    def test_connection(self):
        self._active_task = "connection"
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

    def set_ai_state(self, text: str, kind: str = "idle"):
        colors = {
            "idle": ("#edf8f7", self.CYAN, "●"),
            "working": ("#fff8e6", "#a15c00", "◌"),
            "success": ("#eaf8ef", "#18794e", "●"),
            "warning": ("#fff4e5", "#9a6700", "●"),
            "error": ("#fff0f0", "#c53030", "●"),
        }
        background, foreground, marker = colors.get(kind, colors["idle"])
        self.ai_state_var.set(f"{marker}  AI 状态 · {text}")
        self.ai_state_label.configure(bg=background, fg=foreground)
