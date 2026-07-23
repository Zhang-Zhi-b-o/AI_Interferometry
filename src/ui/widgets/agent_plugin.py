"""沉浸式实验辅助智能体面板。"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext, ttk

from src.ui.markdown_renderer import insert_markdown


class AgentPluginPanel(tk.LabelFrame):
    BG = "#f5f8fc"
    NAVY = "#17324d"
    BLUE = "#2563eb"
    CYAN = "#087f8c"
    TEXT = "#1f2937"
    MUTED = "#64748b"
    BORDER = "#dbe5f0"
    FONT = "Microsoft YaHei UI"

    def __init__(self, parent: tk.Widget):
        super().__init__(
            parent, text="", bg=self.BG, fg=self.NAVY,
            relief=tk.FLAT, bd=0, highlightthickness=0,
        )
        self.on_ask = lambda question, include_status: None
        self.on_test = lambda: None
        self.on_cancel = lambda: None
        self.include_status_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="●  尚未测试 DeepSeek")
        self.context_var = tk.StringVar(value="实验状态：等待连接仪器")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="实验进度 0% · 等待实时状态")
        self.guidance_var = tk.StringVar(value="下一步：打开设备后将显示现场指导")
        self.ai_state_var = tk.StringVar(value="●  AI 状态 · 就绪")
        self.thinking_var = tk.StringVar(value="")
        self._thinking_job = None
        self._thinking_step = 0
        self._active_task = "general"
        self._busy = False

        self._build_header()
        self._build_input()
        self._build_actions()
        self._build_chat()
        self.bind("<Configure>", self._on_panel_resize)
        self.append(
            "系统",
            "欢迎进入迈克尔逊实验工作台。我可以陪你预习实验、指导当前步骤、分析白光条纹、"
            "计算误差，并按固定格式整理实验报告。助手会读取现场状态，设备动作仍由操作区执行。",
        )

    def _build_header(self):
        header = tk.Frame(
            self, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        header.pack(fill=tk.X, padx=10, pady=(9, 0))
        identity = tk.Frame(header, bg="#ffffff")
        identity.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 6), pady=8)
        tk.Label(
            identity, text="MICHELSON COPILOT", bg="#ffffff", fg=self.NAVY,
            font=(self.FONT, 11, "bold"), anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            identity, text="实时状态分析与实验指导", bg="#ffffff", fg=self.MUTED,
            font=(self.FONT, 8), anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))
        status_row = tk.Frame(header, bg="#ffffff")
        status_row.pack(side=tk.RIGHT, padx=9, pady=7)
        self.status_label = tk.Label(
            status_row, textvariable=self.status_var, bg="#ffffff", fg="#b7791f",
            font=(self.FONT, 8, "bold"), anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(status_row, text="测试连接", command=self.test_connection,
                  relief=tk.FLAT, bd=0, bg="#e8f0fe", fg="#1d4ed8",
                  activebackground="#dbeafe", activeforeground="#1d4ed8",
                  cursor="hand2", font=(self.FONT, 8, "bold"),
                  padx=9, pady=4).pack(side=tk.LEFT)

        self.context_label = tk.Label(
            self, textvariable=self.context_var, bg="#eaf2ff",
            fg="#24558c", anchor="w", justify=tk.LEFT,
            font=(self.FONT, 8), padx=10,
        )
        self.context_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=5)
        progress_row = tk.Frame(
            self, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        self.progress_row = progress_row
        progress_row.pack(fill=tk.X, padx=10, pady=(6, 0))
        tk.Label(
            progress_row, textvariable=self.progress_text_var,
            bg="#ffffff", fg=self.NAVY, anchor="w",
            font=(self.FONT, 9, "bold"),
        ).pack(fill=tk.X, padx=10, pady=(7, 4))
        progress_style = ttk.Style(self)
        progress_style.configure(
            "Copilot.Horizontal.TProgressbar", troughcolor="#e7eef7",
            background=self.BLUE, bordercolor="#e7eef7",
            lightcolor=self.BLUE, darkcolor=self.BLUE, thickness=8,
        )
        ttk.Progressbar(
            progress_row, variable=self.progress_var, maximum=100,
            style="Copilot.Horizontal.TProgressbar",
        ).pack(fill=tk.X, padx=10, pady=(0, 5))
        self.guidance_label = tk.Label(
            progress_row, textvariable=self.guidance_var,
            bg="#ffffff", fg="#475569", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8),
        )
        self.guidance_label.pack(fill=tk.X, padx=10, pady=(0, 7))
        self.ai_state_label = tk.Label(
            self, textvariable=self.ai_state_var, bg="#eaf8f5", fg=self.CYAN,
            anchor="w", font=(self.FONT, 8, "bold"), padx=10)
        self.ai_state_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=4)

    def _build_chat(self):
        self.output = scrolledtext.ScrolledText(
            self, height=15, wrap=tk.WORD, bg="#ffffff", fg=self.TEXT,
            insertbackground=self.NAVY, relief=tk.FLAT, bd=0,
            highlightthickness=1, highlightbackground=self.BORDER,
            highlightcolor=self.BLUE,
            font=(self.FONT, 10), state=tk.DISABLED,
            padx=14, pady=12, spacing1=3, spacing2=2, spacing3=8)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 5))
        self.output.tag_configure("user_role", foreground=self.BLUE,
                                  font=(self.FONT, 10, "bold"))
        self.output.tag_configure("assistant_role", foreground=self.CYAN,
                                  font=(self.FONT, 10, "bold"))
        self.output.tag_configure("system_role", foreground="#7a5b00",
                                  font=(self.FONT, 10, "bold"))
        self.output.tag_configure("timestamp", foreground="#8995a3",
                                  font=(self.FONT, 8))
        self.output.tag_configure("message", foreground=self.TEXT,
                                  lmargin1=10, lmargin2=10, rmargin=10,
                                  spacing3=8)
        self.output.tag_configure("heading1", foreground=self.NAVY,
                                  font=(self.FONT, 15, "bold"),
                                  spacing1=9, spacing3=5)
        self.output.tag_configure("heading2", foreground="#173f6b",
                                  font=(self.FONT, 13, "bold"),
                                  spacing1=8, spacing3=4)
        self.output.tag_configure("heading3", foreground="#24558c",
                                  font=(self.FONT, 11, "bold"),
                                  spacing1=6, spacing3=3)
        self.output.tag_configure("bold", font=(self.FONT, 10, "bold"))
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
        self.output.tag_configure("table", font=(self.FONT, 10),
                                  background="#f7f9fc", lmargin1=12, lmargin2=12)
        self.output.tag_configure("divider", foreground="#c7d4e5")

    def _build_actions(self):
        action_shell = tk.Frame(self, bg=self.BG)
        self.action_shell = action_shell
        action_shell.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(
            action_shell, text="快捷任务", bg=self.BG, fg=self.NAVY, anchor="w",
            font=(self.FONT, 9, "bold"),
        ).pack(fill=tk.X, padx=12, pady=(2, 1))
        quick = tk.Frame(action_shell, bg=self.BG)
        quick.pack(fill=tk.X, padx=8, pady=(1, 5))
        for column in range(4):
            quick.columnconfigure(column, weight=1)
        for index, (label, question) in enumerate([
            ("预习指导", "带我预习迈克尔逊干涉实验的目的、原理、关键公式、安全事项和预期现象。"),
            ("过程辅助", "读取完整实时状态和近期日志，按五步实验流程判断我现在处于哪一步，并只告诉我接下来应执行的动作、操作顺序和完成标志。"),
            ("误差计算", "根据当前记录和我提供的数据进行误差与不确定度计算；缺少数据时明确列出缺少项。"),
            ("生成报告", "按固定格式生成迈克尔逊干涉实验报告，使用已有状态与读数，缺失内容标记为待补充。"),
        ]):
            button = tk.Button(
                quick, text=label, command=lambda q=question: self.ask(q),
                relief=tk.FLAT, bd=0, bg="#eaf2ff", fg="#1e4f87",
                activebackground="#dbeafe", activeforeground="#163f70",
                cursor="hand2", highlightthickness=1,
                highlightbackground="#d6e3f5", font=(self.FONT, 8, "bold"))
            button.grid(row=index // 4, column=index % 4, sticky="ew",
                        padx=3, pady=3, ipady=4)

    def _build_input(self):
        composer = tk.Frame(
            self, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        composer.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(2, 9))
        self.input = tk.Text(
            composer, height=3, wrap=tk.WORD, bg="#ffffff",
            fg=self.TEXT, insertbackground=self.NAVY, font=(self.FONT, 10),
            relief=tk.FLAT, bd=0, highlightthickness=1,
            highlightbackground="#cbd8e8", highlightcolor=self.BLUE,
            padx=8, pady=7)
        self.input.pack(fill=tk.X, padx=7, pady=(7, 5))
        self.input.insert("1.0", "描述你观察到的现象，或询问下一步实验操作……")
        self.input.configure(fg="#8794a5")
        self.input.bind("<FocusIn>", self._clear_placeholder)
        self.input.bind("<Control-Return>", lambda event: self.ask())

        controls = tk.Frame(composer, bg="#ffffff")
        controls.pack(fill=tk.X, padx=7, pady=(0, 7))
        tk.Checkbutton(
            controls, text="附加实时实验状态", variable=self.include_status_var,
            bg="#ffffff", fg="#475569", activebackground="#ffffff",
            selectcolor="#ffffff", font=(self.FONT, 9)).pack(side=tk.LEFT)
        tk.Label(controls, textvariable=self.thinking_var, bg="#ffffff",
                 fg=self.CYAN, font=(self.FONT, 8)).pack(side=tk.LEFT, padx=7)
        self.ask_button = tk.Button(
            controls, text="发送  Ctrl+Enter", command=self.ask,
            relief=tk.FLAT, bd=0, bg=self.BLUE, fg="#fff",
            activebackground="#0c61d6", activeforeground="#fff",
            cursor="hand2", font=(self.FONT, 9, "bold"))
        self.ask_button.pack(side=tk.RIGHT, ipadx=8, ipady=4)
        self.cancel_button = tk.Button(
            controls, text="停止", command=lambda: self.on_cancel(),
            relief=tk.FLAT, bd=0, bg="#e8edf3", fg="#52606d",
            activebackground="#cbd5e1", cursor="hand2",
            font=(self.FONT, 9), state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 6), ipadx=6, ipady=4)

    def _on_panel_resize(self, event) -> None:
        wraplength = max(240, int(event.width) - 38)
        self.context_label.configure(wraplength=wraplength)
        self.guidance_label.configure(wraplength=wraplength)
        # 紧凑尺寸优先保留聊天记录、输入框和发送按钮。快捷任务与详细
        # 进度卡仅在空间足够时显示，重新放大后自动恢复。
        if event.height < 570:
            if self.action_shell.winfo_manager():
                self.action_shell.pack_forget()
        elif not self.action_shell.winfo_manager():
            self.action_shell.pack(
                side=tk.BOTTOM, fill=tk.X, before=self.output)

        if event.height < 470:
            if self.progress_row.winfo_manager():
                self.progress_row.pack_forget()
        elif not self.progress_row.winfo_manager():
            self.progress_row.pack(
                fill=tk.X, padx=10, pady=(6, 0), before=self.ai_state_label)

    def _clear_placeholder(self, _event=None):
        if self.input.get("1.0", tk.END).strip().startswith("描述你观察到的现象"):
            self.input.delete("1.0", tk.END)
            self.input.configure(fg=self.TEXT)

    def set_experiment_context(self, context: dict):
        camera = context.get("camera", {})
        vision = context.get("vision", {})
        motor = context.get("motor", {})
        progress = context.get("experiment_progress", {})
        detected = len(vision.get("detections", {}))
        self.context_var.set(
            f"{progress.get('step_number', '--')}/5 {progress.get('stage', '实验状态')}  │  "
            f"双相机{'就绪' if camera.get('interferometer_running') and camera.get('micrometer_running') else '未就绪'}  │  "
            f"模型{'已加载' if vision.get('model_loaded') else '未加载'}  │  "
            f"目标 {detected}  │  电机{'已连接' if motor.get('connected') else '未连接'}"
        )
        percent = max(0, min(100, int(progress.get("progress_percent", 0))))
        self.progress_var.set(percent)
        self.progress_text_var.set(
            f"实验进度 {percent}% · {progress.get('stage', '等待状态')}")
        self.guidance_var.set(
            f"下一步：{progress.get('next_action', '等待实时状态')}  ｜  "
            f"完成：{progress.get('completion_criterion', '--')}")

    @property
    def is_busy(self) -> bool:
        return self._busy

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
