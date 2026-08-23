"""沉浸式实验辅助智能体面板。"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext, ttk

from src.agent.tools import diagnose_context, parse_options
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
        self.on_confirm_motion = lambda tool_name: None
        self.on_reject_motion = lambda tool_name: None
        self.on_emergency_stop = lambda: None
        self.on_toggle_autonomous = lambda enabled: None
        self.on_toggle_dry_run = lambda enabled: None
        self.include_status_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="●  尚未测试 DeepSeek")
        self.context_var = tk.StringVar(value="实验状态：等待连接仪器")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="实验进度 0% · 等待实时状态")
        self.guidance_var = tk.StringVar(value="下一步：打开设备后将显示现场指导")
        self.ai_insight_var = tk.StringVar(value="AI 洞察 · 等待实时状态")
        self.suggestion_var = tk.StringVar(value="下一步任务：打开设备后将显示现场指导")
        self.ai_state_var = tk.StringVar(value="●  AI 状态 · 就绪")
        self.thinking_var = tk.StringVar(value="")
        self._thinking_job = None
        self._thinking_step = 0
        self._active_task = "general"
        self._busy = False
        self.autonomous_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.plan_var = tk.StringVar(value="")
        self.activity_var = tk.StringVar(value="")
        self._motion_tool_name = ""
        self._activity_log: list[str] = []
        self._font_size = 10
        self._section_order = ("status", "quick", "chat", "input")
        self._section_heights = {
            "status": 205, "quick": 115, "chat": 220, "input": 145}
        self._collapsed_sections: set[str] = set()

        self.content_pane = tk.PanedWindow(
            self, orient=tk.VERTICAL, bg="#c7d4e3", bd=0,
            sashwidth=5, sashrelief=tk.FLAT, showhandle=False,
        )
        self.status_area = tk.Frame(self.content_pane, bg=self.BG)
        self.quick_area = tk.Frame(self.content_pane, bg=self.BG)
        self.chat_area = tk.Frame(self.content_pane, bg=self.BG)
        self.input_area = tk.Frame(self.content_pane, bg=self.BG)
        self._section_frames = {
            "status": self.status_area,
            "quick": self.quick_area,
            "chat": self.chat_area,
            "input": self.input_area,
        }
        self._section_buttons: dict[str, tk.Button] = {}
        self.status_content = self._make_scrollable_area(self.status_area)
        self.quick_content = self._make_scrollable_area(self.quick_area)
        self._build_confirmation_row()
        self._build_header()
        self._build_agent_controls()
        self._build_section_toolbar()
        self._build_input()
        self._build_actions()
        self._build_chat()
        self.content_pane.add(self.status_area, minsize=60, stretch="never")
        self.content_pane.add(
            self.quick_area, minsize=self._section_heights["quick"],
            stretch="never")
        self.content_pane.add(self.chat_area, minsize=110, stretch="always")
        self.content_pane.add(self.input_area, minsize=95, stretch="never")
        self.content_pane.pack(fill=tk.BOTH, expand=True)
        self.after_idle(self._set_initial_pane_ratio)
        self.after(60, self._lock_quick_area_height)
        self.content_pane.bind(
            "<ButtonRelease-1>",
            lambda _event: self.after_idle(self._lock_quick_area_height),
            add="+")
        self.bind("<Configure>", self._on_panel_resize)
        self.append(
            "系统",
            "欢迎进入迈克尔逊实验工作台。我可以陪你预习实验、指导当前步骤、分析白光条纹、"
            "计算误差，并按固定格式整理实验报告。助手会读取现场状态，设备动作仍由操作区执行。",
        )

    def _make_scrollable_area(self, parent: tk.Widget) -> tk.Frame:
        """创建拥有独立滚动条和鼠标滚轮行为的轻量区域。"""
        canvas = tk.Canvas(
            parent, bg=self.BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content = tk.Frame(canvas, bg=self.BG)
        window_id = canvas.create_window(
            (0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda event, c=canvas, item=window_id:
                c.itemconfigure(item, width=event.width))

        def on_mousewheel(event, target=canvas):
            target.yview_scroll(-1 * int(event.delta / 120), "units")
            return "break"

        parent.bind(
            "<Enter>",
            lambda _event, c=canvas, handler=on_mousewheel:
                c.bind_all("<MouseWheel>", handler))
        parent.bind(
            "<Leave>",
            lambda _event, c=canvas: c.unbind_all("<MouseWheel>"))
        return content

    def _build_confirmation_row(self):
        """运动工具待确认行：默认隐藏，智能体请求运动时弹出。"""
        self.confirm_row = tk.Frame(
            self, bg="#fff4e5", highlightthickness=1,
            highlightbackground="#f0b429",
        )
        self._motion_summary_var = tk.StringVar(value="")
        self._motion_summary_var.set("")
        self._motion_summary_label = tk.Label(
            self.confirm_row, textvariable=self._motion_summary_var,
            bg="#fff4e5", fg="#9a6700", anchor="w", justify=tk.LEFT,
            font=(self.FONT, 9, "bold"),
        )
        self._motion_summary_label.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=8)
        self.confirm_btn = tk.Button(
            self.confirm_row, text="确认执行",
            command=self._on_confirm_click,
            relief=tk.FLAT, bd=0, bg="#18794e", fg="#fff",
            activebackground="#116149", activeforeground="#fff",
            cursor="hand2", font=(self.FONT, 9, "bold"),
        )
        self.confirm_btn.pack(side=tk.RIGHT, padx=(6, 10), pady=8, ipadx=8, ipady=3)
        self.reject_btn = tk.Button(
            self.confirm_row, text="拒绝",
            command=self._on_reject_click,
            relief=tk.FLAT, bd=0, bg="#e8edf3", fg="#52606d",
            activebackground="#cbd5e1", cursor="hand2",
            font=(self.FONT, 9),
        )
        self.reject_btn.pack(side=tk.RIGHT, padx=(0, 2), pady=8, ipadx=8, ipady=3)

    def _build_agent_controls(self):
        """自主执行开关 + 仅规划开关 + 急停按钮 + 计划与工具活动流。"""
        controls = tk.Frame(
            self.status_content, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        controls.pack(fill=tk.X, padx=10, pady=(6, 0))
        tk.Label(
            controls, text="智能体执行", bg="#ffffff", fg=self.NAVY,
            anchor="w", font=(self.FONT, 9, "bold"),
        ).pack(fill=tk.X, padx=10, pady=(7, 2))
        row = tk.Frame(controls, bg="#ffffff")
        row.pack(fill=tk.X, padx=10, pady=(0, 7))
        tk.Checkbutton(
            row, text="自主执行", variable=self.autonomous_var,
            command=lambda: self.on_toggle_autonomous(self.autonomous_var.get()),
            bg="#ffffff", fg="#475569", activebackground="#ffffff",
            selectcolor="#ffffff", font=(self.FONT, 9)).pack(side=tk.LEFT)
        tk.Checkbutton(
            row, text="仅规划", variable=self.dry_run_var,
            command=lambda: self.on_toggle_dry_run(self.dry_run_var.get()),
            bg="#ffffff", fg="#475569", activebackground="#ffffff",
            selectcolor="#ffffff", font=(self.FONT, 9)).pack(side=tk.LEFT, padx=(8, 0))
        self.emergency_stop_btn = tk.Button(
            row, text="急停", command=lambda: self.on_emergency_stop(),
            relief=tk.FLAT, bd=0, bg="#c53030", fg="#fff",
            activebackground="#a02424", activeforeground="#fff",
            cursor="hand2", font=(self.FONT, 9, "bold"),
        )
        self.emergency_stop_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)

        self.plan_label = tk.Label(
            self.status_content, textvariable=self.plan_var, bg="#f6f9fd",
            fg="#24558c", anchor="w", justify=tk.LEFT, wraplength=420,
            font=(self.FONT, 8), padx=10,
        )
        self.plan_label.pack(fill=tk.X, padx=10, pady=(5, 0), ipady=4)
        self.activity_label = tk.Label(
            self.status_content, textvariable=self.activity_var, bg="#f6f9fd",
            fg="#475569", anchor="w", justify=tk.LEFT, wraplength=420,
            font=(self.FONT, 8), padx=10,
        )
        self.activity_label.pack(fill=tk.X, padx=10, pady=(3, 0), ipady=4)

    def _on_confirm_click(self):
        tool_name = self._motion_tool_name
        self.hide_motion_confirmation()
        if tool_name:
            self.on_confirm_motion(tool_name)

    def _on_reject_click(self):
        tool_name = self._motion_tool_name
        self.hide_motion_confirmation()
        if tool_name:
            self.on_reject_motion(tool_name)

    def show_motion_confirmation(self, tool_name: str, summary: str) -> None:
        """弹出运动工具确认行；须在主线程调用。"""
        self._motion_tool_name = tool_name
        self._motion_summary_var.set(f"⚠ 待确认运动操作：{summary}")
        self.confirm_row.pack(side=tk.TOP, fill=tk.X, before=self.header)

    def hide_motion_confirmation(self) -> None:
        self._motion_tool_name = ""
        self._motion_summary_var.set("")
        self.confirm_row.pack_forget()

    def set_plan(self, plan: str) -> None:
        self.plan_var.set(plan.strip() if plan else "")

    def append_tool_activity(self, text: str) -> None:
        """向工具活动流追加一条，只保留最近若干条。"""
        self._activity_log.append(text)
        self._activity_log = self._activity_log[-8:]
        self.activity_var.set("工具活动\n" + "\n".join(
            f"· {line}" for line in self._activity_log))

    def _build_header(self):
        header = tk.Frame(
            self, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        self.header = header
        header.pack(fill=tk.X, padx=10, pady=(9, 0))
        identity = tk.Frame(header, bg="#ffffff")
        identity.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=6)
        tk.Label(
            identity, text="MICHELSON COPILOT", bg="#ffffff", fg=self.NAVY,
            font=(self.FONT, 10, "bold"), anchor="w",
        ).pack(fill=tk.X)
        status_row = tk.Frame(header, bg="#ffffff")
        status_row.pack(side=tk.RIGHT, padx=7, pady=5)
        tk.Button(
            status_row, text="A−", command=lambda: self.change_font_size(-1),
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.NAVY,
            activebackground="#dbe5f0", cursor="hand2",
            font=(self.FONT, 8, "bold"), padx=6, pady=4,
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(
            status_row, text="A", command=self.reset_font_size,
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.NAVY,
            activebackground="#dbe5f0", cursor="hand2",
            font=(self.FONT, 8, "bold"), padx=6, pady=4,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            status_row, text="A+", command=lambda: self.change_font_size(1),
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.NAVY,
            activebackground="#dbe5f0", cursor="hand2",
            font=(self.FONT, 8, "bold"), padx=6, pady=4,
        ).pack(side=tk.LEFT, padx=(2, 7))
        self.status_label = tk.Label(
            self.status_content, textvariable=self.status_var,
            bg="#fff8e6", fg="#b7791f",
            font=(self.FONT, 8, "bold"), anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=(5, 0), ipady=3)
        tk.Button(status_row, text="连接", command=self.test_connection,
                  relief=tk.FLAT, bd=0, bg="#e8f0fe", fg="#1d4ed8",
                  activebackground="#dbeafe", activeforeground="#1d4ed8",
                  cursor="hand2", font=(self.FONT, 8, "bold"),
                  padx=9, pady=4).pack(side=tk.LEFT)

        self.context_label = tk.Label(
            self.status_content, textvariable=self.context_var, bg="#eaf2ff",
            fg="#24558c", anchor="w", justify=tk.LEFT,
            font=(self.FONT, 8), padx=10,
        )
        self.context_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=5)
        progress_row = tk.Frame(
            self.status_content, bg="#ffffff", highlightthickness=1,
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
            self.status_content, textvariable=self.ai_state_var,
            bg="#eaf8f5", fg=self.CYAN,
            anchor="w", font=(self.FONT, 8, "bold"), padx=10)
        self.ai_state_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=4)
        self.ai_insight_label = tk.Label(
            self.status_content, textvariable=self.ai_insight_var,
            bg="#f3f0ff", fg="#5b3cc4", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8), padx=10)
        self.ai_insight_label.pack(fill=tk.X, padx=10, pady=(4, 0), ipady=4)
        self.suggestion_label = tk.Label(
            self.status_content, textvariable=self.suggestion_var,
            bg="#eef7f0", fg="#166534", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8), padx=10)
        self.suggestion_label.pack(fill=tk.X, padx=10, pady=(4, 6), ipady=4)

    def _build_section_toolbar(self) -> None:
        toolbar = tk.Frame(self, bg="#edf3fa")
        toolbar.pack(fill=tk.X, padx=10, pady=(5, 0))
        labels = {
            "status": "状态栏",
            "quick": "快捷指令栏",
            "chat": "对话栏",
            "input": "输入栏",
        }
        for index, name in enumerate(self._section_order):
            toolbar.columnconfigure(index, weight=1)
            button = tk.Button(
                toolbar, text=f"{labels[name]} －",
                command=lambda section=name: self.toggle_section(section),
                relief=tk.FLAT, bd=0, bg="#e5edf7", fg="#234a73",
                activebackground="#d5e3f3", activeforeground="#17324d",
                cursor="hand2", font=(self.FONT, 8, "bold"), pady=4,
            )
            button.grid(row=0, column=index, sticky="ew", padx=2, pady=3)
            self._section_buttons[name] = button

    def _build_chat(self):
        self.output = scrolledtext.ScrolledText(
            self.chat_area, height=15, wrap=tk.WORD, bg="#ffffff", fg=self.TEXT,
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

    @property
    def font_size(self) -> int:
        return self._font_size

    def change_font_size(self, delta: int) -> None:
        self.set_font_size(self._font_size + int(delta))

    def reset_font_size(self) -> None:
        self.set_font_size(10)

    def set_font_size(self, size: int) -> None:
        """调整助手对话、Markdown 内容和输入框字号。"""
        self._font_size = max(8, min(24, int(size)))
        size = self._font_size
        self.output.configure(font=(self.FONT, size))
        self.input.configure(font=(self.FONT, size))
        tag_fonts = {
            "user_role": (self.FONT, size, "bold"),
            "assistant_role": (self.FONT, size, "bold"),
            "system_role": (self.FONT, size, "bold"),
            "timestamp": (self.FONT, max(7, size - 2)),
            "heading1": (self.FONT, size + 5, "bold"),
            "heading2": (self.FONT, size + 3, "bold"),
            "heading3": (self.FONT, size + 1, "bold"),
            "bold": (self.FONT, size, "bold"),
            "code": ("Consolas", max(8, size - 1)),
            "code_block": ("Consolas", max(8, size - 1)),
            "math": ("Cambria Math", size),
            "math_display": ("Cambria Math", size + 1),
            "table": (self.FONT, size),
        }
        for tag, font in tag_fonts.items():
            self.output.tag_configure(tag, font=font)

    def _build_actions(self):
        action_shell = tk.Frame(self.quick_content, bg=self.BG)
        self.action_shell = action_shell
        action_shell.pack(fill=tk.X)
        tk.Label(
            action_shell, text="引导流程", bg=self.BG, fg=self.NAVY, anchor="w",
            font=(self.FONT, 9, "bold"),
        ).pack(fill=tk.X, padx=12, pady=(2, 1))
        quick = tk.Frame(action_shell, bg=self.BG)
        quick.pack(fill=tk.X, padx=8, pady=(1, 5))
        for column in range(3):
            quick.columnconfigure(column, weight=1)
        for index, (label, question) in enumerate([
            ("预习指导", "带我预习迈克尔逊干涉实验的目的、原理、关键公式、安全事项和预期现象，最后用两个自检问题考我。"),
            ("调出条纹", "请一步一步带我调出白光干涉条纹：从激光非定域条纹、等厚直条纹，到白光彩色条纹和中央黑条纹，每步都告诉我操作和观察标志，并问我观察到什么。"),
            ("过程辅助", "读取完整实时状态和近期日志，判断我现在处于哪一步，然后只给我当前这一步的操作和观察标志，等我确认后再继续。"),
            ("误差计算", "根据当前记录和我提供的数据进行误差与不确定度计算；缺少数据时明确列出缺少项。"),
            ("生成报告", "按固定格式生成迈克尔逊干涉实验报告，使用已有状态与读数，缺失内容标记为待补充。"),
        ]):
            button = tk.Button(
                quick, text=label, command=lambda q=question: self.ask(q),
                relief=tk.FLAT, bd=0, bg="#eaf2ff", fg="#1e4f87",
                activebackground="#dbeafe", activeforeground="#163f70",
                cursor="hand2", highlightthickness=1,
                highlightbackground="#d6e3f5", font=(self.FONT, 8, "bold"))
            button.grid(row=index // 3, column=index % 3, sticky="ew",
                        padx=3, pady=3, ipady=4)

    def _build_input(self):
        # 可点选项按钮行：默认隐藏，仅当助手回答带「【选项】」标记时由
        # _render_options 显示在输入框上方，供实验者点选反馈。
        self._options_row = tk.Frame(self.input_area, bg="#ffffff")
        composer = tk.Frame(
            self.input_area, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        self.composer = composer
        composer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 7))
        input_shell = tk.Frame(composer, bg="#ffffff")
        input_shell.pack(fill=tk.BOTH, expand=True, padx=7, pady=(7, 5))
        self.input = tk.Text(
            input_shell, height=3, wrap=tk.WORD, bg="#ffffff",
            fg=self.TEXT, insertbackground=self.NAVY, font=(self.FONT, 10),
            relief=tk.FLAT, bd=0, highlightthickness=1,
            highlightbackground="#cbd8e8", highlightcolor=self.BLUE,
            padx=8, pady=7)
        input_scrollbar = ttk.Scrollbar(
            input_shell, orient=tk.VERTICAL, command=self.input.yview)
        self.input.configure(yscrollcommand=input_scrollbar.set)
        input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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

    def _set_initial_pane_ratio(self) -> None:
        if self.content_pane.winfo_height() <= 1:
            return
        available = self.content_pane.winfo_height()
        input_top = max(300, available - self._section_heights["input"])
        chat_top = max(170, input_top - self._section_heights["chat"])
        quick_top = max(80, chat_top - self._section_heights["quick"])
        for index, position in enumerate((quick_top, chat_top, input_top)):
            try:
                self.content_pane.sash_place(index, 0, position)
            except tk.TclError:
                break
        self.after_idle(self._lock_quick_area_height)

    def _lock_quick_area_height(self) -> None:
        """快捷指令栏保持固定高度，拖动只改变相邻的可调区域。"""
        panes = tuple(map(str, self.content_pane.panes()))
        quick_path = str(self.quick_area)
        if quick_path not in panes or len(panes) < 2:
            return
        quick_index = panes.index(quick_path)
        fixed_height = self._section_heights["quick"]
        sash_width = int(self.content_pane.cget("sashwidth"))
        try:
            if quick_index < len(panes) - 1:
                top = (
                    0 if quick_index == 0
                    else self.content_pane.sash_coord(
                        quick_index - 1)[1] + sash_width
                )
                self.content_pane.sash_place(
                    quick_index, 0, top + fixed_height)
            else:
                bottom = self.content_pane.winfo_height()
                self.content_pane.sash_place(
                    quick_index - 1, 0, bottom - fixed_height)
        except tk.TclError:
            return

    def toggle_section(self, name: str) -> None:
        """折叠或展开一个独立区域，其他区域仍可继续拖动调整。"""
        if name not in self._section_frames:
            return
        frame = self._section_frames[name]
        labels = {
            "status": "状态栏", "quick": "快捷指令栏",
            "chat": "对话栏", "input": "输入栏",
        }
        if name in self._collapsed_sections:
            panes = tuple(map(str, self.content_pane.panes()))
            next_frame = None
            start = self._section_order.index(name) + 1
            for later_name in self._section_order[start:]:
                candidate = self._section_frames[later_name]
                if str(candidate) in panes:
                    next_frame = candidate
                    break
            options = {
                "minsize": 55 if name in {"status", "quick"} else 90,
                "stretch": "always" if name == "chat" else "never",
            }
            if next_frame is None:
                self.content_pane.add(frame, **options)
            else:
                self.content_pane.add(frame, before=next_frame, **options)
            self._collapsed_sections.remove(name)
            self._section_buttons[name].configure(text=f"{labels[name]} －")
            if name == "quick":
                self.after_idle(self._lock_quick_area_height)
        else:
            self._remember_section_heights()
            self.content_pane.forget(frame)
            self._collapsed_sections.add(name)
            self._section_buttons[name].configure(text=f"{labels[name]} ＋")

    def _remember_section_heights(self) -> None:
        for name, frame in self._section_frames.items():
            if name == "quick":
                continue
            if name not in self._collapsed_sections and frame.winfo_height() > 1:
                self._section_heights[name] = frame.winfo_height()

    def _on_panel_resize(self, event) -> None:
        wraplength = max(240, int(event.width) - 38)
        self.context_label.configure(wraplength=wraplength)
        self.guidance_label.configure(wraplength=wraplength)
        self.ai_insight_label.configure(wraplength=wraplength)
        self.suggestion_label.configure(wraplength=wraplength)
        self.plan_label.configure(wraplength=wraplength)
        self.activity_label.configure(wraplength=wraplength)
        # 确认行右侧固定两个按钮，标签可用宽度更窄，需扣除约 130px
        self._motion_summary_label.configure(
            wraplength=max(120, wraplength - 130))
        if "quick" not in self._collapsed_sections:
            self.after_idle(self._lock_quick_area_height)

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
        self.ai_insight_var.set(diagnose_context(context))

    def set_suggestion(self, text: str, source: str = "") -> None:
        """设置主动建议标签；source 用于标注来源（DeepSeek / 本地提示）。"""
        prefix = f"{source} · " if source else ""
        self.suggestion_var.set(prefix + text.strip())

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
        options: list[str] = []
        if role == "助手":
            text, options = parse_options(text)
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
        if options:
            self._render_options(options)

    def _render_options(self, options: list[str]) -> None:
        """在输入框上方渲染一排可点选项按钮。"""
        self._clear_options()
        if not options:
            return
        for option in options:
            button = tk.Button(
                self._options_row, text=option,
                command=lambda o=option: self._choose_option(o),
                relief=tk.FLAT, bd=0, bg="#e8f4f2", fg=self.CYAN,
                activebackground="#cfe9e5", activeforeground=self.NAVY,
                cursor="hand2", highlightthickness=1,
                highlightbackground="#bfe0db", font=(self.FONT, 8, "bold"))
            button.pack(side=tk.LEFT, padx=(0, 6), pady=3, ipadx=8, ipady=3)
        self._options_row.pack(
            side=tk.TOP, fill=tk.X, padx=10, pady=(4, 0), before=self.composer)

    def _clear_options(self) -> None:
        for child in self._options_row.winfo_children():
            child.destroy()
        self._options_row.pack_forget()

    def _choose_option(self, option: str) -> None:
        """点选选项即作为对上一个问题的反馈发出。"""
        if self._busy:
            return
        self._clear_options()
        self._active_task = "general"
        self.append("你", f"（选择）{option}")
        self.set_busy(True)
        self.on_ask(
            f"（选择）{option}（这是对你上一个问题的回答）",
            self.include_status_var.get())

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
