"""沉浸式实验辅助智能体面板。"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk

from src.agent.conversation_export import ConversationEntry
from src.agent.experiment_guidance import INTENT_LABELS
from src.agent.tools import diagnose_context, parse_options
from src.ui.markdown_renderer import insert_markdown
from src.vision.fringe_guidance import render_laser_alignment_instruction


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
        self.on_set_guidance_stage = lambda stage: None
        self.on_apply_guidance = lambda: None
        self.on_auto_center = lambda command: None
        self.on_set_intent = lambda kind: None
        self.on_set_response_mode = lambda mode: None
        self.on_mark_adjustment = lambda: None
        self.on_compare_adjustment = lambda: None
        self.on_review_image = lambda: None
        self.on_toggle_laser_alignment = lambda enabled: None
        self.on_toggle_laser_ai_guidance = lambda enabled: None
        self.on_laser_recheck = lambda: None
        self.on_export_experiment_record = lambda: None
        self.on_export_chat = lambda: None
        self.include_status_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="●  尚未测试 DeepSeek")
        self.context_var = tk.StringVar(value="实验状态：等待连接仪器")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="实验进度 0% · 等待实时状态")
        self.guidance_var = tk.StringVar(value="下一步：打开设备后将显示现场指导")
        self.ai_insight_var = tk.StringVar(value="AI 洞察 · 等待实时状态")
        self.suggestion_var = tk.StringVar(value="下一步任务：打开设备后将显示现场指导")
        self.proactive_var = tk.StringVar(value="主动响应 · 等待现场状态")
        self.proactive_budget_var = tk.StringVar(value="后台模型调用 0 次")
        self.intent_var = tk.StringVar(value=INTENT_LABELS["white_light_centering"])
        self.response_mode_var = tk.StringVar(value="标准")
        self.adjustment_result_var = tk.StringVar(value="尚未记录调节前状态")
        self.fringe_quality_var = tk.StringVar(value="测量质量门 · 等待条纹分析")
        self.fringe_metrics_var = tk.StringVar(value="角度 --  │  间距 --  │  运动 --")
        self.fringe_summary_var = tk.StringVar(value="启动预测后显示条纹诊断与调节建议。")
        self.laser_alignment_active_var = tk.BooleanVar(value=False)
        self.laser_alignment_var = tk.StringVar(
            value="点击“调节激光条纹”后，根据实时画面显示具体旋钮与方向。")
        self.laser_step_var = tk.StringVar(value="激光预调 · 等待开始")
        self.laser_state_var = tk.StringVar(value="未启动")
        self.laser_diagnosis_var = tk.StringVar(value="点击下方按钮开始实时判断。")
        self.laser_action_var = tk.StringVar(value="系统只会显示一个安全的小步操作。")
        self.laser_expected_var = tk.StringVar(value="等待分析")
        self.laser_stop_var = tk.StringVar(value="证据不足时不要转动旋钮")
        self.laser_metrics_var = tk.StringVar(
            value="倾角 --  │  明纹 --  │  间距 --  │  证据不足")
        self.laser_comparison_var = tk.StringVar(
            value="停手后系统会自动比较本次调节。")
        self.laser_ai_guidance_var = tk.BooleanVar(value=False)
        self.laser_ai_var = tk.StringVar(value="自动 AI 指导未开启")
        self.guidance_stage_var = tk.StringVar(value="advisory")
        self.guidance_stage_note_var = tk.StringVar(value="阶段 1：只读诊断")
        self.adaptive_var = tk.StringVar(value="自适应：尚无设备响应样本")
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
        self._guidance_action_available = False
        self._auto_center_running = False
        self._activity_log: list[str] = []
        self._conversation: list[ConversationEntry] = []
        self._font_size = 10
        self._font_baseline_size = 10
        self._widget_font_bases: list[tuple[tk.Widget, dict]] = []
        self._section_order = ("status", "quick", "chat", "input")
        self._section_heights = {
            "status": 330, "quick": 120, "chat": 220, "input": 145}
        self._collapsed_sections: set[str] = set()
        self._advanced_status_visible = False

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
        self._capture_widget_fonts()
        self.set_font_size(11)
        self.after_idle(self._set_initial_pane_ratio)
        self.after(60, self._lock_quick_area_height)
        self.content_pane.bind(
            "<ButtonRelease-1>",
            lambda _event: self.after_idle(self._lock_quick_area_height),
            add="+")
        self.bind("<Configure>", self._on_panel_resize)
        self.after_idle(self._collapse_secondary_sections)
        self.append(
            "系统",
            "实验助手已接入实时条纹质量门和四阶段调节。你可以让我判断当前步骤、深度分析"
            "角度与间距、确认执行白名单建议、启动安全闭环、解释数据并生成实验报告。",
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
            self.advanced_status_content, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        self.agent_controls = controls
        controls.pack(fill=tk.X, padx=10, pady=(6, 0))
        tk.Label(
            controls, text="智能体执行与硬件安全", bg="#ffffff", fg=self.NAVY,
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
            self.advanced_status_content, textvariable=self.plan_var, bg="#f6f9fd",
            fg="#24558c", anchor="w", justify=tk.LEFT, wraplength=420,
            font=(self.FONT, 8), padx=10,
        )
        self.activity_label = tk.Label(
            self.advanced_status_content, textvariable=self.activity_var, bg="#f6f9fd",
            fg="#475569", anchor="w", justify=tk.LEFT, wraplength=420,
            font=(self.FONT, 8), padx=10,
        )

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
        value = plan.strip() if plan else ""
        self.plan_var.set(value)
        if value:
            if not self.plan_label.winfo_manager():
                options = dict(
                    fill=tk.X, padx=10, pady=(5, 0), ipady=4)
                if self.activity_label.winfo_manager():
                    options["before"] = self.activity_label
                self.plan_label.pack(**options)
        else:
            self.plan_label.pack_forget()

    def append_tool_activity(self, text: str) -> None:
        """向工具活动流追加一条，只保留最近若干条。"""
        self._activity_log.append(text)
        self._activity_log = self._activity_log[-8:]
        self.activity_var.set("工具活动\n" + "\n".join(
            f"· {line}" for line in self._activity_log))
        if not self.activity_label.winfo_manager():
            self.activity_label.pack(
                fill=tk.X, padx=10, pady=(3, 0), ipady=4)

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

        self._build_laser_focus_card()
        self.advanced_status_button = tk.Button(
            self.status_content, text="高级状态与控制  ＋",
            command=self._toggle_advanced_status,
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.MUTED,
            activebackground="#e2e8f0", activeforeground=self.NAVY,
            cursor="hand2", font=(self.FONT, 8, "bold"), pady=4)
        self.advanced_status_button.pack(fill=tk.X, padx=10, pady=(6, 0))
        self.advanced_status_content = tk.Frame(
            self.status_content, bg=self.BG)

        self._build_intent_controls()

        self.context_label = tk.Label(
            self.advanced_status_content, textvariable=self.context_var, bg="#eaf2ff",
            fg="#24558c", anchor="w", justify=tk.LEFT,
            font=(self.FONT, 8), padx=10,
        )
        self.context_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=5)
        progress_row = tk.Frame(
            self.advanced_status_content, bg="#ffffff", highlightthickness=1,
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
        self._build_fringe_dashboard()
        self.ai_state_label = tk.Label(
            self.advanced_status_content, textvariable=self.ai_state_var,
            bg="#eaf8f5", fg=self.CYAN,
            anchor="w", font=(self.FONT, 8, "bold"), padx=10)
        self.ai_state_label.pack(fill=tk.X, padx=10, pady=(6, 0), ipady=4)
        self.ai_insight_label = tk.Label(
            self.advanced_status_content, textvariable=self.ai_insight_var,
            bg="#f3f0ff", fg="#5b3cc4", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8), padx=10)
        self.ai_insight_label.pack(fill=tk.X, padx=10, pady=(4, 0), ipady=4)
        self.proactive_label = tk.Label(
            self.advanced_status_content, textvariable=self.proactive_var,
            bg="#eef7f0", fg="#166534", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8), padx=10)
        self.proactive_label.pack(fill=tk.X, padx=10, pady=(4, 0), ipady=4)
        self.proactive_budget_label = tk.Label(
            self.advanced_status_content, textvariable=self.proactive_budget_var,
            bg=self.BG, fg=self.MUTED, anchor="e",
            font=(self.FONT, 7), padx=10,
        )
        self.proactive_budget_label.pack(fill=tk.X, padx=10, pady=(1, 0))
        self.suggestion_label = tk.Label(
            self.advanced_status_content, textvariable=self.suggestion_var,
            bg="#eef7f0", fg="#166534", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8), padx=10)
        self.suggestion_label.pack(fill=tk.X, padx=10, pady=(4, 6), ipady=4)

    def _build_laser_focus_card(self) -> None:
        """始终置顶的单步激光操作卡；只显示当前最重要的一项操作。"""
        card = tk.Frame(
            self.status_content, bg="#ffffff", highlightthickness=2,
            highlightbackground="#f2cf66")
        self.laser_focus_card = card
        card.pack(fill=tk.X, padx=10, pady=(6, 0))
        header = tk.Frame(card, bg="#ffffff")
        header.pack(fill=tk.X, padx=10, pady=(8, 3))
        tk.Label(header, textvariable=self.laser_step_var,
                 bg="#ffffff", fg=self.NAVY,
                 font=(self.FONT, 10, "bold"), anchor="w").pack(
                     side=tk.LEFT, fill=tk.X, expand=True)
        self.laser_state_label = tk.Label(
            header, textvariable=self.laser_state_var,
            bg="#eef2f7", fg=self.MUTED,
            font=(self.FONT, 8, "bold"), padx=7, pady=2)
        self.laser_state_label.pack(side=tk.RIGHT)
        tk.Label(card, text="当前判断", bg="#ffffff", fg=self.MUTED,
                 font=(self.FONT, 8, "bold"), anchor="w").pack(
                     fill=tk.X, padx=10, pady=(3, 0))
        self.laser_diagnosis_label = tk.Label(
            card, textvariable=self.laser_diagnosis_var,
            bg="#ffffff", fg=self.TEXT, anchor="w", justify=tk.LEFT,
            wraplength=410, font=(self.FONT, 10, "bold"), padx=10, pady=4)
        self.laser_diagnosis_label.pack(fill=tk.X)
        action_box = tk.Frame(card, bg="#eaf2ff")
        action_box.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(action_box, text="下一步", bg="#eaf2ff", fg=self.BLUE,
                 font=(self.FONT, 8, "bold"), anchor="w").pack(
                     fill=tk.X, padx=8, pady=(6, 0))
        self.laser_action_label = tk.Label(
            action_box, textvariable=self.laser_action_var,
            bg="#eaf2ff", fg="#174ea6", anchor="w", justify=tk.LEFT,
            wraplength=390, font=(self.FONT, 10, "bold"), padx=8, pady=5)
        self.laser_action_label.pack(fill=tk.X)
        self.laser_expected_label = tk.Label(
            card, textvariable=self.laser_expected_var,
            bg="#ffffff", fg="#18794e", anchor="w", justify=tk.LEFT,
            wraplength=410, font=(self.FONT, 8), padx=10, pady=2)
        self.laser_expected_label.pack(fill=tk.X)
        self.laser_stop_label = tk.Label(
            card, textvariable=self.laser_stop_var,
            bg="#ffffff", fg="#b42318", anchor="w", justify=tk.LEFT,
            wraplength=410, font=(self.FONT, 8), padx=10, pady=2)
        self.laser_stop_label.pack(fill=tk.X)
        self.laser_metrics_label = tk.Label(
            card, textvariable=self.laser_metrics_var,
            bg="#172033", fg="#ffffff", anchor="w", justify=tk.LEFT,
            font=("Consolas", 8, "bold"), padx=8, pady=5)
        self.laser_metrics_label.pack(fill=tk.X, padx=8, pady=(5, 3))
        self.laser_comparison_label = tk.Label(
            card, textvariable=self.laser_comparison_var,
            bg="#eef7f0", fg="#166534", anchor="w", justify=tk.LEFT,
            wraplength=410, font=(self.FONT, 8), padx=8, pady=4)
        self.laser_comparison_label.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.laser_ai_label = tk.Label(
            card, textvariable=self.laser_ai_var,
            bg="#f3f0ff", fg="#5b3cc4", anchor="w", justify=tk.LEFT,
            wraplength=410, font=(self.FONT, 8), padx=8, pady=4)
        self.laser_ai_label.pack(fill=tk.X, padx=8, pady=(0, 4))
        controls = tk.Frame(card, bg="#ffffff")
        controls.pack(fill=tk.X, padx=8, pady=(1, 8))
        self.laser_alignment_button = tk.Button(
            controls, text="调节激光条纹", command=self._toggle_laser_alignment,
            relief=tk.FLAT, bd=0, bg="#f59e0b", fg="#ffffff",
            activebackground="#d97706", activeforeground="#ffffff",
            cursor="hand2", font=(self.FONT, 8, "bold"), padx=9, pady=5)
        self.laser_alignment_button.pack(side=tk.LEFT, padx=(0, 4))
        self.laser_recheck_button = tk.Button(
            controls, text="我已完成，重新判断",
            command=lambda: self.on_laser_recheck(),
            relief=tk.FLAT, bd=0, bg=self.BLUE, fg="#ffffff",
            activebackground="#1d4ed8", activeforeground="#ffffff",
            cursor="hand2", font=(self.FONT, 8, "bold"), padx=8, pady=5)
        self.laser_recheck_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.laser_ai_button = tk.Checkbutton(
            controls, text="自动 AI 指导",
            variable=self.laser_ai_guidance_var,
            command=self._toggle_laser_ai_guidance,
            bg="#f3f0ff", fg="#5b3cc4", selectcolor="#ddd6fe",
            activebackground="#f3f0ff", font=(self.FONT, 8, "bold"))
        self.laser_ai_button.pack(side=tk.RIGHT, padx=(4, 0))
        footer = tk.Frame(card, bg="#ffffff")
        footer.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(
            footer, text="提问", command=self._open_question_sections,
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.NAVY,
            cursor="hand2", font=(self.FONT, 8), padx=10, pady=3).pack(
                side=tk.LEFT)
        self.export_record_button = tk.Button(
            footer, text="导出实验记录",
            command=lambda: self.on_export_experiment_record(),
            relief=tk.FLAT, bd=0, bg="#e8f0fe", fg=self.BLUE,
            cursor="hand2", font=(self.FONT, 8, "bold"), padx=10, pady=3)
        self.export_record_button.pack(side=tk.RIGHT)

    def _build_intent_controls(self) -> None:
        """实验目的和主动响应模式；修改后由主程序写入实时快照。"""
        card = tk.Frame(
            self.advanced_status_content, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER)
        self.intent_card = card
        card.pack(fill=tk.X, padx=10, pady=(6, 0))
        tk.Label(
            card, text="实验目的", bg="#ffffff", fg=self.NAVY,
            font=(self.FONT, 8, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(9, 5), pady=6)
        intent_box = ttk.Combobox(
            card, textvariable=self.intent_var, state="readonly",
            values=tuple(INTENT_LABELS.values()), width=25,
            font=(self.FONT, 8))
        intent_box.grid(row=0, column=1, sticky="ew", padx=4, pady=5)
        intent_box.bind("<<ComboboxSelected>>", self._on_intent_selected)
        mode_box = ttk.Combobox(
            card, textvariable=self.response_mode_var, state="readonly",
            values=("安静", "标准", "教学"), width=6,
            font=(self.FONT, 8))
        mode_box.grid(row=0, column=2, sticky="e", padx=(4, 9), pady=5)
        mode_box.bind("<<ComboboxSelected>>", self._on_response_mode_selected)
        card.columnconfigure(1, weight=1)

    def _on_intent_selected(self, _event=None) -> None:
        reverse = {label: code for code, label in INTENT_LABELS.items()}
        self.on_set_intent(reverse.get(
            self.intent_var.get(), "white_light_centering"))

    def _on_response_mode_selected(self, _event=None) -> None:
        modes = {"安静": "quiet", "标准": "standard", "教学": "teaching"}
        self.on_set_response_mode(modes.get(
            self.response_mode_var.get(), "standard"))

    def _build_fringe_dashboard(self) -> None:
        """集中呈现条纹质量门、四阶段执行和闭环入口。"""
        card = tk.Frame(
            self.advanced_status_content, bg="#ffffff", highlightthickness=1,
            highlightbackground=self.BORDER)
        card.pack(fill=tk.X, padx=10, pady=(6, 0))
        header = tk.Frame(card, bg="#ffffff")
        header.pack(fill=tk.X, padx=10, pady=(7, 2))
        tk.Label(
            header, text="条纹质量与装置调节", bg="#ffffff", fg=self.NAVY,
            anchor="w", font=(self.FONT, 9, "bold"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fringe_quality_label = tk.Label(
            header, textvariable=self.fringe_quality_var,
            bg="#eef2f7", fg=self.MUTED, anchor="e",
            font=(self.FONT, 8, "bold"), padx=7, pady=2)
        self.fringe_quality_label.pack(side=tk.RIGHT)

        self.fringe_metrics_label = tk.Label(
            card, textvariable=self.fringe_metrics_var,
            bg="#f6f9fd", fg="#24558c", anchor="w", justify=tk.LEFT,
            font=("Consolas", 8), padx=8)
        self.fringe_metrics_label.pack(fill=tk.X, padx=10, pady=(3, 0), ipady=4)
        self.fringe_summary_label = tk.Label(
            card, textvariable=self.fringe_summary_var,
            bg="#ffffff", fg="#475569", anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8))
        self.fringe_summary_label.pack(fill=tk.X, padx=10, pady=(4, 5))

        self.laser_alignment_label = self.laser_action_label

        stages = tk.Frame(card, bg="#ffffff")
        stages.pack(fill=tk.X, padx=8, pady=(0, 4))
        stage_definitions = (
            ("advisory", "1 只读"),
            ("confirm", "2 确认"),
            ("closed_loop", "3 闭环"),
            ("adaptive", "4 自适应"),
        )
        for column, (code, label) in enumerate(stage_definitions):
            stages.columnconfigure(column, weight=1)
            tk.Radiobutton(
                stages, text=label, value=code,
                variable=self.guidance_stage_var,
                command=self._on_guidance_stage_selected,
                indicatoron=False, relief=tk.FLAT, bd=0,
                bg="#eef3f8", fg="#39536d", selectcolor="#dbeafe",
                activebackground="#dbeafe", activeforeground=self.NAVY,
                font=(self.FONT, 8, "bold"), pady=4,
            ).grid(row=0, column=column, sticky="ew", padx=2)
        tk.Label(
            card, textvariable=self.guidance_stage_note_var,
            bg="#ffffff", fg=self.BLUE, anchor="w",
            font=(self.FONT, 8, "bold"),
        ).pack(fill=tk.X, padx=10, pady=(1, 4))

        action_row = tk.Frame(card, bg="#ffffff")
        action_row.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.apply_guidance_button = tk.Button(
            action_row, text="当前无可执行建议", state=tk.DISABLED,
            command=lambda: self.on_apply_guidance(),
            relief=tk.FLAT, bd=0, bg=self.BLUE, fg="#ffffff",
            activebackground="#0c61d6", activeforeground="#ffffff",
            disabledforeground="#94a3b8", cursor="hand2",
            font=(self.FONT, 8, "bold"))
        self.apply_guidance_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=4)
        self.auto_center_button = tk.Button(
            action_row, text="启动闭环", command=self._toggle_auto_center,
            relief=tk.FLAT, bd=0, bg="#e8f4f2", fg=self.CYAN,
            activebackground="#cfe9e5", activeforeground=self.NAVY,
            cursor="hand2", font=(self.FONT, 8, "bold"), width=10)
        self.auto_center_button.pack(side=tk.RIGHT, ipady=4)
        self.adaptive_label = tk.Label(
            card, textvariable=self.adaptive_var,
            bg="#f6f9fd", fg=self.MUTED, anchor="w", justify=tk.LEFT,
            wraplength=420, font=("Consolas", 8), padx=8)
        self.adaptive_label.pack(fill=tk.X, padx=10, pady=(0, 7), ipady=3)
        compare_row = tk.Frame(card, bg="#ffffff")
        compare_row.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Button(
            compare_row, text="记录调节前", command=lambda: self.on_mark_adjustment(),
            relief=tk.FLAT, bd=0, bg="#eef2f7", fg=self.NAVY,
            activebackground="#dbe5f0", cursor="hand2",
            font=(self.FONT, 8), padx=8, pady=3,
        ).pack(side=tk.LEFT)
        tk.Button(
            compare_row, text="比较调节后", command=lambda: self.on_compare_adjustment(),
            relief=tk.FLAT, bd=0, bg="#e8f0fe", fg=self.BLUE,
            activebackground="#dbeafe", cursor="hand2",
            font=(self.FONT, 8, "bold"), padx=8, pady=3,
        ).pack(side=tk.LEFT, padx=5)
        self.image_review_button = tk.Button(
            compare_row, text="AI识图复核", command=lambda: self.on_review_image(),
            relief=tk.FLAT, bd=0, bg="#f3f0ff", fg="#5b3cc4",
            activebackground="#e9e2ff", cursor="hand2",
            font=(self.FONT, 8, "bold"), padx=8, pady=3,
        )
        self.image_review_button.pack(side=tk.RIGHT)
        tk.Label(
            card, textvariable=self.adjustment_result_var,
            bg="#ffffff", fg=self.MUTED, anchor="w", justify=tk.LEFT,
            wraplength=420, font=(self.FONT, 8),
        ).pack(fill=tk.X, padx=10, pady=(0, 7))
        self._refresh_guidance_action_button()
        self._refresh_auto_center_button()

    def _on_guidance_stage_selected(self) -> None:
        stage = self.guidance_stage
        self._update_guidance_stage_note(stage)
        self._refresh_guidance_action_button()
        self._refresh_auto_center_button()
        self.on_set_guidance_stage(stage)

    @property
    def guidance_stage(self) -> str:
        value = str(self.guidance_stage_var.get())
        return value if value in {
            "advisory", "confirm", "closed_loop", "adaptive"} else "advisory"

    def _update_guidance_stage_note(self, stage: str) -> None:
        notes = {
            "advisory": "阶段 1：只分析，不改变画面或设备",
            "confirm": "阶段 2：固定白名单建议，执行前逐项确认",
            "closed_loop": "阶段 3：确认启动后由安全状态机自动搜索并寻中",
            "adaptive": "阶段 4：闭环寻中，并受限学习设备响应参数",
        }
        self.guidance_stage_note_var.set(notes.get(stage, notes["advisory"]))

    def _refresh_guidance_action_button(self, label: str | None = None) -> None:
        if self.guidance_stage == "advisory":
            self.apply_guidance_button.configure(
                text="只读模式不执行建议", state=tk.DISABLED)
        elif self._guidance_action_available:
            text = label or getattr(
                self, "_guidance_action_label", "执行当前建议")
            self.apply_guidance_button.configure(
                text=f"执行建议：{text}", state=tk.NORMAL)
        else:
            self.apply_guidance_button.configure(
                text="当前无可执行建议", state=tk.DISABLED)

    def _toggle_auto_center(self) -> None:
        self.on_auto_center("stop" if self._auto_center_running else "start")

    def _toggle_laser_alignment(self) -> None:
        enabled = not bool(self.laser_alignment_active_var.get())
        self.set_laser_alignment_active(enabled)
        if not enabled and self.laser_ai_guidance_var.get():
            self.set_laser_ai_guidance_enabled(False)
            self.on_toggle_laser_ai_guidance(False)
        self.on_toggle_laser_alignment(enabled)

    def _toggle_laser_ai_guidance(self) -> None:
        enabled = bool(self.laser_ai_guidance_var.get())
        if enabled and not self.laser_alignment_active_var.get():
            self.set_laser_alignment_active(True)
            self.on_toggle_laser_alignment(True)
        self.set_laser_ai_guidance_enabled(enabled)
        self.on_toggle_laser_ai_guidance(enabled)

    def _toggle_advanced_status(self) -> None:
        self._advanced_status_visible = not self._advanced_status_visible
        if self._advanced_status_visible:
            self.advanced_status_content.pack(fill=tk.X)
        else:
            self.advanced_status_content.pack_forget()
        self.advanced_status_button.configure(
            text="高级状态与控制  －" if self._advanced_status_visible
            else "高级状态与控制  ＋")

    def _collapse_secondary_sections(self) -> None:
        for name in ("quick", "chat", "input"):
            if name not in self._collapsed_sections:
                self.toggle_section(name)

    def _open_question_sections(self) -> None:
        for name in ("chat", "input"):
            if name in self._collapsed_sections:
                self.toggle_section(name)

    def set_laser_ai_guidance_enabled(self, enabled: bool) -> None:
        self.laser_ai_guidance_var.set(bool(enabled))
        if not enabled:
            self.set_laser_ai_guidance("自动 AI 指导未开启", "offline")

    def set_laser_ai_guidance(self, text: str, state: str = "ready") -> None:
        colors = {
            "working": ("#fff8e6", "#9a6700"),
            "ready": ("#f3f0ff", "#5b3cc4"),
            "error": ("#fff0f0", "#b42318"),
            "offline": ("#eef2f7", self.MUTED),
        }
        bg, fg = colors.get(state, colors["ready"])
        self.laser_ai_var.set(str(text or "").strip())
        self.laser_ai_label.configure(bg=bg, fg=fg)

    def set_laser_workflow(self, workflow: dict) -> None:
        """显示确定性状态机结果；不解释或改写旋钮方向。"""
        if not workflow:
            return
        self.laser_step_var.set(
            f"激光预调 · 第 {workflow.get('step_number', '--')}/"
            f"{workflow.get('total_steps', '--')} 步 · "
            f"{workflow.get('step_title', '等待判断')}")
        state = str(workflow.get("state") or "observing")
        labels = {
            "blocked": "条件未满足", "observing": "正在观察",
            "action_required": "需要操作", "evaluating": "正在评估",
            "passed": "已完成",
        }
        colors = {
            "blocked": ("#fff0f0", "#b42318"),
            "observing": ("#fff8e6", "#9a6700"),
            "action_required": ("#eaf2ff", "#174ea6"),
            "evaluating": ("#f3f0ff", "#5b3cc4"),
            "passed": ("#eaf8ef", "#18794e"),
        }
        self.laser_state_var.set(labels.get(state, state))
        bg, fg = colors.get(state, colors["observing"])
        self.laser_state_label.configure(bg=bg, fg=fg)
        self.laser_diagnosis_var.set(str(
            workflow.get("diagnosis") or "等待实时判断"))
        action = str(workflow.get("action") or "保持装置不动")
        self.laser_action_var.set(action)
        self.laser_alignment_var.set(action)
        self.laser_expected_var.set(
            "预期：" + str(workflow.get("expected_change") or "等待下一帧"))
        self.laser_stop_var.set(
            "停止：" + str(workflow.get("stop_condition") or "每次只做一个小步"))
        metrics = workflow.get("metrics") or {}
        target = workflow.get("target") or {}
        angle = metrics.get("angle_deg")
        spacing = metrics.get("spacing_px")
        count = int(metrics.get("bright_fringe_count") or 0)
        min_count = int(target.get("min_bright_fringes") or 4)
        max_count = int(target.get("max_bright_fringes") or 10)
        density = (
            "合适" if min_count <= count <= max_count
            else "偏密" if count > max_count else "偏疏")
        evidence = (
            "证据可靠" if angle is not None and metrics.get("spacing_valid")
            else "证据不足")
        angle_text = "--" if angle is None else f"{float(angle):+.1f}°"
        spacing_text = "--" if spacing is None else f"{float(spacing):.1f}px"
        self.laser_metrics_var.set(
            f"倾角 {angle_text}  │  明纹 {count}条·{density}  │  "
            f"间距 {spacing_text}  │  {evidence}")
        comparison = workflow.get("comparison") or {}
        if comparison:
            text = (
                f"{comparison.get('summary', '已自动比较')} "
                f"{comparison.get('recommendation', '')}").strip()
            self.laser_comparison_var.set(text)
            if comparison.get("outcome") in {"worsened", "mixed"}:
                self.laser_comparison_label.configure(
                    bg="#fff0f0", fg="#b42318")
            else:
                self.laser_comparison_label.configure(
                    bg="#eef7f0", fg="#166534")


    def set_laser_alignment_active(self, enabled: bool) -> None:
        self.laser_alignment_active_var.set(bool(enabled))
        self.laser_alignment_button.configure(
            text="结束激光条纹调节" if enabled else "调节激光条纹",
            bg="#0f766e" if enabled else "#f59e0b",
            activebackground="#115e59" if enabled else "#d97706")
        if not enabled:
            self.laser_alignment_var.set(
                "点击“调节激光条纹”后，根据实时画面显示具体旋钮与方向。")
            self.laser_alignment_label.configure(bg="#fff8e6", fg="#7c4a03")
            self.laser_step_var.set("激光预调 · 等待开始")
            self.laser_state_var.set("未启动")

    def _refresh_auto_center_button(self) -> None:
        can_start = self.guidance_stage in {"closed_loop", "adaptive"}
        self.auto_center_button.configure(
            text="停止闭环" if self._auto_center_running else "启动闭环",
            state=tk.NORMAL if self._auto_center_running or can_start else tk.DISABLED,
            bg="#fee4e2" if self._auto_center_running else "#e8f4f2",
            fg="#b42318" if self._auto_center_running else self.CYAN,
        )

    def _build_section_toolbar(self) -> None:
        toolbar = tk.Frame(self, bg="#edf3fa")
        toolbar.pack(fill=tk.X, padx=10, pady=(5, 0))
        labels = {
            "status": "现场总览",
            "quick": "实验任务",
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
        toolbar = tk.Frame(self.chat_area, bg=self.BG)
        toolbar.pack(fill=tk.X, padx=10, pady=(5, 0))
        tk.Label(
            toolbar, text="实验助手对话", bg=self.BG, fg=self.NAVY,
            anchor="w", font=(self.FONT, 9, "bold"),
        ).pack(side=tk.LEFT)
        self.export_chat_button = tk.Button(
            toolbar, text="导出对话", command=lambda: self.on_export_chat(),
            relief=tk.FLAT, bd=0, bg="#e8f0fe", fg="#1d4ed8",
            activebackground="#dbeafe", activeforeground="#1d4ed8",
            cursor="hand2", font=(self.FONT, 8, "bold"), padx=9, pady=3,
        )
        self.export_chat_button.pack(side=tk.RIGHT)
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
        self.set_font_size(11)

    def set_font_size(self, size: int) -> None:
        """等比例调整整个实验助手，同时保留标题和说明文字的层级。"""
        self._font_size = max(8, min(24, int(size)))
        size = self._font_size
        delta = size - self._font_baseline_size
        for widget, base in self._widget_font_bases:
            try:
                base_size = abs(int(base.get("size") or self._font_baseline_size))
                scaled_size = max(7, min(30, base_size + delta))
                styles = []
                if base.get("weight") == "bold":
                    styles.append("bold")
                if base.get("slant") == "italic":
                    styles.append("italic")
                if base.get("underline"):
                    styles.append("underline")
                if base.get("overstrike"):
                    styles.append("overstrike")
                widget.configure(font=(
                    base.get("family") or self.FONT,
                    scaled_size,
                    *styles,
                ))
            except (tk.TclError, ValueError, TypeError):
                continue
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

    def _capture_widget_fonts(self) -> None:
        """记录各控件的初始字体，后续缩放基于原层级而非反复累加。"""
        captured: list[tuple[tk.Widget, dict]] = []

        def visit(widget: tk.Widget) -> None:
            try:
                raw_font = widget.cget("font")
            except (tk.TclError, AttributeError):
                raw_font = None
            if raw_font:
                try:
                    captured.append((
                        widget,
                        tkfont.Font(root=self, font=raw_font).actual(),
                    ))
                except tk.TclError:
                    pass
            for child in widget.winfo_children():
                visit(child)

        visit(self)
        self._widget_font_bases = captured

    def _build_actions(self):
        action_shell = tk.Frame(self.quick_content, bg=self.BG)
        self.action_shell = action_shell
        action_shell.pack(fill=tk.X)
        tk.Label(
            action_shell, text="常用实验任务", bg=self.BG, fg=self.NAVY, anchor="w",
            font=(self.FONT, 9, "bold"),
        ).pack(fill=tk.X, padx=12, pady=(2, 1))
        quick = tk.Frame(action_shell, bg=self.BG)
        quick.pack(fill=tk.X, padx=8, pady=(1, 5))
        for column in range(3):
            quick.columnconfigure(column, weight=1)
        for index, (label, question) in enumerate([
            ("现场下一步", "读取完整实时状态、条纹质量门和近期日志，只告诉我当前最应该完成的一步、操作方法和观察标志。"),
            ("深度分析条纹", "根据实时角度、曲率、法向间距、清晰度、运动速度和中心偏差，分析当前条纹质量并给出有优先级的调节建议。"),
            ("调出条纹", "请一步一步带我调出白光干涉条纹：从激光非定域条纹、等厚直条纹，到白光彩色条纹和中央黑条纹，每步都告诉我操作和观察标志，并问我观察到什么。"),
            ("预习原理", "带我预习迈克尔逊干涉实验的目的、原理、关键公式、安全事项和预期现象，最后用两个自检问题考我。"),
            ("数据与误差", "检查当前测量记录是否足够，基于真实数据进行误差和不确定度计算；缺少数据时明确列出缺少项。"),
            ("生成报告", "按固定格式生成迈克尔逊干涉实验报告，使用已有状态、条纹质量和读数，缺失内容标记为待补充。"),
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
            "status": "现场总览", "quick": "实验任务",
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
        self.fringe_summary_label.configure(wraplength=wraplength)
        self.laser_alignment_label.configure(wraplength=wraplength)
        self.laser_diagnosis_label.configure(wraplength=wraplength)
        self.laser_action_label.configure(wraplength=wraplength)
        self.laser_expected_label.configure(wraplength=wraplength)
        self.laser_stop_label.configure(wraplength=wraplength)
        self.laser_comparison_label.configure(wraplength=wraplength)
        self.laser_ai_label.configure(wraplength=wraplength)
        self.adaptive_label.configure(wraplength=wraplength)
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
        guidance = vision.get("fringe_guidance") or {}
        laser_active = bool(vision.get("laser_alignment_active", False))
        if laser_active != bool(self.laser_alignment_active_var.get()):
            self.set_laser_alignment_active(laser_active)
        laser_ai_enabled = bool(
            vision.get("laser_ai_guidance_enabled", False))
        if laser_ai_enabled != bool(self.laser_ai_guidance_var.get()):
            self.set_laser_ai_guidance_enabled(laser_ai_enabled)
        workflow = vision.get("laser_guidance_session") or {}
        if workflow:
            self.set_laser_workflow(workflow)
        adaptive = vision.get("adaptive_response") or {}
        intent = context.get("experiment_intent") or {}
        intent_kind = str(intent.get("kind") or "white_light_centering")
        self.intent_var.set(INTENT_LABELS.get(
            intent_kind, INTENT_LABELS["white_light_centering"]))
        self.response_mode_var.set({
            "quiet": "安静", "standard": "标准", "teaching": "教学",
        }.get(str(intent.get("response_mode") or "standard"), "标准"))
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

        stage = str(
            guidance.get("execution_stage")
            or vision.get("guidance_execution_stage")
            or "advisory")
        if stage not in {"advisory", "confirm", "closed_loop", "adaptive"}:
            stage = "advisory"
        self.guidance_stage_var.set(stage)
        self._update_guidance_stage_note(stage)

        metrics = guidance.get("metrics") or {}
        angle = self._format_metric(metrics.get("angle_deg"), "+.1f", "°")
        spacing = self._format_metric(metrics.get("spacing_px"), ".2f", "px")
        cv = self._format_metric(
            metrics.get("spacing_cv_percent"), ".1f", "%")
        movement_names = {
            "stable": "稳定", "left": "左移", "right": "右移",
            "unknown": "未知", "no_fringe": "无条纹",
        }
        movement = movement_names.get(
            str(metrics.get("movement") or "unknown"),
            str(metrics.get("movement") or "未知"))
        offset = vision.get("center_offset_px")
        offset_text = self._format_metric(offset, "+.1f", "px")
        self.fringe_metrics_var.set(
            f"角度 {angle}  │  间距 {spacing}  │  CV {cv}  │  "
            f"运动 {movement}  │  中心偏差 {offset_text}")

        if not guidance:
            quality_text = "测量质量门 · 等待分析"
            quality_colors = ("#eef2f7", self.MUTED)
            summary = "启动模型预测后显示角度、间距、稳定性和调节建议。"
        elif guidance.get("measurement_ready"):
            quality_text = (
                f"质量门通过 · {float(guidance.get('quality_score') or 0):.0%}")
            quality_colors = ("#eaf8ef", "#18794e")
            summary = str(guidance.get("summary") or "可以进行测量记录。")
        else:
            phase = str(guidance.get("phase") or "observing")
            phase_names = {
                "searching": "搜索条纹", "quality_recovery": "恢复画质",
                "adjusting": "等待调节", "observing": "等待稳定",
            }
            quality_text = (
                f"质量门未通过 · {phase_names.get(phase, '继续观察')} · "
                f"{float(guidance.get('quality_score') or 0):.0%}")
            high_issue = any(
                item.get("severity") == "high"
                for item in (guidance.get("issues") or []))
            quality_colors = (
                ("#fff0f0", "#c53030") if high_issue
                else ("#fff8e6", "#9a6700"))
            summary = str(guidance.get("summary") or "当前暂不建议测量。")
            recommendations = guidance.get("recommendations") or []
            if recommendations:
                summary += f" 下一步：{recommendations[0]}"
        self.fringe_quality_var.set(quality_text)
        self.fringe_quality_label.configure(
            bg=quality_colors[0], fg=quality_colors[1])
        self.fringe_summary_var.set(summary)

        if laser_active:
            alignment = guidance.get("laser_vertical_alignment") or {}
            fallback_instruction = render_laser_alignment_instruction(alignment)
            self.laser_alignment_var.set(fallback_instruction)
            if not workflow:
                self.laser_action_var.set(fallback_instruction)
            laser_ready = bool(alignment.get("ready", False))
            self.laser_alignment_label.configure(
                bg="#eaf8ef" if laser_ready else "#fff8e6",
                fg="#18794e" if laser_ready else "#7c4a03")

        actions = guidance.get("actions") or []
        primary = actions[0] if actions else {}
        self._guidance_action_label = str(primary.get("label") or "执行当前建议")
        self._guidance_action_available = bool(primary)
        self._refresh_guidance_action_button(self._guidance_action_label)

        self._auto_center_running = bool(motor.get("auto_enabled"))
        self._refresh_auto_center_button()
        settle = adaptive.get("learned_settle_seconds")
        settle_text = "--" if settle is None else f"{float(settle):.2f}s"
        self.adaptive_var.set(
            f"自适应：置信度 {float(adaptive.get('confidence') or 0):.0%}"
            f" │ 响应样本 {int(adaptive.get('response_samples') or 0)}"
            f" │ 学习停稳 {settle_text}"
            f" │ {'正在应用' if stage == 'adaptive' else '仅观察'}")

    @staticmethod
    def _format_metric(value, spec: str, suffix: str) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        return f"{format(number, spec)}{suffix}"

    def set_suggestion(self, text: str, source: str = "") -> None:
        """设置主动建议标签；source 用于标注来源（DeepSeek / 本地提示）。"""
        prefix = f"{source} · " if source else ""
        self.suggestion_var.set(prefix + text.strip())

    def set_proactive_guidance(
        self, decision: dict, *, llm_calls: int = 0,
    ) -> None:
        """更新本地实时决策卡；不会向聊天区追加重复消息。"""
        issues = decision.get("issues") or []
        priority = str(decision.get("priority") or "normal")
        evidence = decision.get("evidence") or []
        evidence_text = "；".join(str(item) for item in evidence[:2])
        text = (
            f"主动响应 · {decision.get('diagnosis', '等待判断')}\n"
            f"操作：{decision.get('action', '等待状态更新')}"
        )
        if evidence_text:
            text += f"\n依据：{evidence_text}"
        self.proactive_var.set(text)
        if priority == "blocking":
            colors = ("#fff0f0", "#c53030")
        elif issues:
            colors = ("#fff8e6", "#9a6700")
        else:
            colors = ("#eef7f0", "#166534")
        self.proactive_label.configure(bg=colors[0], fg=colors[1])
        self.proactive_budget_var.set(
            f"本地实时判断 · 本次会话后台模型调用 {int(llm_calls)} 次")

    def set_proactive_budget(self, llm_calls: int) -> None:
        self.proactive_budget_var.set(
            f"本地实时判断 · 本次会话后台模型调用 {int(llm_calls)} 次")

    def set_adjustment_result(self, text: str) -> None:
        self.adjustment_result_var.set(text)

    def set_image_review_state(self, running: bool) -> None:
        self.image_review_button.configure(
            text="识图复核中…" if running else "AI识图复核",
            state=tk.DISABLED if running else tk.NORMAL,
        )

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
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._conversation.append(ConversationEntry(
            role=role,
            text=text.strip(),
            timestamp=timestamp,
            options=tuple(options),
        ))
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

    def conversation_entries(self) -> tuple[ConversationEntry, ...]:
        """返回当前完整会话的只读快照，供导出使用。"""
        return tuple(self._conversation)

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
                highlightbackground="#bfe0db",
                font=(self.FONT, max(8, self._font_size - 2), "bold"))
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
        if hasattr(self, "image_review_button"):
            self.image_review_button.configure(
                state=tk.DISABLED if busy else tk.NORMAL)
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
