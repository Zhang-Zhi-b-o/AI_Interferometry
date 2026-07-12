"""实验辅助智能体面板。"""
from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext


class AgentPluginPanel(tk.LabelFrame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent, text="实验助手（只读）", bg="#ffffff", fg="#000000")
        self.on_ask = lambda question, include_status: None
        self.include_status_var = tk.BooleanVar(value=True)

        self.output = scrolledtext.ScrolledText(
            self, height=12, wrap=tk.WORD, bg="#fafafa", fg="#222222",
            font=("Microsoft YaHei UI", 9), state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        quick = tk.Frame(self, bg="#fff")
        quick.pack(fill=tk.X, padx=8, pady=2)
        for label, question in [
            ("原理", "简要说明迈克尔逊干涉的基本原理。"),
            ("无条纹", "当前看不到干涉条纹，应该按什么顺序排查？"),
            ("不确定度", "本实验主要有哪些测量不确定度来源？"),
        ]:
            tk.Button(quick, text=label, command=lambda q=question: self.ask(q),
                      relief=tk.FLAT, bd=0, bg="#e8e8e8", cursor="hand2").pack(
                side=tk.LEFT, padx=(0, 4))

        self.input = tk.Text(self, height=3, wrap=tk.WORD, font=("Microsoft YaHei UI", 9))
        self.input.pack(fill=tk.X, padx=8, pady=4)
        self.input.bind("<Control-Return>", lambda event: self.ask())

        controls = tk.Frame(self, bg="#fff")
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Checkbutton(
            controls, text="附加当前实验状态", variable=self.include_status_var,
            bg="#fff", activebackground="#fff", selectcolor="#fff").pack(side=tk.LEFT)
        self.ask_button = tk.Button(
            controls, text="提问 (Ctrl+Enter)", command=self.ask,
            relief=tk.FLAT, bd=0, bg="#111", fg="#fff", cursor="hand2")
        self.ask_button.pack(side=tk.RIGHT)

    def ask(self, preset: str | None = None):
        question = preset or self.input.get("1.0", tk.END).strip()
        if not question:
            return
        if not preset:
            self.input.delete("1.0", tk.END)
        self.append("你", question)
        self.set_busy(True)
        self.on_ask(question, self.include_status_var.get())

    def append(self, role: str, text: str):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, f"{role}：\n{text.strip()}\n\n")
        self.output.configure(state=tk.DISABLED)
        self.output.see(tk.END)

    def set_busy(self, busy: bool):
        self.ask_button.configure(state=tk.DISABLED if busy else tk.NORMAL,
                                  text="思考中..." if busy else "提问 (Ctrl+Enter)")
