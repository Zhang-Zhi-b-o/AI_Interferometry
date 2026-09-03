"""实验助手对话的结构化记录与 UTF-8 导出。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ConversationEntry:
    """一条未经界面样式处理的实验助手消息。"""

    role: str
    text: str
    timestamp: str
    options: tuple[str, ...] = ()


def render_conversation(
    entries: Iterable[ConversationEntry],
    *,
    format_name: str = "markdown",
    exported_at: datetime | None = None,
) -> str:
    """将会话渲染为 Markdown 或纯文本。"""
    records = tuple(entries)
    exported_at = exported_at or datetime.now().astimezone()
    exported_text = exported_at.strftime("%Y-%m-%d %H:%M:%S %z")

    if format_name == "text":
        lines = [
            "AI Interferometry 实验助手对话",
            f"导出时间：{exported_text}",
            f"消息数量：{len(records)}",
            "",
        ]
        for entry in records:
            lines.extend((f"[{entry.timestamp}] {entry.role}", entry.text.strip()))
            if entry.options:
                lines.append("可选回复：" + " / ".join(entry.options))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    if format_name != "markdown":
        raise ValueError(f"不支持的对话导出格式：{format_name}")

    lines = [
        "# AI Interferometry 实验助手对话",
        "",
        f"> 导出时间：{exported_text}",
        f"> 消息数量：{len(records)}",
        "",
    ]
    for entry in records:
        lines.extend((f"## {entry.role} · {entry.timestamp}", "", entry.text.strip(), ""))
        if entry.options:
            lines.extend(("可选回复：", ""))
            lines.extend(f"- {option}" for option in entry.options)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_conversation(
    path: str | Path,
    entries: Iterable[ConversationEntry],
) -> Path:
    """按目标扩展名导出会话；``.txt`` 为纯文本，其余为 Markdown。"""
    target = Path(path)
    if not target.name:
        raise ValueError("导出路径不能为空")
    format_name = "text" if target.suffix.lower() == ".txt" else "markdown"
    target.write_text(
        render_conversation(entries, format_name=format_name),
        encoding="utf-8",
        newline="\n",
    )
    return target
