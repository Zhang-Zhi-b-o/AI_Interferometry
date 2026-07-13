"""将助手常用 Markdown/LaTeX 子集转换为 Tk Text 富文本片段。"""
from __future__ import annotations

import re


Segment = tuple[str, tuple[str, ...]]

_INLINE = re.compile(
    r"(\*\*.+?\*\*|`.+?`|\\\(.+?\\\)|\$[^$]+?\$)"
)
_SUBSCRIPT = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_MATH_COMMANDS = {
    r"\Delta": "Δ", r"\delta": "δ", r"\lambda": "λ",
    r"\theta": "θ", r"\omega": "ω", r"\pi": "π",
    r"\sigma": "σ", r"\mu": "μ", r"\pm": "±",
    r"\times": "×", r"\cdot": "·", r"\approx": "≈",
    r"\le": "≤", r"\ge": "≥", r"\neq": "≠",
    r"\rightarrow": "→", r"\infty": "∞",
}


def latex_to_text(value: str) -> str:
    """把常见实验公式转成 Cambria Math 可直接显示的 Unicode 文本。"""
    text = value.strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2]
    elif text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2]
    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    for command, symbol in _MATH_COMMANDS.items():
        text = text.replace(command, symbol)
    text = re.sub(
        r"_\{?([+-]?[0-9]+)\}?",
        lambda match: match.group(1).translate(_SUBSCRIPT), text)
    text = re.sub(
        r"\^\{?([+-]?[0-9]+)\}?",
        lambda match: match.group(1).translate(_SUPERSCRIPT), text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _inline_segments(text: str, extra_tags: tuple[str, ...] = ()) -> list[Segment]:
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    segments: list[Segment] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            segments.append((text[position:match.start()], extra_tags))
        token = match.group(0)
        if token.startswith("**"):
            segments.append((token[2:-2], extra_tags + ("bold",)))
        elif token.startswith("`"):
            segments.append((token[1:-1], extra_tags + ("code",)))
        else:
            segments.append((latex_to_text(token), extra_tags + ("math",)))
        position = match.end()
    if position < len(text):
        segments.append((text[position:], extra_tags))
    return segments


def markdown_segments(markdown: str) -> list[Segment]:
    """解析适合实验回答的 Markdown 子集，返回文本和样式标签。"""
    result: list[Segment] = []
    in_code = False
    for raw_line in markdown.strip().splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            result.append((line + "\n", ("code_block",)))
            continue
        if not line.strip():
            result.append(("\n", ()))
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            result.extend(_inline_segments(heading.group(2), (f"heading{level}",)))
            result.append(("\n", (f"heading{level}",)))
            continue
        if ((line.strip().startswith("$$") and line.strip().endswith("$$"))
                or (line.strip().startswith(r"\[") and line.strip().endswith(r"\]"))):
            result.append((latex_to_text(line.strip()) + "\n", ("math_display",)))
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet:
            result.append(("•  ", ("bullet",)))
            result.extend(_inline_segments(bullet.group(1), ("bullet",)))
            result.append(("\n", ("bullet",)))
            continue
        ordered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if ordered:
            result.append((f"{ordered.group(1)}.  ", ("bullet",)))
            result.extend(_inline_segments(ordered.group(2), ("bullet",)))
            result.append(("\n", ("bullet",)))
            continue
        if line.lstrip().startswith(">"):
            result.append(("│  ", ("quote",)))
            result.extend(_inline_segments(line.lstrip()[1:].strip(), ("quote",)))
            result.append(("\n", ("quote",)))
            continue
        if "|" in line and re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):
            continue
        if "|" in line and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            result.extend(_inline_segments("   │   ".join(cells), ("table",)))
            result.append(("\n", ("table",)))
            continue
        if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
            result.append(("─" * 36 + "\n", ("divider",)))
            continue
        result.extend(_inline_segments(line))
        result.append(("\n", ()))
    return result


def insert_markdown(text_widget, markdown: str, base_tag: str = "message") -> None:
    """将解析结果插入已处于 NORMAL 状态的 Tk Text。"""
    for content, tags in markdown_segments(markdown):
        text_widget.insert("end", content, (base_tag, *tags))
