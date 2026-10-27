"""输出检查：数字仅可复述当前工具/状态或明确引用的资料，不借历史对话兜底。"""
from __future__ import annotations

import json
import re

REF = re.compile(r"\[(S\d+-p\d+-\d+)\]")
NUMBER = re.compile(r"(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def supported_answer(text: str, outputs: list, references: list[dict]) -> bool:
    known = {r["id"]: r for r in references}
    cited = REF.findall(text)
    if set(cited) - known.keys():
        return False
    # 数字按原始表示检查，允许尾零，不把编号当实测数值。
    allowed = json.dumps(outputs, ensure_ascii=False, default=str)
    allowed += " ".join(known[key]["text"] for key in cited)
    tokens = {token.lstrip("+").rstrip("0").rstrip(".") if "." in token else token.lstrip("+")
              for token in NUMBER.findall(REF.sub("", allowed))}
    for token in NUMBER.findall(REF.sub("", text)):
        normal = token.lstrip("+").rstrip("0").rstrip(".") if "." in token else token.lstrip("+")
        if normal not in tokens:
            return False
    # 出现其他形态的出处编号时不接受，避免伪造参考文献。
    return not re.search(r"\[(?:K\d+|S\d+)[^\]]*\]", REF.sub("", text))
