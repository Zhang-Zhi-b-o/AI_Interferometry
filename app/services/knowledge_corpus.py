"""语料构建：PDF / Markdown → 带页码出处的文本块（无 GUI 依赖）。

只做确定性抽取，不调模型。构建结果由 `app.services.knowledge_index` 使用。
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

from app.core.config import ROOT

INDEX_PATH = ROOT / "data" / "knowledge" / "index.json"

#: (kind, 相对路径)。竞赛 PPT 不可用，此处只用期刊论文、自有报告、仪器说明与知识库。
SOURCES = [
    ("paper", "knowledge/papers/白光和激光扩展光源协同调节白光等厚干涉条纹_王希成.pdf"),
    ("paper", "knowledge/papers/迈克耳孙干涉仪白光干涉现象分析_窦玲玉.pdf"),
    ("paper", "knowledge/papers/迈克尔逊干涉实验的仿真研究_乔亮.pdf"),
    ("paper", "knowledge/papers/迈克尔逊干涉测薄膜厚度_鄢仁文.pdf"),
    ("paper", "knowledge/papers/非接触法自动测量薄膜厚度_彭真真.pdf"),
    ("paper", "knowledge/papers/基于白光干涉的介质薄膜厚度测量_徐黎轩.pdf"),
    ("paper", "knowledge/papers/白光干涉测量系统计量特性的评价与校准_蔡潇雨.pdf"),
    ("paper", "knowledge/papers/基于白光和激光干涉的高灵敏微位移测量技术研究_李劲松.pdf"),
    ("report", "knowledge/papers/研究报告.pdf"),
    ("report", "knowledge/papers/研究报告18.57最终版(1).pdf"),
    ("manual", "knowledge/papers/仪器说明文档.pdf"),
]
KB_DIR = ROOT / "engine" / "agent" / "knowledge_base"
GUIDE_DIR = ROOT / "knowledge" / "guides"

#: 单片少于这么多字符视为图片页，直接跳过。
MIN_PAGE_CHARS = 60

HEADING = re.compile(
    r"^(?:第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+[、．.]|\d+(?:\.\d+)*[、．.\s]\s*\S"
    r"|摘\s*要|引\s*言|前\s*言|结\s*论|参考文献|致\s*谢|实验原理|实验方法|结果与?分析|误差分析)")
NOISE = ("参考文献", "致谢", "目录", "版权", "封底")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def heading_of(page_text: str) -> str:
    for line in page_text.splitlines()[:6]:
        line = line.strip()
        if 2 <= len(line) <= 40 and HEADING.match(line):
            return line
    return ""


def split_page(text: str, limit: int = 600) -> list[str]:
    """按段落与句号切块，合并到 limit 字符以内。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= limit:
            if chunks and len(chunks[-1]) + len(para) <= limit:
                chunks[-1] += "\n" + para
            else:
                chunks.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[。！？；;])", para):
            if not sentence:
                continue
            if len(current) + len(sentence) > limit and current:
                chunks.append(current)
                current = ""
            current += sentence
        if current:
            chunks.append(current)
    return [c.strip() for c in chunks if len(c.strip()) >= 40]


def load_pdf(path: Path, kind: str, doc_id: str) -> tuple[list[dict], dict]:
    import fitz

    stats = {"id": doc_id, "path": str(path.relative_to(ROOT)), "kind": kind,
             "pages": 0, "text_pages": 0, "chunks": 0}
    try:
        doc = fitz.open(path)
    except Exception as exc:  # 文件损坏不该阻断整个构建
        stats["error"] = str(exc)
        return [], stats
    stats["pages"] = doc.page_count
    chunks: list[dict] = []
    for number in range(doc.page_count):
        page_text = clean(doc.load_page(number).get_text("text"))
        if len(page_text) < MIN_PAGE_CHARS:
            continue
        stats["text_pages"] += 1
        section = heading_of(page_text)
        if any(word in section for word in NOISE):
            continue
        for index, part in enumerate(split_page(page_text)):
            chunks.append({
                "id": f"{doc_id}-p{number + 1:03d}-{index + 1:02d}",
                "source": doc_id, "title": path.stem, "kind": kind,
                "page": number + 1, "section": section, "text": part,
            })
    stats["chunks"] = len(chunks)
    doc.close()
    return chunks, stats


def load_markdown(path: Path, kind: str, doc_id: str) -> tuple[list[dict], dict]:
    raw = clean(path.read_text(encoding="utf-8"))
    raw = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.S)  # 去掉 front matter
    chunks = [
        {"id": f"{doc_id}-p001-{index + 1:02d}", "source": doc_id,
         "title": path.stem, "kind": kind, "page": 1, "section": "", "text": part}
        for index, part in enumerate(split_page(raw))
    ]
    return chunks, {"id": doc_id, "path": str(path.relative_to(ROOT)), "kind": kind,
                    "pages": 1, "text_pages": 1, "chunks": len(chunks)}


def build() -> dict:
    chunks: list[dict] = []
    stats: list[dict] = []
    for index, (kind, relative) in enumerate(SOURCES, start=1):
        path = ROOT / relative
        doc_id = f"S{index:02d}"
        if not path.is_file():
            stats.append({"id": doc_id, "path": relative, "kind": kind, "error": "missing"})
            continue
        part_chunks, part_stats = load_pdf(path, kind, doc_id)
        # 扫描版识别：可读文本页不足三成时标出来，已抽到的部分仍然保留。
        if part_stats.get("pages") and part_stats["text_pages"] < part_stats["pages"] * 0.3:
            part_stats["warning"] = "文本层稀疏（可能是扫描版）"
        chunks.extend(part_chunks)
        stats.append(part_stats)

    index = len(SOURCES)
    for directory, kind in ((KB_DIR, "kb"), (GUIDE_DIR, "guide")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            index += 1
            part_chunks, part_stats = load_markdown(path, kind, f"S{index:02d}")
            chunks.extend(part_chunks)
            stats.append(part_stats)

    return {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "documents": [s for s in stats if not s.get("error")],
        "skipped": [s for s in stats if s.get("error") or s.get("warning")],
        "chunks": chunks,
    }


def load_or_build(path: Path | None = None) -> dict:
    """读取已有索引；缺失时现场构建（约 1–2 秒），并写回缓存。"""
    import json

    target = Path(path or INDEX_PATH)
    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if data.get("chunks"):
                return data
        except Exception:  # 缓存坏了就重建，不报错给用户
            pass
    data = build()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return data
