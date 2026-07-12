"""轻量本地知识库：Markdown 加载、切片和关键词检索。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class KnowledgeChunk:
    source_id: str
    title: str
    text: str
    url: str = ""
    tags: tuple[str, ...] = ()


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    ascii_words = set(re.findall(r"[a-z0-9_+-]{2,}", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    grams = {chinese[i:i + 2] for i in range(max(0, len(chinese) - 1))}
    return ascii_words | grams


class KnowledgeBase:
    def __init__(self, root: Path, chunk_chars: int = 700):
        self.root = Path(root)
        self.chunk_chars = max(200, chunk_chars)
        self.chunks: list[KnowledgeChunk] = []
        self.reload()

    def reload(self):
        self.chunks.clear()
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*.md")):
            self.chunks.extend(self._load_file(path))

    def _load_file(self, path: Path) -> list[KnowledgeChunk]:
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        if raw.startswith("---"):
            _, header, raw = raw.split("---", 2)
            for line in header.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip('"')
        title = meta.get("title", path.stem)
        source_id = meta.get("id", path.stem)
        url = meta.get("url", "")
        tags = tuple(x.strip() for x in meta.get("tags", "").strip("[]").split(",") if x.strip())
        sections = re.split(r"(?=^##?\s+)", raw, flags=re.MULTILINE)
        result = []
        for section in sections:
            text = re.sub(r"\s+", " ", section).strip()
            if not text:
                continue
            for start in range(0, len(text), self.chunk_chars):
                part = text[start:start + self.chunk_chars].strip()
                if part:
                    result.append(KnowledgeChunk(source_id, title, part, url, tags))
        return result

    def search(self, query: str, top_k: int = 4) -> list[KnowledgeChunk]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        ranked = []
        for chunk in self.chunks:
            body_terms = _terms(chunk.text)
            title_terms = _terms(chunk.title + " " + " ".join(chunk.tags))
            overlap = len(query_terms & body_terms)
            title_overlap = len(query_terms & title_terms)
            if overlap or title_overlap:
                score = overlap + title_overlap * 3
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:max(1, top_k)]]
