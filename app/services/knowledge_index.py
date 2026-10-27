"""本地知识检索：TF-IDF 字符 n-gram，返回带页码出处的文本块。

只用已安装的 scikit-learn，不联网、不下载模型、不需要向量库。
542 块规模下建索引约 0.3 秒，冷启动一次后常驻。
"""
from __future__ import annotations

import re
import threading
from typing import Iterable

from app.services import knowledge_corpus

KIND_LABEL = {"paper": "论文", "report": "自有报告", "manual": "仪器说明",
              "kb": "知识库", "guide": "操作指导"}
MAX_QUERY_CHARS = 120


class KnowledgeIndex:
    def __init__(self, path=None, chunks: Iterable[dict] | None = None):
        data = knowledge_corpus.load_or_build(path) if chunks is None else {"chunks": list(chunks)}
        sources = {item["id"]: item for item in data.get("documents", [])}
        self.chunks: list[dict] = [
            {**item, "path": sources.get(item.get("source"), {}).get("path", item.get("path", ""))}
            for item in data["chunks"]
            if not re.search(r"(?:\.\s*){8,}", item.get("text", ""))]
        self.built_at = data.get("built_at", "")
        self._vectorizer = None
        self._matrix = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._matrix is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = [self._document_text(chunk) for chunk in self.chunks]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True,
            min_df=2, max_features=30000)
        self._matrix = self._vectorizer.fit_transform(documents)

    @staticmethod
    def _document_text(chunk: dict) -> str:
        return f"{chunk.get('section', '')} {chunk.get('title', '')} {chunk['text']}"

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, kinds: Iterable[str] | None = None) -> list[dict]:
        """按相关度返回文本块；``kinds`` 可限定只用论文/知识库等。"""
        text = re.sub(r"\s+", " ", str(query)).strip()[:MAX_QUERY_CHARS]
        if not text or not self.chunks:
            return []
        with self._lock:
            self._ensure()
            from sklearn.metrics.pairwise import cosine_similarity

            scores = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        allowed = set(kinds) if kinds else None
        order = scores.argsort()[::-1]
        results: list[dict] = []
        for position in order:
            if scores[position] <= 0.0:
                break
            chunk = self.chunks[int(position)]
            if allowed is not None and chunk.get("kind") not in allowed:
                continue
            results.append({**chunk, "score": round(float(scores[position]), 4)})
            if len(results) >= max(1, int(top_k)):
                break
        return results

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.get("kind", "?")] = counts.get(chunk.get("kind", "?"), 0) + 1
        return {"chunks": len(self.chunks), "by_kind": counts, "built_at": self.built_at}


def citation(item: dict) -> str:
    """统一的出处格式，界面、提示词和报告共用。"""
    title = item.get("title", "")
    page = item.get("page")
    location = f" 第{page}页" if page else ""
    return f"[{item.get('id', '')}] {title}{location}"


def as_evidence(items: list[dict], limit: int = 3) -> str:
    """把检索结果压成给模型看的证据段落。"""
    lines = []
    for item in items[:limit]:
        text = item.get("text", "").replace("\n", " ")
        lines.append(f"{citation(item)}：{text[:400]}")
    return "\n".join(lines)


_default: KnowledgeIndex | None = None
_default_lock = threading.Lock()


def default_index() -> KnowledgeIndex:
    global _default
    with _default_lock:
        if _default is None:
            _default = KnowledgeIndex()
        return _default


def prewarm() -> None:
    """后台预热索引，避免第一次提问才建 TF-IDF 矩阵。"""
    def run() -> None:
        try:
            default_index().search("预热", top_k=1)
        except Exception:  # 预热失败不影响主流程，首次检索会重试
            return

    threading.Thread(target=run, name="knowledge-prewarm", daemon=True).start()
