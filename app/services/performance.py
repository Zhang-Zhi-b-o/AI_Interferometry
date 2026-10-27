"""助手性能记录和固定题集评测；不以检索相似度或回答成功返回替代任务验收。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time
from uuid import uuid4

CASE_PATH = Path(__file__).resolve().parents[1] / "evaluation_cases.json"
MODES = {"online": "在线模型", "offline": "离线本地", "fallback": "模型降级"}


def latency_summary(values: list[float]) -> dict:
    numbers = sorted(values)
    if not numbers:
        return {"count": 0, "median_s": None, "p95_s": None, "mean_s": None}
    return {"count": len(numbers), "median_s": statistics.median(numbers),
            "p95_s": numbers[math.ceil(0.95 * len(numbers)) - 1],
            "mean_s": statistics.mean(numbers)}


def summarize_turns(turns: list[dict]) -> list[dict]:
    groups = {}
    for turn in turns:
        key = (turn["mode"], turn.get("model", ""), turn["data_source"])
        groups.setdefault(key, []).append(turn)
    result = []
    for (mode, model, source), items in groups.items():
        judged = [t for t in items if t["status"] != "cancelled"
                  and t.get("completed") is not None]
        successes = sum(t["completed"] is True for t in judged)
        result.append({"mode": mode, "model": model, "data_source": source,
                       "total": len(items), "judged": len(judged), "successes": successes,
                       "task_rate": successes / len(judged) if judged else None,
                       "pending": sum(t["status"] != "cancelled" and t.get("completed") is None
                                      for t in items),
                       "cancelled": sum(t["status"] == "cancelled" for t in items),
                       "errors": sum(t["status"] == "error" for t in items),
                       "latency": latency_summary([t["elapsed_s"] for t in items
                                                   if t["status"] != "cancelled"])})
    return result


class PerformanceLog:
    """保存本会话全部轮次；修改验收结果按轮次 ID 覆盖，不重复计入分母。"""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._turns: dict[str, dict] = {}
        self._retrieval = None
        self._lock = threading.Lock()
        self.persistence_error = ""

    def _append(self, event):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError as exc:
            self.persistence_error = str(exc)

    def record_turn(self, record: dict) -> dict:
        item = copy.deepcopy(record)
        item.setdefault("id", uuid4().hex)
        item.setdefault("at", time.time())
        item.setdefault("completed", False if item["status"] == "error" else None)
        item.setdefault("reviewer", "system" if item["status"] == "error" else "")
        item.setdefault("review_note", "请求异常，任务未完成" if item["status"] == "error" else "")
        if item["mode"] not in MODES or item["status"] not in {"answered", "cancelled", "error"}:
            raise ValueError("未知性能记录类型")
        if not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] < 0:
            raise ValueError("耗时必须是非负有限秒数")
        with self._lock:
            if item["id"] in self._turns:
                raise ValueError("轮次 ID 重复")
            self._turns[item["id"]] = item
            self._append({"event": "turn", "record": item})
        return copy.deepcopy(item)

    def review(self, turn_id: str, completed: bool, note: str = ""):
        if not isinstance(completed, bool):
            raise ValueError("验收结果必须是布尔值")
        with self._lock:
            turn = self._turns[turn_id]
            if turn["status"] == "cancelled":
                raise ValueError("用户取消的轮次不参与完成率")
            turn.update(completed=completed, reviewer="user", review_note=note,
                        reviewed_at=time.time())
            self._append({"event": "review", "id": turn_id, "completed": completed,
                          "note": note, "at": turn["reviewed_at"]})

    def set_retrieval(self, run: dict):
        with self._lock:
            self._retrieval = copy.deepcopy(run)
            self._append({"event": "retrieval", "run": run})

    def snapshot(self) -> dict:
        with self._lock:
            turns = copy.deepcopy(list(self._turns.values()))
            retrieval = copy.deepcopy(self._retrieval)
            error = self.persistence_error
        return {"turns": turns, "groups": summarize_turns(turns), "retrieval": retrieval,
                "log_path": str(self.path) if self.path else None,
                "persistence_error": error}


def evaluate_retrieval(*, index=None, cases_path: Path = CASE_PATH, cancel_event=None,
                       on_progress=None) -> dict:
    """不改检索参数；前 3 个知识块命中任一预标注块则该题记 1。错误仍在分母中。"""
    from app.services.knowledge_index import default_index

    dataset_bytes = Path(cases_path).read_bytes()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    cases = dataset["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("题集为空或题号重复")
    # 加载时间单列；第一道题的耗时包含首次 TF-IDF 矩阵构建。
    load_started = time.perf_counter()
    index = index if index is not None else default_index()
    load_s = time.perf_counter() - load_started
    known = {chunk["id"]: chunk for chunk in index.chunks}
    digest = hashlib.sha256()
    for chunk in sorted(index.chunks, key=lambda c: c["id"]):
        digest.update(json.dumps(chunk, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    rows = []
    for case in cases:
        if cancel_event is not None and cancel_event.is_set():
            from app.engine.agent.provider import ProviderCancelled
            raise ProviderCancelled("检索评测已取消")
        expected = case["expected_ids"]
        if not expected or not case.get("anchor"):
            raise ValueError("每题必须包含正确知识块与正文标注依据")
        missing = [key for key in expected if key not in known]
        valid = not missing and any(case["anchor"] in known[key]["text"] for key in expected)
        started = time.perf_counter()
        error = ""
        try:
            found = index.search(case["query"], top_k=3)
        except Exception as exc:
            found = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed_s = time.perf_counter() - started
        hit = bool(set(expected) & {r["id"] for r in found[:3]}) if valid else None
        rows.append({**case, "hit": hit, "label_valid": valid, "missing_ids": missing,
                     "error": error, "elapsed_s": elapsed_s,
                     "retrieved": [{key: item.get(key) for key in
                                    ("id", "source", "title", "page", "path", "score")}
                                   for item in found[:3]]})
        if on_progress:
            on_progress(len(rows), len(cases))
    invalid = sum(not r["label_valid"] for r in rows)
    hits = sum(r["hit"] is True for r in rows)
    return {"version": dataset["version"], "dataset_description": dataset.get("description", ""),
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "corpus_sha256": digest.hexdigest(), "corpus_chunks": len(index.chunks),
            "method": "TF-IDF char_wb 2–3 gram / cosine / top_k=3",
            "at": time.time(), "count": len(rows), "hits": hits,
            "invalid_labels": invalid, "errors": sum(bool(r["error"]) for r in rows),
            "hit_at_3": hits / len(rows) if not invalid else None,
            "index_load_s": load_s, "latency": latency_summary([r["elapsed_s"] for r in rows]),
            "rows": rows,
            "timing_note": "加载索引单独计时；首题可能包含矩阵冷启动，后续题可能使用内存缓存。"}
