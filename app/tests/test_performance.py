"""性能指标必须有真实分母、完整状态和可复核标注；不需要付费模型或设备。"""
import json
import threading
from types import SimpleNamespace as NS

from docx import Document
import pytest

from app.agent.runtime import AgentRuntime
from app.services.performance import (
    PerformanceLog, evaluate_retrieval, latency_summary, summarize_turns,
)
from app.services.performance_report import export_performance, summary_rows
from app.engine.agent.provider import ChatResult, ProviderError, ProviderCancelled
from tests.test_app_v2_agent import context


def turn(**values):
    return {"question": "执行任务", "answer": "任务结果", "mode": "online", "model": "test-model",
            "status": "answered", "data_source": "live", "elapsed_s": 1.0, **values}


def test_task_denominator_pending_cancel_errors_and_review(tmp_path):
    path = tmp_path / "session.jsonl"
    log = PerformanceLog(path)
    first = log.record_turn(turn())
    log.record_turn(turn(status="cancelled", elapsed_s=50))
    failed = log.record_turn(turn(status="error", elapsed_s=3))
    pending = log.record_turn(turn())
    log.review(first["id"], True, "原始目标已达到")
    summary = log.snapshot()["groups"][0]
    assert summary["total"] == 4
    assert summary["judged"] == 2 and summary["successes"] == 1
    assert summary["task_rate"] == 0.5
    assert summary["pending"] == 1 and summary["cancelled"] == 1
    assert summary["latency"] == {"count": 3, "median_s": 1, "p95_s": 3, "mean_s": 5 / 3}
    log.review(first["id"], False, "复核未达到目标")
    assert log.snapshot()["groups"][0]["judged"] == 2
    assert log.snapshot()["groups"][0]["task_rate"] == 0
    log.review(failed["id"], True, "人工复核")
    assert log.snapshot()["groups"][0]["task_rate"] == 0.5
    copy = log.snapshot()
    copy["turns"][0]["question"] = "changed"
    assert log.snapshot()["turns"][0]["question"] != "changed"
    assert pending["completed"] is None
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 7 and events[-1]["event"] == "review"
    with pytest.raises(ValueError):
        log.review(log.snapshot()["turns"][1]["id"], True)


def test_empty_groups_modes_and_nearest_rank():
    assert summarize_turns([]) == []
    assert latency_summary([])["median_s"] is None
    assert latency_summary(list(range(1, 21))) == {
        "count": 20, "median_s": 10.5, "p95_s": 19, "mean_s": 10.5}
    log = PerformanceLog()
    for mode, source in (("online", "live"), ("offline", "demo"), ("fallback", "live")):
        log.record_turn(turn(mode=mode, data_source=source))
    assert len(log.snapshot()["groups"]) == 3
    assert all(g["task_rate"] is None for g in log.snapshot()["groups"])
    assert "未评定" in str(summary_rows(log.snapshot()))


def test_record_failure_preserves_memory(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")
    log = PerformanceLog(blocked / "log.jsonl")
    log.record_turn(turn())
    assert log.snapshot()["persistence_error"]
    assert len(log.snapshot()["turns"]) == 1


def test_runtime_records_offline_online_fallback_error_and_cancel(monkeypatch):
    monkeypatch.setattr("app.agent.runtime.default_index",
                        lambda: NS(search=lambda *a, **kw: []))
    ctx = context()
    ctx.config.data["agent"] = {"chat_mode": "local"}
    runtime = AgentRuntime(ctx, provider=NS(available=False))
    runtime.ask("下一步")
    assert runtime.last_turn["mode"] == "offline"
    assert runtime.last_turn["completed"] is None
    assert runtime.last_turn["elapsed_s"] >= 0
    ctx.config.data["agent"]["chat_mode"] = "online"

    class Provider:
        available = True
        model = "fake-model"

        def chat_with_tools(self, *a, **kw):
            return ChatResult(content="请先检查画面。")

    runtime.provider = Provider()
    runtime.ask("下一步")
    assert runtime.last_turn["mode"] == "online"

    class Broken(Provider):
        def chat_with_tools(self, *a, **kw):
            raise ProviderError("network error")

    runtime.provider = Broken()
    with pytest.raises(ProviderError, match="没有切换为本地回答"):
        runtime.ask("下一步")
    assert runtime.last_turn["mode"] == "online"
    assert runtime.last_turn["status"] == "error"

    class Failed(Provider):
        def chat_with_tools(self, *a, **kw):
            raise ValueError("bad response")

    runtime.provider = Failed()
    with pytest.raises(ValueError):
        runtime.ask("下一步")
    assert runtime.last_turn["status"] == "error"
    assert runtime.last_turn["completed"] is False
    cancel = threading.Event()
    cancel.set()
    assert "已取消" in runtime.ask("下一步", cancel_event=cancel)
    assert runtime.last_turn["status"] == "cancelled"
    assert ctx.calls == []


def cases_file(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": "test-v1", "cases": cases}), encoding="utf-8")
    return path


def test_retrieval_hit3_known_labels_miss_error_and_invalid(tmp_path):
    chunks = [{"id": f"S{i}-p001-01", "text": "标注依据"} for i in range(1, 6)]
    cases = [{"id": f"R{i}", "query": f"query{i}", "expected_ids": ["S1-p001-01"],
              "anchor": "标注依据"} for i in range(1, 4)]
    path = cases_file(tmp_path, cases)

    class Index:
        def __init__(self):
            self.chunks = chunks

        def search(self, query, top_k):
            assert top_k == 3
            if query == "query1":
                return [chunks[1], chunks[2], chunks[0]]
            if query == "query2":
                # 第四条命中不计入 Hit@3。
                return chunks[1:] + [chunks[0]]
            raise OSError("query failed")

    result = evaluate_retrieval(index=Index(), cases_path=path)
    assert result["hit_at_3"] == 1 / 3
    assert result["hits"] == 1 and result["count"] == 3
    assert result["errors"] == 1
    assert result["latency"]["count"] == 3
    cases[0]["expected_ids"] = ["missing"]
    result = evaluate_retrieval(index=Index(), cases_path=cases_file(tmp_path, cases))
    assert result["invalid_labels"] == 1 and result["hit_at_3"] is None
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ProviderCancelled):
        evaluate_retrieval(index=Index(), cases_path=path, cancel_event=cancel)


def test_export_docx_and_json_preserve_counts_and_no_fake_success(tmp_path):
    log = PerformanceLog()
    log.record_turn(turn())
    good = log.record_turn(turn(elapsed_s=2.5))
    log.review(good["id"], True, "已检查结果")
    paths = export_performance(log.snapshot(), tmp_path)
    doc = Document(paths["docx"])
    text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(
        c.text for table in doc.tables for row in table.rows for c in row.cells)
    assert "完成 1/已验收 1；待验收 1" in text
    assert "100.0%" in text
    assert "Hit@3" in text and "请先运行检索评测" in text
    data = json.loads(open(paths["json"], encoding="utf-8").read())
    assert data["groups"][0]["judged"] == 1
    assert data["turns"] == log.snapshot()["turns"]
