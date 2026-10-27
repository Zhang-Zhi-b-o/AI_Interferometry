"""会话事件：有界内存快照与 JSONL 留档，不持有设备对象。"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import threading
import time
from uuid import uuid4


class Timeline:
    def __init__(self, directory: Path | None = None, *, capacity: int = 200):
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.path = directory / f"session-{self.session_id}.jsonl" if directory else None
        self._events = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sequence = 0
        self.persistence_error = ""
        self.data_source = "live"

    def add(self, kind: str, *, parameters=None, result=None, elapsed_s=None) -> dict:
        with self._lock:
            self._sequence += 1
            event = {"id": f"E{self._sequence:04d}", "t": time.time(), "kind": kind,
                     "data_source": self.data_source,
                     "parameters": parameters or {}, "result": result,
                     "elapsed_s": elapsed_s}
            # 独立 JSON 值：调用者后续修改列表/字典不污染已记录事件。
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
            event = json.loads(encoded)
            self._events.append(event)
            if self.path:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(encoded + "\n")
                except OSError as exc:
                    # 记录失败不能阻止急停或断开；在快照/报告中明确披露。
                    self.persistence_error = str(exc)
            return json.loads(encoded)

    def snapshot(self, limit: int = 200) -> list[dict]:
        with self._lock:
            return json.loads(json.dumps(list(self._events)[-max(1, min(limit, 200)):],
                                         ensure_ascii=False))
