"""模型只接收 JSON 状态，不接收画面或设备对象。"""
from __future__ import annotations

import json
import time


def snapshot(ctx) -> dict:
    payload = {
        "data_source": "demo" if getattr(ctx, "demo_mode", False) else "live",
        "state": ctx.state.snapshot(),
        "roi": list(ctx.roi) if ctx.roi else None,
        "services": {
            "camera_open": ctx.camera.is_open,
            "model_loaded": ctx.detect.is_loaded,
            "reading_connected": ctx.reading.connected,
            "motor_connected": ctx.motion.connected,
            "search_running": ctx.search.running,
            "target_running": ctx.target.running,
        },
        "recent_readings": list(ctx.history[-5:]),
    }
    at = ctx.state.reading_at
    payload["reading_age_seconds"] = None if at is None else max(0, time.time() - at)
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
