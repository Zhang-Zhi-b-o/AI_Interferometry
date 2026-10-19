"""纯 Python 内核：配置、状态、后台任务。"""
from __future__ import annotations

from app.core.config import AppConfig, load_config
from app.core.state import AppState, Signal
from app.core.tasks import Background

__all__ = ["AppConfig", "load_config", "AppState", "Signal", "Background"]
