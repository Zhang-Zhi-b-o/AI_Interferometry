"""旧“独立实验助手”的兼容入口。

实验助手现已合并到主工作台：本模块不再创建第二个窗口、
第二个 AgentService 或独立会话，只把旧入口重定向到主应用。
"""
from __future__ import annotations

from src.ui.app import YoloCamApp
from src.ui.app import run_app as _run_main_app


# 保留旧导入名，但它与主应用是同一个类，不是第二套助手。
StandaloneExperimentAssistant = YoloCamApp


def run_app() -> None:
    """启动包含唯一实验助手的主工作台。"""
    _run_main_app()


__all__ = ["StandaloneExperimentAssistant", "run_app"]
