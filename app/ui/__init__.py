"""界面层：主题、组件、页面。

界面只依赖 ``app/core`` 与 ``app/services`` 的公开接口；不允许在这里 import
``app.engine.hardware`` / ``app.engine.camera`` / ``ultralytics``。
"""
from __future__ import annotations

__all__ = ["theme", "main_window"]
