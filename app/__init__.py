"""迈克尔逊白光干涉智能测量工作台 v2。

分层约定（改代码前先读这条）::

    app/core      纯 Python：配置、状态、后台任务。不依赖 Qt，也不依赖硬件。
    app/services  设备与算法适配：相机、YOLO、条纹、读数、电机、计算、导出。
                  不依赖 Qt；只能用回调把结果交出去。
    app/ui        PySide6 界面：主题、组件、页面。只调用 services，不直接碰硬件。
    app/agent     会话、九工具与本地实验流程建议。

一句话：**改界面不要动 services，改算法不要动 ui**。
"""
from __future__ import annotations

__version__ = "2.0.0-dev"
APP_NAME = "白光干涉测量工作台"
APP_NAME_EN = "AI INTERFEROMETRY WORKBENCH"
