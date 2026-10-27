"""服务层：设备与算法的适配器。**不导入 Qt**，只通过回调把结果交出去。

所有服务都遵守两条约定：
1. 构造时注入 ``on_*`` 回调（可空），服务自己起线程，绝不阻塞调用方；
2. ``close()`` / ``stop()`` 幂等，程序退出时按依赖顺序调用。

想换掉某个实现（例如把 cv2 采集换成旧 ``app.engine.camera.CameraManager``），
只需要保持类名和方法签名不变，界面代码一行都不用改。
"""
from __future__ import annotations

__all__ = [
    "CameraService",
    "DetectService",
    "FringeService",
    "ReadingService",
    "MotionService",
    "SearchService",
    "ComputeService",
    "RecordService",
    "ThicknessMapService",
    "CalibrationTable",
]
