"""Manual workspace module definitions.

This file owns the information architecture of the manual workspace.  Widget
construction remains in :mod:`src.ui.app` because several panels need runtime
configuration, while grouping and ordering live here as data.
"""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk

from src.ui.widgets import (
    AgentPluginPanel,
    AutoCenterControlPanel,
    CameraPluginPanel,
    FringeCenterPluginPanel,
    LogPanel,
    MicrometerPluginPanel,
    ModelPluginPanel,
    MotorControlPanel,
    StatusPanel,
    VideoRecorderPanel,
)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    panel_class: type[tk.Widget]
    attribute: str


@dataclass(frozen=True)
class ManualModuleSpec:
    key: str
    title: str
    description: str
    panels: tuple[PanelSpec, ...]


MANUAL_MODULES = (
    ManualModuleSpec(
        "vision", "01  视觉观察", "相机、模型预测、中心条纹、ROI 与录像",
        (
            PanelSpec("status", "实时状态", StatusPanel, "status"),
            PanelSpec("camera", "摄像头控制", CameraPluginPanel, "camera_plugin"),
            PanelSpec("model", "模型与预测", ModelPluginPanel, "model_plugin"),
            PanelSpec(
                "fringe_center", "中心条纹分析", FringeCenterPluginPanel,
                "fringe_center_plugin",
            ),
            PanelSpec("recorder", "视频录制", VideoRecorderPanel, "recorder"),
        ),
    ),
    ManualModuleSpec(
        "motion", "02  运动控制", "电机连接、人工操作与自动寻找中心条纹",
        (
            PanelSpec("motor", "电机控制", MotorControlPanel, "motor_panel"),
            PanelSpec(
                "auto_control", "自动寻找中心条纹", AutoCenterControlPanel,
                "manual_auto_center_panel",
            ),
        ),
    ),
    ManualModuleSpec(
        "measurement", "03  测量记录", "微分表画面、OCR、读数时间戳与实验记录",
        (
            PanelSpec(
                "micrometer", "视觉微分表读数", MicrometerPluginPanel,
                "micrometer_panel",
            ),
        ),
    ),
    ManualModuleSpec(
        "assistant", "04  实验助手", "状态问答、数据计算、报告与运行日志",
        (
            PanelSpec("agent", "实验助手", AgentPluginPanel, "agent_panel"),
            PanelSpec("log", "运行日志", LogPanel, "log"),
        ),
    ),
)


PANEL_MODULE = {
    panel.key: module.key
    for module in MANUAL_MODULES
    for panel in module.panels
}
