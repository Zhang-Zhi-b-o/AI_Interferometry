"""相机采集模块"""
from app.engine.camera.manager import CameraManager
from app.engine.camera.registry import CAMERA_REGISTRY, CameraLease, CameraRegistry

__all__ = ["CameraManager", "CAMERA_REGISTRY", "CameraLease", "CameraRegistry"]
