"""相机采集模块"""
from src.camera.manager import CameraManager
from src.camera.registry import CAMERA_REGISTRY, CameraLease, CameraRegistry

__all__ = ["CameraManager", "CAMERA_REGISTRY", "CameraLease", "CameraRegistry"]
