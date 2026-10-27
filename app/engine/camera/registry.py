"""进程内 USB 相机所有权注册表。"""
from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class CameraLease:
    index: int
    owner: str


class CameraRegistry:
    """保证同一相机索引在一个进程内只有一个所有者。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: dict[int, str] = {}

    def acquire(self, index: int, owner: str) -> bool:
        index = int(index)
        owner = str(owner).strip() or "anonymous"
        with self._lock:
            current = self._owners.get(index)
            if current is not None:
                return False
            self._owners[index] = owner
            return True

    def release(self, index: int, owner: str) -> None:
        with self._lock:
            if self._owners.get(int(index)) == str(owner):
                self._owners.pop(int(index), None)

    def owner_of(self, index: int) -> str | None:
        with self._lock:
            return self._owners.get(int(index))

    def leases(self) -> tuple[CameraLease, ...]:
        with self._lock:
            return tuple(
                CameraLease(index, owner)
                for index, owner in sorted(self._owners.items())
            )


CAMERA_REGISTRY = CameraRegistry()
