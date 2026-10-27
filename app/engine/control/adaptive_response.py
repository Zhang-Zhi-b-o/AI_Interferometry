"""电机—画面响应的受限在线学习器。

本模块只学习观测量并返回有界参数，不产生也不发送电机命令。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass
class _GearResponse:
    samples: int = 0
    speed_px_s: float = 0.0

    def add(self, speed: float, alpha: float = 0.18) -> None:
        speed = abs(float(speed))
        if self.samples == 0:
            self.speed_px_s = speed
        else:
            self.speed_px_s = (1.0 - alpha) * self.speed_px_s + alpha * speed
        self.samples += 1


@dataclass
class AdaptiveResponseLearner:
    """学习档位的像素速度和停车后的稳定时间。"""

    min_speed_px_s: float = 1.0
    max_settle_seconds: float = 2.0
    gear_responses: dict[int, _GearResponse] = field(default_factory=dict)
    settle_samples: int = 0
    settle_seconds: float = 0.0
    profile_key: str = ""
    _last_direction: str = "stopped"
    _stopped_at: float | None = None

    def observe(
        self,
        *,
        now: float,
        direction: str,
        gear: int | None,
        velocity_px_s: float | None,
        stable: bool,
        blurred: bool = False,
        held: bool = False,
        profile_key: str | None = None,
    ) -> None:
        """加入一帧可观测证据；模糊或历史保持帧不用于学习。"""
        if profile_key:
            self.set_profile(profile_key)
        direction = direction if direction in {"forward", "reverse"} else "stopped"
        now = float(now)
        if self._last_direction != "stopped" and direction == "stopped":
            self._stopped_at = now
        elif direction != "stopped":
            self._stopped_at = None

        if (direction != "stopped" and gear is not None and not blurred and not held
                and velocity_px_s is not None and math.isfinite(float(velocity_px_s))
                and abs(float(velocity_px_s)) >= self.min_speed_px_s):
            safe_gear = max(1, min(10, int(gear)))
            self.gear_responses.setdefault(safe_gear, _GearResponse()).add(
                abs(float(velocity_px_s)))

        if (direction == "stopped" and self._stopped_at is not None and stable
                and not blurred and not held):
            elapsed = now - self._stopped_at
            if 0.03 <= elapsed <= self.max_settle_seconds:
                if self.settle_samples == 0:
                    self.settle_seconds = elapsed
                else:
                    self.settle_seconds = (
                        0.8 * self.settle_seconds + 0.2 * elapsed)
                self.settle_samples += 1
                self._stopped_at = None
            elif elapsed > self.max_settle_seconds:
                self._stopped_at = None
        self._last_direction = direction

    def set_profile(self, profile_key: str) -> None:
        """相机分辨率或缩放改变时丢弃不可比的历史响应。"""
        normalized = str(profile_key).strip()
        if not normalized:
            return
        if self.profile_key and self.profile_key != normalized:
            self.gear_responses.clear()
            self.settle_samples = 0
            self.settle_seconds = 0.0
            self._last_direction = "stopped"
            self._stopped_at = None
        self.profile_key = normalized

    @property
    def response_samples(self) -> int:
        return sum(item.samples for item in self.gear_responses.values())

    @property
    def confidence(self) -> float:
        response = min(1.0, self.response_samples / 20.0)
        settling = min(1.0, self.settle_samples / 5.0)
        return round(0.7 * response + 0.3 * settling, 3)

    def optimized_params(
        self,
        base: dict[str, Any],
        *,
        spacing_px: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回参数副本及变更说明；所有变更都在原参数附近受限。"""
        result = dict(base)
        changes: dict[str, Any] = {}
        if self.settle_samples >= 3:
            learned = max(0.08, min(1.5, self.settle_seconds * 1.20))
            original = float(base.get("stop_detect_settle_seconds", 0.3))
            bounded = max(original * 0.6, min(original * 1.8, learned))
            result["stop_detect_settle_seconds"] = round(bounded, 3)
            changes["stop_detect_settle_seconds"] = round(bounded, 3)

        slow_gear = max(1, min(10, int(base.get("slow_gear", 10))))
        response = self.gear_responses.get(slow_gear)
        if response is not None and response.samples >= 5:
            settle = float(result.get("stop_detect_settle_seconds", 0.3))
            tolerance = float(base.get("tolerance_px", 15.0))
            spacing_guard = (
                max(0.0, float(spacing_px)) * 0.5
                if spacing_px is not None and math.isfinite(float(spacing_px)) else 0.0)
            learned_zone = response.speed_px_s * settle * 1.5 + 2.0 * tolerance
            learned_zone = max(learned_zone, spacing_guard, 10.0)
            original_zone = float(base.get("slow_zone_px", 160.0))
            bounded_zone = max(
                original_zone * 0.6,
                min(original_zone * 1.8, learned_zone, 2000.0),
            )
            result["slow_zone_px"] = round(bounded_zone, 2)
            changes["slow_zone_px"] = round(bounded_zone, 2)
        return result, changes

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "profile_key": self.profile_key,
            "confidence": self.confidence,
            "response_samples": self.response_samples,
            "settle_samples": self.settle_samples,
            "learned_settle_seconds": (
                round(self.settle_seconds, 3) if self.settle_samples else None),
            "gear_speed_px_s": {
                str(gear): round(item.speed_px_s, 2)
                for gear, item in sorted(self.gear_responses.items())
            },
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveResponseLearner":
        learner = cls()
        target = Path(path)
        if not target.exists():
            return learner
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            learner.profile_key = str(payload.get("profile_key") or "")
            learner.settle_samples = max(0, int(payload.get("settle_samples", 0)))
            value = payload.get("learned_settle_seconds")
            learner.settle_seconds = max(0.0, float(value or 0.0))
            total = max(0, int(payload.get("response_samples", 0)))
            speeds = payload.get("gear_speed_px_s") or {}
            if speeds:
                per_gear = max(1, total // len(speeds))
                for gear, speed in speeds.items():
                    learner.gear_responses[max(1, min(10, int(gear)))] = _GearResponse(
                        samples=per_gear, speed_px_s=max(0.0, float(speed)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return cls()
        return learner
