"""数显微分表 LCD 定位、OCR 与连续帧读数稳定化。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Iterable
from pathlib import Path

import cv2
import numpy as np


_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class MicrometerOCRResult:
    """单帧 OCR 及多帧稳定化结果。"""

    text: str = ""
    value_mm: float | None = None
    score: float = 0.0
    stable_value_mm: float | None = None
    stable: bool = False
    roi_xyxy: tuple[int, int, int, int] | None = None
    frame: np.ndarray | None = None
    crop: np.ndarray | None = None
    format_hint: str = ""
    message: str = "尚未识别"
    reading_held: bool = False
    rejected: bool = False
    rejection_reason: str = ""
    stable_captured_at: float | None = None
    stable_captured_monotonic: float | None = None
    captured_at: float | None = None
    captured_monotonic: float | None = None


def _manual_roi(frame: np.ndarray, roi: Iterable[float]) -> tuple[int, int, int, int]:
    """把归一化 ``x, y, width, height`` 转换为安全的像素 ROI。"""
    h, w = frame.shape[:2]
    values = list(roi)
    if len(values) != 4:
        values = [0.0, 0.0, 1.0, 1.0]
    x, y, rw, rh = (float(v) for v in values)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    rw = max(0.01, min(1.0 - x, rw))
    rh = max(0.01, min(1.0 - y, rh))
    x1, y1 = int(round(x * w)), int(round(y * h))
    x2, y2 = int(round((x + rw) * w)), int(round((y + rh) * h))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def locate_lcd(
    frame: np.ndarray,
    *,
    auto_roi: bool = True,
    manual_roi: Iterable[float] = (0.0, 0.0, 1.0, 1.0),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """定位长条形 LCD；未找到时回退到配置的手动 ROI。"""
    if frame is None or frame.size == 0:
        raise ValueError("摄像头画面为空")
    fallback = _manual_roi(frame, manual_roi)
    if not auto_roi:
        x1, y1, x2, y2 = fallback
        return frame[y1:y2, x1:x2].copy(), fallback

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(clahe, (5, 5), 0), 35, 120)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    frame_area = float(height * width)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        if rh < 30 or rw < 120:
            continue
        aspect = rw / max(1.0, float(rh))
        area_ratio = (rw * rh) / frame_area
        if not 2.0 <= aspect <= 6.8 or not 0.015 <= area_ratio <= 0.75:
            continue
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / max(1.0, float(rw * rh))
        if rectangularity < 0.15:
            continue
        cx, cy = x + rw / 2.0, y + rh / 2.0
        center_distance = abs(cx / width - 0.5) + abs(cy / height - 0.5)
        score = area_ratio * 5.0 + rectangularity - center_distance * 0.25
        if best is None or score > best[0]:
            margin_x = int(rw * 0.025)
            margin_y = int(rh * 0.06)
            box = (
                max(0, x + margin_x),
                max(0, y + margin_y),
                min(width, x + rw - margin_x),
                min(height, y + rh - margin_y),
            )
            best = score, box

    box = best[1] if best is not None else fallback
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        x1, y1, x2, y2 = fallback
        crop = frame[y1:y2, x1:x2]
        box = fallback
    return crop.copy(), box


def normalize_meter_text(text: str, decimal_places: int = 3) -> str | None:
    """清洗 OCR 文本；仅在小数点缺失时按固定小数位辅助补回。"""
    cleaned = str(text or "").strip().replace(",", ".").replace(" ", "")
    cleaned = "".join(ch for ch in cleaned if ch in "-0123456789.")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    elif cleaned.startswith("-."):
        cleaned = "-0" + cleaned[1:]
    if cleaned.count(".") == 0 and decimal_places > 0:
        sign = "-" if cleaned.startswith("-") else ""
        digits = cleaned[1:] if sign else cleaned
        if digits.isdigit() and len(digits) > decimal_places:
            cleaned = f"{sign}{digits[:-decimal_places]}.{digits[-decimal_places:]}"
    if cleaned.count("-") > 1 or ("-" in cleaned and not cleaned.startswith("-")):
        return None
    if cleaned.count(".") > 1 or not _NUMBER_RE.fullmatch(cleaned):
        return None
    return cleaned


class ReadingStabilizer:
    """连续确认读数，并在异常帧期间保持最后可信稳定值。"""

    def __init__(self, window_size: int = 7, required: int = 3,
                 decimal_places: int = 3, max_step: float = 0.05,
                 jump_required: int = 6,
                 scale_ratio_tolerance: float = 0.03) -> None:
        self.window_size = max(1, int(window_size))
        self.required = max(1, min(int(required), self.window_size))
        self.decimal_places = max(0, int(decimal_places))
        self.max_step = max(10 ** (-self.decimal_places), float(max_step))
        self.jump_required = max(self.required + 1, int(jump_required))
        self.scale_ratio_tolerance = max(
            0.001, min(0.2, float(scale_ratio_tolerance)))
        self._values: deque[float] = deque(maxlen=self.window_size)
        self.stable_value: float | None = None
        self._jump_candidate: float | None = None
        self._jump_count = 0
        self.last_rejected = False
        self.last_reason = ""
        self.last_candidate: float | None = None

    def reset(self) -> None:
        self._values.clear()
        self.stable_value = None
        self._jump_candidate = None
        self._jump_count = 0
        self.last_rejected = False
        self.last_reason = ""
        self.last_candidate = None

    def _is_scale_error(self, candidate: float) -> bool:
        if self.stable_value is None or abs(self.stable_value) < 1e-12:
            return False
        stable = abs(self.stable_value)
        value = abs(candidate)
        ratio = value / stable
        tolerance = self.scale_ratio_tolerance
        return abs(ratio - 10.0) <= 10.0 * tolerance or abs(ratio - 0.1) <= tolerance

    def update(self, value: float | None) -> tuple[float | None, bool, int]:
        self.last_rejected = False
        self.last_reason = ""
        self.last_candidate = None if value is None else round(
            float(value), self.decimal_places)
        if value is None:
            return self.stable_value, False, 0
        rounded = round(float(value), self.decimal_places)

        if self.stable_value is not None:
            jump = abs(rounded - self.stable_value) > self.max_step
            # 靠近零点时，两个完全正常的三位小数值也可能恰好相差约
            # 10 倍（例如 -0.021 → -0.002）。只有比例异常同时伴随
            # 足够大的绝对跳变时，才判断为小数点/位数误识别。
            if jump and self._is_scale_error(rounded):
                self.last_rejected = True
                self.last_reason = "疑似 ×10/÷10 小数点或位数误识别"
                self._values.clear()
                self._jump_candidate = None
                self._jump_count = 0
                return self.stable_value, False, 0

            if jump:
                if rounded == self._jump_candidate:
                    self._jump_count += 1
                else:
                    self._jump_candidate = rounded
                    self._jump_count = 1
                if self._jump_count < self.jump_required:
                    self.last_rejected = True
                    self.last_reason = (
                        f"读数突变超过 {self.max_step:.{self.decimal_places}f} mm")
                    self._values.clear()
                    return self.stable_value, False, self._jump_count
                # 极端情况下真实位置可能在掉帧期间发生变化；只有很多帧
                # 持续一致才允许建立新的基准，避免永久锁死旧值。
                self.stable_value = rounded
                self._values.clear()
                self._jump_candidate = None
                self._jump_count = 0
                return self.stable_value, True, self.jump_required

            self._jump_candidate = None
            self._jump_count = 0

        self._values.append(rounded)
        recent = list(self._values)[-self.required:]
        count = 0
        for previous in reversed(self._values):
            if previous != rounded:
                break
            count += 1
        stable = len(recent) >= self.required and all(
            previous == rounded for previous in recent)
        if stable:
            self.stable_value = rounded
        return self.stable_value, stable, count


class MicrometerOCR:
    """使用 RapidOCR 的 PP-OCR ONNX 模型识别裁剪后的 LCD 数字行。"""

    def __init__(self, *, model_path: str | Path | None = None,
                 min_score: float = 0.45, decimal_places: int = 3,
                 stable_window: int = 7, stable_required: int = 3,
                 max_step_mm: float = 0.05, jump_required: int = 6,
                 scale_ratio_tolerance: float = 0.03) -> None:
        self.model_path = str(model_path) if model_path else None
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.decimal_places = max(0, int(decimal_places))
        self.stabilizer = ReadingStabilizer(
            stable_window, stable_required, self.decimal_places,
            max_step=max_step_mm, jump_required=jump_required,
            scale_ratio_tolerance=scale_ratio_tolerance)
        self._engine = None

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "未安装 RapidOCR，请执行 pip install -e .") from exc
        params = {"Rec.model_path": self.model_path} if self.model_path else None
        self._engine = RapidOCR(params=params)

    def recognize(
        self,
        frame: np.ndarray,
        *,
        auto_roi: bool = True,
        manual_roi: Iterable[float] = (0.0, 0.0, 1.0, 1.0),
    ) -> MicrometerOCRResult:
        self.load()
        crop, box = locate_lcd(
            frame, auto_roi=auto_roi, manual_roi=manual_roi)
        variants = [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4)).apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        best_text, best_score = "", 0.0
        best_normalized: str | None = None
        best_rank = -1.0
        best_hint = ""
        for image in variants:
            output = self._engine(
                image, use_det=False, use_cls=False, use_rec=True)
            texts = tuple(getattr(output, "txts", ()) or ())
            scores = tuple(getattr(output, "scores", ()) or ())
            if not texts or not scores:
                continue
            raw_text = str(texts[0])
            raw_score = float(scores[0])
            normalized = normalize_meter_text(raw_text, self.decimal_places)
            normalized_raw = raw_text.strip().replace(",", ".").replace(" ", "")
            explicit_decimal = "." in normalized_raw
            fraction_digits = (
                len(normalized.rsplit(".", 1)[1])
                if normalized is not None and "." in normalized else 0
            )
            # 三位小数只参与候选排序，不作为硬性通过条件。明确识别到的
            # 小数点优先；小数点缺失时允许按末三位补回，但稍降排序权重。
            hint = ""
            format_adjustment = 0.0
            if normalized is None:
                format_adjustment = -0.1
            elif explicit_decimal:
                if fraction_digits == self.decimal_places:
                    format_adjustment = 0.025
                else:
                    format_adjustment = -0.015
                    hint = f"画面结果不是{self.decimal_places}位小数"
            elif normalized is not None and self.decimal_places > 0:
                format_adjustment = -0.01
                hint = f"小数点按末{self.decimal_places}位辅助补全"
            rank = raw_score + format_adjustment
            if rank > best_rank:
                best_rank = rank
                best_text = raw_text
                best_score = raw_score
                best_normalized = normalized
                best_hint = hint

        normalized = best_normalized
        value = None
        if normalized is not None and best_score >= self.min_score:
            try:
                value = float(normalized)
            except ValueError:
                value = None
        stable_value, stable, repeat_count = self.stabilizer.update(value)
        held = stable_value is not None and not stable
        if value is None:
            message = (
                f"保持稳定读数 {stable_value:.{self.decimal_places}f} mm；"
                f"本帧无效：{best_text or '未识别'}"
                if stable_value is not None
                else f"读数无效：{best_text or '未识别'}")
        elif stable:
            message = f"稳定读数 {stable_value:.{self.decimal_places}f} mm"
        elif self.stabilizer.last_rejected and stable_value is not None:
            message = (
                f"保持稳定读数 {stable_value:.{self.decimal_places}f} mm；"
                f"已忽略 {normalized}：{self.stabilizer.last_reason}")
        elif stable_value is not None:
            message = (
                f"保持稳定读数 {stable_value:.{self.decimal_places}f} mm；"
                f"候选 {normalized}（{repeat_count}/{self.stabilizer.required}）")
        else:
            message = f"正在确认 {normalized}（{repeat_count}/{self.stabilizer.required}）"
        if best_hint and value is not None:
            message += f"；{best_hint}"
        return MicrometerOCRResult(
            text=normalized or best_text,
            value_mm=value,
            score=best_score,
            stable_value_mm=stable_value,
            stable=stable,
            roi_xyxy=box,
            crop=crop,
            format_hint=best_hint,
            message=message,
            reading_held=held,
            rejected=self.stabilizer.last_rejected,
            rejection_reason=self.stabilizer.last_reason,
        )
