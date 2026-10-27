"""合并人工数据、多视场双类别模型的统一运行时筛选策略。"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

import numpy as np


MODEL_INFERENCE_CONFIDENCE = 0.25
MODEL_NMS_IOU = 0.70
REGULAR_CONFIDENCE = 0.50
HIGH_RELIABILITY_CONFIDENCE = 0.60
TEMPORAL_MIN_CONFIDENCE = 0.30
CONTAINMENT_THRESHOLD = 0.80
OVERLAP_IOU_THRESHOLD = 0.65
CROSS_CLASS_IOU_THRESHOLD = 0.75
CROSS_CLASS_CONFIDENCE_MARGIN = 0.15
TEMPORAL_IOU_THRESHOLD = 0.50
TEMPORAL_CENTER_DISTANCE = 0.10
TEMPORAL_SIZE_CHANGE = 0.30
MIN_SIDE_RATIO = 0.015
MIN_AREA_RATIO = 0.003
MAX_ZERO_ORDER = 2
MAX_NEAR_FRINGE = 4

_ZERO_KEYWORDS = ("zero", "order", "black", "dark", "零级", "黑")
_NEAR_KEYWORDS = ("near", "fringe", "color", "colour", "近", "彩", "条纹")
_FAR_KEYWORDS = ("far", "远场", "远条纹")


@dataclass(frozen=True)
class DetectionCandidate:
    index: int
    role: str
    confidence: float
    box_xywhn: tuple[float, float, float, float]


@dataclass(frozen=True)
class TemporalReference:
    """最近可靠帧中的目标；坐标均为全帧归一化坐标。"""

    role: str
    box_xywhn: tuple[float, float, float, float]


@dataclass
class DetectionStrategyResult:
    kept_indices: list[int] = field(default_factory=list)
    reliable_references: list[TemporalReference] = field(default_factory=list)
    raw_count: int = 0
    removed_low_confidence: int = 0
    removed_weak_unconfirmed: int = 0
    removed_geometry: int = 0
    removed_overlap: int = 0
    removed_count_limit: int = 0
    removed_cross_class: int = 0
    removed_unknown_class: int = 0
    rescued_temporal: int = 0
    review_reasons: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def add_review(self, reason: str) -> None:
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)

    def as_dict(self) -> dict:
        return {
            "kept_indices": list(self.kept_indices),
            "raw_count": self.raw_count,
            "removed_low_confidence": self.removed_low_confidence,
            "removed_weak_unconfirmed": self.removed_weak_unconfirmed,
            "removed_geometry": self.removed_geometry,
            "removed_overlap": self.removed_overlap,
            "removed_count_limit": self.removed_count_limit,
            "removed_cross_class": self.removed_cross_class,
            "removed_unknown_class": self.removed_unknown_class,
            "rescued_temporal": self.rescued_temporal,
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "background": not self.kept_indices,
        }


def _class_role(class_id: int, class_name: str) -> str | None:
    normalized = str(class_name).lower()
    if any(keyword in normalized for keyword in _ZERO_KEYWORDS):
        return "zero_order"
    if any(keyword in normalized for keyword in _FAR_KEYWORDS):
        return None
    if any(keyword in normalized for keyword in _NEAR_KEYWORDS):
        return "near_fringe"
    if int(class_id) == 0:
        return "zero_order"
    if int(class_id) == 1:
        return "near_fringe"
    return None


def _xywhn(box_xyxy, frame_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    return (
        ((x1 + x2) / 2.0) / max(1.0, float(width)),
        ((y1 + y2) / 2.0) / max(1.0, float(height)),
        (x2 - x1) / max(1.0, float(width)),
        (y2 - y1) / max(1.0, float(height)),
    )


def _corners(box: tuple[float, float, float, float]) -> tuple[float, ...]:
    x, y, width, height = box
    return x - width / 2, y - height / 2, x + width / 2, y + height / 2


def _valid_geometry(box: tuple[float, float, float, float]) -> bool:
    """只拒绝非法/极小框，允许位于任意视场以及接近整幅图的大框。"""
    if not all(isfinite(value) for value in box):
        return False
    x, y, width, height = box
    x1, y1, x2, y2 = _corners(box)
    return (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 <= x1 <= x2 <= 1.0
        and 0.0 <= y1 <= y2 <= 1.0
        and width >= MIN_SIDE_RATIO
        and height >= MIN_SIDE_RATIO
        and width * height >= MIN_AREA_RATIO
    )


def _overlap_boxes(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = _corners(first)
    bx1, by1, bx2, by2 = _corners(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    second_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    minimum_area = min(first_area, second_area)
    containment = intersection / minimum_area if minimum_area else 0.0
    union = first_area + second_area - intersection
    return containment, intersection / union if union else 0.0


def _overlap(first: DetectionCandidate, second: DetectionCandidate) -> tuple[float, float]:
    return _overlap_boxes(first.box_xywhn, second.box_xywhn)


def _same_target(candidate: DetectionCandidate, reference: TemporalReference) -> bool:
    if candidate.role != reference.role:
        return False
    _, iou = _overlap_boxes(candidate.box_xywhn, reference.box_xywhn)
    if iou >= TEMPORAL_IOU_THRESHOLD:
        return True
    x, y, width, height = candidate.box_xywhn
    rx, ry, rwidth, rheight = reference.box_xywhn
    size_close = (
        abs(width - rwidth) / max(width, rwidth) <= TEMPORAL_SIZE_CHANGE
        and abs(height - rheight) / max(height, rheight) <= TEMPORAL_SIZE_CHANGE
    )
    return size_close and hypot(x - rx, y - ry) <= TEMPORAL_CENTER_DISTANCE


def _deduplicate(
    candidates: list[DetectionCandidate], result: DetectionStrategyResult,
) -> list[DetectionCandidate]:
    kept: list[DetectionCandidate] = []
    for item in sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True):
        if any(
            (lambda overlap: overlap[0] >= CONTAINMENT_THRESHOLD
             or overlap[1] >= OVERLAP_IOU_THRESHOLD)(_overlap(item, previous))
            for previous in kept
        ):
            result.removed_overlap += 1
        else:
            kept.append(item)
    return kept


def apply_standard_detection_strategy(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    frame_shape: tuple[int, ...],
    temporal_references: list[TemporalReference] | None = None,
) -> DetectionStrategyResult:
    """应用多视场、置信度分级、同类去重和相邻帧弱框挽救策略。"""
    count = min(len(boxes_xyxy), len(confidences), len(class_ids), len(class_names))
    result = DetectionStrategyResult(raw_count=count)
    grouped: dict[str, list[DetectionCandidate]] = {
        "zero_order": [], "near_fringe": []}
    weak_seen = False

    for index in range(count):
        role = _class_role(int(class_ids[index]), class_names[index])
        if role is None:
            result.removed_unknown_class += 1
            continue
        confidence = float(confidences[index])
        if not isfinite(confidence) or confidence < TEMPORAL_MIN_CONFIDENCE:
            result.removed_low_confidence += 1
            continue
        candidate = DetectionCandidate(
            index=index,
            role=role,
            confidence=confidence,
            box_xywhn=_xywhn(boxes_xyxy[index], frame_shape),
        )
        if not _valid_geometry(candidate.box_xywhn):
            result.removed_geometry += 1
            continue
        weak_seen |= confidence < REGULAR_CONFIDENCE
        grouped[role].append(candidate)

    references = list(temporal_references or [])
    accepted: dict[str, list[DetectionCandidate]] = {
        "zero_order": [], "near_fringe": []}
    for role, candidates in grouped.items():
        for item in _deduplicate(candidates, result):
            if item.confidence >= REGULAR_CONFIDENCE:
                accepted[role].append(item)
                if item.confidence < HIGH_RELIABILITY_CONFIDENCE:
                    result.add_review("accepted_confidence_0.50_to_0.60")
            elif any(_same_target(item, reference) for reference in references):
                accepted[role].append(item)
                result.rescued_temporal += 1
                result.add_review("temporal_rescue_0.30_to_0.50")
            else:
                result.removed_weak_unconfirmed += 1

    if len(accepted["zero_order"]) > MAX_ZERO_ORDER:
        result.removed_count_limit += len(accepted["zero_order"]) - MAX_ZERO_ORDER
        result.add_review("zero_order_over_limit")
        accepted["zero_order"] = accepted["zero_order"][:MAX_ZERO_ORDER]
    if len(accepted["near_fringe"]) > MAX_NEAR_FRINGE:
        result.removed_count_limit += len(accepted["near_fringe"]) - MAX_NEAR_FRINGE
        result.add_review("near_fringe_over_limit")
        accepted["near_fringe"] = accepted["near_fringe"][:MAX_NEAR_FRINGE]

    if len(accepted["zero_order"]) == 2:
        result.add_review("two_zero_order")
    if len(accepted["near_fringe"]) >= 3:
        result.add_review("three_or_four_near_fringe")

    remove_indices: set[int] = set()
    for zero in accepted["zero_order"]:
        for near in accepted["near_fringe"]:
            _, iou = _overlap(zero, near)
            if iou < CROSS_CLASS_IOU_THRESHOLD:
                continue
            result.add_review("cross_class_high_overlap")
            if abs(zero.confidence - near.confidence) >= CROSS_CLASS_CONFIDENCE_MARGIN:
                loser = zero if zero.confidence < near.confidence else near
                remove_indices.add(loser.index)

    kept = [
        item for role in ("zero_order", "near_fringe")
        for item in accepted[role] if item.index not in remove_indices
    ]
    result.removed_cross_class = len(remove_indices)
    kept.sort(key=lambda item: (0 if item.role == "zero_order" else 1,
                                -item.confidence))
    result.kept_indices = [item.index for item in kept]
    result.reliable_references = [
        TemporalReference(item.role, item.box_xywhn)
        for item in kept if item.confidence >= REGULAR_CONFIDENCE
    ]
    if not kept and weak_seen:
        result.add_review("empty_with_unconfirmed_weak_candidate")
    return result
