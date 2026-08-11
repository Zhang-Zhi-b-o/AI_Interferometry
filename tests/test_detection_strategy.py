import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from src.vision.detection_strategy import (
    TemporalReference,
    apply_standard_detection_strategy,
)
from src.vision.detector import YOLODetector


def box_xyxy(x, y, width, height, size=1000):
    return np.array([
        (x - width / 2) * size,
        (y - height / 2) * size,
        (x + width / 2) * size,
        (y + height / 2) * size,
    ], dtype=np.float32)


def predicted(boxes, confidences, class_ids):
    names = {0: "zero_order", 1: "near_fringe"}
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor(np.asarray(boxes)),
            conf=torch.tensor(confidences),
            cls=torch.tensor(class_ids),
        ),
        names=names,
    )


class DetectionStrategyTests(unittest.TestCase):
    def test_zero_order_deduplicates_and_allows_two_views(self):
        boxes = np.array([
            box_xyxy(0.20, 0.50, 0.08, 0.20),
            box_xyxy(0.80, 0.50, 0.08, 0.20),
            box_xyxy(0.80, 0.50, 0.06, 0.18),
            box_xyxy(0.50, 0.50, 0.08, 0.20),
        ])
        result = apply_standard_detection_strategy(
            boxes, np.array([0.80, 0.90, 0.70, 0.40]),
            np.array([0, 0, 0, 0]), ["zero_order"] * 4,
            (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [1, 0])
        self.assertEqual(result.removed_overlap, 1)
        self.assertEqual(result.removed_weak_unconfirmed, 1)
        self.assertIn("two_zero_order", result.review_reasons)

    def test_near_fringe_keeps_four_highest_non_duplicates(self):
        boxes = np.array([
            box_xyxy(x, 0.50, 0.05, 0.20)
            for x in (0.10, 0.28, 0.46, 0.64, 0.82)
        ])
        result = apply_standard_detection_strategy(
            boxes, np.array([0.91, 0.81, 0.71, 0.61, 0.51]),
            np.array([1] * 5), ["near_fringe"] * 5,
            (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [0, 1, 2, 3])
        self.assertEqual(result.removed_count_limit, 1)
        self.assertIn("near_fringe_over_limit", result.review_reasons)
        self.assertIn("three_or_four_near_fringe", result.review_reasons)

    def test_allows_full_frame_box_but_rejects_tiny_or_invalid_box(self):
        boxes = np.array([
            box_xyxy(0.50, 0.50, 1.00, 1.00),
            box_xyxy(0.50, 0.50, 0.01, 0.01),
            np.array([-10, 100, 300, 500], dtype=np.float32),
        ])
        result = apply_standard_detection_strategy(
            boxes, np.array([0.90, 0.90, 0.90]), np.array([1, 1, 1]),
            ["near_fringe"] * 3, (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [0])
        self.assertEqual(result.removed_geometry, 2)

    def test_temporal_reference_rescues_weak_candidate(self):
        box = box_xyxy(0.70, 0.50, 0.10, 0.20)
        result = apply_standard_detection_strategy(
            np.array([box]), np.array([0.36]), np.array([1]),
            ["near_fringe"], (1000, 1000, 3),
            temporal_references=[
                TemporalReference("near_fringe", (0.69, 0.50, 0.10, 0.20))],
        )
        self.assertEqual(result.kept_indices, [0])
        self.assertEqual(result.rescued_temporal, 1)
        self.assertTrue(result.needs_review)

    def test_unconfirmed_weak_candidate_is_empty_but_marked_for_review(self):
        result = apply_standard_detection_strategy(
            np.array([box_xyxy(0.50, 0.50, 0.10, 0.20)]),
            np.array([0.40]), np.array([0]), ["zero_order"],
            (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [])
        self.assertIn(
            "empty_with_unconfirmed_weak_candidate", result.review_reasons)

    def test_cross_class_conflict_uses_confidence_not_area(self):
        boxes = np.array([
            box_xyxy(0.50, 0.50, 0.40, 0.40),
            box_xyxy(0.50, 0.50, 0.40, 0.40),
        ])
        result = apply_standard_detection_strategy(
            boxes, np.array([0.85, 0.65]), np.array([0, 1]),
            ["zero_order", "near_fringe"], (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [0])
        self.assertEqual(result.removed_cross_class, 1)
        self.assertIn("cross_class_high_overlap", result.review_reasons)

    def test_semantic_names_take_priority_over_reversed_class_ids(self):
        boxes = np.array([
            box_xyxy(0.30, 0.50, 0.10, 0.20),
            box_xyxy(0.70, 0.50, 0.10, 0.20),
        ])
        result = apply_standard_detection_strategy(
            boxes, np.array([0.8, 0.9]), np.array([0, 1]),
            ["color_fringe", "black_zero"], (1000, 1000, 3),
        )
        self.assertEqual(result.kept_indices, [1, 0])

    def test_detector_uses_new_inference_values_and_previous_reliable_frame(self):
        detector = YOLODetector("unused.pt", confidence=0.9, iou=0.1, device="cpu")
        detector._model = Mock()
        detector._model.names = {0: "zero_order", 1: "near_fringe"}
        target_box = box_xyxy(0.60, 0.50, 0.10, 0.20)
        detector._model.predict.side_effect = [
            [predicted([target_box], [0.80], [1])],
            [predicted([target_box], [0.40], [1])],
        ]

        first = detector.detect(np.zeros((1000, 1000, 3), dtype=np.uint8))
        second = detector.detect(np.zeros((1000, 1000, 3), dtype=np.uint8))

        call = detector._model.predict.call_args
        self.assertEqual(call.kwargs["conf"], 0.25)
        self.assertEqual(call.kwargs["iou"], 0.70)
        self.assertEqual(len(first["boxes_xyxy"]), 1)
        self.assertEqual(len(second["boxes_xyxy"]), 1)
        self.assertEqual(second["strategy"]["rescued_temporal"], 1)


if __name__ == "__main__":
    unittest.main()
