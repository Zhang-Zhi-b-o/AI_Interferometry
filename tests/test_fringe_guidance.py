import unittest

import numpy as np

from src.vision.fringe_guidance import (
    analyse_guidance_geometry,
    build_fringe_guidance,
)


def _ready_geometry(angle=0.4):
    return {
        "angle": {
            "tilt_deg": angle,
            "correction_deg": -angle,
            "confidence": 0.92,
            "curvature": 0.002,
        },
        "spacing": {
            "spacing_px": 40.2,
            "cv_percent": 2.1,
            "confidence": 0.91,
            "quality_valid": True,
            "min_fringes_ok": True,
            "cv_ok": True,
            "rejection_ok": True,
            "num_fringes": 7,
            "num_valid_intervals": 6,
        },
    }


class FringeGuidanceTests(unittest.TestCase):
    def test_no_fringe_requests_optical_and_roi_check(self):
        result = build_fringe_guidance(
            recognition={"has_fringe": False},
            motion={"has_fringe": False, "movement": "unknown"},
        )
        self.assertTrue(result["read_only"])
        self.assertEqual(result["phase"], "searching")
        self.assertFalse(result["measurement_ready"])
        self.assertTrue(any("光路" in text for text in result["recommendations"]))

    def test_blurred_moving_frame_is_not_measurement_ready(self):
        result = build_fringe_guidance(
            recognition={
                "has_fringe": True, "confidence": 0.9,
                "blurred": True, "velocity_px_s": 60.0,
            },
            motion={"has_fringe": True, "movement": "right"},
            texture={"sharpness": 0.18, "blurred": True},
            geometry=_ready_geometry(),
        )
        self.assertFalse(result["measurement_ready"])
        self.assertEqual(result["phase"], "quality_recovery")
        issue_text = " ".join(item["text"] for item in result["issues"])
        self.assertIn("运动模糊", issue_text)
        self.assertTrue(any("降低电机速度" in text
                            for text in result["recommendations"]))

    def test_stable_high_quality_frame_is_ready(self):
        result = build_fringe_guidance(
            recognition={
                "has_fringe": True, "confidence": 0.95,
                "blurred": False, "velocity_px_s": 0.5,
            },
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.92, "blurred": False},
            geometry=_ready_geometry(),
            center_x=320.0,
            frame_width=640,
        )
        self.assertTrue(result["measurement_ready"])
        self.assertEqual(result["phase"], "measurement_ready")
        self.assertGreater(result["quality_score"], 0.8)
        self.assertIn("保存原始帧", result["recommendations"][0])

    def test_tilt_generates_image_correction_guidance(self):
        result = build_fringe_guidance(
            recognition={"has_fringe": True, "confidence": 0.85},
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.8},
            geometry=_ready_geometry(angle=12.0),
        )
        self.assertTrue(any("倾斜" in item["text"] for item in result["issues"]))
        self.assertTrue(any("校正" in text for text in result["recommendations"]))
        action = next(item for item in result["actions"]
                      if item["code"] == "apply_angle_correction")
        self.assertTrue(action["requires_confirmation"])
        self.assertAlmostEqual(action["params"]["delta_deg"], -12.0)

    def test_no_fringe_can_offer_confirmed_closed_loop_search(self):
        result = build_fringe_guidance(
            recognition={"has_fringe": False},
            motion={"has_fringe": False},
            motor_connected=True,
        )
        self.assertEqual(result["actions"][0]["code"], "start_auto_search")
        self.assertEqual(result["actions"][0]["risk"], "motor")

    def test_blur_actions_are_whitelisted_and_ordered_safely(self):
        result = build_fringe_guidance(
            recognition={
                "has_fringe": True, "confidence": 0.8,
                "blurred": True, "velocity_px_s": 30,
            },
            motion={"has_fringe": True, "movement": "right"},
            texture={"sharpness": 0.1, "blurred": True},
            geometry=_ready_geometry(),
            auto_enabled=True,
        )
        codes = [item["code"] for item in result["actions"]]
        self.assertEqual(codes[0], "stop_auto_center")
        self.assertIn("enable_motion_enhancement", codes)

    def test_geometry_analysis_returns_angle_and_spacing(self):
        height, width, period = 220, 360, 36.0
        x = np.arange(width, dtype=np.float64)[None, :]
        y = np.arange(height, dtype=np.float64)[:, None]
        shift = np.tan(np.radians(10.0)) * y
        gray = 128.0 + 100.0 * np.sin(2.0 * np.pi * (x - shift) / period)
        image = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)

        result = analyse_guidance_geometry(image)

        self.assertIn("angle", result)
        self.assertIn("spacing", result)
        self.assertAlmostEqual(result["angle"]["tilt_deg"], 10.0, delta=2.0)
        self.assertIsNotNone(result["spacing"]["spacing_px"])
        self.assertGreater(len(result["fringes"]), 3)
        self.assertIn("dominant_name_zh", result["color_summary"])

    def test_roi_fringe_positions_use_full_frame_coordinates(self):
        height, width, period = 240, 420, 36.0
        x = np.arange(width, dtype=np.float64)[None, :]
        gray = 128.0 + 100.0 * np.sin(2.0 * np.pi * x / period)
        image = np.repeat(np.repeat(gray, height, axis=0)[:, :, None], 3, axis=2)
        result = analyse_guidance_geometry(
            image.astype(np.uint8), roi=(100, 20, 240, 180))

        self.assertTrue(result["fringes"])
        for fringe in result["fringes"]:
            center = fringe["position"]["center_px"]
            self.assertGreaterEqual(center[0], 100)
            self.assertGreaterEqual(center[1], 20)
            self.assertLessEqual(center[1], 200)
        self.assertEqual(
            result["coordinate_system"]["origin"], "full_frame_top_left")

    def test_laser_alignment_uses_a_knob_against_tilt(self):
        geometry = _ready_geometry(angle=12.0)
        geometry["fringes"] = [
            {"kind": "bright", "id": f"bright-{i}"} for i in range(4)]
        result = build_fringe_guidance(
            recognition={"has_fringe": True, "confidence": 0.9},
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.9}, geometry=geometry)

        alignment = result["laser_vertical_alignment"]
        self.assertEqual(alignment["stage"], "straighten")
        self.assertEqual(
            alignment["knob"], "上方旋钮（位于动镜背面左上侧）")
        self.assertEqual(alignment["direction"], "逆时针")
        self.assertTrue(alignment["read_only"])

    def test_laser_alignment_ready_for_stable_vertical_fringes(self):
        geometry = _ready_geometry(angle=1.0)
        geometry["fringes"] = [
            {"kind": "bright", "id": f"bright-{i}"} for i in range(4)]
        result = build_fringe_guidance(
            recognition={"has_fringe": True, "confidence": 0.95},
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.9}, geometry=geometry)
        self.assertTrue(result["laser_vertical_alignment"]["ready"])
        self.assertEqual(result["laser_vertical_alignment"]["stage"], "ready")

    def test_dense_vertical_fringes_select_b_clockwise(self):
        geometry = _ready_geometry(angle=1.0)
        geometry["fringes"] = [
            {"kind": "bright", "id": f"bright-{i}"} for i in range(12)]
        result = build_fringe_guidance(
            recognition={"has_fringe": True, "confidence": 0.95},
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.9}, geometry=geometry)
        alignment = result["laser_vertical_alignment"]
        self.assertEqual(
            alignment["knob"], "下方旋钮（位于动镜背面右下侧）")
        self.assertEqual(alignment["direction"], "顺时针")
        self.assertIn("过密", alignment["observation"])

    def test_sparse_vertical_fringes_select_b_counterclockwise(self):
        geometry = _ready_geometry(angle=1.0)
        geometry["fringes"] = [
            {"kind": "bright", "id": f"bright-{i}"} for i in range(2)]
        result = build_fringe_guidance(
            recognition={"has_fringe": True, "confidence": 0.95},
            motion={"has_fringe": True, "movement": "stable"},
            texture={"sharpness": 0.9}, geometry=geometry)
        alignment = result["laser_vertical_alignment"]
        self.assertEqual(
            alignment["knob"], "下方旋钮（位于动镜背面右下侧）")
        self.assertEqual(alignment["direction"], "逆时针")
        self.assertIn("过疏", alignment["observation"])


if __name__ == "__main__":
    unittest.main()
