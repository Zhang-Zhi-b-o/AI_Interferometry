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


if __name__ == "__main__":
    unittest.main()
