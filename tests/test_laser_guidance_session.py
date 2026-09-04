import unittest

from src.agent.laser_guidance_session import (
    LaserGuidanceConfig,
    LaserGuidanceSession,
)


def _context(*, camera=True, prediction=True, roi=True, fringe=True,
             angle=8.0, count=7, spacing_valid=True, movement="stable"):
    return {
        "camera": {"interferometer_running": camera},
        "vision": {
            "prediction_running": prediction,
            "roi_defined": roi,
            "fringe_present": fringe,
            "fringe_movement": movement,
            "fringe_guidance": {
                "quality_score": 0.8,
                "metrics": {
                    "angle_deg": angle, "spacing_px": 40.0,
                    "spacing_cv_percent": 2.0, "curvature": 0.01,
                    "sharpness": 0.8, "movement": movement,
                },
                "laser_vertical_alignment": {
                    "bright_fringe_count": count,
                    "spacing_valid": spacing_valid,
                    "observation": "测试判断", "action": "测试操作",
                    "expected_change": "测试预期", "stop_condition": "测试停止",
                    "knob": "上方旋钮（位于动镜背面左上侧）",
                    "direction": "逆时针",
                },
            },
        },
    }


class LaserGuidanceSessionTests(unittest.TestCase):
    def test_invalid_evidence_clears_knob_and_direction(self):
        result = LaserGuidanceSession().observe(
            _context(prediction=False, roi=False), now=0)
        self.assertEqual(result["state"], "blocked")
        self.assertIsNone(result["knob"])
        self.assertIsNone(result["direction"])
        self.assertIn("不要转动", result["stop_condition"])

    def test_tilted_fringe_requests_straightening_step(self):
        result = LaserGuidanceSession().observe(_context(angle=8), now=0)
        self.assertEqual(result["step_title"], "调直条纹")
        self.assertEqual(result["state"], "action_required")
        self.assertIn("上方旋钮", result["knob"])

    def test_completion_requires_consecutive_stable_passes(self):
        session = LaserGuidanceSession(LaserGuidanceConfig(
            consecutive_passes=3))
        context = _context(angle=1, count=7, spacing_valid=True)
        self.assertFalse(session.observe(context, now=0)["ready"])
        self.assertFalse(session.observe(context, now=0.5)["ready"])
        self.assertTrue(session.observe(context, now=1)["ready"])

    def test_movement_and_settle_produce_automatic_comparison(self):
        session = LaserGuidanceSession(LaserGuidanceConfig(
            consecutive_passes=5, settle_seconds=1))
        session.observe(_context(angle=8), now=0)
        session.observe(_context(angle=6, movement="right"), now=0.2)
        session.observe(_context(angle=4), now=0.5)
        result = session.observe(_context(angle=4), now=1.6)
        self.assertIsNotNone(result["comparison"])
        self.assertTrue(any(
            item["event"] == "adjustment_comparison"
            for item in session.events()))


if __name__ == "__main__":
    unittest.main()
