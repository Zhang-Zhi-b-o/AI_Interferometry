import tempfile
import unittest
from pathlib import Path

from src.control.adaptive_response import AdaptiveResponseLearner


class AdaptiveResponseLearnerTests(unittest.TestCase):
    def test_learns_clear_motion_speed_but_ignores_blur(self):
        learner = AdaptiveResponseLearner()
        learner.observe(
            now=1.0, direction="forward", gear=9, velocity_px_s=50,
            stable=False)
        learner.observe(
            now=1.1, direction="forward", gear=9, velocity_px_s=500,
            stable=False, blurred=True)

        snapshot = learner.snapshot()
        self.assertEqual(snapshot["response_samples"], 1)
        self.assertAlmostEqual(snapshot["gear_speed_px_s"]["9"], 50.0)

    def test_learns_settle_time_after_stop(self):
        learner = AdaptiveResponseLearner()
        learner.observe(
            now=1.0, direction="forward", gear=10, velocity_px_s=20,
            stable=False)
        learner.observe(
            now=2.0, direction="stopped", gear=None, velocity_px_s=5,
            stable=False)
        learner.observe(
            now=2.4, direction="stopped", gear=None, velocity_px_s=0,
            stable=True)

        self.assertEqual(learner.settle_samples, 1)
        self.assertAlmostEqual(learner.settle_seconds, 0.4)

    def test_optimized_params_are_bounded(self):
        learner = AdaptiveResponseLearner(settle_samples=5, settle_seconds=1.5)
        for index in range(8):
            learner.observe(
                now=float(index), direction="forward", gear=10,
                velocity_px_s=900, stable=False)
        base = {
            "slow_gear": 10, "slow_zone_px": 100,
            "tolerance_px": 10, "stop_detect_settle_seconds": 0.3,
        }

        params, changes = learner.optimized_params(base, spacing_px=40)

        self.assertLessEqual(params["stop_detect_settle_seconds"], 0.54)
        self.assertLessEqual(params["slow_zone_px"], 180)
        self.assertEqual(base["slow_zone_px"], 100)
        self.assertIn("slow_zone_px", changes)

    def test_snapshot_can_be_persisted(self):
        learner = AdaptiveResponseLearner()
        learner.observe(
            now=1.0, direction="reverse", gear=8, velocity_px_s=-42,
            stable=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "response.json"
            learner.save(path)
            restored = AdaptiveResponseLearner.load(path)

        self.assertEqual(restored.response_samples, 1)
        self.assertAlmostEqual(
            restored.snapshot()["gear_speed_px_s"]["8"], 42.0)

    def test_camera_profile_change_discards_incomparable_samples(self):
        learner = AdaptiveResponseLearner()
        learner.observe(
            now=1.0, direction="forward", gear=9, velocity_px_s=50,
            stable=False, profile_key="width=640;zoom=2.00")
        learner.observe(
            now=2.0, direction="forward", gear=9, velocity_px_s=20,
            stable=False, profile_key="width=1280;zoom=1.00")

        snapshot = learner.snapshot()
        self.assertEqual(snapshot["response_samples"], 1)
        self.assertEqual(snapshot["profile_key"], "width=1280;zoom=1.00")


if __name__ == "__main__":
    unittest.main()
