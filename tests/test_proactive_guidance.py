import unittest

from src.agent.experiment_guidance import (
    ExperimentIntent,
    build_guidance_decision,
)
from src.agent.proactive import ProactiveCoordinator
from src.vision.fringe_adjustment import compare_fringe_adjustment


def _context(**overrides):
    context = {
        "experiment_intent": {
            "kind": "fringe_observation",
            "objective": "调出清晰稳定条纹",
            "response_mode": "standard",
            "confirmed": True,
        },
        "experiment_progress": {
            "stage": "条纹调节", "next_action": "改善条纹",
            "completion_criterion": "质量门通过",
        },
        "camera": {"interferometer_running": True, "fps": 30.0},
        "vision": {
            "prediction_running": True, "fringe_present": True,
            "fringe_guidance": {
                "measurement_ready": True, "summary": "条纹稳定",
                "metrics": {"angle_deg": 1.0, "spacing_px": 40.0,
                            "spacing_cv_percent": 3.0, "movement": "stable"},
            },
        },
        "motor": {"connected": True, "auto_enabled": False},
        "micrometer": {"connected": True, "reading_age_seconds": 1.0},
        "measurement": {},
    }
    for key, value in overrides.items():
        context[key] = value
    return context


class ExperimentGuidanceTests(unittest.TestCase):
    def test_intent_is_validated_and_bounded(self):
        intent = ExperimentIntent.from_mapping({
            "kind": "bad", "required_repeats": 1000, "response_mode": "bad"})
        self.assertEqual(intent.kind, "white_light_centering")
        self.assertEqual(intent.required_repeats, 100)
        self.assertEqual(intent.response_mode, "standard")

    def test_stale_reading_blocks_recording(self):
        context = _context(micrometer={
            "connected": True, "reading_age_seconds": 8.0})
        decision = build_guidance_decision(context)
        self.assertFalse(decision.can_record)
        self.assertEqual(decision.priority, "blocking")
        self.assertEqual(decision.issues[0].code, "STALE_MICROMETER")

    def test_malformed_optional_metrics_do_not_crash_guidance(self):
        context = _context(
            camera={"interferometer_running": True, "fps": "bad"},
            micrometer={"connected": True, "reading_age_seconds": "bad"},
            measurement={"record_count": "bad"},
        )
        decision = build_guidance_decision(context)
        self.assertIn("LOW_CAMERA_RATE", {issue.code for issue in decision.issues})

    def test_ready_state_allows_recording(self):
        decision = build_guidance_decision(_context())
        self.assertTrue(decision.can_record)
        self.assertEqual(decision.priority, "ready")
        self.assertIn("保存当前原始画面", decision.action)

    def test_repeat_shortfall_is_explained_without_blocking_recording(self):
        context = _context()
        context["measurement"] = {"record_count": 2}
        decision = build_guidance_decision(context)
        self.assertTrue(decision.can_record)
        self.assertIn("INSUFFICIENT_REPEATS", {
            issue.code for issue in decision.issues})


class ProactiveCoordinatorTests(unittest.TestCase):
    def test_same_semantic_state_does_not_repeat_local_update(self):
        coordinator = ProactiveCoordinator()
        first = coordinator.observe(_context(), now=10.0)
        second = coordinator.observe(_context(), now=11.0)
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)

    def test_multiple_issues_request_llm_once_with_budget(self):
        bad = _context(
            camera={"interferometer_running": True, "fps": 2.0},
            vision={"prediction_running": True, "fringe_present": False},
        )
        coordinator = ProactiveCoordinator(min_llm_interval=60)
        update = coordinator.observe(bad, now=100.0)
        self.assertTrue(update.llm_reason)
        self.assertTrue(coordinator.reserve_llm(update.request_key, now=100.0))
        self.assertFalse(coordinator.reserve_llm(update.request_key, now=101.0))

    def test_quiet_mode_never_requests_background_llm(self):
        bad = _context(
            experiment_intent={"response_mode": "quiet"},
            camera={"interferometer_running": True, "fps": 2.0},
            vision={"prediction_running": True, "fringe_present": False},
        )
        update = ProactiveCoordinator().observe(bad, now=10.0)
        self.assertEqual(update.llm_reason, "")
        self.assertIsNone(update.request_key)

    def test_intent_change_requests_one_model_explanation(self):
        coordinator = ProactiveCoordinator()
        coordinator.observe(_context(), now=10.0)
        changed = _context(experiment_intent={
            "kind": "fringe_spacing", "response_mode": "standard"})
        update = coordinator.observe(changed, now=20.0)
        self.assertIn("实验目的", update.llm_reason)

    def test_stalled_stage_requests_model_only_once(self):
        coordinator = ProactiveCoordinator(stalled_stage_seconds=30)
        coordinator.observe(_context(), now=10.0)
        first = coordinator.observe(_context(), now=41.0)
        second = coordinator.observe(_context(), now=80.0)
        self.assertIn("长时间", first.llm_reason)
        self.assertEqual(second.llm_reason, "")

    def test_session_budget_caps_background_calls(self):
        coordinator = ProactiveCoordinator(
            min_llm_interval=1, repeat_suppression=1,
            max_calls_per_window=5, max_calls_per_session=2)
        self.assertTrue(coordinator.reserve_llm(("a",), now=10.0))
        self.assertTrue(coordinator.reserve_llm(("b",), now=12.0))
        self.assertFalse(coordinator.reserve_llm(("c",), now=14.0))


class FringeAdjustmentTests(unittest.TestCase):
    def test_effective_adjustment_is_reported_without_guessing_knob_direction(self):
        result = compare_fringe_adjustment(
            {"angle_deg": 8.0, "spacing_cv_percent": 12.0,
             "quality_score": 0.45},
            {"angle_deg": 3.0, "spacing_cv_percent": 6.0,
             "quality_score": 0.70},
        )
        self.assertEqual(result["outcome"], "improved")
        self.assertIn("相同方向", result["recommendation"])

    def test_missing_baseline_does_not_invent_result(self):
        result = compare_fringe_adjustment({}, {"angle_deg": 2.0})
        self.assertEqual(result["outcome"], "insufficient")

    def test_moving_fringe_must_settle_before_comparison(self):
        result = compare_fringe_adjustment(
            {"angle_deg": 8.0, "movement": "stable"},
            {"angle_deg": 3.0, "movement": "right"},
        )
        self.assertEqual(result["outcome"], "insufficient")
        self.assertIn("等待条纹稳定", result["recommendation"])


if __name__ == "__main__":
    unittest.main()
