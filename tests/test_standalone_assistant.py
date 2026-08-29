import unittest
from unittest.mock import patch

from standalone_experiment_assistant.data import STEPS, answer_for
from standalone_experiment_assistant.app import (
    StandaloneExperimentAssistant,
    run_app,
)
from src.ui.app import YoloCamApp


class AssistantMergeTests(unittest.TestCase):
    def test_legacy_class_is_the_main_application(self):
        self.assertIs(StandaloneExperimentAssistant, YoloCamApp)

    def test_legacy_launcher_redirects_to_main_application(self):
        with patch(
                "standalone_experiment_assistant.app._run_main_app") as start:
            run_app()
        start.assert_called_once_with()


class StandaloneAssistantDataTests(unittest.TestCase):
    def test_steps_cover_complete_progress(self):
        self.assertGreaterEqual(len(STEPS), 7)
        self.assertEqual(STEPS[0]["progress"], 5)
        self.assertEqual(STEPS[-1]["progress"], 100)
        self.assertTrue(all(
            STEPS[index]["progress"] < STEPS[index + 1]["progress"]
            for index in range(len(STEPS) - 1)
        ))

    def test_each_step_has_required_display_data(self):
        for step in STEPS:
            self.assertTrue(step["title"])
            self.assertTrue(step["next_action"])
            self.assertTrue(step["criterion"])
            self.assertEqual(len(step["devices"]), 4)

    def test_answers_follow_selected_step(self):
        answer = answer_for("下一步做什么", 4)
        self.assertIn("第 5 步", answer)
        self.assertIn(STEPS[4]["title"], answer)

    def test_uncertainty_answer_uses_local_readings(self):
        answer = answer_for("计算误差和不确定度", len(STEPS) - 1)
        self.assertIn("平均值", answer)
        self.assertIn("A 类", answer)
        self.assertIn("B 类", answer)
