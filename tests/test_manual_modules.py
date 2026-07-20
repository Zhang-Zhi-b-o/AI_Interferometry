import unittest

from src.ui.manual_modules import MANUAL_MODULES, PANEL_MODULE


class ManualModuleLayoutTests(unittest.TestCase):
    def test_workspace_has_exactly_four_modules(self):
        self.assertEqual(
            [module.key for module in MANUAL_MODULES],
            ["vision", "motion", "measurement", "assistant"],
        )

    def test_manual_features_are_assigned_once(self):
        keys = [
            panel.key
            for module in MANUAL_MODULES
            for panel in module.panels
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(PANEL_MODULE))
        self.assertIn("auto_control", keys)
        self.assertNotIn("automatic_experiment", keys)


if __name__ == "__main__":
    unittest.main()
