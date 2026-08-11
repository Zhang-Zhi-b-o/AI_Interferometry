import unittest

from src.ui.manual_modules import MANUAL_MODULES, PANEL_MODULE
from src.ui.widgets.plugin_toggles import plugin_definitions


class ManualModuleLayoutTests(unittest.TestCase):
    def test_workspace_has_exactly_five_modules(self):
        self.assertEqual(
            [module.key for module in MANUAL_MODULES],
            ["vision", "motion", "measurement", "assistant", "temporary"],
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
        self.assertIn("temporary_measurement", keys)
        self.assertIn("thickness_measurement", keys)
        self.assertNotIn("automatic_experiment", keys)

    def test_temporary_navigation_can_be_hidden_by_config(self):
        visible = [key for key, _label, _default in plugin_definitions(True)]
        hidden = [key for key, _label, _default in plugin_definitions(False)]
        self.assertEqual(visible[-1], "temporary")
        self.assertNotIn("temporary", hidden)


if __name__ == "__main__":
    unittest.main()
