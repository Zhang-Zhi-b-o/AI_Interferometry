import unittest

import numpy as np

from src.agent.device_tools import ToolContext, build_tool_registry


class FringeAnalysisToolTests(unittest.TestCase):
    def test_laser_fringe_tool_exposes_per_fringe_details(self):
        height, width = 180, 320
        x = np.arange(width, dtype=np.float64)[None, :]
        red = (128 + 110 * np.sin(2 * np.pi * x / 40)).clip(0, 255)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 2] = np.repeat(red.astype(np.uint8), height, axis=0)
        registry = build_tool_registry(ToolContext(
            latest_frame=lambda: frame,
            get_snapshot=lambda: {"vision": {"fringe_guidance": {
                "laser_vertical_alignment": {"stage": "ready"}}}},
        ))

        result = registry.get("laser_fringe_analyze").fn({})

        self.assertGreater(len(result["fringes"]), 3)
        self.assertIn("color", result["fringes"][0])
        self.assertIn("centerline", result["fringes"][0]["shape"])
        self.assertEqual(result["laser_vertical_alignment"]["stage"], "ready")
        self.assertIn("只读", result["note"])


if __name__ == "__main__":
    unittest.main()
