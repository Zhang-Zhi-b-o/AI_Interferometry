import unittest

from src.ui.widgets.floating_assistant import (
    FloatingGeometry,
    clamp_floating_geometry,
)


class FloatingAssistantGeometryTests(unittest.TestCase):
    def test_geometry_is_kept_inside_page(self):
        geometry = clamp_floating_geometry(
            FloatingGeometry(900, 700, 480, 600), 1200, 800)
        self.assertEqual(geometry, FloatingGeometry(720, 200, 480, 600))

    def test_size_is_clamped_to_minimum_and_available_page(self):
        small = clamp_floating_geometry(
            FloatingGeometry(-20, -10, 100, 120), 1000, 700)
        self.assertEqual(small, FloatingGeometry(0, 0, 360, 300))
        oversized = clamp_floating_geometry(
            FloatingGeometry(50, 50, 1600, 900), 1000, 700)
        self.assertEqual(oversized, FloatingGeometry(0, 0, 1000, 700))


if __name__ == "__main__":
    unittest.main()
