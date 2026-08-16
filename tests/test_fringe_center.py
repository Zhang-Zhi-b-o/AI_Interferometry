"""中心条纹识别（精修 find_center_in_region 与稳健备份 find_center_by_band）的单元测试。"""
import unittest

import numpy as np

from src.vision.fringe_center import find_center_by_band


def _vertical_grating(width=400, height=300, period=40.0, phase=0.0):
    """生成竖向明暗条纹灰度图（BGR 三通道同值）。

    phase=0 时亮峰在 x=0.25P+kP（即 10、50、90...），暗谷在
    x=0.75P+kP（即 30、70、110...）。
    """
    x = np.arange(width, dtype=np.float64)
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * (x / period) - phase)
    gray = np.repeat(val[None, :], height, axis=0)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


class FindCenterByBandTests(unittest.TestCase):
    def test_locate_dark_valley_near_reference(self):
        info = find_center_by_band(_vertical_grating(), expected_center_x=30.0)
        self.assertIsNotNone(info)
        self.assertLess(abs(info["center_x"] - 30.0), 8.0)
        self.assertEqual(info["orientation"], "vertical")

    def test_confidence_within_unit_range(self):
        info = find_center_by_band(_vertical_grating(), expected_center_x=30.0)
        self.assertGreaterEqual(info["confidence"], 0.0)
        self.assertLessEqual(info["confidence"], 1.0)

    def test_returns_region_shape(self):
        info = find_center_by_band(_vertical_grating(), expected_center_x=50.0)
        self.assertEqual(info["roi_width"], 400)
        self.assertEqual(info["roi_height"], 300)
        self.assertGreater(info["period"], 0.0)

    def test_search_bounds_restrict_center(self):
        # 把边界限定在 10~30，最近的暗谷应落在该区间内。
        info = find_center_by_band(
            _vertical_grating(), expected_center_x=200.0,
            search_bounds=(10.0, 30.0))
        self.assertGreaterEqual(info["center_x"], 10.0)
        self.assertLessEqual(info["center_x"], 30.0)

    def test_uniform_image_raises(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        with self.assertRaises(ValueError):
            find_center_by_band(uniform)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            find_center_by_band(np.array([]))


if __name__ == "__main__":
    unittest.main()
