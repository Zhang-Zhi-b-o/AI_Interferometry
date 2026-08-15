"""中心条纹宽度测量（实时画面条纹分析）的单元测试。"""
import unittest

import numpy as np

from src.vision.fringe_width import (
    locate_central_band, measure_center_fringe_width,
    measure_center_fringe_width_2d, measure_fringe_width_by_count)


def _vertical_grating(width=400, height=300, period=40.0, phase=0.0):
    """生成竖向明暗条纹灰度图（BGR 三通道同值）。"""
    x = np.arange(width, dtype=np.float64)
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * (x / period) - phase)
    gray = np.repeat(val[None, :], height, axis=0)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


class MeasureCenterFringeWidthTests(unittest.TestCase):
    def test_finds_bands_on_vertical_grating(self):
        result = measure_center_fringe_width(_vertical_grating())
        self.assertGreaterEqual(result["num_bands"], 2)
        self.assertIsNotNone(result["center_band"])
        band = result["center_band"]
        self.assertGreater(band["width"], 0.0)
        self.assertIn(band["kind"], ("bright", "dark"))
        # 每段条纹宽度约等于半周期
        self.assertAlmostEqual(band["width"], result["period_px"] / 2.0, delta=8.0)

    def test_center_x_picks_dark_valley(self):
        # phase=0 时 sin(2πx/P) 在 x=0.75P+kP 取到暗谷（画面中部取 190px）
        result = measure_center_fringe_width(
            _vertical_grating(), center_x=190.0)
        band = result["center_band"]
        self.assertIsNotNone(band)
        self.assertEqual(band["kind"], "dark")
        self.assertLess(abs(band["center_x"] - 190.0), 6.0)

    def test_center_x_picks_bright_peak(self):
        # phase=0 时亮峰在 x=0.25P+kP（画面中部取 210px）
        result = measure_center_fringe_width(
            _vertical_grating(), center_x=210.0)
        band = result["center_band"]
        self.assertIsNotNone(band)
        self.assertEqual(band["kind"], "bright")
        self.assertLess(abs(band["center_x"] - 210.0), 6.0)

    def test_uniform_image_has_no_bands(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        result = measure_center_fringe_width(uniform)
        self.assertEqual(result["num_bands"], 0)
        self.assertIsNone(result["center_band"])

    def test_grayscale_input(self):
        gray = _vertical_grating()[:, :, 0]
        result = measure_center_fringe_width(gray)
        self.assertGreaterEqual(result["num_bands"], 2)
        self.assertIsNotNone(result["center_band"])

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            measure_center_fringe_width(np.array([]))
        with self.assertRaises(ValueError):
            measure_center_fringe_width(None)

    def test_default_reference_is_frame_center(self):
        result = measure_center_fringe_width(_vertical_grating())
        self.assertAlmostEqual(result["reference_x"], 200.0, places=1)

    def test_returns_all_bands_with_boundaries(self):
        result = measure_center_fringe_width(_vertical_grating())
        bands = result["bands"]
        self.assertEqual(len(bands), result["num_bands"])
        self.assertGreaterEqual(len(bands), 2)
        for b in bands:
            self.assertIn(b["kind"], ("bright", "dark"))
            self.assertGreaterEqual(b["right"], b["left"])
            self.assertAlmostEqual(b["width"], b["right"] - b["left"], places=4)
        # 中心条纹应是 bands 列表中的同一段
        center = result["center_band"]
        self.assertIsNotNone(center)
        self.assertIn(center, bands)


class LocateCentralBandTests(unittest.TestCase):
    def test_finds_dark_valley_near_reference(self):
        result = locate_central_band(
            _vertical_grating(), center_x=190.0, kind="dark")
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "dark")
        self.assertLess(abs(result["center_x"] - 190.0), 8.0)
        self.assertGreater(result["confidence"], 0.0)

    def test_finds_bright_peak_near_reference(self):
        result = locate_central_band(
            _vertical_grating(), center_x=210.0, kind="bright")
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "bright")
        self.assertLess(abs(result["center_x"] - 210.0), 8.0)

    def test_any_kind_returns_band_within_bounds(self):
        result = locate_central_band(
            _vertical_grating(), center_x=190.0, search_bounds=(180.0, 200.0))
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["center_x"], 180.0)
        self.assertLessEqual(result["center_x"], 200.0)

    def test_search_bounds_exclude_outside_band(self):
        # 把边界限定在远离 190 的区域，候选应被过滤到边界内。
        result = locate_central_band(
            _vertical_grating(), center_x=190.0, search_bounds=(40.0, 60.0))
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["center_x"], 40.0)
        self.assertLessEqual(result["center_x"], 60.0)

    def test_uniform_image_returns_none(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        self.assertIsNone(locate_central_band(uniform))

    def test_invalid_input_returns_none(self):
        self.assertIsNone(locate_central_band(np.array([])))
        self.assertIsNone(locate_central_band(None))


def _curved_grating(width=400, height=300, period=40.0, curve=40.0):
    """生成带弯曲的竖向明暗条纹（相位随 y 变化）。"""
    x = np.arange(width, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None]
    shift = curve * np.sin(np.pi * y / height)  # 中段向一侧偏移，形成弯曲
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * (x - shift) / period)
    return np.repeat(val[:, :, None], 3, axis=2).astype(np.uint8)


class Measure2dTests(unittest.TestCase):
    def test_vertical_grating_bands_have_centerline(self):
        result = measure_center_fringe_width_2d(_vertical_grating())
        self.assertGreaterEqual(result["num_bands"], 2)
        self.assertIsNotNone(result["center_band"])
        for b in result["bands"]:
            self.assertGreaterEqual(len(b["centerline"]), 2)
            self.assertGreater(b["width"], 0.0)

    def test_center_band_is_dark_valley(self):
        result = measure_center_fringe_width_2d(
            _vertical_grating(), center_x=190.0)
        band = result["center_band"]
        self.assertIsNotNone(band)
        self.assertEqual(band["kind"], "dark")
        self.assertLess(abs(band["center_x"] - 190.0), 8.0)

    def test_curved_fringe_produces_nonvertical_contour(self):
        # 弯曲条纹的中心线应随 y 变化 x，而不是竖直直线。
        result = measure_center_fringe_width_2d(_curved_grating())
        self.assertGreaterEqual(result["num_bands"], 2)
        curved = False
        for b in result["bands"]:
            xs = [p[0] for p in b["centerline"]]
            if max(xs) - min(xs) > 2.0:
                curved = True
                break
        self.assertTrue(curved, "弯曲条纹应产生随高度变化的轮廓")

    def test_edges_exclude_black_background(self):
        # 左右纯黑背景不应被切成条纹：最左/最右一段的边界应明显内缩。
        img = _vertical_grating()
        img[:, :40] = 0
        img[:, -40:] = 0
        result = measure_center_fringe_width_2d(img)
        self.assertGreaterEqual(result["num_bands"], 2)
        first = result["bands"][0]
        last = result["bands"][-1]
        self.assertGreater(first["left"], 20.0)
        self.assertLess(last["right"], 380.0)


class MeasureFringeWidthByCountTests(unittest.TestCase):
    def test_width_equals_span_over_count(self):
        result = measure_fringe_width_by_count(_vertical_grating())
        self.assertGreater(result["fringe_count"], 0)
        self.assertIsNotNone(result["fringe_width"])
        # 定义即「视场宽度 / 条纹数量」
        self.assertAlmostEqual(
            result["fringe_width"],
            result["span_px"] / result["fringe_count"], places=3)
        # 对等间距竖向光栅，间隔应接近自相关周期（亮纹到亮纹 ≈ 一个周期）
        self.assertAlmostEqual(
            result["fringe_width"], result["period_px"], delta=8.0)

    def test_explicit_range_restricts_count(self):
        result = measure_fringe_width_by_count(
            _vertical_grating(), x_range=(0, 100))
        self.assertGreaterEqual(result["region"][0], 0.0)
        self.assertLessEqual(result["region"][1], 100.0)
        # 区间内至少能数到一条亮纹
        self.assertGreaterEqual(result["fringe_count"], 1)

    def test_no_fringe_returns_none_width(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        result = measure_fringe_width_by_count(uniform)
        self.assertEqual(result["fringe_count"], 0)
        self.assertIsNone(result["fringe_width"])

    def test_invalid_fringe_kind_raises(self):
        with self.assertRaises(ValueError):
            measure_fringe_width_by_count(_vertical_grating(), fringe="bad")


if __name__ == "__main__":
    unittest.main()
