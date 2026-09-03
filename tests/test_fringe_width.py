"""中心条纹宽度测量（实时画面条纹分析）的单元测试。"""
import unittest

import numpy as np

from src.vision.fringe_width import (
    locate_central_band, measure_center_fringe_width,
    measure_center_fringe_width_2d, measure_fringe_width_by_count,
    measure_fringe_spacing_2d, measure_fringe_spacing_robust)


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
    def test_each_fringe_has_colour_position_and_centerline_shape(self):
        height, width = 220, 360
        intensity = (128 + 110 * np.sin(
            2 * np.pi * np.arange(width)[None, :] / 40)).clip(0, 255)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 2] = np.repeat(intensity.astype(np.uint8), height, axis=0)

        result = measure_center_fringe_width_2d(image)

        self.assertGreaterEqual(result["num_bright"], 3)
        for index, band in enumerate(result["bands"]):
            self.assertEqual(band["index"], index)
            self.assertIn("center_px", band["position"])
            self.assertEqual(
                band["shape"]["representation"], "centerline_polyline")
            self.assertGreater(len(band["shape"]["centerline"]), 2)
            self.assertGreater(band["shape"]["length_px"], height * 0.8)
            self.assertEqual(band["color"]["meaning"], "camera_appearance")
            self.assertGreater(band["color"]["sample_count"], 0)
        bright = next(band for band in result["bands"]
                      if band["kind"] == "bright")
        self.assertEqual(bright["color"]["name_zh"], "红色")
        self.assertGreater(bright["color"]["rgb"][0], 180)

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

    def test_single_visible_laser_fringe_does_not_crash(self):
        width, height = 400, 300
        x = np.arange(width, dtype=np.float64)[None, :]
        gray = 20.0 + 180.0 * np.exp(-0.5 * ((x - 200.0) / 18.0) ** 2)
        gray = np.repeat(gray, height, axis=0)
        image = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)

        result = measure_center_fringe_width_2d(image)

        self.assertGreaterEqual(result["num_bright"], 1)
        self.assertIsNotNone(result["center_band"])


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


def _tilted_grating(width=800, height=400, period=40.0, tilt_deg=30.0):
    """生成倾斜明暗条纹：亮纹沿法向 n=(cosθ, -sinθ) 等间距排列。

    亮纹中心线满足 ``x·cosθ - y·sinθ = m·period``，即 ``x = m·period/cosθ + y·tanθ``，
    相对竖直方向倾角为 ``tilt_deg``（正=``\\``）。
    """
    x = np.arange(width, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None]
    th = np.radians(tilt_deg)
    s = x * np.cos(th) - y * np.sin(th)
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * s / period)
    return np.repeat(val[:, :, None], 3, axis=2).astype(np.uint8)


def _grating_with_missing_fringe(width=800, height=300, period=40.0):
    """在竖向光栅中部抹掉一条亮纹，制造一个约 2 倍周期的异常大间隔。"""
    x = np.arange(width, dtype=np.float64)
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * x / period)
    # 中部某条亮纹（峰在 x=400）两侧半周期内压暗，等效漏掉一条亮纹。
    center = 400.0
    lo = int(center - period * 0.75)
    hi = int(center + period * 0.75)
    val[lo:hi] = 128.0 - 110.0  # 压成暗背景
    gray = np.repeat(val[None, :], height, axis=0)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


class MeasureFringeSpacingRobustTests(unittest.TestCase):
    def test_spacing_matches_period(self):
        result = measure_fringe_spacing_robust(_vertical_grating())
        self.assertIsNotNone(result["spacing_px"])
        self.assertGreaterEqual(result["num_fringes"], 2)
        self.assertAlmostEqual(result["spacing_px"], 40.0, delta=8.0)
        # 均匀光栅的 MAD 应远小于间距本身。
        self.assertLess(result["spacing_mad_px"], result["spacing_px"] * 0.25)
        self.assertEqual(result["rejected_count"], 0)

    def test_gap_list_matches_fringe_count(self):
        result = measure_fringe_spacing_robust(_vertical_grating())
        gaps = result["gap_px"]
        self.assertEqual(len(gaps), result["num_fringes"] - 1)
        self.assertEqual(
            result["valid_count"] + result["rejected_count"], len(gaps))

    def test_first_last_matches_uniform_spacing(self):
        result = measure_fringe_spacing_robust(_vertical_grating())
        self.assertIsNotNone(result["spacing_first_last_px"])
        self.assertAlmostEqual(
            result["spacing_first_last_px"], result["spacing_px"], delta=6.0)

    def test_rejects_outlier_gap(self):
        # 抹掉一条亮纹会引入约 2 倍间距的大间隔，中位数应不受影响且被剔除。
        result = measure_fringe_spacing_robust(_grating_with_missing_fringe())
        self.assertIsNotNone(result["spacing_px"])
        self.assertAlmostEqual(result["spacing_px"], 40.0, delta=8.0)
        self.assertGreaterEqual(result["rejected_count"], 1)
        # 最大间隔明显大于中位数，属于被 MAD 剔除的漏纹异常。
        self.assertGreater(max(result["gap_px"]), result["spacing_px"] * 1.5)

    def test_mm_conversion(self):
        result = measure_fringe_spacing_robust(
            _vertical_grating(), mm_per_px=0.01)
        self.assertIsNotNone(result["spacing_mm"])
        self.assertAlmostEqual(
            result["spacing_mm"], result["spacing_px"] * 0.01, places=4)

    def test_count_mm_conversion(self):
        result = measure_fringe_width_by_count(
            _vertical_grating(), mm_per_px=0.01)
        self.assertIsNotNone(result["fringe_width_mm"])
        self.assertAlmostEqual(
            result["fringe_width_mm"], result["fringe_width"] * 0.01, places=4)

    def test_no_fringe_returns_none(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        result = measure_fringe_spacing_robust(uniform)
        self.assertIsNone(result["spacing_px"])
        self.assertEqual(result["num_fringes"], 0)

    def test_invalid_fringe_kind_raises(self):
        with self.assertRaises(ValueError):
            measure_fringe_spacing_robust(_vertical_grating(), fringe="bad")


class MeasureFringeSpacing2dTests(unittest.TestCase):
    def test_vertical_spacing_matches_period(self):
        result = measure_fringe_spacing_2d(_vertical_grating())
        self.assertIsNotNone(result["spacing_px"])
        self.assertGreaterEqual(result["num_fringes"], 4)
        self.assertAlmostEqual(result["spacing_px"], 40.0, delta=8.0)
        # 均匀光栅各估计值应相互接近，主值定义即「首末/间隔」。
        self.assertAlmostEqual(
            result["spacing_first_last_px"], result["spacing_px"], delta=6.0)
        self.assertLess(result["cv_percent"], 10.0)
        self.assertEqual(result["num_rejected"], 0)

    def test_tilted_recovers_normal_spacing(self):
        # 条纹固有法向间距为 40，倾斜 30° 后水平间距被放大到 40/cos(30°)≈46.2。
        # 法向投影应还原真实间距 40；而基于列平均亮度的「视场÷条纹数」对倾斜
        # 条纹会失效（列平均把倾斜条纹抹平，count_estimate_px 为 None），正好
        # 印证主算法采用二维中心线 + 法向投影的必要性。
        result = measure_fringe_spacing_2d(_tilted_grating(tilt_deg=30.0))
        self.assertIsNotNone(result["spacing_px"])
        self.assertAlmostEqual(result["spacing_px"], 40.0, delta=4.0)
        self.assertAlmostEqual(result["angle_deg"], 30.0, delta=8.0)

    def test_near_horizontal_recovers_normal_spacing(self):
        for tilt in (-88.0, -75.0, 75.0, 88.0):
            result = measure_fringe_spacing_2d(
                _tilted_grating(tilt_deg=tilt))
            self.assertIsNotNone(result["spacing_px"])
            self.assertAlmostEqual(result["spacing_px"], 40.0, delta=4.0)
            self.assertAlmostEqual(result["angle_deg"], tilt, delta=3.0)
            self.assertTrue(result["quality_valid"])

    def test_rejects_outlier_gap(self):
        result = measure_fringe_spacing_2d(_grating_with_missing_fringe())
        self.assertIsNotNone(result["spacing_px"])
        self.assertAlmostEqual(result["spacing_px"], 40.0, delta=8.0)
        self.assertGreaterEqual(result["num_rejected"], 1)
        self.assertGreater(len(result["rejected_intervals"]), 0)
        # 被剔除的最大间隔应明显大于中位数（漏纹异常）。
        self.assertGreater(
            max(result["rejected_intervals"]), result["spacing_px"] * 1.5)

    def test_interval_bookkeeping(self):
        result = measure_fringe_spacing_2d(_vertical_grating())
        n = result["num_fringes"]
        self.assertEqual(result["num_intervals"], n - 1)
        self.assertEqual(
            result["num_valid_intervals"] + result["num_rejected"],
            result["num_intervals"])
        self.assertEqual(len(result["individual_spacings_px"]), n - 1)
        self.assertEqual(len(result["interval_valid"]), n - 1)
        self.assertEqual(len(result["fringe_centers"]), n)

    def test_mm_conversion(self):
        result = measure_fringe_spacing_2d(
            _vertical_grating(), pixel_scale_mm=0.01)
        self.assertIsNotNone(result["spacing_mm"])
        self.assertAlmostEqual(
            result["spacing_mm"], result["spacing_px"] * 0.01, places=4)
        self.assertAlmostEqual(result["pixel_scale_mm"], 0.01, places=6)

    def test_roi_crop_offsets_centers(self):
        # 在左侧裁剪一块 ROI，返回的中心坐标应加回 ROI 左上角偏移。
        result = measure_fringe_spacing_2d(
            _vertical_grating(), roi=(50, 0, 300, 300))
        self.assertIsNotNone(result["spacing_px"])
        for c in result["fringe_centers"]:
            self.assertGreaterEqual(c["x"], 50.0)
            self.assertLessEqual(c["x"], 350.0)

    def test_clean_grating_high_confidence(self):
        result = measure_fringe_spacing_2d(_vertical_grating())
        self.assertTrue(result["quality_valid"])
        self.assertGreaterEqual(result["confidence"], 0.8)

    def test_no_fringe_returns_none(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        result = measure_fringe_spacing_2d(uniform)
        self.assertIsNone(result["spacing_px"])
        self.assertEqual(result["num_fringes"], 0)
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["quality_valid"])

    def test_invalid_fringe_kind_raises(self):
        with self.assertRaises(ValueError):
            measure_fringe_spacing_2d(_vertical_grating(), fringe="bad")

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            measure_fringe_spacing_2d(np.array([]))


if __name__ == "__main__":
    unittest.main()
