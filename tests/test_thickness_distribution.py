"""单帧薄膜厚度分布（彩色条纹相位解包）的单元测试。"""
import unittest

import cv2
import numpy as np

from src.vision.thickness_distribution import (
    analyze_thickness_distribution,
    sample_colour,
    sample_colour_band,
)


def _rainbow(width=320, height=240, cycles=6):
    """构造多周期彩色渐变图，模拟白光干涉的彩色条纹。"""
    x = np.arange(width, dtype=np.float64)
    hue = ((x / width) * 179.0 * cycles) % 180.0
    h = np.tile(hue.astype(np.uint8)[None, :], (height, 1))
    s = np.full((height, width), 180, np.uint8)
    v = np.full((height, width), 200, np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class AnalyzeThicknessDistributionTests(unittest.TestCase):
    def test_returns_expected_keys_and_shapes(self):
        img = _rainbow()
        result = analyze_thickness_distribution(img)
        h, w = img.shape[:2]
        self.assertEqual(result["thickness"].shape, (h, w))
        self.assertEqual(result["confidence"].shape, (h, w))
        self.assertEqual(result["mask"].shape, (h, w))
        self.assertEqual(result["overlay"].shape, img.shape)
        self.assertEqual(result["heatmap"].shape, (h, w, 3))
        self.assertIn("metrics", result)

    def test_recovers_non_trivial_distribution(self):
        img = _rainbow()
        result = analyze_thickness_distribution(img)
        metrics = result["metrics"]
        self.assertGreater(metrics["valid_pixels"], 100)
        # 多周期色相渐变应产生可测的厚度起伏。
        self.assertGreater(metrics["pv_robust_um"], 0.02)

    def test_relative_step_size_matches_wavelength_formula(self):
        img = _rainbow()
        wavelength = 589.3
        refractive = 1.523
        result = analyze_thickness_distribution(
            img, wavelength_nm=wavelength, refractive_index=refractive)
        expected_step = wavelength / (2.0 * (refractive - 1.0)) / 1000.0
        self.assertAlmostEqual(result["step_um"], expected_step, places=9)
        self.assertEqual(result["mode"], "relative")

    def test_invert_flips_thickness_direction(self):
        img = _rainbow()
        base = analyze_thickness_distribution(img)
        flipped = analyze_thickness_distribution(img, invert=True)
        # 中位数为 0 的厚度图，反转后符号相反。
        base_vals = base["thickness"][base["mask"]]
        flip_vals = flipped["thickness"][flipped["mask"]]
        self.assertTrue(np.allclose(base_vals, -flip_vals, atol=1e-6))

    def test_refractive_index_must_exceed_one(self):
        with self.assertRaisesRegex(ValueError, "refractive_index"):
            analyze_thickness_distribution(_rainbow(), refractive_index=1.0)

    def test_empty_image_raises(self):
        with self.assertRaises(ValueError):
            analyze_thickness_distribution(np.array([]))


class SampleColourTests(unittest.TestCase):
    def test_returns_rgb_triplet_in_range(self):
        r, g, b = sample_colour(_rainbow())
        self.assertTrue(all(isinstance(v, int) for v in (r, g, b)))
        self.assertTrue(all(0 <= v <= 255 for v in (r, g, b)))


class SampleColourBandTests(unittest.TestCase):
    def test_returns_rgb_triplet_in_range(self):
        r, g, b = sample_colour_band(_rainbow(), 160)
        self.assertTrue(all(isinstance(v, int) for v in (r, g, b)))
        self.assertTrue(all(0 <= v <= 255 for v in (r, g, b)))

    def test_solid_colour_matches(self):
        img = np.zeros((40, 60, 3), np.uint8)
        img[:] = (200, 120, 40)  # BGR → 返回 (R, G, B) = (40, 120, 200)
        r, g, b = sample_colour_band(img, 30)
        self.assertEqual((r, g, b), (40, 120, 200))

    def test_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            sample_colour_band(np.zeros((10, 10, 3), np.uint8), 100)

    def test_excludes_dark_background(self):
        # 中间一条亮色带、上下为黑色背景；筛选后应只取亮带颜色而非背景黑。
        img = np.zeros((80, 100, 3), np.uint8)
        img[25:55, :, :] = (200, 120, 40)  # BGR → (R, G, B) = (40, 120, 200)
        r, g, b = sample_colour_band(img, 50)
        self.assertEqual((r, g, b), (40, 120, 200))


class ReferenceSubtractionTests(unittest.TestCase):
    def test_marks_has_reference(self):
        img = _rainbow()
        result = analyze_thickness_distribution(img, reference_image=_rainbow())
        self.assertTrue(result["metrics"]["has_reference"])

    def test_same_reference_yields_flat_map(self):
        img = _rainbow()
        result = analyze_thickness_distribution(img, reference_image=img)
        self.assertAlmostEqual(result["metrics"]["pv_robust_um"], 0.0, places=6)

    def test_reference_of_different_size_raises(self):
        img = _rainbow(320, 240)
        with self.assertRaisesRegex(ValueError, "基准图尺寸"):
            analyze_thickness_distribution(
                img, reference_image=_rainbow(160, 120))


if __name__ == "__main__":
    unittest.main()
