"""条纹角度鲁棒估计（自动画面倾斜校正）的单元测试。"""
import unittest

import numpy as np

from src.vision.angle import rotate_expand
from src.vision.fringe_angle import estimate_fringe_angle_2d


def _tilted_grating(angle_deg=0.0, width=640, height=420, period=40.0):
    """生成倾斜 angle_deg 度的竖向明暗条纹（相位随 y 平移 tan(θ)·y）。"""
    x = np.arange(width, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None]
    shift = np.tan(np.radians(angle_deg)) * y
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * (x - shift) / period)
    return np.repeat(val[:, :, None], 3, axis=2).astype(np.uint8)


def _curved_grating(width=640, height=420, period=40.0, curve=60.0):
    """生成整体竖直、但中段左右弯曲的条纹。"""
    x = np.arange(width, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None]
    shift = curve * np.sin(np.pi * y / height)
    val = 128.0 + 110.0 * np.sin(2.0 * np.pi * (x - shift) / period)
    return np.repeat(val[:, :, None], 3, axis=2).astype(np.uint8)


def _laser_grating_with_guide(
    angle_deg=78.0, width=720, height=480, period=52.0,
):
    """低对比度红色激光条纹、圆形光斑、散斑和青色中心辅助线。"""
    rng = np.random.default_rng(20260903)
    x = np.arange(width, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None]
    th = np.radians(angle_deg)
    phase = x * np.cos(th) - y * np.sin(th)
    radius = np.hypot(x - width / 2.0, y - height / 2.0)
    envelope = np.clip(1.0 - radius / (0.52 * min(width, height)), 0.0, 1.0)
    red = 45.0 + envelope * (
        75.0 + 38.0 * np.sin(2.0 * np.pi * phase / period))
    red += rng.normal(0.0, 5.0, size=red.shape)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 2] = np.clip(red, 0, 255).astype(np.uint8)
    image[:, width // 2 - 1:width // 2 + 2] = (255, 255, 0)
    return image


class EstimateFringeAngleTests(unittest.TestCase):
    def test_vertical_grating_is_near_zero(self):
        result = estimate_fringe_angle_2d(_tilted_grating(0.0))
        self.assertIsNotNone(result["tilt_deg"])
        self.assertLess(abs(result["tilt_deg"]), 1.0)
        self.assertGreater(result["confidence"], 0.5)
        self.assertGreaterEqual(result["num_lines"], 2)

    def test_recovers_known_tilt(self):
        for angle in (10.0, 20.0, 30.0, 40.0):
            result = estimate_fringe_angle_2d(_tilted_grating(angle))
            self.assertIsNotNone(result["tilt_deg"])
            self.assertAlmostEqual(result["tilt_deg"], angle, delta=1.5)

    def test_correction_straightens(self):
        for angle in (10.0, 30.0, 40.0):
            img = _tilted_grating(angle)
            est = estimate_fringe_angle_2d(img)
            corrected = rotate_expand(img, est["correction_deg"])
            after = estimate_fringe_angle_2d(corrected)
            self.assertIsNotNone(after["tilt_deg"])
            self.assertLess(abs(after["tilt_deg"]), 1.5)

    def test_correction_sign_opposes_tilt(self):
        # 倾斜角与校正角应反向，否则旋转方向错误。
        result = estimate_fringe_angle_2d(_tilted_grating(25.0))
        self.assertGreater(result["tilt_deg"], 20.0)
        self.assertLess(result["correction_deg"], -20.0)

    def test_curved_grating_has_high_curvature(self):
        result = estimate_fringe_angle_2d(_curved_grating())
        # 整体竖直，平均倾角应接近 0；弯曲度应明显大于竖直直条纹。
        self.assertIsNotNone(result["tilt_deg"])
        self.assertLess(abs(result["tilt_deg"]), 2.0)
        self.assertIsNotNone(result["curvature"])
        straight = estimate_fringe_angle_2d(_tilted_grating(0.0))
        self.assertGreater(result["curvature"], straight["curvature"])

    def test_uniform_image_has_no_angle(self):
        uniform = np.full((300, 400, 3), 128, dtype=np.uint8)
        result = estimate_fringe_angle_2d(uniform)
        self.assertIsNone(result["tilt_deg"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["num_lines"], 0)

    def test_grayscale_input(self):
        gray = _tilted_grating(15.0)[:, :, 0]
        result = estimate_fringe_angle_2d(gray)
        self.assertIsNotNone(result["tilt_deg"])
        self.assertAlmostEqual(result["tilt_deg"], 15.0, delta=1.5)

    def test_near_horizontal_laser_fringes_with_cyan_guide(self):
        result = estimate_fringe_angle_2d(_laser_grating_with_guide())
        self.assertIsNotNone(result["tilt_deg"])
        self.assertAlmostEqual(result["tilt_deg"], 78.0, delta=3.0)
        self.assertGreater(result["confidence"], 0.35)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            estimate_fringe_angle_2d(np.array([]))
        with self.assertRaises(ValueError):
            estimate_fringe_angle_2d(None)


if __name__ == "__main__":
    unittest.main()
