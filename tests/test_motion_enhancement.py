import unittest

import cv2
import numpy as np

from src.vision.motion_enhancement import MotionFrameEnhancer


class MotionFrameEnhancerTests(unittest.TestCase):
    def test_preserves_shape_type_and_input(self):
        frame = np.full((80, 120, 3), 80, dtype=np.uint8)
        frame[:, 45:55] = (160, 80, 30)
        original = frame.copy()
        enhancer = MotionFrameEnhancer()

        result = enhancer.apply(frame, clarity_score=10, clarity_baseline=100)

        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)
        self.assertTrue(np.array_equal(frame, original))

    def test_horizontal_edge_contrast_increases_for_blurred_vertical_fringe(self):
        sharp = np.zeros((100, 180), dtype=np.uint8)
        for x in range(0, sharp.shape[1], 24):
            sharp[:, x:x + 12] = 180
        blurred = cv2.GaussianBlur(sharp, (13, 1), sigmaX=3)
        frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        enhancer = MotionFrameEnhancer({
            "sharpen_strength": 1.2,
            "max_sharpen_strength": 2.0,
            "contrast_gain": 1.0,
        })

        result = enhancer.apply(
            frame, clarity_score=10, clarity_baseline=100, blur_ratio=0.55)
        before = cv2.Sobel(blurred, cv2.CV_32F, 1, 0).var()
        after_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        after = cv2.Sobel(after_gray, cv2.CV_32F, 1, 0).var()

        self.assertGreater(after, before)
        self.assertGreater(enhancer.last_strength, enhancer.base_strength)

    def test_low_contrast_colored_vertical_fringes_become_more_visible(self):
        height, width = 120, 240
        x = np.arange(width, dtype=np.float32)
        wave = np.sin(2 * np.pi * x / 30.0)
        illumination = np.linspace(-22, 28, width, dtype=np.float32)
        frame = np.empty((height, width, 3), dtype=np.float32)
        frame[:, :, 0] = 105 + illumination + 8 * wave
        frame[:, :, 1] = 108 + illumination - 3 * wave
        frame[:, :, 2] = 110 + illumination - 7 * wave
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (11, 1), sigmaX=3)
        enhancer = MotionFrameEnhancer({"contrast_gain": 1.0})

        result = enhancer.apply(
            frame, clarity_score=8, clarity_baseline=100, blur_ratio=0.55)
        before_profile = (
            frame[:, :, 0].mean(axis=0) - frame[:, :, 2].mean(axis=0))
        after_profile = (
            result[:, :, 0].mean(axis=0) - result[:, :, 2].mean(axis=0))

        self.assertGreater(after_profile.std(), before_profile.std() * 1.5)
        self.assertGreater(enhancer.last_stripe_strength, enhancer.stripe_strength)
        self.assertGreater(enhancer.last_color_gain, enhancer.color_gain)

    def test_disabled_returns_original_frame(self):
        frame = np.full((20, 30, 3), 90, dtype=np.uint8)
        enhancer = MotionFrameEnhancer({"enabled": False})
        self.assertIs(enhancer.apply(frame), frame)
        self.assertEqual(enhancer.last_strength, 0.0)


if __name__ == "__main__":
    unittest.main()
