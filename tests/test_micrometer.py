import threading
import unittest

import numpy as np

from src.hardware.micrometer import MicrometerReader
from src.vision.micrometer_ocr import (
    MicrometerOCR,
    MicrometerOCRResult,
    ReadingStabilizer,
    locate_lcd,
    normalize_meter_text,
)


class MicrometerVisionTests(unittest.TestCase):
    def test_normalizes_fixed_three_decimal_reading(self):
        self.assertEqual(normalize_meter_text("001234", 3), "001.234")
        self.assertEqual(normalize_meter_text(".125", 3), "0.125")
        self.assertEqual(normalize_meter_text("-0125", 3), "-0.125")
        self.assertEqual(normalize_meter_text("-.002", 3), "-0.002")
        self.assertIsNone(normalize_meter_text("12.3.4", 3))
        # 已经识别到小数点时不强制改写为三位，格式规则只是辅助条件。
        self.assertEqual(normalize_meter_text("12.34", 3), "12.34")

    def test_stabilizer_requires_repeated_frames(self):
        stabilizer = ReadingStabilizer(window_size=5, required=3, decimal_places=3)
        self.assertEqual(stabilizer.update(1.234), (None, False, 1))
        stabilizer.update(1.235)
        stabilizer.update(1.234)
        stabilizer.update(1.234)
        stable, ready, count = stabilizer.update(1.234)
        self.assertEqual(stable, 1.234)
        self.assertTrue(ready)
        self.assertEqual(count, 3)

    def test_stabilizer_holds_last_value_during_invalid_frames(self):
        stabilizer = ReadingStabilizer(required=3, decimal_places=3)
        for _ in range(3):
            stabilizer.update(1.234)
        stable, ready, count = stabilizer.update(None)
        self.assertEqual(stable, 1.234)
        self.assertFalse(ready)
        self.assertEqual(count, 0)

    def test_stabilizer_rejects_repeated_times_ten_error(self):
        stabilizer = ReadingStabilizer(
            required=3, decimal_places=3, max_step=0.05,
            jump_required=6)
        for _ in range(3):
            stabilizer.update(1.234)
        for _ in range(12):
            stable, ready, _count = stabilizer.update(12.34)
            self.assertEqual(stable, 1.234)
            self.assertFalse(ready)
            self.assertTrue(stabilizer.last_rejected)
            self.assertIn("×10/÷10", stabilizer.last_reason)

    def test_near_zero_three_decimal_change_is_not_mistaken_for_divide_by_ten(self):
        stabilizer = ReadingStabilizer(
            required=3, decimal_places=3, max_step=0.05,
            jump_required=6)
        for _ in range(3):
            stabilizer.update(-0.021)
        for _ in range(2):
            stable, ready, _count = stabilizer.update(-0.002)
            self.assertEqual(stable, -0.021)
            self.assertFalse(ready)
            self.assertFalse(stabilizer.last_rejected)
        stable, ready, _count = stabilizer.update(-0.002)
        self.assertEqual(stable, -0.002)
        self.assertTrue(ready)

    def test_stabilizer_confirms_small_physical_change(self):
        stabilizer = ReadingStabilizer(
            required=3, decimal_places=3, max_step=0.05)
        for _ in range(3):
            stabilizer.update(1.234)
        self.assertEqual(stabilizer.update(1.235)[0], 1.234)
        self.assertEqual(stabilizer.update(1.235)[0], 1.234)
        stable, ready, count = stabilizer.update(1.235)
        self.assertEqual(stable, 1.235)
        self.assertTrue(ready)
        self.assertEqual(count, 3)

    def test_large_change_needs_extra_confirmation(self):
        stabilizer = ReadingStabilizer(
            required=3, decimal_places=3, max_step=0.05,
            jump_required=6)
        for _ in range(3):
            stabilizer.update(1.234)
        for _ in range(5):
            stable, ready, _count = stabilizer.update(1.500)
            self.assertEqual(stable, 1.234)
            self.assertFalse(ready)
        stable, ready, _count = stabilizer.update(1.500)
        self.assertEqual(stable, 1.500)
        self.assertTrue(ready)

    def test_manual_roi_is_applied_when_auto_is_disabled(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop, box = locate_lcd(
            frame, auto_roi=False, manual_roi=(0.25, 0.2, 0.5, 0.4))
        self.assertEqual(box, (50, 20, 150, 60))
        self.assertEqual(crop.shape[:2], (40, 100))

    def test_ocr_reads_recognition_output_and_infers_decimal(self):
        class Output:
            txts = ("001234",)
            scores = (0.91,)

        class Engine:
            def __call__(self, *_args, **_kwargs):
                return Output()

        ocr = MicrometerOCR(decimal_places=3, stable_required=1)
        ocr._engine = Engine()
        result = ocr.recognize(
            np.zeros((50, 180, 3), dtype=np.uint8), auto_roi=False)
        self.assertEqual(result.text, "001.234")
        self.assertEqual(result.stable_value_mm, 1.234)
        self.assertTrue(result.stable)
        self.assertIn("末3位辅助补全", result.format_hint)

    def test_explicit_non_three_decimal_result_is_kept_for_temporal_confirmation(self):
        class Output:
            txts = ("12.34",)
            scores = (0.92,)

        class Engine:
            def __call__(self, *_args, **_kwargs):
                return Output()

        ocr = MicrometerOCR(decimal_places=3, stable_required=1)
        ocr._engine = Engine()
        result = ocr.recognize(
            np.zeros((50, 180, 3), dtype=np.uint8), auto_roi=False)
        self.assertEqual(result.text, "12.34")
        self.assertEqual(result.value_mm, 12.34)
        self.assertTrue(result.stable)
        self.assertIn("不是3位小数", result.format_hint)


class MicrometerReaderTests(unittest.TestCase):
    def test_reader_preserves_stable_value_when_a_frame_fails(self):
        reader = MicrometerReader()
        accepted = reader._preserve_stable_value(MicrometerOCRResult(
            value_mm=1.234, stable_value_mm=1.234, stable=True,
            message="稳定读数 1.234 mm",
        ))
        self.assertEqual(accepted.stable_value_mm, 1.234)

        held = reader._preserve_stable_value(MicrometerOCRResult(
            message="微分表摄像头读取失败",
        ))
        self.assertEqual(held.stable_value_mm, 1.234)
        self.assertTrue(held.reading_held)
        self.assertIn("保持稳定读数", held.message)

    def test_background_reader_publishes_stable_value_and_closes_camera(self):
        camera_instances = []

        class Camera:
            def __init__(self, **_kwargs):
                self.stopped = False
                camera_instances.append(self)

            def start(self):
                return True

            def read(self):
                return np.zeros((20, 50, 3), dtype=np.uint8)

            def stop(self):
                self.stopped = True

        class OCR:
            def recognize(self, *_args, **_kwargs):
                return MicrometerOCRResult(
                    text="1.234", value_mm=1.234, score=0.9,
                    stable_value_mm=1.234, stable=True,
                    message="稳定读数 1.234 mm")

        reader = MicrometerReader(
            interval_ms=50, ocr=OCR(), camera_factory=Camera)
        self.assertTrue(reader.connect())
        received = threading.Event()
        results = []

        def on_result(result):
            results.append(result)
            received.set()

        reader.start(on_result)
        self.assertTrue(received.wait(1.0))
        self.assertIsNotNone(results[0].frame)
        self.assertEqual(results[0].frame.shape, (20, 50, 3))
        self.assertEqual(reader.read_value_mm(), 1.234)
        self.assertIsNotNone(results[0].captured_at)
        self.assertIsNotNone(results[0].captured_monotonic)
        self.assertEqual(
            results[0].stable_captured_at, results[0].captured_at)
        self.assertEqual(
            results[0].stable_captured_monotonic,
            results[0].captured_monotonic)
        reader.close()
        self.assertFalse(reader.connected)
        self.assertTrue(camera_instances[0].stopped)


if __name__ == "__main__":
    unittest.main()
