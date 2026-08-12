import unittest

from src.measurement import ThicknessMeasurement, calculate_thickness_mm


class ThicknessMeasurementTests(unittest.TestCase):
    def test_uses_requested_formula_and_mm_units(self):
        result = calculate_thickness_mm(1.000, 1.500)
        self.assertAlmostEqual(result, 0.500 / (10 * (1.4586 - 1)), places=12)

    def test_preserves_d2_minus_d1_direction(self):
        self.assertLess(calculate_thickness_mm(2.0, 1.0), 0)

    def test_refractive_index_must_be_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "折射率"):
            calculate_thickness_mm(1.0, 2.0, refractive_index=1.0)

    def test_records_are_selected_by_stable_ids(self):
        measurement = ThicknessMeasurement()
        first = measurement.add(1.100, captured_at=100.0)
        second = measurement.add(1.200, captured_at=101.0)
        self.assertEqual((first.key, second.key), ("R1", "R2"))
        self.assertAlmostEqual(
            measurement.calculate("R1", "R2"),
            calculate_thickness_mm(1.100, 1.200),
        )
        measurement.remove("R1")
        self.assertEqual([record.key for record in measurement.records], ["R2"])

    def test_same_record_cannot_be_used_twice(self):
        measurement = ThicknessMeasurement()
        measurement.add(1.100, captured_at=100.0)
        with self.assertRaisesRegex(ValueError, "不同"):
            measurement.calculate("R1", "R1")

    def test_same_micrometer_frame_cannot_be_recorded_twice(self):
        measurement = ThicknessMeasurement()
        measurement.add(1.100, captured_at=100.0)
        with self.assertRaisesRegex(ValueError, "采集帧"):
            measurement.add(1.100, captured_at=100.0)


if __name__ == "__main__":
    unittest.main()
