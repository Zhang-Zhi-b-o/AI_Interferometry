"""测量不确定度与确定性数据分析工具的单元测试。"""
import math
import unittest

from src.measurement import calculate_thickness_mm
from src.measurement.uncertainty import (
    DEFAULT_REFRACTIVE_INDEX_TOLERANCE,
    analyze_glass_thickness,
    combine_standard_uncertainties,
    coverage_factor,
    format_measurement,
    grubbs_test,
    sample_mean,
    sample_std,
    type_a_standard_uncertainty,
    type_b_normal,
    type_b_rectangular,
)
from src.agent.tools import (
    build_deterministic_section,
    detect_intent,
    extract_glass_rounds,
)


class BasicStatisticsTests(unittest.TestCase):
    def test_sample_mean(self):
        self.assertAlmostEqual(sample_mean([1, 2, 3]), 2.0)

    def test_sample_std_bessel(self):
        # 样本标准差使用 n-1 分母：[1,2,3] -> s=1
        self.assertAlmostEqual(sample_std([1, 2, 3]), 1.0)

    def test_type_a_is_std_over_sqrt_n(self):
        self.assertAlmostEqual(
            type_a_standard_uncertainty([1, 2, 3]), 1.0 / math.sqrt(3))

    def test_type_a_requires_two_values(self):
        with self.assertRaises(ValueError):
            type_a_standard_uncertainty([1.0])

    def test_type_b_rectangular(self):
        self.assertAlmostEqual(type_b_rectangular(3.0), 3.0 / math.sqrt(3))

    def test_type_b_normal(self):
        self.assertAlmostEqual(type_b_normal(0.02, k=2), 0.01)

    def test_combine_rss(self):
        self.assertAlmostEqual(
            combine_standard_uncertainties(3.0, 4.0), 5.0)


class CoverageAndFormatTests(unittest.TestCase):
    def test_coverage_factor_large_n_approaches_two(self):
        self.assertAlmostEqual(coverage_factor(30), 2.042, places=3)
        self.assertAlmostEqual(coverage_factor(1000), 2.0)

    def test_coverage_factor_small_n(self):
        # nu=2 -> t(0.95) = 4.303
        self.assertAlmostEqual(coverage_factor(2), 4.303, places=3)

    def test_format_measurement_aligns_decimals(self):
        # 不确定度保留两位有效数字，结果对齐到同一位
        self.assertEqual(format_measurement(0.10876, 0.00234), "0.1088 ± 0.0023")

    def test_format_zero_uncertainty(self):
        self.assertIn("± 0", format_measurement(0.5, 0.0))


class GrubbsTests(unittest.TestCase):
    def test_no_outlier_in_uniform_data(self):
        result = grubbs_test([1.00, 1.01, 1.02, 0.99, 1.00])
        self.assertIsNotNone(result)
        self.assertFalse(result["is_outlier"])

    def test_detects_extreme_outlier(self):
        result = grubbs_test([1.00, 1.01, 1.02, 0.99, 1.00, 1.60])
        self.assertIsNotNone(result)
        self.assertTrue(result["is_outlier"])
        self.assertEqual(result["suspicious_value"], 1.60)

    def test_too_few_or_too_many_values_returns_none(self):
        self.assertIsNone(grubbs_test([1.0, 2.0]))
        self.assertIsNone(grubbs_test(list(range(1, 40))))


class GlassThicknessAnalysisTests(unittest.TestCase):
    def _round(self, d1, d2, n=1.4586):
        return calculate_thickness_mm(d1, d2, n)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            analyze_glass_thickness([])

    def test_single_measurement_has_no_type_a(self):
        h = self._round(1.000, 1.500)
        result = analyze_glass_thickness([h], d1_values=[1.0], d2_values=[1.5])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["type_a_mm"], 0.0)
        self.assertIsNone(result["thickness_std_mm"])
        # 只有 B 类时包含因子按正态取 k≈2
        self.assertEqual(result["coverage_factor"], 2.0)
        # 单次测量无法评定 A 类，应给出提示
        self.assertTrue(any("一次测量" in w for w in result["warnings"]))

    def test_identical_measurements_zero_type_a(self):
        h = self._round(1.000, 1.500)
        result = analyze_glass_thickness(
            [h, h, h], d1_values=[1.0, 1.0, 1.0], d2_values=[1.5, 1.5, 1.5])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["type_a_mm"], 0.0)
        self.assertGreater(result["combined_uc_mm"], 0.0)
        # N=3 -> 自由度 2 -> k=4.303
        self.assertAlmostEqual(result["coverage_factor"], 4.303, places=3)

    def test_b_class_micrometer_contribution(self):
        # 灵敏系数 |∂h/∂d| = 1/[20(n-1)]；d1、d2 各一次独立，RSS 得 sqrt(2) 倍。
        n = 1.5
        h = self._round(1.000, 1.500, n=n)  # (0.5)/(10)=0.05
        result = analyze_glass_thickness(
            [h], d1_values=[1.0], d2_values=[1.5],
            refractive_index=n, micrometer_accuracy_mm=0.0001,
            refractive_index_tolerance=0.001)
        dhd = 1.0 / (20.0 * (n - 1.0))  # 0.1
        expected = math.sqrt(2.0) * dhd * (0.0001 / math.sqrt(3.0))
        self.assertAlmostEqual(
            result["type_b_mm"]["micrometer_contribution_mm"], expected, places=12)

    def test_result_text_present(self):
        h1 = self._round(1.000, 1.500)
        h2 = self._round(1.001, 1.501)
        result = analyze_glass_thickness(
            [h1, h2], d1_values=[1.0, 1.001], d2_values=[1.5, 1.501])
        self.assertIn("±", result["result_text"])
        self.assertIn("mm", result["result_text"])


class AgentToolsTests(unittest.TestCase):
    def test_detect_intent(self):
        self.assertEqual(detect_intent("帮我计算误差和不确定度"), "calculation")
        self.assertEqual(detect_intent("生成实验报告"), "report")
        self.assertEqual(detect_intent("下一步做什么"), "general")

    def test_extract_rounds_from_context(self):
        context = {
            "measurement": {"experiment_assistant": {
                "session": {"rounds": [{"thickness_mm": 0.1}]}}}}
        rounds = extract_glass_rounds(context)
        self.assertIsNotNone(rounds)
        self.assertEqual(len(rounds), 1)

    def test_extract_rounds_empty_returns_none(self):
        self.assertIsNone(extract_glass_rounds({}))
        self.assertIsNone(extract_glass_rounds(
            {"measurement": {"experiment_assistant": {"session": {"rounds": []}}}}))

    def test_build_deterministic_section_with_data(self):
        h = calculate_thickness_mm(1.000, 1.500)
        context = {
            "measurement": {"experiment_assistant": {
                "session": {
                    "refractive_index": 1.4586,
                    "rounds": [{
                        "sequence": 1, "d1_mm": 1.000, "d2_mm": 1.500,
                        "thickness_mm": h, "note": "",
                    }],
                },
                "statistics": {"count": 1, "mean_mm": h},
            }}}
        section = build_deterministic_section(context)
        self.assertIn("程序已计算的确定性结果", section)
        self.assertIn("不确定度", section)
        self.assertIn("±", section)

    def test_build_deterministic_section_without_data(self):
        self.assertEqual(build_deterministic_section({}), "")


if __name__ == "__main__":
    unittest.main()
