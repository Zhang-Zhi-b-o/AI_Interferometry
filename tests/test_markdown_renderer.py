import unittest

from src.ui.markdown_renderer import latex_to_text, markdown_segments


class MarkdownRendererTests(unittest.TestCase):
    def test_renders_headings_bold_lists_and_math_without_raw_markers(self):
        source = (
            "### 2. 实验原理\n"
            "**核心思想**：两束光发生干涉。\n"
            "- 光程差为 \\( \\Delta L = 2(d_2-d_1) \\)。")
        segments = markdown_segments(source)
        text = "".join(content for content, _ in segments)
        tags = {tag for _, item_tags in segments for tag in item_tags}
        self.assertNotIn("###", text)
        self.assertNotIn("**", text)
        self.assertNotIn(r"\(", text)
        self.assertIn("Δ L = 2(d₂-d₁)", text)
        self.assertTrue({"heading3", "bold", "bullet", "math"} <= tags)

    def test_converts_common_latex(self):
        self.assertEqual(latex_to_text(r"\(\lambda = \frac{2\Delta d}{N}\)"),
                         "λ = (2Δ d)/(N)")

    def test_formats_report_table(self):
        segments = markdown_segments("| 次数 | 位移 |\n|---|---|\n| 1 | 0.10 mm |")
        text = "".join(content for content, _ in segments)
        self.assertNotIn("---", text)
        self.assertIn("次数   │   位移", text)


if __name__ == "__main__":
    unittest.main()
