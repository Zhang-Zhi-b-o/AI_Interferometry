"""``michelson`` 命令行纯逻辑子命令的单元测试。"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from src import cli


def run_cli(*argv) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(list(argv))
    return buf.getvalue()


class CliMeasureTests(unittest.TestCase):
    def test_glass_thickness_outputs_positive_number(self):
        out = run_cli("measure", "glass-thickness", "1.0", "1.5")
        data = json.loads(out)
        self.assertIn("thickness_mm", data)
        self.assertGreater(data["thickness_mm"], 0)
        self.assertEqual(data["d1_mm"], 1.0)
        self.assertEqual(data["d2_mm"], 1.5)

    def test_uncertainty_outputs_dict(self):
        out = run_cli("measure", "uncertainty", "0.100", "0.101", "0.099")
        data = json.loads(out)
        self.assertIsInstance(data, dict)


class CliAnalyzeTests(unittest.TestCase):
    def test_load_image_reads_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :, 1] = 128
            cv2.imencode(".png", image)[1].tofile(str(path))
            loaded = cli._load_image(str(path))
            self.assertEqual(loaded.shape, (24, 32, 3))


class CliControlAndToolsTests(unittest.TestCase):
    def test_center_search_simulates_state_machine(self):
        out = run_cli("control", "center-search", "--steps", "3")
        self.assertIn("start", out)

    def test_tools_lists_registry_with_risk(self):
        out = run_cli("tools")
        self.assertIn("get_context", out)
        self.assertIn("measurement_start", out)
        self.assertIn("read", out)
        self.assertIn("motion", out)


if __name__ == "__main__":
    unittest.main()
