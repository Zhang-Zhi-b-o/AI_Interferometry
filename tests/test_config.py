import tempfile
import unittest
from pathlib import Path

from src.config import Config, ConfigError


class ConfigValidationTests(unittest.TestCase):
    def test_project_config_is_valid(self):
        config = Config()
        self.assertGreater(config.get("motor", "timeout"), 0)

    def test_reports_multiple_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "camera:\n  resolution: [0, 10]\n  fps: 0\n"
                "vision:\n  model_path: missing.pt\n  confidence_threshold: 2\n"
                "motor:\n  timeout: -1\n",
                encoding="utf-8")
            with self.assertRaises(ConfigError) as raised:
                Config(str(path))
            message = str(raised.exception)
            self.assertIn("camera.resolution", message)
            self.assertIn("vision.confidence_threshold", message)
            self.assertIn("motor.timeout", message)

    def test_rejects_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config(str(path))


if __name__ == "__main__":
    unittest.main()
