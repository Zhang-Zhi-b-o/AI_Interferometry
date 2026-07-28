import tempfile
import unittest
from pathlib import Path

from src.config import Config, ConfigError
from src.ui.recording_preset import load_recording_preset


class ConfigValidationTests(unittest.TestCase):
    def test_project_config_is_valid(self):
        config = Config()
        self.assertGreater(config.get("ui", "window_size")[0], 0)
        self.assertIsNone(config.get("motor"))

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

    def test_temporary_measurement_enabled_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "temporary_measurement:\n  enabled: yes-please\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ConfigError, "temporary_measurement.enabled 必须是布尔值",
            ):
                Config(str(path))

    def test_video_demo_preset_is_complete(self):
        preset = load_recording_preset()
        self.assertEqual(
            preset["auto_center"]["search_mode"], "single_direction")
        self.assertTrue(Path(
            preset["yolo"]["model_path"]).as_posix().endswith(".pt"))

    def test_video_demo_preset_does_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_demo.yaml"
            path.write_text(
                "main_camera:\n  index: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "缺少参数"):
                load_recording_preset(path)


if __name__ == "__main__":
    unittest.main()
