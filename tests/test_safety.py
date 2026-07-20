import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import serial

from src.hardware.motor import MotorController
from src.vision.detector import YOLODetector
from src.vision.fringe_center import CenterTracker, find_center_in_region
from src.vision.fringe_motion import FringeMotionTracker
from src.vision.class_names import get_class_confidences, get_non_center_guide


class DetectorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.detector = YOLODetector("unused.pt", device="cpu")
        self.detector._model = Mock()
        self.detector._model.names = {0: "color", 1: "black"}
        self.frame = np.zeros((100, 120, 3), dtype=np.uint8)

    def test_roi_outside_frame_is_rejected_without_inference(self):
        result = self.detector.detect(self.frame, roi=(999, 999, 20, 20))
        self.assertEqual(len(result["boxes_xyxy"]), 0)
        self.detector._model.predict.assert_not_called()

    def test_model_class_ids_are_resolved_by_name(self):
        self.assertEqual(self.detector.find_class_ids("black"), {1})
        self.assertEqual(self.detector.find_class_ids("color"), {0})

    def test_current_model_names_map_to_control_roles(self):
        result = {
            "class_names": ["near_fringe", "zero_order", "far_fringe"],
            "confs": np.array([0.7, 0.8, 0.6]),
        }
        self.assertEqual(
            get_class_confidences(result), {"color": 0.7, "black": 0.8})

    def test_non_center_guide_prefers_near_fringe_and_excludes_zero(self):
        result = {
            "boxes_xyxy": np.array([
                [600, 0, 680, 100],
                [850, 0, 950, 100],
                [300, 0, 400, 100],
            ]),
            "class_names": ["zero_order", "far_fringe", "near_fringe"],
            "confs": np.array([0.95, 0.9, 0.75]),
        }
        guide = get_non_center_guide(result, 1280)
        self.assertEqual(guide["class_name"], "near_fringe")
        self.assertEqual(guide["x"], 350.0)
        self.assertEqual(guide["count"], 2)

    def test_non_center_guide_keeps_previous_box_when_multiple_boxes_exist(self):
        result = {
            "boxes_xyxy": np.array([
                [200, 10, 300, 90],
                [600, 10, 700, 90],
            ]),
            "class_names": ["near_fringe", "near_fringe"],
            "confs": np.array([0.8, 0.9]),
        }
        guide = get_non_center_guide(result, 1280, previous_x=245)
        self.assertEqual(guide["x"], 250)
        self.assertEqual(guide["count"], 2)


class MotorProtocolTests(unittest.TestCase):
    @patch("src.hardware.motor.serial.tools.list_ports.comports")
    def test_detect_port_prefers_configured_existing_port(self, comports):
        comports.return_value = [
            SimpleNamespace(device="COM5", description="USB-SERIAL CH340", hwid="1A86:7523"),
            SimpleNamespace(device="COM8", description="Bluetooth", hwid="BTH"),
        ]
        self.assertEqual(MotorController.detect_port("COM8"), "COM8")

    @patch("src.hardware.motor.serial.tools.list_ports.comports")
    def test_detect_port_finds_unique_usb_serial_adapter(self, comports):
        comports.return_value = [
            SimpleNamespace(device="COM5", description="USB-SERIAL CH340", hwid="1A86:7523"),
            SimpleNamespace(device="COM8", description="Bluetooth", hwid="BTH"),
        ]
        self.assertEqual(MotorController.detect_port("auto"), "COM5")

    @patch("src.hardware.motor.serial.tools.list_ports.comports")
    def test_detect_port_does_not_guess_between_ambiguous_ports(self, comports):
        comports.return_value = [
            SimpleNamespace(device="COM7", description="Serial A", hwid="A"),
            SimpleNamespace(device="COM8", description="Serial B", hwid="B"),
        ]
        self.assertIsNone(MotorController.detect_port("auto"))

    def test_json_status_parser(self):
        status = MotorController._parse_status(
            '{"running":true,"direction":"forward","gear":5}')
        self.assertTrue(status["valid"])
        self.assertTrue(status["running"])
        self.assertEqual(status["direction"], "forward")
        self.assertEqual(status["speed"], 5)
        self.assertEqual(status["omega"], 630)
        self.assertEqual(status["pulse_freq"], 2800)

    def test_legacy_status_parser_remains_compatible(self):
        status = MotorController._parse_status("RUN,SPD:5,OMEGA:630deg/s")
        self.assertTrue(status["valid"])
        self.assertTrue(status["running"])
        self.assertEqual(status["speed"], 5)
        self.assertEqual(status["omega"], 630)

    def test_start_reports_write_success(self):
        motor = MotorController()
        motor._connected = True
        motor._ser = Mock()
        motor._ser.is_open = True
        self.assertTrue(motor.start())
        motor._ser.write.assert_called_once_with(b"R")

    def test_set_speed_falls_back_when_status_has_no_gear(self):
        motor = MotorController()
        motor.query_status = Mock(return_value={"valid": False, "speed": 0})
        motor.speed_down = Mock(return_value=True)
        motor.speed_up = Mock(return_value=True)

        self.assertTrue(motor.set_speed(8))
        self.assertEqual(motor.speed_down.call_count, 10)
        self.assertEqual(motor.speed_up.call_count, 2)

    def test_reverse_and_toggle_direction_use_documented_commands(self):
        motor = MotorController()
        motor._connected = True
        motor._ser = Mock()
        motor._ser.is_open = True
        self.assertTrue(motor.start_reverse())
        self.assertTrue(motor.toggle_direction())
        self.assertEqual(
            [call.args[0] for call in motor._ser.write.call_args_list],
            [b"r", b"D"],
        )

    def test_malformed_or_empty_status_is_safe(self):
        status = MotorController._parse_status("garbage,SPD:nope,OMEGA:?")
        self.assertFalse(status["valid"])
        self.assertFalse(status["running"])
        self.assertEqual(status["speed"], 0)
        self.assertEqual(status["omega"], 0)

    def test_echoed_nested_camel_case_json_is_parsed(self):
        status = MotorController._parse_status(
            '?\r\n{"motor":{"isRunning":1,"dir":"CW",'
            '"speedLevel":"5档","pulseFreq":"2800 Hz"}}')
        self.assertTrue(status["valid"])
        self.assertTrue(status["running"])
        self.assertEqual(status["direction"], "forward")
        self.assertEqual(status["speed"], 5)
        self.assertEqual(status["pulse_freq"], 2800)
        self.assertEqual(status["omega"], 630)

    def test_protocol_direction_characters_remain_case_sensitive(self):
        forward = MotorController._parse_status('{"run":1,"dir":"R","gear":1}')
        reverse = MotorController._parse_status('{"run":1,"dir":"r","gear":1}')
        self.assertEqual(forward["direction"], "forward")
        self.assertEqual(reverse["direction"], "reverse")

    def test_query_skips_command_echo_before_json(self):
        motor = MotorController()
        motor._connected = True
        motor._ser = Mock()
        motor._ser.is_open = True
        motor._ser.readline.side_effect = [
            b"?\r\n",
            b'{"run":true,"gear":8,"direction":"reverse"}\r\n',
        ]

        status = motor.query_status()

        motor._ser.reset_input_buffer.assert_called_once_with()
        motor._ser.write.assert_called_once_with(b"?")
        self.assertTrue(status["valid"])
        self.assertTrue(status["running"])
        self.assertEqual(status["speed"], 8)
        self.assertEqual(status["direction"], "reverse")

    def test_serial_failure_marks_controller_disconnected(self):
        motor = MotorController()
        motor._connected = True
        motor._ser = Mock()
        motor._ser.is_open = True
        motor._ser.write.side_effect = serial.SerialException("disconnected")
        self.assertFalse(motor.start())
        self.assertFalse(motor.is_connected)


class FringeCenterInputTests(unittest.TestCase):
    @staticmethod
    def _colored_vertical_fringe(center=104, blue_center=True):
        x = np.arange(208, dtype=np.float64)
        envelope = np.exp(-((x - center) / 42.0) ** 2)
        phase = 2 * np.pi * (x - center) / 15.0
        if blue_center:
            channels = [
                0.42 + 0.36 * envelope * np.cos(phase),
                0.42 + 0.28 * envelope * np.cos(phase + 2.1),
                0.42 + 0.28 * envelope * np.cos(phase - 2.1),
            ]
        else:
            channels = [
                0.42 + 0.28 * envelope * np.cos(phase + 2.1),
                0.42 + 0.28 * envelope * np.cos(phase - 2.1),
                0.42 + 0.36 * envelope * np.cos(phase),
            ]
        image = np.stack(channels, axis=1)
        image = np.tile(image[None, :, :], (90, 1, 1))
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def _offset_colored_center(center=150, dominant_channel=0):
        """构造框中心偏移且中心色可变化的白光竖条纹。"""
        x = np.arange(240, dtype=np.float64)
        envelope = np.exp(-((x - center) / 55.0) ** 2)
        phase = 2 * np.pi * (x - center) / 18.0
        channels = np.stack([
            0.45 + 0.25 * envelope * np.cos(phase + shift)
            for shift in (0.0, 2.1, -2.1)
        ], axis=1)
        center_mask = np.exp(-0.5 * ((x - center) / 3.5) ** 2)[:, None]
        center_colour = np.full((240, 3), 0.07)
        center_colour[:, dominant_channel] = 0.36
        channels = channels * (1 - center_mask) + center_colour * center_mask
        image = np.tile(channels[None, :, :], (90, 1, 1))
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            find_center_in_region(np.array([]))

    def test_flat_image_is_rejected(self):
        image = np.full((80, 120, 3), 127, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "对比度过低"):
            find_center_in_region(image)

    def test_vertical_fringe_returns_bounded_center(self):
        x = np.arange(160)
        envelope = np.exp(-((x - 80) / 32) ** 2)
        profile = 0.5 + 0.35 * envelope * np.cos(2 * np.pi * (x - 80) / 12)
        gray = np.tile(profile, (80, 1))
        image = np.repeat((gray * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        result = find_center_in_region(image)
        self.assertEqual(result["orientation"], "vertical")
        self.assertLess(abs(result["center_x"] - 80), 12)
        self.assertGreater(result["confidence"], 0.1)

    def test_blue_center_is_located_without_dark_stripe_assumption(self):
        image = self._colored_vertical_fringe(center=104, blue_center=True)
        result = find_center_in_region(
            image, expected_center_x=100, search_radius=24)
        self.assertEqual(result["orientation"], "vertical")
        self.assertLess(abs(result["center_x"] - 104), 8)

    def test_center_location_is_not_tied_to_a_specific_color(self):
        blue = find_center_in_region(
            self._colored_vertical_fringe(104, True),
            expected_center_x=103,
            search_radius=24,
        )
        red = find_center_in_region(
            self._colored_vertical_fringe(104, False),
            expected_center_x=103,
            search_radius=24,
        )
        self.assertLess(abs(blue["center_x"] - red["center_x"]), 3)

    def test_zero_box_is_search_boundary_not_center_answer(self):
        image = self._offset_colored_center(center=150, dominant_channel=0)
        result = find_center_in_region(
            image,
            expected_center_x=112,
            search_bounds=(80, 190),
        )
        self.assertGreater(abs(result["center_x"] - 112), 25)
        self.assertLess(abs(result["center_x"] - 150), 4)

    def test_box_search_is_invariant_to_center_hue(self):
        centers = []
        for channel in range(3):
            result = find_center_in_region(
                self._offset_colored_center(150, channel),
                expected_center_x=112,
                search_bounds=(80, 190),
            )
            centers.append(result["center_x"])
        self.assertLess(max(centers) - min(centers), 3)
        self.assertLess(max(abs(center - 150) for center in centers), 4)


class CenterTrackerTests(unittest.TestCase):
    def test_tracker_smooths_jitter_and_holds_short_dropout(self):
        tracker = CenterTracker(hold_frames=2, max_jump_px=20)
        outputs = [tracker.update(x, 0.7)["center"] for x in (100, 106, 98, 104)]
        self.assertLess(np.std(outputs), np.std((100, 106, 98, 104)))
        held = tracker.update(None)
        self.assertTrue(held["held"])
        self.assertIsNotNone(held["center"])

    def test_tracker_rejects_low_confidence_large_jump(self):
        tracker = CenterTracker(hold_frames=2, max_jump_px=20)
        tracker.update(100, 0.8)
        rejected = tracker.update(180, 0.4)
        self.assertFalse(rejected["accepted"])
        self.assertLess(abs(rejected["center"] - 100), 1)


class FringeMotionTrackerTests(unittest.TestCase):
    def test_reports_no_fringe_without_position(self):
        tracker = FringeMotionTracker()
        result = tracker.update(has_fringe=False, position_x=None)
        self.assertFalse(result["has_fringe"])
        self.assertEqual(result["movement"], "unknown")

    def test_detects_rightward_motion(self):
        tracker = FringeMotionTracker(window_size=5, movement_threshold_px=3)
        for position in (100, 102, 108):
            result = tracker.update(
                has_fringe=True, position_x=position, source="guide")
        self.assertEqual(result["movement"], "right")
        self.assertEqual(result["delta_x_px"], 8)

    def test_detects_leftward_motion(self):
        tracker = FringeMotionTracker(window_size=5, movement_threshold_px=3)
        for position in (110, 106, 101):
            result = tracker.update(
                has_fringe=True, position_x=position, source="center")
        self.assertEqual(result["movement"], "left")

    def test_small_jitter_is_stable(self):
        tracker = FringeMotionTracker(window_size=5, movement_threshold_px=3)
        for position in (100, 101, 99.5, 101.5):
            result = tracker.update(
                has_fringe=True, position_x=position, source="guide")
        self.assertEqual(result["movement"], "stable")

    def test_source_switch_starts_a_new_motion_window(self):
        tracker = FringeMotionTracker(window_size=5, movement_threshold_px=3)
        for position in (100, 105, 110):
            tracker.update(has_fringe=True, position_x=position, source="guide")
        result = tracker.update(
            has_fringe=True, position_x=112, source="center")
        self.assertEqual(result["movement"], "unknown")
        self.assertEqual(result["delta_x_px"], 0)


if __name__ == "__main__":
    unittest.main()
