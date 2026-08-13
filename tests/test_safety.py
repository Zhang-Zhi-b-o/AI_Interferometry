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
from src.vision.fringe_recognition import (
    FringeRecognitionTracker,
    analyse_fringe_texture,
)
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

    def test_reconnect_reopens_same_port_and_restores_connected(self):
        motor = MotorController(port="COM3")
        motor._connected = False
        motor._ser = Mock()
        motor._ser.is_open = False
        with patch("serial.Serial") as serial_cls:
            serial_cls.return_value.is_open = True
            with patch.object(
                MotorController, "detect_port", return_value="COM3"
            ) as detect:
                self.assertTrue(motor.reconnect())
                detect.assert_called_once_with("COM3")
                serial_cls.assert_called_once_with(
                    "COM3", motor.baudrate, timeout=motor.timeout)
        self.assertTrue(motor.is_connected)

    def test_reconnect_falls_back_to_redetected_port(self):
        motor = MotorController(port="COM3")
        motor._connected = False
        motor._ser = Mock()
        motor._ser.is_open = False
        with patch("serial.Serial") as serial_cls:
            serial_cls.return_value.is_open = True
            with patch.object(
                MotorController, "detect_port", return_value="COM9"
            ) as detect:
                self.assertTrue(motor.reconnect())
                serial_cls.assert_called_once_with(
                    "COM9", motor.baudrate, timeout=motor.timeout)
        self.assertEqual(motor.port, "COM9")
        self.assertTrue(motor.is_connected)

    def test_reconnect_returns_false_on_serial_error(self):
        motor = MotorController(port="COM3")
        motor._connected = False
        motor._ser = Mock()
        motor._ser.is_open = False
        with patch("serial.Serial", side_effect=serial.SerialException("gone")):
            with patch.object(
                MotorController, "detect_port", return_value="COM3"
            ):
                self.assertFalse(motor.reconnect())
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

    def test_curved_fringe_relies_more_on_yolo_prior(self):
        """弯曲条纹 (verticality < 0.55) 应更依赖 YOLO prior，结果靠近框中心。"""
        import cv2

        image = np.zeros((120, 320, 3), dtype=np.uint8)
        # 绘制倾斜条纹：每行 x 偏移不同，降低 verticality
        for row in range(image.shape[0]):
            offset = int(25 * np.sin(row / 18.0))  # 正弦弯曲
            for col in range(60, 280):
                phase = (col - 160 - offset) / 12.0
                val = int(128 + 80 * np.cos(2 * np.pi * phase))
                val = np.clip(val, 0, 255)
                image[row, col] = (val, max(0, val - 20), max(0, val - 40))

        result = find_center_in_region(
            image,
            expected_center_x=160,
            search_bounds=(80, 240),
        )
        # 弯曲条纹：中心应落在 YOLO 框范围内的合理位置
        self.assertEqual(result["orientation"], "vertical")
        self.assertGreaterEqual(result["center_x"], 80)
        self.assertLessEqual(result["center_x"], 240)
        # verticality 应明显低于完美竖直条纹
        self.assertLess(result["verticality"], 0.90)


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

    def test_tracker_ema_faster_tracking(self):
        """新的 EMA (0.35*prev + 0.65*median) 应更快跟踪单调移动。"""
        tracker = CenterTracker(hold_frames=3, max_jump_px=60)
        outputs = []
        for x, conf in [(100, 0.8), (110, 0.8), (120, 0.8), (130, 0.8)]:
            outputs.append(tracker.update(x, conf)["center"])
        # 由于 5 元素中值滤波，输出滞后不可避免，但每帧应有进展
        self.assertTrue(all(o is not None for o in outputs))
        # 输出应单调递增
        for i in range(1, len(outputs)):
            self.assertGreater(outputs[i], outputs[i - 1])
        # 第 4 帧时输出应明显偏离起始值（跟上运动趋势）
        self.assertGreater(outputs[-1], 110)

    def test_reset_from_yolo_seeds_tracker(self):
        """reset_from_yolo 应以 YOLO 位置为锚点重置追踪器。"""
        tracker = CenterTracker()
        tracker.update(200, 0.8)
        result = tracker.reset_from_yolo(500.0, 0.75)
        self.assertEqual(result["center"], 500.0)
        self.assertEqual(result["confidence"], 0.75)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["held"])
        # 后继帧应从此锚点继续
        next_result = tracker.update(502, 0.7)
        self.assertTrue(next_result["accepted"])
        self.assertAlmostEqual(next_result["center"], 501, delta=3)

    def test_tracker_accepts_high_confidence_jump(self):
        """新阈值 0.65：高置信度跳变被接受而非拒绝。"""
        tracker = CenterTracker(hold_frames=2, max_jump_px=45)
        tracker.update(100, 0.8)
        accepted = tracker.update(160, 0.70)
        self.assertTrue(accepted["accepted"])
        self.assertGreater(abs(accepted["center"] - 100), 10)


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


class FringeRecognitionTests(unittest.TestCase):
    @staticmethod
    def _stripe_scene(curved=False, blurred=False):
        import cv2

        image = np.zeros((240, 400, 3), dtype=np.uint8)
        image[45:195, 80:330] = (80, 100, 115)
        for x in range(95, 325, 13):
            points = []
            for y in range(55, 190):
                offset = (
                    10 * np.sin((y - 55) / 34)
                    if curved else 0.13 * (y - 120)
                )
                points.append((int(x + offset), y))
            colour = (
                30 + (x * 3) % 220,
                60 + (x * 5) % 190,
                80 + (x * 7) % 170,
            )
            cv2.polylines(
                image, [np.asarray(points, dtype=np.int32)],
                False, colour, 3,
            )
        if blurred:
            image = cv2.GaussianBlur(image, (21, 21), 5)
        return image

    def test_texture_detects_tilted_colour_fringe(self):
        result = analyse_fringe_texture(self._stripe_scene())
        self.assertGreater(result["confidence"], 0.65)
        self.assertAlmostEqual(result["position_x"], 210, delta=35)

    def test_texture_preserves_full_resolution_coordinates(self):
        import cv2

        image = cv2.resize(self._stripe_scene(), (800, 480))
        result = analyse_fringe_texture(image)
        self.assertAlmostEqual(result["position_x"], 420, delta=70)

    def test_texture_detects_curved_colour_fringe(self):
        result = analyse_fringe_texture(self._stripe_scene(curved=True))
        self.assertGreater(result["confidence"], 0.65)
        self.assertGreater(result["angle_spread_deg"], 5)

    def test_texture_marks_motion_blur_without_erasing_evidence(self):
        result = analyse_fringe_texture(self._stripe_scene(blurred=True))
        self.assertTrue(result["blurred"])
        self.assertGreater(result["confidence"], 0.45)

    def test_uniform_scene_is_not_a_fringe(self):
        image = np.full((240, 400, 3), 70, dtype=np.uint8)
        self.assertLess(analyse_fringe_texture(image)["confidence"], 0.20)

    def test_tracker_uses_visual_fallback_after_yolo_miss(self):
        tracker = FringeRecognitionTracker()
        tracker.update(
            yolo_has_fringe=True, yolo_position_x=100,
            yolo_confidence=0.8, now=0.0,
        )
        result = tracker.update(
            yolo_has_fringe=False,
            texture={"confidence": 0.35, "position_x": 104,
                     "blurred": False},
            now=0.1,
        )
        self.assertTrue(result["has_fringe"])
        self.assertEqual(result["source"], "visual")
        self.assertFalse(result["held"])

    def test_tracker_can_start_from_strong_visual_evidence(self):
        tracker = FringeRecognitionTracker()
        texture = analyse_fringe_texture(self._stripe_scene(curved=True))
        result = tracker.update(
            yolo_has_fringe=False, texture=texture, now=0.0)
        self.assertTrue(result["has_fringe"])
        self.assertEqual(result["source"], "visual")

    def test_tracker_predicts_through_short_blurred_dropout(self):
        tracker = FringeRecognitionTracker(missing_hold_frames=2)
        tracker.update(
            yolo_has_fringe=True, yolo_position_x=100,
            yolo_confidence=0.8, now=0.0,
        )
        tracker.update(
            yolo_has_fringe=True, yolo_position_x=110,
            yolo_confidence=0.8, now=0.1,
        )
        result = tracker.update(
            yolo_has_fringe=False,
            texture={"confidence": 0.05, "position_x": None,
                     "blurred": True},
            now=0.2,
        )
        self.assertTrue(result["has_fringe"])
        self.assertTrue(result["held"])
        self.assertEqual(result["source"], "history")
        self.assertAlmostEqual(result["position_x"], 120, delta=1)
        self.assertAlmostEqual(result["velocity_px_s"], 100, delta=1)

    def test_tracker_rejects_discontinuous_visual_candidate(self):
        tracker = FringeRecognitionTracker(missing_hold_frames=0)
        tracker.update(
            yolo_has_fringe=True, yolo_position_x=100,
            yolo_confidence=0.8, now=0.0,
        )
        result = tracker.update(
            yolo_has_fringe=False,
            texture={"confidence": 0.9, "position_x": 310,
                     "blurred": False},
            now=0.1,
        )
        self.assertFalse(result["has_fringe"])

if __name__ == "__main__":
    unittest.main()
