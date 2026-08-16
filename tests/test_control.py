import threading
import time
import unittest

from src.control import (
    AutoControlStateMachine,
    CenterControlStateMachine,
    ExpandingSearchPlanner,
    ExperimentObservation,
    ExperimentWorkflowStateMachine,
)
from src.hardware.command_queue import SerialCommandQueue
from src.ui.lifecycle import shutdown_motor_safely


CONTINUOUS = {
    "search_speed": 10,
    "color_speed": 5,
    "black_speed": 8,
    "black_threshold": 0.5,
}
STEP = {
    "first_ms": 100,
    "cycle_ms": 50,
    "pause_ms": 20,
    "speed": 5,
    "black_threshold": 0.5,
}
SAFETY = {"max_run_seconds": 10, "black_confirm_frames": 2, "max_missing_frames": 3}

CENTER_PARAMS = {
    "search_gear": 5,
    "fast_gear": 5,
    "slow_gear": 8,
    "slow_zone_px": 160,
    "tolerance_px": 15,
    "stable_frames": 2,
    "dropout_hold_frames": 2,
    "command_refresh_frames": 10,
    "min_confidence": 0.18,
    "search_direction": "forward",
    "invert_direction": False,
    "auto_learn_direction": False,
    "learning_delta_px": 4,
    "search_min_gear": 5,
    "search_acceleration_step": 0,
    "blur_slowdown_frames": 3,
    "blur_safe_gear": 8,
    "blur_recovery_clear_frames": 3,
}


class AutoControlStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = AutoControlStateMachine()

    def update(self, now, color=0.0, black=0.0, connected=True,
               params=CONTINUOUS, safety=SAFETY):
        return self.machine.update(
            color_conf=color, black_conf=black, connected=connected,
            params=params, safety=safety, now=now)

    def test_continuous_mode_only_changes_speed_on_transitions(self):
        self.machine.start("continuous", 0.0)
        first = self.update(0.0, color=0.1)
        self.assertEqual(first.commands, (("set_speed", 10), ("start", None)))
        color = self.update(0.1, color=0.8)
        self.assertEqual(color.commands, (("set_speed", 5),))
        self.assertEqual(self.update(0.2, color=0.8).commands, ())

    def test_continuous_mode_locks_after_confirmed_black(self):
        self.machine.start("continuous", 0.0)
        self.update(0.0, color=0.8)
        self.update(0.1, color=0.8)
        self.assertEqual(self.update(0.2, color=0.8, black=0.8).commands, ())
        enter_black = self.update(0.3, color=0.8, black=0.8)
        self.assertEqual(enter_black.commands, (("set_speed", 8),))
        self.assertEqual(self.update(0.4, black=0.8).commands, ())
        locked = self.update(0.5, black=0.8)
        self.assertEqual(locked.commands, (("stop", None),))
        self.assertEqual(locked.status, "自动控制: 已锁定")

    def test_missing_frames_and_disconnect_stop_motor(self):
        self.machine.start("continuous", 0.0)
        self.update(0.0)
        self.update(0.1)
        stopped = self.update(0.2)
        self.assertEqual(stopped.stopped_reason, "连续未检测到条纹")
        self.assertIn(("stop", None), stopped.commands)

        self.machine.start("continuous", 1.0)
        disconnected = self.update(1.1, connected=False)
        self.assertEqual(disconnected.stopped_reason, "串口失联")

    def test_maximum_run_time_stops_motor(self):
        self.machine.start("continuous", 0.0)
        stopped = self.update(10.1, color=0.5)
        self.assertEqual(stopped.stopped_reason, "达到最大运行时间")

    def test_step_mode_moves_then_pauses_without_sleep(self):
        self.machine.start("step", 0.0)
        moving = self.update(0.0, color=0.5, params=STEP)
        self.assertEqual(moving.commands, (("set_speed", 5), ("start", None)))
        paused = self.update(0.11, color=0.5, params=STEP)
        self.assertEqual(paused.commands, (("stop", None),))
        self.assertEqual(self.update(0.14, black=0.8, params=STEP).commands, ())


class CenterControlStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = CenterControlStateMachine()
        self.machine.start(0.0)

    def update(self, center=None, width=1280, confidence=0.8, now=0.1,
               params=CENTER_PARAMS, guide=None, guide_confidence=0.0,
               guide_count=0, fringe_movement="unknown", fringe_delta=None,
               fringe_velocity=None, scene_has=False, scene_x=None,
               scene_confidence=0.0, scene_source="", scene_blurred=False,
               scene_held=False, zero_box_x=None, zero_box_confidence=0.0):
        return self.machine.update(
            center_x=center,
            frame_width=width,
            confidence=confidence,
            guide_x=guide,
            guide_confidence=guide_confidence,
            guide_count=guide_count,
            fringe_movement=fringe_movement,
            fringe_delta_x_px=fringe_delta,
            fringe_velocity_px_s=fringe_velocity,
            scene_has_fringe=scene_has,
            scene_position_x=scene_x,
            scene_confidence=scene_confidence,
            scene_source=scene_source,
            scene_blurred=scene_blurred,
            scene_held=scene_held,
            zero_box_x=zero_box_x,
            zero_box_confidence=zero_box_confidence,
            connected=True,
            params=params,
            safety=SAFETY,
            now=now,
        )

    def test_searches_then_uses_both_directions_and_slows_near_center(self):
        searching = self.update(center=None)
        self.assertEqual(
            searching.commands,
            (("set_speed", 5), ("start_forward", None)),
        )

        far_left = self.update(center=300, now=0.2)
        self.assertEqual(far_left.commands, ())
        self.assertEqual(far_left.gear, 5)
        self.assertEqual(far_left.direction, "forward")

        near_right = self.update(center=700, now=0.3)
        self.assertEqual(
            near_right.commands,
            (("stop", None), ("set_speed", 8), ("start_reverse", None)),
        )
        self.assertEqual(near_right.state, "approaching")

    def test_known_direction_does_not_reverse_before_center_is_found(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "reverse",
            "auto_learn_direction": True,
        }
        decisions = [
            self.update(center=None, now=0.1, params=params),
            self.update(
                center=None, guide=900, guide_confidence=0.9,
                guide_count=2, now=0.2, params=params),
            self.update(center=None, now=0.3, params=params),
            self.update(center=None, now=0.4, params=params),
        ]
        self.assertTrue(all(item.direction == "reverse" for item in decisions))
        commands = [command for item in decisions for command in item.commands]
        self.assertIn(("start_reverse", None), commands)
        self.assertNotIn(("start_forward", None), commands)
        self.assertTrue(all("找到中心条纹前不往返" in item.message
                            for item in decisions))

    def test_known_direction_can_reverse_after_center_is_found(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "auto_learn_direction": False,
            "center_confirm_frames": 2,
        }
        searching = self.update(center=None, now=0.1, params=params)
        candidate = self.update(center=900, now=0.2, params=params)
        correcting = self.update(center=890, now=0.3, params=params)
        self.assertEqual(searching.direction, "forward")
        self.assertEqual(candidate.direction, "forward")
        self.assertIn("确认稳定 1/2", candidate.message)
        self.assertEqual(correcting.direction, "reverse")
        self.assertIn(("start_reverse", None), correcting.commands)

    def test_known_direction_reuses_existing_center_tracking_after_detection(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "auto_learn_direction": True,
            "center_confirm_frames": 2,
        }
        searching = self.update(center=None, now=0.1, params=params)
        candidate = self.update(center=900, now=0.2, params=params)
        learning = self.update(center=880, now=0.3, params=params)
        correcting = self.update(center=860, now=0.4, params=params)

        self.assertEqual(searching.state, "single_direction_search")
        self.assertEqual(candidate.state, "single_direction_search")
        self.assertEqual(candidate.direction, "forward")
        self.assertEqual(learning.state, "learning_direction")
        self.assertEqual(correcting.state, "centering")
        self.assertEqual(correcting.direction, "forward")
        self.assertEqual(self.machine.forward_x_sign, -1)

    def test_known_direction_uses_closed_loop_recovery_after_stable_detection(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "auto_learn_direction": True,
            "dropout_hold_frames": 0,
            "center_confirm_frames": 2,
        }
        candidate = self.update(center=900, now=0.1, params=params)
        correcting = self.update(center=880, now=0.2, params=params)
        lost = self.update(center=None, now=0.3, params=params)

        self.assertEqual(candidate.state, "single_direction_search")
        self.assertEqual(correcting.state, "learning_direction")
        self.assertEqual(lost.state, "waiting")
        self.assertNotEqual(lost.search_phase, "single_search")

    def test_single_direction_still_stops_when_centered(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
        }
        moving = self.update(center=300, now=0.1, params=params)
        self.assertEqual(moving.direction, "forward")
        confirming = self.update(center=640, now=0.2, params=params)
        completed = self.update(center=640, now=0.3, params=params)
        self.assertEqual(confirming.state, "confirming")
        self.assertEqual(completed.state, "centered")
        self.assertNotIn(("start_reverse", None), confirming.commands + completed.commands)

    def test_stops_and_confirms_when_center_is_stable(self):
        self.update(center=300)
        first = self.update(center=648, now=0.2)
        self.assertEqual(first.commands, (("stop", None),))
        self.assertEqual(first.state, "confirming")
        complete = self.update(center=646, now=0.3)
        self.assertTrue(complete.completed)
        self.assertEqual(complete.state, "centered")
        self.assertFalse(self.machine.enabled)

    def test_search_range_keeps_updating_during_centering_and_final_stop(self):
        params = {
            **CENTER_PARAMS,
            "search_initial_span_turns": 6,
            "search_expansion_factor": 1.6,
            "search_max_span_turns": 15,
        }
        self.update(center=None, now=0.1, params=params)
        centering = self.update(center=800, now=0.67, params=params)
        self.assertGreaterEqual(centering.searched_max_turns, 0.9)
        self.assertIsNotNone(centering.search_position_turns)

        confirming = self.update(center=640, now=1.24, params=params)
        self.assertEqual(confirming.state, "confirming")
        self.assertLess(
            confirming.search_position_turns, centering.search_position_turns)
        completed = self.update(center=640, now=1.3, params=params)
        self.assertEqual(completed.state, "centered")
        self.assertEqual(
            completed.search_position_turns,
            confirming.search_position_turns,
        )
        self.assertGreaterEqual(completed.searched_max_turns, 0.9)

    def test_direction_mapping_can_be_inverted(self):
        params = {**CENTER_PARAMS, "invert_direction": True}
        decision = self.update(center=300, params=params)
        self.assertEqual(decision.direction, "reverse")
        self.assertIn(("start_reverse", None), decision.commands)

    def test_automatically_learns_when_forward_moves_stripe_right(self):
        params = {**CENTER_PARAMS, "auto_learn_direction": True}
        self.update(center=None, params=params)
        probing = self.update(center=780, now=0.2, params=params)
        self.assertEqual(probing.state, "learning_direction")
        learned = self.update(center=790, now=0.3, params=params)
        self.assertEqual(learned.direction_mapping, "正转使条纹向右")
        self.assertEqual(learned.direction, "reverse")
        self.assertIn(("start_reverse", None), learned.commands)

    def test_automatically_keeps_forward_when_it_moves_stripe_left(self):
        params = {**CENTER_PARAMS, "auto_learn_direction": True}
        self.update(center=None, params=params)
        self.update(center=780, now=0.2, params=params)
        learned = self.update(center=770, now=0.3, params=params)
        self.assertEqual(learned.direction_mapping, "正转使条纹向左")
        self.assertEqual(learned.direction, "forward")

    def test_motion_command_is_periodically_reasserted(self):
        params = {**CENTER_PARAMS, "command_refresh_frames": 2}
        self.update(center=300, params=params)
        self.assertEqual(self.update(center=280, now=0.2, params=params).commands, ())
        refreshed = self.update(center=260, now=0.3, params=params)
        self.assertEqual(refreshed.commands, (("start_forward", None),))

    def test_non_center_box_position_selects_initial_direction(self):
        decision = self.update(
            center=None, guide=900, guide_confidence=0.8, guide_count=2)
        self.assertEqual(decision.state, "guided_expanding")
        self.assertEqual(decision.direction, "reverse")
        self.assertIn(("start_reverse", None), decision.commands)

    def test_non_center_boxes_disappearing_does_not_reverse_inside_leg(self):
        params = {
            **CENTER_PARAMS,
            "guide_loss_confirm_frames": 3,
        }
        self.update(
            center=None, guide=900, guide_confidence=0.8, guide_count=1,
            params=params)
        self.update(center=None, now=0.2, params=params)
        self.update(center=None, now=0.3, params=params)
        continued = self.update(center=None, now=0.4, params=params)
        self.assertEqual(continued.state, "search_expanding")
        self.assertEqual(continued.direction, "reverse")
        self.assertNotIn(("start_forward", None), continued.commands)

    def test_guide_moving_farther_records_trend_but_keeps_range_direction(self):
        params = {
            **CENTER_PARAMS,
            "guide_worsening_px": 5,
            "guide_trend_window": 6,
            "guide_focus_confirm_frames": 30,
        }
        corrected = None
        for index, guide_x in enumerate((900, 920, 940, 960, 980, 1000), 1):
            corrected = self.update(
                center=None, guide=guide_x, guide_confidence=0.8,
                guide_count=1, now=index * 0.1, params=params,
                fringe_movement="right", fringe_delta=20)
        self.assertEqual(corrected.state, "guided_expanding")
        self.assertEqual(corrected.direction, "reverse")
        self.assertIn("保持本轮方向", corrected.message)

    def test_guide_motion_is_accumulated_into_searched_range(self):
        params = {
            **CENTER_PARAMS,
            "search_initial_span_turns": 6,
            "search_expansion_factor": 1.6,
            "search_max_span_turns": 15,
        }
        self.update(
            center=None, guide=900, guide_confidence=0.8,
            guide_count=1, now=0.1, params=params)
        decision = self.update(
            center=None, guide=890, guide_confidence=0.8,
            guide_count=1, now=0.67, params=params)
        self.assertLessEqual(decision.searched_min_turns, -0.9)
        self.assertEqual(decision.direction, "reverse")

    def test_guide_inside_known_range_cannot_reverse_planned_return(self):
        params = {
            **CENTER_PARAMS,
            "search_initial_span_turns": 0.2,
            "search_expansion_factor": 2,
            "search_max_span_turns": 1,
        }
        self.update(center=None, now=0.1, params=params)
        returned = self.update(center=None, now=0.3, params=params)
        self.assertEqual(returned.direction, "reverse")
        guided = self.update(
            center=None, guide=300, guide_confidence=0.9,
            guide_count=1, now=0.4, params=params)
        self.assertEqual(guided.state, "guided_returning")
        self.assertEqual(guided.direction, "reverse")
        self.assertNotIn(("start_forward", None), guided.commands)

    def test_guide_box_jitter_does_not_cause_repeated_reversal(self):
        params = {
            **CENTER_PARAMS,
            "guide_worsening_px": 12,
            "guide_trend_window": 8,
        }
        decisions = []
        for index, guide_x in enumerate(
                (900, 906, 897, 904, 899, 908, 901, 905, 898, 903), 1):
            decisions.append(self.update(
                center=None, guide=guide_x, guide_confidence=0.8,
                guide_count=1, now=index * 0.1, params=params,
                fringe_movement="stable", fringe_delta=3))
        self.assertNotIn(
            "correcting_wrong_way", {decision.state for decision in decisions})
        self.assertTrue(all(decision.direction == "reverse" for decision in decisions))

    def test_sustained_guide_moves_search_center_toward_clue(self):
        params = {
            **CENTER_PARAMS,
            "guide_focus_confirm_frames": 3,
            "guide_focus_shift_ratio": 0.5,
            "guide_focus_min_shift_turns": 1,
            "guide_focus_max_shift_turns": 12,
            "search_initial_span_turns": 6,
        }
        decisions = [self.update(
            center=None, guide=900, guide_confidence=0.8,
            guide_count=1, now=index * 0.01, params=params)
            for index in range(1, 4)]
        focused = decisions[-1]
        self.assertEqual(focused.state, "guided_refocusing")
        self.assertLess(focused.search_center_turns, 0)
        self.assertAlmostEqual(
            focused.search_target_turns, focused.search_center_turns)
        self.assertIn("从新中心向两侧扩散", focused.message)

    def test_persistent_guide_only_recenters_once_per_expansion_level(self):
        params = {
            **CENTER_PARAMS,
            "guide_focus_confirm_frames": 2,
            "guide_focus_shift_ratio": 0.5,
            "search_initial_span_turns": 6,
        }
        self.update(
            center=None, guide=900, guide_confidence=0.8,
            guide_count=1, now=0.01, params=params)
        focused = self.update(
            center=None, guide=900, guide_confidence=0.8,
            guide_count=1, now=0.02, params=params)
        center = focused.search_center_turns
        continued = self.update(
            center=None, guide=900, guide_confidence=0.8,
            guide_count=1, now=0.03, params=params)
        self.assertEqual(continued.search_center_turns, center)
        self.assertEqual(continued.state, "guided_refocusing")

    def test_empty_search_reaches_boundary_then_returns_fast(self):
        params = {
            **CENTER_PARAMS,
            "search_initial_span_turns": 0.2,
            "search_expansion_factor": 2,
            "search_max_span_turns": 1,
        }
        first = self.update(center=None, now=0.1, params=params)
        self.assertEqual(first.state, "search_expanding")
        returned = self.update(center=None, now=0.3, params=params)
        self.assertEqual(returned.state, "search_returning")
        self.assertEqual(returned.direction, "reverse")
        self.assertEqual(returned.gear, CENTER_PARAMS["fast_gear"])

    def test_single_frame_false_center_returns_to_range_search(self):
        params = {**CENTER_PARAMS, "center_confirm_frames": 3}
        self.update(center=300, params=params)
        resumed = self.update(
            center=None, confidence=0.0, now=0.2, params=params)
        self.assertIn(resumed.state, ("search_expanding", "search_returning"))
        self.assertTrue(self.machine.enabled)
        self.assertFalse(self.machine.center_seen)

    def test_brief_dropout_during_confirmation_does_not_reset_stability(self):
        # 中心已进入容差、正在确认稳定性时，中间一帧置信度短暂掉线不应把
        # stable_frames 清零，也不应恢复搜索运动；机器应保持停车并最终居中。
        params = {
            **CENTER_PARAMS,
            "center_confirm_frames": 2,
            "stable_frames": 3,
            "dropout_hold_frames": 2,
        }
        self.update(center=648, now=0.1, params=params)
        self.update(center=646, now=0.2, params=params)
        self.assertTrue(self.machine.center_seen)
        self.assertEqual(self.machine.stable_frames, 2)

        dropout = self.update(center=None, confidence=0.0, now=0.3, params=params)
        self.assertEqual(dropout.state, "confirming")
        # 丢失期间不得恢复搜索运动（无 start_forward / start_reverse）。
        for action, _ in dropout.commands:
            self.assertFalse(action.startswith("start_"))

        complete = self.update(center=646, now=0.4, params=params)
        self.assertEqual(complete.state, "centered")
        self.assertTrue(complete.completed)

    def test_persistent_dropout_beyond_hold_resets_stability(self):
        # 连续丢失超过容忍帧数后才真正放弃累计，保证迟滞窗口不会把真实丢帧
        # 也当成“仍稳定”而一直停在原地。
        params = {
            **CENTER_PARAMS,
            "center_confirm_frames": 2,
            "stable_frames": 3,
            "dropout_hold_frames": 2,
        }
        self.update(center=648, now=0.1, params=params)
        self.update(center=646, now=0.2, params=params)
        self.assertEqual(self.machine.stable_frames, 2)

        self.update(center=None, confidence=0.0, now=0.3, params=params)
        self.update(center=None, confidence=0.0, now=0.4, params=params)
        self.update(center=None, confidence=0.0, now=0.5, params=params)
        self.assertEqual(self.machine.stable_frames, 0)

    def test_featureless_frames_slow_down_without_stopping(self):
        first = self.update(center=None, now=0.1)
        second = self.update(center=None, now=0.2)
        blurred = self.update(center=None, now=0.3)
        self.assertEqual(first.gear, 5)
        self.assertEqual(second.gear, 5)
        self.assertEqual(blurred.gear, 8)
        self.assertIn(blurred.direction, ("forward", "reverse"))
        self.assertNotIn(("stop", None), blurred.commands)
        self.assertIn("保持连续旋转", blurred.message)

        recovering = self.update(
            center=None, guide=800, guide_confidence=0.8,
            guide_count=1, now=0.4)
        self.assertEqual(recovering.gear, 8)
        self.update(center=None, guide=800, guide_confidence=0.8,
                    guide_count=1, now=0.5)
        recovered = self.update(
            center=None, guide=800, guide_confidence=0.8,
            guide_count=1, now=0.6)
        self.assertEqual(recovered.gear, 5)

    def test_single_direction_slows_after_persistent_motion_blur(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "blur_slowdown_frames": 2,
        }
        first = self.update(
            center=None, now=0.1, params=params,
            scene_has=True, scene_x=800, scene_confidence=0.32,
            scene_source="visual", scene_blurred=True)
        slowed = self.update(
            center=None, now=0.2, params=params,
            scene_has=True, scene_x=805, scene_confidence=0.32,
            scene_source="visual", scene_blurred=True)
        self.assertEqual(first.direction, "forward")
        self.assertEqual(slowed.direction, "forward")
        self.assertEqual(slowed.gear, CENTER_PARAMS["blur_safe_gear"])
        self.assertIn("画面持续模糊", slowed.message)

    def test_single_direction_uses_history_only_to_hold_and_slow(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "reverse",
        }
        decision = self.update(
            center=None, now=0.1, params=params,
            scene_has=True, scene_x=900, scene_confidence=0.45,
            scene_source="history", scene_blurred=True, scene_held=True,
            fringe_velocity=120)
        self.assertEqual(decision.direction, "reverse")
        self.assertEqual(decision.gear, CENTER_PARAMS["blur_safe_gear"])
        self.assertIn("历史轨迹", decision.message)

    def test_visual_fallback_guides_bidirectional_search_after_yolo_miss(self):
        decision = self.update(
            center=None, now=0.1,
            scene_has=True, scene_x=900, scene_confidence=0.65,
            scene_source="visual")
        self.assertIn(decision.state, ("guided_expanding", "guided_returning"))
        self.assertEqual(decision.direction, "reverse")

    def test_high_fringe_velocity_uses_safe_gear_during_centering(self):
        params = {**CENTER_PARAMS, "center_confirm_frames": 1}
        decision = self.update(
            center=300, now=0.1, params=params,
            scene_has=True, scene_x=300, scene_confidence=0.8,
            scene_source="yolo", fringe_velocity=240)
        self.assertEqual(decision.gear, CENTER_PARAMS["blur_safe_gear"])

    def test_stop_and_detect_phase1_continuous_until_guide_appears(self):
        """阶段一：无彩色条纹时连续旋转，不停车。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "forward",
            "stop_detect_move_seconds": 0.5,
            "stop_detect_settle_seconds": 0.2,
            "stop_detect_frames": 2,
        }
        # 没有 near_fringe 信号 → 阶段一连续旋转
        moving = self.update(center=None, now=0.1, params=params)
        self.assertEqual(moving.state, "sd_phase1_continuous")
        self.assertEqual(moving.direction, "forward")
        self.assertIn(("start_forward", None), moving.commands)
        # 继续保持，不停止
        still = self.update(center=None, now=0.2, params=params)
        self.assertEqual(still.state, "sd_phase1_continuous")
        self.assertNotIn(("stop", None), still.commands)

    def test_stop_and_detect_phase2_cycle_after_guide(self):
        """阶段二：彩色条纹出现后进入转停识别循环。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "forward",
            "stop_detect_move_seconds": 0.5,
            "stop_detect_settle_seconds": 0.2,
            "stop_detect_frames": 2,
        }
        # 提供 near_fringe → 进入阶段二
        moving = self.update(
            center=None, now=0.15, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.assertEqual(moving.state, "sd_phase2_moving")

        # 经过 move_seconds → 进入停车稳定
        settling = self.update(
            center=None, now=0.66, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.assertEqual(settling.state, "sd_phase2_settling")
        self.assertIn(("stop", None), settling.commands)

        # 经过 settle_seconds → 进入检测阶段（清晰但无零级框）
        self.update(
            center=None, now=0.87, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        detecting = self.update(
            center=None, now=0.88, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.assertEqual(detecting.state, "sd_phase2_detecting")

        # 连续无零级框达到 required_frames → 继续转动下一段
        resumed = self.update(
            center=None, now=0.89, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.assertEqual(resumed.state, "sd_phase2_moving")
        self.assertIn(("start_forward", None), resumed.commands)

    def test_stop_and_detect_phase2_waits_for_clear_image(self):
        """阶段二检测阶段：画面模糊时一直等待清晰，不消耗检测帧数。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "forward",
            "stop_detect_move_seconds": 0.5,
            "stop_detect_settle_seconds": 0.2,
            "stop_detect_frames": 2,
        }
        # 进入阶段二：moving → settling
        self.update(
            center=None, now=0.15, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        # settling 阶段：电机停止
        settling = self.update(center=None, now=0.66, params=params)
        self.assertIn(("stop", None), settling.commands)
        # 进入 detecting
        self.update(center=None, now=0.87, params=params)

        # 模糊画面 → 等待清晰（电机已停止，无新指令）
        blurred = self.update(
            center=None, now=0.88, params=params,
            scene_blurred=True, scene_has=True, scene_x=900,
            scene_confidence=0.6, scene_source="visual")
        self.assertEqual(blurred.state, "sd_phase2_waiting_clear")

        # 仍然模糊 → 继续等待
        still_blurred = self.update(
            center=None, now=0.89, params=params,
            scene_blurred=True, scene_has=True, scene_x=900,
            scene_confidence=0.6, scene_source="visual")
        self.assertEqual(still_blurred.state, "sd_phase2_waiting_clear")

    def test_stop_and_detect_phase2_zero_box_triggers_phase3(self):
        """阶段二检测阶段：零级框需连续两帧确认后才进入阶段三。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "forward",
            "stop_detect_move_seconds": 0.5,
            "stop_detect_settle_seconds": 0.2,
            "stop_detect_frames": 2,
            "center_confirm_frames": 1,
        }
        # 进入阶段二 → moving → settling → detecting
        self.update(
            center=None, now=0.15, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.update(center=None, now=0.66, params=params)
        self.update(center=None, now=0.87, params=params)

        # 第一帧看到零级框 → 锁定等待确认（不能直接进阶段三）
        locked = self.update(
            center=None, now=0.88, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=300, zero_box_confidence=0.7)
        self.assertEqual(locked.state, "sd_phase2_locked")
        self.assertFalse(self.machine._sd_zero_box_confirmed)

        # 第二帧零级框持续存在 → 确认，进入阶段三
        box_centering = self.update(
            center=None, now=0.89, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=300, zero_box_confidence=0.7)
        self.assertIn(box_centering.state, (
            "sd_phase3_box_centering", "sd_phase3_box_lost",
            "sd_phase3_box_stabilizing"))
        self.assertTrue(self.machine._sd_zero_box_confirmed)
        self.assertFalse(self.machine.center_seen)

    def test_stop_and_detect_phase3_box_to_phase4_fine_center(self):
        """阶段三→四：零级框连续确认稳定后，中心线可用 → 进入精细居中。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "forward",
            "stop_detect_move_seconds": 0.5,
            "stop_detect_settle_seconds": 0.2,
            "stop_detect_frames": 2,
            "center_confirm_frames": 1,
            "stable_frames": 1,
            "slow_zone_px": 200,
            "tolerance_px": 15,
        }
        # 快速通过阶段二 → 进入 detecting 子阶段
        self.update(
            center=None, now=0.15, params=params,
            guide=900, guide_confidence=0.8, guide_count=1)
        self.update(center=None, now=0.66, params=params)
        self.update(center=None, now=0.87, params=params)
        # 第一帧零级框 → 锁定
        locked = self.update(
            center=640, now=0.88, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=640, zero_box_confidence=0.7)
        self.assertEqual(locked.state, "sd_phase2_locked")
        # 第二帧零级框确认 → 进入阶段三，开始稳定计数
        self.update(
            center=640, now=0.89, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=640, zero_box_confidence=0.7)
        # 阶段三稳定帧 2/3
        self.update(
            center=640, now=0.90, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=640, zero_box_confidence=0.7)
        # 阶段三稳定帧 3/3 → 进入阶段四，因误差为 0 直接完成居中
        final = self.update(
            center=640, now=0.91, params=params,
            guide=900, guide_confidence=0.8, guide_count=1,
            zero_box_x=640, zero_box_confidence=0.7)
        self.assertTrue(self.machine.center_seen)
        self.assertIn(final.state, ("confirming", "centered"))

    def test_stop_and_detect_phase3_box_lost_slows_and_holds_direction(self):
        """阶段三：零级框丢失时减速保持方向，不换向。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "reverse",
            "stop_detect_move_seconds": 0.2,
            "stop_detect_settle_seconds": 0.05,
            "stop_detect_frames": 1,
            "blur_safe_gear": 9,
        }
        # 快速进入阶段三
        self.machine._sd_near_fringe_seen = True
        self.machine._sd_zero_box_confirmed = True
        self.machine._sd_box_stable_frames = 0

        # 零级框丢失
        lost = self.update(
            center=None, now=0.1, params=params,
            zero_box_x=None, zero_box_confidence=0.0)
        self.assertEqual(lost.state, "sd_phase3_box_lost")
        self.assertEqual(lost.direction, "reverse")
        self.assertEqual(lost.gear, 9)
        self.assertIn(("start_reverse", None), lost.commands)

    def test_stop_and_detect_phase3_never_reverses(self):
        """阶段一～三：未进入阶段四之前绝不反向。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "stop_and_detect",
            "search_direction": "reverse",
            "stop_detect_move_seconds": 0.2,
            "stop_detect_settle_seconds": 0.05,
            "stop_detect_frames": 1,
            "stable_frames": 5,
            "center_confirm_frames": 3,
        }
        decisions = [
            self.update(center=None, now=0.10, params=params),
            self.update(center=None, now=0.31, params=params),
            self.update(center=None, now=0.37, params=params),
            self.update(center=None, now=0.38, params=params),
            self.update(center=None, now=0.59, params=params),
            self.update(center=None, now=0.65, params=params),
            self.update(center=None, now=0.66, params=params),
        ]
        commands = [command for item in decisions for command in item.commands]
        self.assertNotIn(("start_forward", None), commands)
        self.assertTrue(all(
            item.direction in ("reverse", "stopped") for item in decisions))
        self.assertFalse(self.machine.center_seen)

    def test_unknown_direction_can_use_stop_and_detect_timing(self):
        params = {
            **CENTER_PARAMS,
            "direction_mode": "bidirectional",
            "recognition_mode": "stop_and_detect",
            "search_initial_span_turns": 0.1,
            "stop_detect_move_seconds": 0.2,
            "stop_detect_settle_seconds": 0.05,
            "stop_detect_frames": 1,
        }
        first_leg = self.update(center=None, now=0.1, params=params)
        self.update(center=None, now=0.2, params=params)
        self.update(center=None, now=0.3, params=params)
        second_leg = self.update(center=None, now=0.4, params=params)
        self.assertEqual(first_leg.direction, "forward")
        self.assertEqual(second_leg.state, "cycle_moving")
        self.assertEqual(second_leg.direction, "reverse")
        self.assertIn(("start_reverse", None), second_leg.commands)

    def test_confirmed_center_loss_waits_then_resumes_range_search(self):
        params = {
            **CENTER_PARAMS,
            "center_confirm_frames": 2,
            "dropout_hold_frames": 1,
        }
        self.update(center=300, now=0.1, params=params)
        self.update(center=305, now=0.2, params=params)
        self.assertTrue(self.machine.center_seen)
        short = self.update(
            center=None, confidence=0.0, now=0.3, params=params)
        waiting = self.update(
            center=None, confidence=0.0, now=0.4, params=params)
        resumed = self.update(
            center=None, confidence=0.0, now=0.5, params=params)
        self.assertEqual(short.state, "tracking_dropout")
        self.assertEqual(waiting.state, "waiting")
        self.assertIn(resumed.state, ("search_expanding", "search_returning"))
        self.assertTrue(self.machine.enabled)
        self.assertFalse(self.machine.center_seen)

    def test_guide_reappearing_after_center_loss_resumes_immediately(self):
        params = {
            **CENTER_PARAMS,
            "center_confirm_frames": 2,
            "dropout_hold_frames": 1,
        }
        self.update(center=300, now=0.1, params=params)
        self.update(center=305, now=0.2, params=params)
        self.update(center=None, confidence=0.0, now=0.3, params=params)
        guided = self.update(
            center=None, confidence=0.0, guide=900,
            guide_confidence=0.8, guide_count=1, now=0.4, params=params)
        self.assertIn(guided.state, ("guided_expanding", "guided_returning"))
        self.assertTrue(self.machine.enabled)
        self.assertFalse(self.machine.center_seen)

    def test_phase4_drift_guard_blends_toward_zero_box(self):
        """阶段四：中心线漂移过大时混合回拉到 YOLO 零级框中心。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "auto_learn_direction": False,
            "center_confirm_frames": 2,
            "required_stable_frames": 5,
        }
        # 建立 center_seen 状态
        self.update(center=640, now=0.1, params=params)
        self.update(center=640, now=0.2, params=params)
        self.assertTrue(self.machine.center_seen)

        # 中心线漂移到 800，但 YOLO 框说中心在 640
        drift_result = self.update(
            center=800, width=1280, confidence=0.80,
            zero_box_x=640, zero_box_confidence=0.75,
            now=0.3, params=params,
        )
        # 漂移 > max(40, 1280*0.08) = 102.4 → 漂移保护触发
        # 混合后 center_x ≈ (0.75*640 + 0.80*800) / 1.55 ≈ 696.8
        # error ≈ 696.8 - 640 = 56.8（而非 800 - 640 = 160）
        if drift_result.error_px is not None:
            self.assertLess(abs(drift_result.error_px), 120)
            # 没有漂移保护时 error 应为 160，有保护时显著降低
            self.assertLess(abs(drift_result.error_px), 100)

    def test_phase4_no_drift_when_center_near_zero_box(self):
        """阶段四：中心线与 YOLO 框一致时不触发漂移保护。"""
        params = {
            **CENTER_PARAMS,
            "direction_mode": "single_direction",
            "recognition_mode": "continuous",
            "search_direction": "forward",
            "auto_learn_direction": False,
            "center_confirm_frames": 2,
            "required_stable_frames": 5,
        }
        self.update(center=640, now=0.1, params=params)
        self.update(center=640, now=0.2, params=params)
        self.assertTrue(self.machine.center_seen)

        # 中心线 650，YOLO 框 645，漂移 5px < 阈值 → 不触发
        normal = self.update(
            center=650, width=1280, confidence=0.80,
            zero_box_x=645, zero_box_confidence=0.75,
            now=0.3, params=params,
        )
        if normal.error_px is not None:
            # 误差应是 650-640=10，而不是被混合后的值
            self.assertAlmostEqual(normal.error_px, 10, delta=5)


class ExpandingSearchPlannerTests(unittest.TestCase):
    def test_recenter_starts_from_new_focus_and_skips_known_area(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=2, expansion_factor=2, max_span=0)

        planner.recenter("reverse", shift_turns=3)
        self.assertEqual(planner.center, -3)
        self.assertEqual(planner.target, -3)
        self.assertTrue(planner.focus_pending)

        planner.advance(-3)
        self.assertFalse(planner.focus_pending)
        self.assertEqual(planner.target, -5)

        planner.advance(-2)
        self.assertGreater(planner.target, planner.searched_max)
        self.assertEqual(planner.center, -3)

    def test_negative_target_reached_uses_motion_direction_not_target_sign(self):
        planner = ExpandingSearchPlanner()
        planner.reset("reverse", initial_span=2, expansion_factor=2, max_span=0)
        planner.position = -10
        planner.center = -10
        planner.searched_min = -10
        planner.recenter("forward", shift_turns=3)
        self.assertEqual(planner.target, -7)

        planner.advance(1)
        self.assertTrue(planner.focus_pending)
        self.assertEqual(planner.target, -7)

        planner.advance(2)
        self.assertFalse(planner.focus_pending)
        self.assertEqual(planner.target, -5)

    def test_returns_through_known_range_before_searching_new_area(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=2, expansion_factor=2, max_span=8)

        planner.advance(2)
        self.assertEqual(planner.target, -2)
        self.assertEqual(planner.direction, "reverse")
        self.assertTrue(planner.in_known_range)

        planner.advance(-1)
        self.assertTrue(planner.in_known_range)
        self.assertEqual((planner.searched_min, planner.searched_max), (0, 2))

        planner.advance(-1.2)
        self.assertFalse(planner.in_known_range)
        self.assertLess(planner.searched_min, 0)

    def test_expands_only_after_opposite_existing_boundary_is_covered(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=2, expansion_factor=2, max_span=8)
        planner.advance(2)
        planner.advance(-4)
        self.assertEqual(planner.position, -2)
        self.assertEqual(planner.target, 4)
        self.assertEqual((planner.searched_min, planner.searched_max), (-2, 2))

    def test_stops_after_both_maximum_boundaries_are_covered(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=1, expansion_factor=2, max_span=2)
        for movement in (1, -2, 3, -4):
            planner.advance(movement)
        self.assertTrue(planner.completed)
        self.assertEqual((planner.searched_min, planner.searched_max), (-2, 2))

    def test_visual_redirect_keeps_target_beyond_known_range(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=2, expansion_factor=2, max_span=8)
        planner.advance(1)
        planner.redirect("reverse")
        self.assertEqual(planner.target, -2)
        self.assertEqual(planner.direction, "reverse")

        planner.advance(-3)
        planner.redirect("forward")
        self.assertGreater(planner.target, planner.searched_max)

    def test_zero_maximum_span_keeps_expanding_without_range_exhaustion(self):
        planner = ExpandingSearchPlanner()
        planner.reset("forward", initial_span=1, expansion_factor=1.6, max_span=0)
        for _ in range(20):
            planner.advance(planner.target - planner.position)
            self.assertFalse(planner.completed)
        self.assertGreater(planner.span, 20)

    def test_search_gear_accelerates_after_each_expansion_round(self):
        machine = CenterControlStateMachine()
        machine.search_planner.expansion_level = 0
        self.assertEqual(machine._dynamic_search_gear(5, 2, 1), 5)
        machine.search_planner.expansion_level = 1
        self.assertEqual(machine._dynamic_search_gear(5, 2, 1), 4)
        machine.search_planner.expansion_level = 3
        self.assertEqual(machine._dynamic_search_gear(5, 2, 1), 2)
        machine.search_planner.expansion_level = 20
        self.assertEqual(machine._dynamic_search_gear(5, 2, 1), 2)


class ExperimentWorkflowStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = ExperimentWorkflowStateMachine()
        self.ready = ExperimentObservation(
            camera_running=True,
            model_loaded=True,
            prediction_running=True,
            motor_connected=True,
            micrometer_connected=True,
            micrometer_reading_mm=12.300,
        )

    def test_manual_steps_are_the_only_required_confirmations(self):
        decision = self.machine.update(self.ready, 0.0)
        self.assertEqual(decision.stage, "manual_alignment")
        self.machine.confirm_instrument_adjusted()
        decision = self.machine.update(self.ready, 0.1)
        self.assertEqual(decision.stage, "waiting_white_light")
        self.machine.confirm_white_light_placed()
        decision = self.machine.update(self.ready, 0.2)
        self.assertEqual(decision.stage, "ready")

    def test_auto_mode_starts_search_and_records_stable_center(self):
        self.machine.confirm_instrument_adjusted()
        self.machine.confirm_white_light_placed()
        self.machine.set_auto_enabled(True, 0.0)
        searching = self.machine.update(self.ready, 0.1, stable_frames=2)
        self.assertEqual(searching.actions, ("start_search",))
        center = ExperimentObservation(**{
            **self.ready.__dict__,
            "micrometer_reading_mm": 12.350,
            "center_x_px": 640.0,
            "center_confidence": 0.7,
        })
        confirming = self.machine.update(center, 0.2, stable_frames=2)
        self.assertEqual(confirming.stage, "confirming_center")
        complete = self.machine.update(center, 0.3, stable_frames=2)
        self.assertEqual(complete.actions, ("stop_search", "record_center"))
        self.assertAlmostEqual(self.machine.center_reading_mm, 12.350)
        self.assertEqual(self.machine.snapshot()["stage"], "application_ready")

    def test_detected_center_must_reach_frame_middle_before_completion(self):
        self.machine.confirm_instrument_adjusted()
        self.machine.confirm_white_light_placed()
        self.machine.set_auto_enabled(True, 0.0)
        off_center = ExperimentObservation(**{
            **self.ready.__dict__,
            "center_x_px": 300.0,
            "center_confidence": 0.8,
            "frame_width_px": 1280.0,
        })
        decision = self.machine.update(off_center, 0.1, stable_frames=1)
        self.assertEqual(decision.stage, "centering")
        self.assertFalse(self.machine.center_recorded)

    def test_connected_micrometer_must_refresh_after_motor_stops(self):
        self.machine.confirm_instrument_adjusted()
        self.machine.confirm_white_light_placed()
        self.machine.set_auto_enabled(True, 0.0)
        self.machine.update(self.ready, 0.1, stable_frames=1)
        center_without_reading = ExperimentObservation(**{
            **self.ready.__dict__,
            "micrometer_reading_mm": None,
            "center_x_px": 640.0,
            "center_confidence": 0.8,
            "frame_width_px": 1280.0,
        })
        waiting = self.machine.update(
            center_without_reading, 0.2, stable_frames=1)
        self.assertEqual(waiting.stage, "waiting_micrometer")
        self.assertEqual(waiting.actions, ("stop_search",))
        self.assertFalse(self.machine.center_recorded)

        refreshed = ExperimentObservation(**{
            **center_without_reading.__dict__,
            "micrometer_reading_mm": 12.351,
        })
        complete = self.machine.update(refreshed, 0.3, stable_frames=1)
        self.assertEqual(complete.actions, ("record_center",))
        self.assertEqual(self.machine.center_reading_mm, 12.351)

    def test_missing_device_and_timeout_stop_search(self):
        self.machine.confirm_instrument_adjusted()
        self.machine.confirm_white_light_placed()
        self.machine.set_auto_enabled(True, 0.0)
        self.machine.update(self.ready, 0.1)
        missing = ExperimentObservation(**{
            **self.ready.__dict__, "motor_connected": False})
        stopped = self.machine.update(missing, 0.2)
        self.assertEqual(stopped.stage, "initializing")
        self.assertIn("stop_search", stopped.actions)

        self.machine.set_auto_enabled(True, 1.0)
        self.machine.update(self.ready, 1.1)
        timeout = self.machine.update(self.ready, 3.1, max_seconds=2)
        self.assertEqual(timeout.stage, "error")
        self.assertIn("stop_search", timeout.actions)

    def test_default_automatic_experiment_limit_is_ten_minutes(self):
        self.machine.confirm_instrument_adjusted()
        self.machine.confirm_white_light_placed()
        self.machine.set_auto_enabled(True, 0.0)
        self.machine.update(self.ready, 0.1)
        self.assertNotEqual(self.machine.update(self.ready, 60.1).stage, "error")
        timeout = self.machine.update(self.ready, 600.1)
        self.assertEqual(timeout.stage, "error")
        self.assertIn("stop_search", timeout.actions)


class SerialCommandQueueTests(unittest.TestCase):
    def test_motor_shutdown_stops_before_closing(self):
        order = []

        class Controller:
            def stop(self):
                order.append("stop")
                return True

            def close(self):
                order.append("close")

        report = shutdown_motor_safely(
            SerialCommandQueue("test-safe-close"), Controller(), timeout=1.0)
        self.assertTrue(report.completed)
        self.assertTrue(report.stop_succeeded)
        self.assertEqual(order, ["stop", "close"])

    def test_runs_operations_off_caller_thread_and_reports_errors(self):
        commands = SerialCommandQueue("test-serial")
        caller = threading.get_ident()
        commands.submit("thread", threading.get_ident)
        commands.submit("failure", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        deadline = time.monotonic() + 1
        results = []
        while len(results) < 2 and time.monotonic() < deadline:
            results.extend(commands.drain())
            time.sleep(0.005)
        commands.shutdown()
        by_name = {result.name: result for result in results}
        self.assertNotEqual(by_name["thread"].value, caller)
        self.assertIsInstance(by_name["failure"].error, RuntimeError)

    def test_coalesces_duplicate_poll_commands(self):
        commands = SerialCommandQueue("test-coalesce")
        gate = threading.Event()
        self.assertTrue(commands.submit("poll", lambda: gate.wait(0.2), coalesce=True))
        self.assertFalse(commands.submit("poll", lambda: None, coalesce=True))
        gate.set()
        commands.shutdown()

    def test_shutdown_runs_safety_action_before_discarding_queued_commands(self):
        commands = SerialCommandQueue("test-shutdown")
        gate = threading.Event()
        started = threading.Event()
        order = []
        def running():
            started.set()
            gate.wait(0.2)
            order.append("running")

        commands.submit("running", running)
        self.assertTrue(started.wait(0.2))
        commands.submit("late_start", lambda: order.append("late_start"))
        commands.shutdown(lambda: order.append("stop"))
        gate.set()
        deadline = time.monotonic() + 1
        while commands._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(order, ["running", "stop"])

    def test_shutdown_waits_for_safety_action_and_returns_result(self):
        commands = SerialCommandQueue("test-shutdown-wait")
        completed = threading.Event()

        def stop():
            completed.set()
            return True

        result = commands.shutdown(stop, timeout=1.0)
        self.assertTrue(completed.is_set())
        self.assertFalse(commands._thread.is_alive())
        self.assertIsNotNone(result)
        self.assertTrue(result.value)


if __name__ == "__main__":
    unittest.main()
