"""Live extrema, fresh alignment cues, no recording, and the one-step UI."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from app.services.perpendicular import PerpendicularService


class _AlignmentCase:
    frame = (640, 480)

    def setUp(self):
        logger_patch = patch('app.services.perpendicular._logger')
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.service = PerpendicularService()
        self.service.set_extreme('left')
        self.service.start(3)
        self.at = 1000.

    def feed(self, values, confidence=.8):
        for x in values:
            self.at += .03
            self.service.observe(x, confidence, self.at, self.frame, now=self.at)

    def state(self):
        return self.service.snapshot(now=self.at)


class AlignmentTests(_AlignmentCase, unittest.TestCase):
    def test_left_reference_updates_immediately_without_return_sweep(self):
        self.feed([140, 130, 120, 110, 100, 100, 100])
        self.assertEqual(self.service.reference_x, 100)
        self.feed([120, 140])
        self.assertEqual(self.service.reference_x, 100)
        self.assertFalse(self.state()['aligned'])
        self.feed([101] * 5)
        self.assertTrue(self.state()['aligned'])

    def test_right_reference_and_signed_delta_use_original_coordinates(self):
        self.service.set_extreme('right')
        self.assertTrue(self.service.active)
        self.feed([460, 480, 500, 500, 500, 480])
        self.assertEqual(self.service.reference_x, 500)
        self.assertEqual(self.state()['delta_px'], -20)
        self.feed([500] * 5)
        self.assertTrue(self.state()['aligned'])

    def test_single_frame_wrong_extremum_is_not_used(self):
        self.feed([140, 140, 140, 20, 140, 130])
        self.assertEqual(self.service.reference_x, 140)
        self.assertEqual(self.state()['rejected_positions'], 1)

    def test_reference_requires_three_distinct_fresh_observations(self):
        self.feed([100])
        self.assertIsNone(self.service.reference_x)
        self.assertFalse(self.state()['aligned'])
        for _ in range(10):
            self.service.observe(100, .8, self.at, (640, 480), now=self.at)
        self.assertFalse(self.state()['aligned'])
        self.feed([100, 100])
        self.assertEqual(self.service.reference_x, 100)
        self.feed([100, 100])
        self.assertTrue(self.state()['aligned'])
        self.feed([120])
        self.assertFalse(self.state()['aligned'])

    def test_new_distant_extremum_clears_old_alignment_votes(self):
        self.feed([140] * 5)
        self.assertTrue(self.state()['aligned'])
        self.feed([120, 110, 100, 100, 100])
        self.assertEqual(self.service.reference_x, 100)
        self.assertFalse(self.state()['aligned'])

    def test_invalid_and_stale_observations_do_not_update_reference(self):
        for value, confidence in [(None, .8), (20, .1), (float('nan'), .8), (-2, .8), (650, .8)]:
            self.feed([100] * 5)
            self.feed([value], confidence)
            self.assertEqual(self.service.reference_x, 100)
            self.assertFalse(self.state()['aligned'])
        self.service.observe(20, .8, self.at - 2, (640, 480), now=self.at)
        self.assertEqual(self.service.reference_x, 100)
        self.feed([100] * 5)
        self.assertFalse(self.service.snapshot(now=self.at + 2)['aligned'])

    def test_monotonic_camera_timestamps_are_accepted(self):
        self.service.observe(100, .8, time.monotonic(), (640, 480))
        self.assertEqual(len(self.service.positions), 1)

    def test_reset_stop_and_direction_change_clear_extreme(self):
        self.feed([100] * 5)
        generation = self.service.generation
        self.service.set_extreme('right')
        self.assertGreater(self.service.generation, generation)
        self.assertTrue(self.service.active)
        self.assertIsNone(self.service.reference_x)
        self.assertEqual(self.service.tolerance_px, 3)
        self.feed([500])
        self.service.start()
        self.assertIsNone(self.service.reference_x)
        self.assertEqual(self.service.extreme, 'right')
        self.service.stop()
        self.assertFalse(self.service.active)
        self.assertIsNone(self.service.reference_x)
        with self.assertRaises(ValueError):
            self.service.set_extreme('up')

    def test_tolerance_changes_apply_without_resetting_reference(self):
        self.feed([100] * 5 + [120])
        self.service.configure_tolerance(30)
        self.feed([120] * 3)
        self.assertTrue(self.state()['aligned'])
        self.assertEqual(self.service.reference_x, 100)
        self.service.configure_tolerance(3)
        self.assertFalse(self.state()['aligned'])
        for bad in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.service.configure_tolerance(bad)

    def test_geometry_change_invalidates_reference(self):
        self.feed([100] * 5)
        self.at += .03
        self.service.observe(100, .8, self.at, (1280, 960), now=self.at)
        self.assertFalse(self.service.active)
        self.assertIsNone(self.service.reference_x)

    def test_long_tracking_keeps_bounded_position_set_and_no_images(self):
        for _ in range(10000):
            self.feed([100])
        self.assertLessEqual(len(self.service.positions), self.service.max_positions)
        self.assertLessEqual(len(self.service.positions), 4001)
        self.assertEqual(len(self.service.positions), len(self.service._position_values))
        self.assertEqual(self.service.reference_x, 100)
        self.assertFalse(any(isinstance(x, np.ndarray) for x in vars(self.service).values()))

    def test_repeated_wrong_stripe_jumps_never_become_right_reference(self):
        self.service.set_extreme('right')
        self.feed([400] * 5)
        self.feed([520] * 60)
        self.assertEqual(self.service.reference_x, 400)
        self.assertEqual(self.state()['rejected_positions'], 60)
        self.feed([400] * 3)
        self.assertTrue(self.state()['aligned'])

    def test_wrong_first_detection_is_not_a_permanent_extreme(self):
        self.service.set_extreme('right')
        self.feed([600, 390, 400, 400, 400])
        self.assertEqual(self.service.reference_x, 400)

    def test_supported_extreme_expires_from_rolling_position_set(self):
        self.feed([100] * 5 + [120] * 5)
        self.assertEqual(self.service.reference_x, 100)
        self.at += self.service.position_window_s + 1
        self.feed([130] * 5)
        self.assertEqual(self.service.reference_x, 130)
        self.assertEqual(len(self.service.positions), 5)

    def test_genuine_gap_allows_reacquiring_a_distant_track(self):
        self.service.set_extreme('right')
        self.feed([400] * 5)
        self.at += 2
        self.feed([520] * 5)
        self.assertEqual(self.service.reference_x, 520)

    def test_default_right_and_camera_zero(self):
        from app.core.config import AppConfig, load_config
        self.assertEqual(PerpendicularService().extreme, 'right')
        self.assertEqual(AppConfig().camera_index, 0)
        self.assertEqual(load_config().camera_index, 0)
        self.assertEqual(AppConfig({'camera': {'index': 2}}).camera_index, 2)

    def test_continuous_box_supported_right_move_reacquires_at_30fps(self):
        self.service.set_extreme('right')
        self.feed([400] * 5)
        for _ in range(16):
            self.at += 1/30
            self.service.observe(520, .6, self.at, (640, 480), now=self.at,
                                 zero_box_x=522, zero_box_confidence=.8,
                                 zero_box_half_width=20)
        self.assertEqual(self.service.reference_x, 520)
        self.assertEqual(self.state()['current_x'], 520)
        self.assertEqual(self.state()['reacquisitions'], 1)
        self.assertTrue(self.state()['aligned'])

    def test_wrong_box_cannot_corroborate_distant_jump(self):
        self.service.set_extreme('right')
        self.feed([400] * 5)
        for _ in range(20):
            self.at += .1
            self.service.observe(520, .8, self.at, (640, 480), now=self.at,
                                 zero_box_x=400, zero_box_confidence=.9,
                                 zero_box_half_width=20)
        self.assertEqual(self.service.reference_x, 400)
        self.assertEqual(self.state()['reacquisitions'], 0)

    def test_separated_wrong_points_cannot_form_right_reference(self):
        self.service.set_extreme('right')
        self.feed([400]*5 + [450] + [400]*5 + [451] + [400]*5 + [449])
        self.assertEqual(self.service.reference_x, 400)

    def test_subpixel_reference_updates_keep_alignment_votes(self):
        self.service.set_extreme('right')
        self.feed([400+i*.05 for i in range(15)])
        self.assertTrue(self.state()['aligned'])
        self.assertAlmostEqual(self.service.reference_x, 400.7)

    def test_early_warning_is_wider_than_arrival_and_expires(self):
        self.service.set_extreme('right')
        self.feed([400]*5 + [360]*3)
        state = self.state()
        self.assertTrue(state['approaching'])
        self.assertFalse(state['aligned'])
        self.assertEqual(state['approach_tolerance_px'], 100)
        self.feed([320]*3)
        self.assertTrue(self.state()['approaching'])
        self.feed([280]*3)
        self.assertFalse(self.state()['approaching'])
        self.service.configure_approach_tolerance(140)
        self.assertTrue(self.state()['approaching'])
        self.assertFalse(self.service.snapshot(now=self.at+2)['approaching'])

    def test_center_confidence_matches_live_recognition_threshold(self):
        self.service.set_extreme('right')
        self.feed([400]*5, confidence=.25)
        self.assertEqual(self.service.reference_x, 400)
        self.assertTrue(self.state()['aligned'])

    def test_early_warning_text_is_not_green_arrival(self):
        from app.ui.widgets.perpendicular import alignment_text
        self.service.set_extreme('right')
        self.feed([400]*5 + [360]*3)
        text, aligned = alignment_text(self.state())
        self.assertIn('快到垂直', text)
        self.assertFalse(aligned)

    def test_reacquisition_uses_recent_stability_after_initial_jitter(self):
        self.service.set_extreme('right')
        self.feed([400]*5)
        for x in [520, 530, 540, 550] + [550]*12:
            self.at += 1/30
            self.service.observe(x, .6, self.at, (640, 480), now=self.at,
                                 zero_box_x=x, zero_box_confidence=.8, zero_box_half_width=20)
        self.assertEqual(self.service.reference_x, 550)
        self.assertEqual(self.state()['reacquisitions'], 1)

    def test_moderate_confidence_box_reacquires_with_longer_confirmation(self):
        self.service.set_extreme('right')
        self.feed([400]*5)
        for _ in range(30):
            self.at += 1/30
            self.service.observe(520, .25, self.at, (640, 480), now=self.at,
                                 zero_box_x=521, zero_box_confidence=.4, zero_box_half_width=20)
        self.assertEqual(self.service.reference_x, 520)
        self.assertEqual(self.state()['reacquisitions'], 1)

    def test_expired_reference_does_not_freeze_the_motion_anchor(self):
        self.service.set_extreme('right')
        self.service.position_window_s = .4
        for _ in range(5):
            self.at += .03
            self.service.observe(400, .8, self.at, (1920, 1080), now=self.at)
        for i in range(1, 51):
            self.at += .03
            self.service.observe(400+8*i, .8, self.at, (1920, 1080), now=self.at)
        self.assertIsNone(self.service.reference_x)
        self.assertEqual(self.state()['current_x'], 800)
        self.assertEqual(self.state()['rejected_positions'], 0)

    def test_search_approach_and_arrival_use_three_distinct_colors(self):
        from app.ui.widgets.perpendicular import alignment_palette
        colors = [alignment_palette(dict(mode='tracking', aligned=aligned, approaching=near))
                  for aligned, near in [(False, False), (False, True), (True, False)]]
        self.assertEqual(len(set(colors)), 3)
        self.assertEqual(colors[0][0], '#1d4ed8')
        self.assertEqual(colors[1][0], '#b45309')
        self.assertEqual(colors[2][0], '#15803d')


class UnreachableExtremeTests(_AlignmentCase, unittest.TestCase):
    """现场 12:02 的情形：扫过 935 后停在 919，参考被钉住两分钟，提示永远差 16 px。

    现在只要当前轨迹稳定停留、旧极限又超过保持期没有复现，就放宽到可复现的极限。
    """

    frame = (1920, 1080)

    def setUp(self):
        super().setUp()
        self.service.start(15)  # 现场默认允许误差

    def sweep_then_settle(self):
        self.service.set_extreme('right')
        self.feed([760, 790, 820, 850, 880, 910, 935, 935, 935, 932, 919])
        self.assertEqual(self.service.reference_x, 935)
        self.assertFalse(self.state()['aligned'])

    def test_transient_extreme_is_released_after_the_track_settles(self):
        self.sweep_then_settle()
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        self.feed([919] * 150)
        state = self.state()
        self.assertEqual(state['released_extremes'], 1)
        self.assertEqual(state['released_x'], 935)
        self.assertEqual(self.service.reference_x, 919)
        self.assertEqual(state['delta_px'], 0)
        self.assertTrue(state['aligned'])
        self.assertTrue(state['release_note'])

    def test_release_drops_the_votes_of_the_old_reference(self):
        self.sweep_then_settle()
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        for _ in range(200):
            self.at += .03
            self.service.observe(919, .8, self.at, self.frame, now=self.at)
            if self.state()['released_extremes']:
                break
        state = self.state()
        self.assertEqual(state['released_extremes'], 1)
        self.assertFalse(state['aligned'])

    def test_reachable_extreme_survives_a_stable_dwell(self):
        self.service.set_extreme('right')
        self.feed([400] * 5)
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        self.feed([398] * 150)
        state = self.state()
        self.assertEqual(state['released_extremes'], 0)
        self.assertEqual(self.service.reference_x, 400)
        self.assertTrue(state['aligned'])

    def test_extreme_is_kept_until_the_hold_expires(self):
        self.service.extreme_hold_s = 30
        self.sweep_then_settle()
        self.at += 2 * self.service.stable_dwell_s
        self.feed([919] * 150)
        self.assertEqual(self.state()['released_extremes'], 0)
        self.assertEqual(self.service.reference_x, 935)
        self.service.extreme_hold_s = 8
        self.feed([919] * 150)
        self.assertEqual(self.state()['released_extremes'], 1)

    def test_swing_toward_the_extreme_is_never_released(self):
        self.service.set_extreme('right')
        self.feed([935] * 5)
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        for i in range(60):
            self.at += .1
            self.service.observe(880 + i * 2, .8, self.at, self.frame, now=self.at)
        self.assertEqual(self.state()['released_extremes'], 0)
        self.assertEqual(self.service.reference_x, 998)

    def test_left_extreme_release_is_symmetric(self):
        self.feed([1000, 970, 940, 910, 880, 850, 820, 790, 760, 730,
                   700, 690, 680, 670, 665, 665, 665, 668, 672])
        self.assertEqual(self.service.reference_x, 665)
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        self.feed([681] * 150)
        self.assertEqual(self.state()['released_extremes'], 1)
        self.assertEqual(self.service.reference_x, 681)
        self.assertTrue(self.state()['aligned'])

    def test_far_away_extreme_is_not_released(self):
        self.service.set_extreme('right')
        self.feed([935] * 5)
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        self.feed([871] * 150)
        state = self.state()
        self.assertEqual(state['released_extremes'], 0)
        self.assertEqual(self.service.reference_x, 935)
        self.assertFalse(state['aligned'])

    def test_release_note_expires_from_the_status_text(self):
        from app.ui.widgets.perpendicular import alignment_text
        self.sweep_then_settle()
        self.at += self.service.extreme_hold_s + 2 * self.service.stable_dwell_s
        self.feed([919] * 150)
        text, _ = alignment_text(self.state())
        self.assertIn('已回落到可复现位置', text)
        later = self.service.snapshot(now=self.at + self.service.release_note_s + 1)
        self.assertFalse(later['release_note'])
        self.assertNotIn('已回落到可复现位置', alignment_text(later)[0])


class AlignmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        logger_patch = patch('app.services.perpendicular._logger')
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def test_automatic_motion_invalidates_manual_reference(self):
        from app.ui.main_window import MainWindow
        service = PerpendicularService()
        service.start()
        window = NS(perpendicular=service, search=NS(active=True))
        MainWindow._pump_detection(window)
        self.assertFalse(service.active)

    def test_live_worker_is_used_without_detection_toggle_or_recording(self):
        from app.ui.main_window import MainWindow
        service = PerpendicularService()
        service.start()
        frame = np.zeros((40, 60, 3), np.uint8)
        at = time.monotonic()
        camera = NS(latest_sample=Mock(return_value=(frame, at)))
        window = NS(perpendicular=service, search=NS(active=False), backlash=NS(active=False),
                    state=NS(detecting=False, update=Mock()), detect=NS(is_loaded=True),
                    _detect_busy=False, _current_frame=frame, _vision_epoch=0,
                    camera=camera, _live_observe=Mock(), ctx=NS(run_async=Mock()))
        with patch('tempfile.TemporaryDirectory', side_effect=AssertionError('no recording')):
            MainWindow._pump_detection(window)
            done = window.ctx.run_async.call_args.kwargs['on_done']
            done(dict(center_x=25., confidence=.8))
        self.assertEqual(len(service.positions), 1)
        self.assertIsNone(service.reference_x)
        service.start()
        done(dict(center_x=10., confidence=.8))
        self.assertIsNone(service.reference_x)

    def test_card_one_step_workflow_and_live_direction_switch(self):
        from PySide6.QtWidgets import QApplication, QPushButton
        from app.ui.widgets.perpendicular import PerpendicularCard, alignment_text
        app = QApplication.instance() or QApplication([])
        service = PerpendicularService()
        ctx = NS(perpendicular=service, camera=NS(is_open=True), detect=NS(is_loaded=True),
                 search=NS(active=False), target=NS(active=False), backlash=NS(active=False),
                 timeline=Mock(), log=Mock())
        card = PerpendicularCard(ctx)
        try:
            self.assertEqual(card.extreme.currentData(), 'right')
            self.assertEqual(len(card.findChildren(QPushButton)), 2)
            self.assertFalse(hasattr(card, 'correct'))
            self.assertFalse(hasattr(card, 'repeatability'))
            with patch('tempfile.TemporaryDirectory', side_effect=AssertionError('no recording')):
                card._start()
                self.assertTrue(service.active)
                at = time.monotonic() - .1
                for i in range(5):
                    service.observe(100, .8, at + i * .01, (640, 480))
                card.refresh()
                self.assertTrue(alignment_text(service.snapshot())[1])
                card.extreme.setCurrentIndex(card.extreme.findData('left'))
                self.assertTrue(service.active)
                self.assertIsNone(service.reference_x)
                card.tolerance.setValue(30)
                self.assertEqual(service.tolerance_px, 30)
                card._stop()
                self.assertFalse(service.active)
            ctx.timeline.add.assert_not_called()
        finally:
            card.timer.stop()
            card.close()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
