"""Track supported extrema of a rolling center-position set, in full-frame pixels."""
from bisect import bisect_left, insort
from collections import deque
import math
import time
import logging

_logger = logging.getLogger('michelson')


class PerpendicularService:
    max_age_s = 1.0
    position_window_s = 120.0
    max_positions = 8192
    min_support = 3
    support_radius_px = 12.0
    # A fast sweep can record a position the settled track never reproduces.
    # Such an extreme stays the target only while it was observed recently;
    # after this hold plus a stable dwell it is released instead of blocking
    # the cue for the whole rolling window. Only nearby extremes are released:
    # a far away one may simply be a target the hand has not reached yet.
    extreme_hold_s = 15.0
    stable_dwell_s = 5.0
    release_note_s = 6.0

    def __init__(self):
        self.extreme = 'right'
        self.tolerance_px = 15.0
        self.approach_tolerance_px = 100.0
        self.reset()

    def reset(self, message='请选择极限方向，点击开始后实时识别'):
        self.generation = getattr(self, 'generation', 0) + 1
        self.mode = 'idle'
        self.message = message
        self.reference_x = None
        self.current_x = None
        self._last_at = 0.0
        self._geometry = None
        self._votes = 0
        self._aligned = False
        self.positions = deque()
        self._position_values = []
        self._track_x = None
        self._track_at = 0.0
        self._candidate = deque(maxlen=32)
        self._support = deque(maxlen=self.min_support)
        self._supported = deque()
        self._supported_values = []
        self.rejected_positions = 0
        self.reacquisitions = 0
        self.released_extremes = 0
        self.released_x = None
        self._released_at = None
        self.reject_reason = ''
        self._last_log_at = None

    def _update_reference(self, now):
        while self.positions and now - self.positions[0][0] > self.position_window_s:
            _, x = self.positions.popleft()
            self._position_values.pop(bisect_left(self._position_values, x))
        while self._supported and now - self._supported[0][0] > self.position_window_s:
            _, x = self._supported.popleft()
            self._supported_values.pop(bisect_left(self._supported_values, x))
        values = self._supported_values
        reference = (values[0] if self.extreme == 'left' else values[-1]) if values else None
        if (reference is None or self.reference_x is None
                or abs(reference - self.reference_x) > self.tolerance_px):
            self._votes = 0
            self._aligned = False
        self.reference_x = reference

    def _settled(self, now):
        """Recent accepted positions cover the dwell window as one tight cluster."""
        cutoff = now - self.stable_dwell_s
        recent = [(captured_at, x) for captured_at, x in reversed(self.positions)
                  if captured_at >= cutoff]
        return (len(recent) >= 5
                and now - recent[-1][0] >= .8 * self.stable_dwell_s
                and max(x for _, x in recent) - min(x for _, x in recent) <= 1.5 * self.support_radius_px)

    def _release_unreachable_extreme(self, now):
        """Drop an extreme the settled track can no longer reach.

        A quick sweep past the target can leave a supported position that the
        steady track never reproduces (ambiguous stripe or motion blur). Keeping
        it would pin the cue short of the target for the whole rolling window.
        """
        values = self._supported_values
        if len(values) < 2 or self.current_x is None or not self._settled(now):
            return
        reference = values[0] if self.extreme == 'left' else values[-1]
        gap = abs(self.current_x - reference)
        if gap <= self.tolerance_px or gap > 2 * self.tolerance_px:
            return
        observed = [captured_at for captured_at, x in self._supported
                    if abs(x - reference) <= self.support_radius_px]
        if not observed or now - max(observed) < self.extreme_hold_s:
            return
        self._drop_supported(lambda x: abs(x - reference) <= self.support_radius_px)
        self.released_extremes += 1
        self.released_x = reference
        self._released_at = now
        self._votes = 0
        self._aligned = False
        _logger.info('找垂直放宽不可达极限: reference=%.2f current=%.2f 保持%.1f秒未复现',
                     reference, self.current_x, now - max(observed))
        self._update_reference(now)

    def _drop_supported(self, should_drop):
        kept = [(captured_at, x) for captured_at, x in self._supported
                if not should_drop(x)]
        for captured_at, x in self._supported:
            if should_drop(x):
                self._supported_values.pop(bisect_left(self._supported_values, x))
        self._supported = deque(kept)

    @property
    def active(self):
        return self.mode == 'tracking'

    @property
    def extreme_label(self):
        return '最左' if self.extreme == 'left' else '最右'

    def configure_tolerance(self, tolerance_px):
        tolerance_px = float(tolerance_px)
        if not math.isfinite(tolerance_px) or tolerance_px <= 0:
            raise ValueError('允许误差必须是正的有限像素数')
        if tolerance_px != self.tolerance_px:
            self._votes = 0
            self._aligned = False
        self.tolerance_px = tolerance_px

    def configure_approach_tolerance(self, tolerance_px):
        tolerance_px = float(tolerance_px)
        if not math.isfinite(tolerance_px) or tolerance_px <= 0:
            raise ValueError('提前提醒范围必须是正的有限像素数')
        self.approach_tolerance_px = tolerance_px

    def set_extreme(self, extreme):
        if extreme not in ('left', 'right'):
            raise ValueError('极限方向必须选择最左或最右')
        if extreme != self.extreme:
            running = self.active
            self.extreme = extreme
            self.reset(f'已选择{self.extreme_label}极限')
            if running:
                self.start()

    def start(self, tolerance_px=None):
        self.configure_tolerance(self.tolerance_px if tolerance_px is None else tolerance_px)
        self.reset()
        self.mode = 'tracking'
        self.message = f'实时寻找{self.extreme_label}位置'
        _logger.info('找垂直启动: extreme=%s tolerance=%.1f approach=%.1f',
                                           self.extreme, self.tolerance_px, self.approach_tolerance_px)

    def stop(self):
        self.reset('找垂直已停止')

    def _miss(self):
        self.current_x = None
        self._votes = 0
        self._aligned = False
        self._support.clear()

    def observe(self, center_x, confidence, captured_at, frame_size, *, now=None,
                zero_box_x=None, zero_box_confidence=None, zero_box_half_width=0):
        if not self.active:
            return
        now = time.monotonic() if now is None else now
        if (not isinstance(captured_at, (int, float)) or not math.isfinite(captured_at)
                or not 0 <= now - captured_at <= self.max_age_s):
            self.reject_reason = '观测已过期'
            self._candidate.clear()
            self._miss()
            return
        if captured_at <= self._last_at:
            return
        if self._last_at and captured_at - self._last_at > self.max_age_s:
            self._miss()
            self._track_x = None
            self._candidate.clear()
        self._last_at = captured_at
        geometry = tuple(frame_size)
        if self._geometry is not None and geometry != self._geometry:
            self.reset('画面尺寸已改变，请重新开始寻找极限')
            return
        self._geometry = geometry
        self._update_reference(captured_at)
        if (center_x is None or confidence is None or not math.isfinite(center_x)
                or not math.isfinite(confidence) or not .18 <= confidence <= 1
                or not 0 <= center_x < geometry[0]):
            if captured_at - self._track_at > self.max_age_s:
                self._track_x = None
            self._miss()
            self._candidate.clear()
            self.reject_reason = '未获得有效中心条纹'
            return
        # Once a track is established, a fresh large jump is a different-stripe candidate.
        # After a genuine observation gap, permit reacquisition instead of locking forever.
        jump_limit = max(60.0, min(120.0, geometry[0] * .06))
        if self._track_x is not None and abs(center_x - self._track_x) > jump_limit:
            # Continuous current zero-order boxes can corroborate a moved center.
            # A repeated image-only jump must not silently become a new extremum.
            corroborated = (zero_box_x is not None and zero_box_confidence is not None
                            and math.isfinite(zero_box_x) and math.isfinite(zero_box_confidence)
                            and .25 <= zero_box_confidence <= 1 and confidence >= .18
                            and math.isfinite(zero_box_half_width) and zero_box_half_width >= 0
                            and abs(center_x-zero_box_x) <= max(20, zero_box_half_width))
            if corroborated:
                if self._candidate and abs(center_x-self._candidate[-1][1]) > self.support_radius_px:
                    self._candidate.clear()
                self._candidate.append((captured_at, float(center_x),
                                        zero_box_confidence >= .5 and confidence >= .35))
            else:
                self._candidate.clear()
            # Use the shortest recent confirmation window, so early jitter cannot
            # veto a subsequently stable center. Weaker valid boxes need longer support.
            recent = []
            confirmed = False
            for item in reversed(self._candidate):
                recent.append(item)
                strong = all(p[2] for p in recent)
                duration = .2 if strong else .5
                count = 5 if strong else 8
                if len(recent) >= count and captured_at-item[0] >= duration:
                    confirmed = max(p[1] for p in recent)-min(p[1] for p in recent) <= self.support_radius_px
                    break
            if not confirmed:
                self.rejected_positions += 1
                self.reject_reason = '中心跳变，等待零级框连续确认'
                self._miss()
                return
            self.reacquisitions += 1
            self._support.clear()
            self._votes = 0
            self._aligned = False
            _logger.info('找垂直重新捕获: old=%.2f new=%.2f', self._track_x, center_x)
        self._candidate.clear()
        self.reject_reason = ''
        self.current_x = float(center_x)
        if len(self.positions) >= self.max_positions:
            _, oldest = self.positions.popleft()
            self._position_values.pop(bisect_left(self._position_values, oldest))
        self.positions.append((captured_at, self.current_x))
        insort(self._position_values, self.current_x)
        self._support.append(self.current_x)
        if len(self._support) == self.min_support and max(self._support)-min(self._support) <= self.support_radius_px:
            supported = min(self._support) if self.extreme == 'left' else max(self._support)
            if len(self._supported) >= self.max_positions:
                _, oldest = self._supported.popleft()
                self._supported_values.pop(bisect_left(self._supported_values, oldest))
            self._supported.append((captured_at, supported))
            insort(self._supported_values, supported)
        self._update_reference(captured_at)
        self._release_unreachable_extreme(captured_at)
        # The motion anchor belongs to the accepted track, not to the rolling extremum.
        # Keep it current even when all old supported extrema have expired.
        if self.reference_x is not None or self._track_x is not None:
            self._track_x, self._track_at = self.current_x, captured_at
        inside = (self.reference_x is not None
                  and abs(self.current_x - self.reference_x) <= self.tolerance_px)
        self._votes = self._votes + 1 if inside else 0
        self._aligned = self._votes >= 3

    def snapshot(self, *, now=None):
        now = time.monotonic() if now is None else now
        if self.active:
            self._update_reference(now)
        fresh = self.active and self.current_x is not None and 0 <= now - self._last_at <= self.max_age_s
        if self.active and not 0 <= now - self._last_at <= self.max_age_s:
            self._miss()
        delta = None if not fresh or self.reference_x is None else self.current_x-self.reference_x
        approach = max(self.approach_tolerance_px, self.tolerance_px)
        if self.active and (self._last_log_at is None or now-self._last_log_at >= 5):
            self._last_log_at = now
            _logger.info('找垂直状态: extreme=%s current=%s reference=%s delta=%s tolerance=%.1f '
                         'aligned=%s approaching=%s rejected=%d reacquired=%d released=%d reason=%s',
                         self.extreme, self.current_x if fresh else None, self.reference_x,
                         None if delta is None else round(delta, 2), self.tolerance_px,
                         bool(self._aligned and fresh),
                         delta is not None and abs(delta) <= approach and not self._aligned,
                         self.rejected_positions, self.reacquisitions, self.released_extremes,
                         self.reject_reason or '正常')
        return dict(mode=self.mode, message=self.message,
                    extreme=self.extreme, extreme_label=self.extreme_label,
                    reference_x=self.reference_x,
                    position_count=len(self.positions), rejected_positions=self.rejected_positions,
                    current_x=self.current_x if fresh else None,
                    tolerance_px=self.tolerance_px, aligned=self._aligned and fresh,
                    approach_tolerance_px=approach,
                    approaching=delta is not None and abs(delta) <= approach and not self._aligned,
                    reacquisitions=self.reacquisitions, reject_reason=self.reject_reason,
                    released_extremes=self.released_extremes, released_x=self.released_x,
                    release_note=self._released_at is not None and now - self._released_at <= self.release_note_s,
                    delta_px=delta)
