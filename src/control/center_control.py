"""基于中心条纹横坐标的双向电机闭环控制。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from src.hardware.motor import MOTOR_GEAR_TABLE


@dataclass(frozen=True)
class CenterControlDecision:
    """一次视觉更新产生的电机动作与可显示状态。"""

    commands: tuple[tuple[str, int | None], ...] = ()
    state: str = "idle"
    message: str = "自动寻中未启动"
    error_px: float | None = None
    direction: str = "stopped"
    gear: int | None = None
    stopped_reason: str = ""
    completed: bool = False
    direction_mapping: str = "learning"
    search_position_turns: float | None = None
    searched_min_turns: float | None = None
    searched_max_turns: float | None = None
    search_target_turns: float | None = None
    search_phase: str = ""
    search_expansion_level: int = 0
    search_span_turns: float | None = None
    search_center_turns: float | None = None


class ExpandingSearchPlanner:
    """在虚拟电机坐标中规划不重复的交替扩展扫描。

    电机位置以标称转速积分得到的估算圈数表示。每次到达新区间边界后，规划器反向快速穿过
    已搜索区间，再越过另一侧边界继续搜索；左右最大范围都覆盖后停止。
    """

    def __init__(self) -> None:
        self.reset("forward", 6.0, 1.6, 0.0)

    def reset(self, initial_direction: str, initial_span: float,
              expansion_factor: float, max_span: float) -> None:
        self.position = 0.0
        self.searched_min = 0.0
        self.searched_max = 0.0
        self.center = 0.0
        self.initial_sign = -1 if initial_direction == "reverse" else 1
        self.initial_span = max(0.1, float(initial_span))
        self.span = self.initial_span
        self.expansion_factor = max(1.1, float(expansion_factor))
        # 最大范围为 0 表示不设置范围上限，仍由运行超时和人工停止保护。
        self.max_span = (
            float("inf") if float(max_span) <= 0
            else max(self.span, float(max_span))
        )
        self.expansion_level = 0
        self.target = self.center + self.initial_sign * self.span
        self.in_known_range = True
        self.completed = False
        self.focus_pending = False
        self.focus_sign = self.initial_sign

    @property
    def direction(self) -> str:
        return "forward" if self.target >= self.position else "reverse"

    def redirect(self, direction: str) -> None:
        """让视觉判断出的方向与范围规划保持一致。"""
        sign = -1 if direction == "reverse" else 1
        required_span = max(
            self.span,
            abs(self.searched_min - self.center),
            abs(self.searched_max - self.center),
        )
        self.span = min(self.max_span, required_span)
        target = self.center + sign * self.span
        while (sign * target <= sign * self.position
               and self.span < self.max_span):
            self.span = min(
                self.max_span, max(self.span + 0.1,
                                   self.span * self.expansion_factor))
            target = self.center + sign * self.span
        self.target = target
        self.focus_pending = False
        self.in_known_range = (
            self.searched_min <= self.position <= self.searched_max)

    def recenter(self, direction: str, shift_turns: float) -> float:
        """把搜索重心向视觉线索方向平移，并从新中心重新向两侧扩散。"""
        sign = -1 if direction == "reverse" else 1
        shift = max(0.1, float(shift_turns))
        new_center = self.center + sign * shift
        # 新中心必须位于当前运动位置前方，避免刚重定位就立即反向。
        minimum_ahead = min(0.5, max(0.1, shift * 0.25))
        if sign * (new_center - self.position) < minimum_ahead:
            new_center = self.position + sign * minimum_ahead
        self.center = new_center
        self.span = self.initial_span
        self.expansion_level = 0
        self.completed = False
        self.focus_pending = True
        self.focus_sign = sign
        self.target = self.center
        self.in_known_range = (
            self.searched_min <= self.position <= self.searched_max)
        return self.center

    def advance(self, signed_turns: float) -> None:
        if self.completed or signed_turns == 0:
            return
        old_min, old_max = self.searched_min, self.searched_max
        old_position = self.position
        self.position += float(signed_turns)
        self.in_known_range = old_min <= self.position <= old_max
        if self.position < self.searched_min:
            self.searched_min = self.position
        if self.position > self.searched_max:
            self.searched_max = self.position

        reached = (
            self.target >= old_position and self.position >= self.target
            or self.target < old_position and self.position <= self.target
        )
        if not reached:
            return

        if self.focus_pending:
            # 先到达新搜索中心，然后优先向线索侧搜索，再交替搜索另一侧。
            self.focus_pending = False
            self.target = self.center + self.focus_sign * self.span
            return

        # 保留实际积分产生的少量过冲，使返回距离与电机真实运行时间一致。
        if (self.searched_min <= self.center - self.max_span
                and self.searched_max >= self.center + self.max_span):
            self.completed = True
            return

        next_sign = -1 if self.target > self.center else 1
        candidate = self.center + next_sign * self.span
        while (self.searched_min <= candidate <= self.searched_max
               and self.span < self.max_span):
            self.span = min(
                self.max_span, max(self.span + 0.1,
                                   self.span * self.expansion_factor))
            self.expansion_level += 1
            candidate = self.center + next_sign * self.span
        self.target = candidate
        # 刚掉头时必然先穿过已经搜索过的区间。
        self.in_known_range = True


class CenterControlStateMachine:
    """将白光中心条纹移动到画面水平中心。

    状态机不访问串口或 Tkinter，只产生正转、反转、调速和停车动作。
    """

    def __init__(self) -> None:
        self.enabled = False
        self.started_at = 0.0
        self.direction = "stopped"
        self.gear: int | None = None
        self.stable_frames = 0
        self.missing_frames = 0
        self.center_seen = False
        self.center_candidate_frames = 0
        self.motion_updates = 0
        self.command_refresh_frames = 10
        self.forward_x_sign = 0
        self.direction_learning_score = 0
        self.learning_reference_x: float | None = None
        self.learning_direction = "stopped"
        self.guide_seen = False
        self.guide_missing_frames = 0
        self.blur_frames = 0
        self.blur_recovery_active = False
        self.clear_visual_frames = 0
        self.guide_history: deque[float] = deque(maxlen=12)
        self.search_planner = ExpandingSearchPlanner()
        self.search_planner_configured = False
        self.search_last_update_at = 0.0
        self.guide_initial_direction_selected = False
        self.guide_focus_direction = ""
        self.guide_focus_frames = 0
        self.guide_focus_last_level = -1
        self.stop_detect_phase = "moving"
        self.stop_detect_phase_at = 0.0
        self.stop_detect_frames = 0

    def start(self, now: float) -> CenterControlDecision:
        self.enabled = True
        self.started_at = float(now)
        self.direction = "stopped"
        self.gear = None
        self.stable_frames = 0
        self.missing_frames = 0
        self.center_seen = False
        self.center_candidate_frames = 0
        self.motion_updates = 0
        self.forward_x_sign = 0
        self.direction_learning_score = 0
        self.learning_reference_x = None
        self.learning_direction = "stopped"
        self.guide_seen = False
        self.guide_missing_frames = 0
        self.blur_frames = 0
        self.blur_recovery_active = False
        self.clear_visual_frames = 0
        self.guide_history.clear()
        self.search_planner_configured = False
        self.search_last_update_at = float(now)
        self.guide_initial_direction_selected = False
        self.guide_focus_direction = ""
        self.guide_focus_frames = 0
        self.guide_focus_last_level = -1
        self.stop_detect_phase = "moving"
        self.stop_detect_phase_at = float(now)
        self.stop_detect_frames = 0
        return CenterControlDecision(
            state="searching", message="正在搜索中心条纹")

    def stop(self, reason: str = "用户停止") -> CenterControlDecision:
        was_moving = self.direction != "stopped"
        was_enabled = self.enabled
        self.enabled = False
        self.direction = "stopped"
        self.gear = None
        self.stable_frames = 0
        commands = (("stop", None),) if was_enabled or was_moving else ()
        return CenterControlDecision(
            commands=commands,
            state="stopped",
            message=reason,
            stopped_reason=reason if was_enabled else "",
            **self._range_fields("stopped"),
        )

    def update(
        self,
        *,
        center_x: float | None,
        frame_width: float | None,
        confidence: float,
        guide_x: float | None = None,
        guide_confidence: float = 0.0,
        guide_count: int = 0,
        fringe_movement: str = "unknown",
        fringe_delta_x_px: float | None = None,
        fringe_velocity_px_s: float | None = None,
        scene_has_fringe: bool = False,
        scene_position_x: float | None = None,
        scene_confidence: float = 0.0,
        scene_source: str = "",
        scene_blurred: bool = False,
        scene_held: bool = False,
        connected: bool,
        params: dict,
        safety: dict,
        now: float,
    ) -> CenterControlDecision:
        if not self.enabled:
            return CenterControlDecision()
        if not connected:
            return self.stop("串口失联")

        max_run_s = max(1.0, float(safety.get("max_run_seconds", 60)))
        if float(now) - self.started_at > max_run_s:
            return self.stop("达到最大运行时间")

        search_gear = self._gear(params.get("search_gear", 9))
        # 不允许“快速档位”超过搜索档位，防止穿越区间或追踪中心时产生运动模糊。
        fast_gear = max(search_gear, self._gear(params.get("fast_gear", 9)))
        slow_gear = max(fast_gear, self._gear(params.get("slow_gear", 10)))
        slow_zone = max(1.0, float(params.get("slow_zone_px", 160)))
        tolerance = max(1.0, min(float(params.get("tolerance_px", 15)), slow_zone))
        required_stable = max(1, int(params.get("stable_frames", 5)))
        min_confidence = max(0.0, float(params.get("min_confidence", 0.18)))
        max_missing = max(1, int(safety.get("max_missing_frames", 90)))
        dropout_hold = max(0, int(params.get("dropout_hold_frames", 3)))
        center_confirm_required = max(
            1, min(required_stable,
                   int(params.get("center_confirm_frames", 3))))
        self.command_refresh_frames = max(
            1, int(params.get("command_refresh_frames", 10)))
        search_direction = self._direction(params.get("search_direction", "forward"))
        invert = bool(params.get("invert_direction", False))
        # 方向策略与识别节拍是两个正交选项。保留旧 search_mode 解析，
        # 使历史配置和外部调用方可以平滑迁移。
        legacy_mode = str(params.get("search_mode", ""))
        direction_mode = str(params.get(
            "direction_mode",
            "single_direction" if legacy_mode in {
                "single_direction", "stop_and_detect"} else "bidirectional",
        ))
        recognition_mode = str(params.get(
            "recognition_mode",
            "stop_and_detect" if legacy_mode == "stop_and_detect" else "continuous",
        ))
        single_direction = direction_mode == "single_direction"
        stop_and_detect = recognition_mode == "stop_and_detect"
        fixed_direction = (
            self._opposite_direction(search_direction)
            if invert else search_direction
        )
        auto_learn_direction = bool(params.get("auto_learn_direction", True))
        learning_delta = max(1.0, float(params.get("learning_delta_px", 8)))
        guide_min_confidence = max(
            0.0, float(params.get("guide_min_confidence", 0.2)))
        guide_loss_confirm_frames = max(
            1, int(params.get("guide_loss_confirm_frames", 10)))
        worsening_px = max(1.0, float(params.get("guide_worsening_px", 12)))
        trend_window = max(6, int(params.get("guide_trend_window", 8)))
        guide_focus_confirm_frames = max(
            1, int(params.get("guide_focus_confirm_frames", 3)))
        guide_focus_shift_ratio = max(
            0.05, min(2.0, float(params.get("guide_focus_shift_ratio", 0.5))))
        guide_focus_min_shift = max(
            0.1, float(params.get("guide_focus_min_shift_turns", 1.0)))
        guide_focus_max_shift = max(
            guide_focus_min_shift,
            float(params.get("guide_focus_max_shift_turns", 12.0)),
        )
        search_initial_span = max(
            0.1, float(params.get("search_initial_span_turns", 6.0)))
        search_expansion_factor = max(
            1.1, float(params.get("search_expansion_factor", 1.6)))
        configured_max_span = float(params.get("search_max_span_turns", 0.0))
        search_max_span = (
            0.0 if configured_max_span <= 0
            else max(search_initial_span, configured_max_span)
        )
        search_min_gear = self._gear(params.get("search_min_gear", search_gear))
        search_acceleration_step = max(
            0, min(3, int(params.get("search_acceleration_step", 0))))
        blur_slowdown_frames = max(
            1, min(60, int(params.get("blur_slowdown_frames", 3))))
        blur_safe_gear = max(
            search_gear, self._gear(params.get("blur_safe_gear", 10)))
        blur_recovery_clear_frames = max(
            1, min(60, int(params.get("blur_recovery_clear_frames", 5))))
        stop_detect_move_s = max(
            0.05, float(params.get("stop_detect_move_seconds", 0.6)))
        stop_detect_settle_s = max(
            0.05, float(params.get("stop_detect_settle_seconds", 0.3)))
        stop_detect_required_frames = max(
            1, int(params.get("stop_detect_frames", 2)))
        scene_valid = (
            bool(scene_has_fringe)
            and scene_position_x is not None
            and frame_width is not None
            and float(frame_width) > 0
            and float(scene_confidence) >= guide_min_confidence
        )
        high_scene_speed = (
            fringe_velocity_px_s is not None
            and abs(float(fringe_velocity_px_s))
            >= max(60.0, learning_delta * 10.0)
        )
        blur_risk = bool(scene_blurred) and (
            bool(scene_held) or float(scene_confidence) < 0.40)
        if not self.search_planner_configured:
            self.search_planner.reset(
                search_direction,
                search_initial_span,
                search_expansion_factor,
                search_max_span,
            )
            self.search_planner_configured = True
            self.search_last_update_at = float(now)
        else:
            self._advance_search_position(float(now))

        valid_center = (
            center_x is not None
            and frame_width is not None
            and float(frame_width) > 0
            and float(confidence) >= min_confidence
        )
        valid_guide = (
            guide_x is not None
            and frame_width is not None
            and float(frame_width) > 0
            and float(guide_confidence) >= guide_min_confidence
            and int(guide_count) > 0
        )
        # YOLO 框缺失时，只有当前帧二维视觉证据可以成为软路标；纯历史
        # 预测只负责保持运动连续，不能单独触发重心迁移或换向。
        if (not valid_guide and scene_valid and not scene_held
                and str(scene_source) in ("visual", "yolo")):
            guide_x = float(scene_position_x)
            guide_confidence = float(scene_confidence)
            guide_count = 1
            valid_guide = True
        has_clear_evidence = bool(
            (scene_has_fringe and not scene_held and not blur_risk)
            or ((valid_center or valid_guide) and not scene_has_fringe)
        )
        has_no_evidence = not (valid_center or valid_guide or scene_has_fringe)
        self._update_blur_state(
            blurred=blur_risk or has_no_evidence,
            clear=has_clear_evidence,
            slowdown_frames=blur_slowdown_frames,
            recovery_frames=blur_recovery_clear_frames,
        )
        if stop_and_detect and not self.center_seen:
            if (single_direction and search_max_span > 0
                    and abs(self.search_planner.position) >= search_max_span):
                return self.stop("单向搜索已达到设定最大范围")
            if not single_direction and self.search_planner.completed:
                return self._search_range_decision(
                    search_gear=search_gear,
                    search_min_gear=search_min_gear,
                    acceleration_step=search_acceleration_step,
                )
            cycle_direction = (
                fixed_direction if single_direction
                else self.search_planner.direction
            )
            gated = self._stop_and_detect_decision(
                now=float(now), valid_center=valid_center,
                direction=cycle_direction, gear=search_gear,
                move_seconds=stop_detect_move_s,
                settle_seconds=stop_detect_settle_s,
                required_frames=stop_detect_required_frames,
            )
            if gated is not None:
                # 转动和稳定阶段的检测可能受运动模糊影响，不能送入中心
                # 候选确认；只有停车识别阶段返回 None 才进入原闭环。
                self.center_candidate_frames = 0
                self.stable_frames = 0
                return gated
        if not valid_center:
            self.stable_frames = 0
            self.missing_frames += 1
            if not self.center_seen:
                self.center_candidate_frames = 0
            if self.center_seen:
                if self.missing_frames <= dropout_hold:
                    commands = self._motion_commands(self.direction, self.gear)
                    return CenterControlDecision(
                        commands=commands,
                        state="tracking_dropout",
                        message=(f"中心条纹短时丢失，保持当前运动 "
                                 f"{self.missing_frames}/{dropout_hold}"),
                        direction=self.direction,
                        gear=self.gear,
                        direction_mapping=self._mapping_text(),
                        **self._range_fields("center_dropout"),
                    )
            if single_direction and not self.center_seen:
                if (search_max_span > 0
                        and abs(self.search_planner.position) >= search_max_span):
                    return self.stop("单向搜索已达到设定最大范围")
                temporal_safe = (
                    self.blur_recovery_active or scene_held or high_scene_speed)
                single_gear = blur_safe_gear if temporal_safe else search_gear
                if scene_held:
                    search_message = "短时漏检，依据历史轨迹保持固定方向并降速"
                elif self.blur_recovery_active:
                    search_message = "画面持续模糊，保持固定方向并使用安全档"
                elif high_scene_speed:
                    search_message = "条纹移动较快，保持固定方向并使用安全档"
                elif scene_valid:
                    search_message = "已看到连续条纹线索，保持固定方向搜索中心"
                else:
                    search_message = "未识别到中心条纹，保持固定方向连续搜索"
                return self._single_direction_decision(
                    fixed_direction, single_gear,
                    message=search_message,
                    phase="single_search",
                )
            if self.search_planner.completed:
                return self._search_range_decision(
                    search_gear=search_gear,
                    search_min_gear=search_min_gear,
                    acceleration_step=search_acceleration_step)

            if valid_guide:
                guide_search_gear = (
                    blur_safe_gear
                    if (self.blur_recovery_active or high_scene_speed)
                    else search_gear)
                guide_min_gear = (
                    blur_safe_gear
                    if (self.blur_recovery_active or high_scene_speed)
                    else search_min_gear)
                if self.center_seen and self.missing_frames > dropout_hold:
                    self.center_seen = False
                    self.center_candidate_frames = 0
                    self.missing_frames = 0
                return self._update_guide(
                    guide_x=float(guide_x),
                    frame_width=float(frame_width),
                    guide_count=int(guide_count),
                    search_direction=search_direction,
                    search_gear=guide_search_gear,
                    search_min_gear=guide_min_gear,
                    acceleration_step=(
                        0 if self.blur_recovery_active else search_acceleration_step),
                    auto_learn_direction=auto_learn_direction,
                    learning_delta=learning_delta,
                    worsening_px=worsening_px,
                    trend_window=trend_window,
                    fringe_movement=fringe_movement,
                    fringe_delta_x_px=fringe_delta_x_px,
                    focus_confirm_frames=guide_focus_confirm_frames,
                    focus_shift_ratio=guide_focus_shift_ratio,
                    focus_min_shift=guide_focus_min_shift,
                    focus_max_shift=guide_focus_max_shift,
                )

            if scene_valid and scene_held:
                # 历史预测最多持续数帧。保持搜索规划器当前方向并降低速度，
                # 等待清晰帧恢复，不把预测点当成新的路标。
                return self._search_range_decision(
                    search_gear=blur_safe_gear,
                    search_min_gear=blur_safe_gear,
                    acceleration_step=0,
                    message_prefix="短时漏检，依据历史轨迹降速保持；",
                )

            lost_guide_message = ""
            if self.guide_seen:
                self.guide_missing_frames += 1
                if self.guide_missing_frames >= guide_loss_confirm_frames:
                    self.guide_seen = False
                    self.guide_missing_frames = 0
                    self.guide_history.clear()
                    self.guide_focus_direction = ""
                    self.guide_focus_frames = 0
                    self.guide_focus_last_level = -1
                    lost_guide_message = "线索已丢失，继续既定范围扫描；"
                else:
                    lost_guide_message = (
                        f"线索短时丢失 {self.guide_missing_frames}/"
                        f"{guide_loss_confirm_frames}，不换向；")

            if self.center_seen:
                # 中心曾出现但路标也不可见时停车等待，避免越过目标。
                commands = self._motion_commands("stopped", None)
                if self.missing_frames >= max_missing:
                    self.center_seen = False
                    self.center_candidate_frames = 0
                    self.missing_frames = 0
                    return self._search_range_decision(
                        search_gear=search_gear,
                        search_min_gear=search_min_gear,
                        acceleration_step=search_acceleration_step,
                        message_prefix=(
                            "中心长时间未恢复，已返回扩大范围搜索；"),
                    )
                return CenterControlDecision(
                    commands=commands,
                    state="waiting",
                    message=f"中心条纹暂时丢失（{self.missing_frames}/{max_missing}）",
                    **self._range_fields("center_waiting"),
                )

            effective_search_gear = (
                blur_safe_gear if self.blur_recovery_active else search_gear)
            effective_min_gear = (
                blur_safe_gear if self.blur_recovery_active else search_min_gear)
            blur_message = (
                f"连续 {self.blur_frames} 帧未看清条纹，"
                f"保持连续旋转并降至档位 {blur_safe_gear}；"
                if self.blur_recovery_active else ""
            )
            return self._search_range_decision(
                search_gear=effective_search_gear,
                search_min_gear=effective_min_gear,
                acceleration_step=(
                    0 if self.blur_recovery_active else search_acceleration_step),
                message_prefix=lost_guide_message + blur_message,
            )

        if self.blur_recovery_active or high_scene_speed:
            fast_gear = max(fast_gear, blur_safe_gear)
            slow_gear = max(slow_gear, blur_safe_gear)
        self.center_candidate_frames += 1
        if self.center_candidate_frames >= center_confirm_required:
            self.center_seen = True
        elif stop_and_detect:
            # 转停方式必须在停车状态完成候选确认，不能因第一帧候选重新启动电机。
            self.missing_frames = 0
            commands = self._motion_commands("stopped", None)
            return CenterControlDecision(
                commands=commands,
                state="confirming_center_candidate",
                message=("停车确认中心候选 "
                         f"{self.center_candidate_frames}/{center_confirm_required}"),
                error_px=float(center_x) - float(frame_width) / 2.0,
                direction_mapping=self._mapping_text(),
                **self._range_fields("confirming_center_candidate"),
            )
        elif single_direction:
            # 已知方向搜索必须先确认中心候选连续稳定，再交给允许换向的
            # 原有闭环。单帧误检不能提前改变人工指定的搜索方向。
            self.missing_frames = 0
            return self._single_direction_decision(
                fixed_direction,
                search_gear,
                message=(
                    "已发现中心条纹候选，继续保持人工指定方向并确认稳定 "
                    f"{self.center_candidate_frames}/{center_confirm_required}"
                ),
                phase="confirming_center_candidate",
                error=float(center_x) - float(frame_width) / 2.0,
            )
        self.missing_frames = 0
        target_x = float(frame_width) / 2.0
        error = float(center_x) - target_x
        distance = abs(error)

        # 搜索模式只影响中心条纹出现前的扫描。找到中心条纹后，单向方案
        # 与双向方案共用同一套方向学习、分区调速和闭环居中逻辑。
        if auto_learn_direction:
            self._learn_direction(float(center_x), learning_delta)

        if distance <= tolerance:
            self.stable_frames += 1
            commands = self._motion_commands("stopped", None)
            if (self.stable_frames >= required_stable
                    and self.center_seen):
                self.enabled = False
                return CenterControlDecision(
                    commands=commands,
                    state="centered",
                    message=f"中心条纹已稳定在画面中央（误差 {error:+.1f} px）",
                    error_px=error,
                    completed=True,
                    direction_mapping=self._mapping_text(),
                    **self._range_fields("centered"),
                )
            return CenterControlDecision(
                commands=commands,
                state="confirming",
                message=(f"已进入中心容差，正在确认稳定性 "
                         f"{self.stable_frames}/{required_stable}；"
                         f"中心确认 {self.center_candidate_frames}/"
                         f"{center_confirm_required}"),
                error_px=error,
                direction_mapping=self._mapping_text(),
                **self._range_fields("confirming"),
            )

        self.stable_frames = 0
        if auto_learn_direction and self.forward_x_sign == 0:
            # 先沿当前方向慢速探测；产生足够像素位移后即可判断映射。
            direction = (
                self.direction if self.direction in ("forward", "reverse")
                else search_direction
            )
            gear = slow_gear
            state = "learning_direction"
            message = (
                f"正在自动判断方向：{self._direction_text(direction)}慢速探测，"
                f"中心偏差 {error:+.1f} px，中心确认 "
                f"{self.center_candidate_frames}/{center_confirm_required}")
        else:
            desired_x_sign = -1 if error > 0 else 1
            if auto_learn_direction:
                direction = (
                    "forward" if self.forward_x_sign == desired_x_sign else "reverse"
                )
            else:
                direction = "forward" if error < 0 else "reverse"
                if invert:
                    direction = "reverse" if direction == "forward" else "forward"
            gear = slow_gear if distance <= slow_zone else fast_gear
            state = "approaching" if gear == slow_gear else "centering"
            message = (f"中心偏差 {error:+.1f} px，"
                       f"{self._direction_text(direction)}，档位 {gear}，"
                       f"中心确认 {self.center_candidate_frames}/"
                       f"{center_confirm_required}")
        commands = self._motion_commands(direction, gear)
        return CenterControlDecision(
            commands=commands,
            state=state,
            message=message,
            error_px=error,
            direction=direction,
            gear=gear,
            direction_mapping=self._mapping_text(),
            **self._range_fields("centering"),
        )

    def _single_direction_decision(
        self, direction: str, gear: int, *, message: str,
        phase: str, error: float | None = None,
    ) -> CenterControlDecision:
        """在中心条纹尚未出现时沿已知方向连续搜索。"""
        commands = self._motion_commands(direction, gear)
        return CenterControlDecision(
            commands=commands,
            state="single_direction_search",
            message=f"{message}（找到中心条纹前不往返）",
            error_px=error,
            direction=direction,
            gear=gear,
            direction_mapping=f"固定{self._direction_text(direction)}",
            **self._range_fields(phase),
        )

    def _stop_and_detect_decision(
        self, *, now: float, valid_center: bool, direction: str, gear: int,
        move_seconds: float, settle_seconds: float, required_frames: int,
    ) -> CenterControlDecision | None:
        """独立的转动—停车—稳定—识别节拍；识别成功后交回主闭环。"""
        elapsed = max(0.0, now - self.stop_detect_phase_at)
        if self.stop_detect_phase == "moving":
            if elapsed < move_seconds:
                commands = self._motion_commands(direction, gear)
                return CenterControlDecision(
                    commands=commands, state="cycle_moving",
                    message=(f"分步模式：{self._direction_text(direction)}转动 "
                             f"{elapsed:.2f}/{move_seconds:.2f}s"),
                    direction=direction, gear=gear,
                    direction_mapping=f"固定{self._direction_text(direction)}",
                    **self._range_fields("cycle_moving"),
                )
            self.stop_detect_phase = "settling"
            self.stop_detect_phase_at = now
            self.stop_detect_frames = 0
            commands = self._motion_commands("stopped", None)
            return CenterControlDecision(
                commands=commands, state="cycle_settling",
                message="分步模式：电机已停止，等待画面恢复清晰",
                **self._range_fields("cycle_settling"),
            )
        if self.stop_detect_phase == "settling":
            commands = self._motion_commands("stopped", None)
            if elapsed >= settle_seconds:
                self.stop_detect_phase = "detecting"
                self.stop_detect_phase_at = now
                self.stop_detect_frames = 0
            return CenterControlDecision(
                commands=commands, state="cycle_settling",
                message=(f"分步模式：停车稳定 "
                         f"{min(elapsed, settle_seconds):.2f}/{settle_seconds:.2f}s"),
                **self._range_fields("cycle_settling"),
            )

        # detecting：只接受停车稳定后的中心结果；当前帧找到中心时返回
        # None，让现有中心确认和闭环居中逻辑继续处理。
        if self.stop_detect_phase == "locked":
            if valid_center:
                return None
            self.stop_detect_phase = "moving"
            self.stop_detect_phase_at = now
            self.stop_detect_frames = 0
            commands = self._motion_commands(direction, gear)
            return CenterControlDecision(
                commands=commands, state="cycle_moving",
                message="分步模式：中心候选未持续出现，继续转动下一段",
                direction=direction, gear=gear,
                direction_mapping=f"固定{self._direction_text(direction)}",
                **self._range_fields("cycle_moving"),
            )
        if valid_center:
            self.stop_detect_phase = "locked"
            self.stop_detect_frames = 0
            return None
        self.stop_detect_frames += 1
        if self.stop_detect_frames >= required_frames:
            self.stop_detect_phase = "moving"
            self.stop_detect_phase_at = now
            self.stop_detect_frames = 0
            commands = self._motion_commands(direction, gear)
            return CenterControlDecision(
                commands=commands, state="cycle_moving",
                message="分步模式：停车识别未找到中心，继续转动下一段",
                direction=direction, gear=gear,
                direction_mapping=f"固定{self._direction_text(direction)}",
                **self._range_fields("cycle_moving"),
            )
        commands = self._motion_commands("stopped", None)
        return CenterControlDecision(
            commands=commands, state="cycle_detecting",
            message=(f"分步模式：清晰画面识别 "
                     f"{self.stop_detect_frames}/{required_frames}"),
            **self._range_fields("cycle_detecting"),
        )

    def _update_guide(
        self,
        *,
        guide_x: float,
        frame_width: float,
        guide_count: int,
        search_direction: str,
        search_gear: int,
        search_min_gear: int,
        acceleration_step: int,
        auto_learn_direction: bool,
        learning_delta: float,
        worsening_px: float,
        trend_window: int,
        fringe_movement: str,
        fringe_delta_x_px: float | None,
        focus_confirm_frames: int,
        focus_shift_ratio: float,
        focus_min_shift: float,
        focus_max_shift: float,
    ) -> CenterControlDecision:
        """用多帧平滑后的非中心条纹路标引导中心出现。"""
        self.guide_seen = True
        self.guide_missing_frames = 0
        if self.guide_history.maxlen != trend_window:
            self.guide_history = deque(self.guide_history, maxlen=trend_window)
        self.guide_history.append(float(guide_x))
        smooth_x = float(median(list(self.guide_history)[-3:]))
        target_x = frame_width / 2.0
        error = smooth_x - target_x

        if auto_learn_direction:
            self._learn_direction(smooth_x, learning_delta)

        trend_delta = 0.0
        if len(self.guide_history) >= trend_window:
            history = list(self.guide_history)
            half = len(history) // 2
            old_distance = median(abs(x - target_x) for x in history[:half])
            new_distance = median(abs(x - target_x) for x in history[half:])
            trend_delta = float(new_distance - old_distance)
        movement_away = (
            (error > 0 and fringe_movement == "right")
            or (error < 0 and fringe_movement == "left")
        )
        movement_known = fringe_movement in ("left", "right")
        enough_motion = (
            fringe_delta_x_px is None
            or abs(float(fringe_delta_x_px)) >= learning_delta
        )
        worsening = (
            len(self.guide_history) >= trend_window
            and trend_delta > worsening_px
            and enough_motion
            and (movement_away or not movement_known)
        )
        desired_x_sign = -1 if error > 0 else 1
        if auto_learn_direction and self.forward_x_sign != 0:
            suggested_direction = (
                "forward" if self.forward_x_sign == desired_x_sign else "reverse")
        else:
            suggested_direction = "forward" if error < 0 else "reverse"
            if abs(error) < 1:
                suggested_direction = search_direction

        planner = self.search_planner
        if suggested_direction != self.guide_focus_direction:
            self.guide_focus_direction = suggested_direction
            self.guide_focus_frames = 1
            # 新方向是一条新线索，允许它建立新的搜索重心。
            self.guide_focus_last_level = -1
        else:
            self.guide_focus_frames += 1

        recentered = False
        focus_shift = 0.0
        if (self.guide_focus_frames >= focus_confirm_frames
                and planner.expansion_level > self.guide_focus_last_level
                and not planner.focus_pending):
            focus_shift = min(
                focus_max_shift,
                max(focus_min_shift, planner.span * focus_shift_ratio),
            )
            planner.recenter(suggested_direction, focus_shift)
            self.guide_focus_last_level = planner.expansion_level
            self.guide_initial_direction_selected = True
            recentered = True

        searched_width = planner.searched_max - planner.searched_min
        if not self.guide_initial_direction_selected and not recentered:
            # 线索只允许在起步阶段选择一次首个搜索方向；之后整段扫描锁向。
            if searched_width < 0.25 and abs(planner.position) < 0.25:
                planner.redirect(suggested_direction)
            self.guide_initial_direction_selected = True

        direction = planner.direction
        dynamic_search_gear = self._dynamic_search_gear(
            search_gear, search_min_gear, acceleration_step)
        in_known = planner.in_known_range and searched_width > 0.1
        if recentered or planner.focus_pending:
            gear = dynamic_search_gear
            state = "guided_refocusing"
            phase = "refocusing"
            if recentered:
                message = (
                    f"连续 {self.guide_focus_frames} 帧线索指向"
                    f"{self._direction_text(suggested_direction)}；搜索中心向该方向"
                    f"平移 {focus_shift:.1f} 圈至 {planner.center:+.1f} 圈，"
                    "到达后从新中心向两侧扩散"
                )
            else:
                message = (
                    f"正在前往线索确定的新搜索中心 {planner.center:+.1f} 圈；"
                    "到达后从中心向两侧扩散"
                )
        elif in_known:
            gear = dynamic_search_gear
            state = "guided_returning"
            phase = "returning"
            message = (
                f"检测到 {guide_count} 个线索；已搜索区间内锁定"
                f"{self._direction_text(direction)}快速通过，不重复换向")
        else:
            gear = dynamic_search_gear
            state = "guided_expanding"
            phase = "expanding"
            trend_text = (
                f"，路标趋势远离 {trend_delta:+.1f}px 但保持本轮方向"
                if worsening else "")
            message = (
                f"依据 {guide_count} 个线索扩展新区间：平滑偏差 "
                f"{error:+.1f}px，锁定{self._direction_text(direction)}"
                f"直到 {planner.target:+.1f} 圈，档位 {gear}{trend_text}")

        commands = self._motion_commands(direction, gear)
        return CenterControlDecision(
            commands=commands,
            state=state,
            message=message,
            error_px=error,
            direction=direction,
            gear=gear,
            direction_mapping=self._mapping_text(),
            search_position_turns=planner.position,
            searched_min_turns=planner.searched_min,
            searched_max_turns=planner.searched_max,
            search_target_turns=planner.target,
            search_phase=phase,
            search_expansion_level=planner.expansion_level,
            search_span_turns=planner.span,
            search_center_turns=planner.center,
        )

    def _search_range_decision(self, *, search_gear: int,
                               search_min_gear: int, acceleration_step: int,
                               message_prefix: str = "") -> CenterControlDecision:
        """按已搜索区间生成不受非中心框抖动影响的扫描动作。"""
        planner = self.search_planner
        if planner.completed:
            decision = self.stop("设定的扩大搜索范围已全部覆盖，未找到中心条纹")
            return CenterControlDecision(
                commands=decision.commands,
                state="search_exhausted",
                message="左右最大搜索范围均已覆盖，已安全停车",
                stopped_reason="搜索范围已全部覆盖",
                search_position_turns=planner.position,
                searched_min_turns=planner.searched_min,
                searched_max_turns=planner.searched_max,
                search_target_turns=planner.target,
                search_phase="complete",
                search_center_turns=planner.center,
            )
        direction = planner.direction
        dynamic_search_gear = self._dynamic_search_gear(
            search_gear, search_min_gear, acceleration_step)
        if planner.focus_pending:
            gear = dynamic_search_gear
            state = "search_refocusing"
            phase = "refocusing"
            message = (
                f"{message_prefix}正在前往线索确定的新搜索中心 "
                f"{planner.center:+.1f} 圈，当前位置 {planner.position:+.1f} 圈，"
                f"档位 {gear}")
        elif planner.in_known_range and (planner.searched_max - planner.searched_min) > 0.1:
            gear = dynamic_search_gear
            state = "search_returning"
            phase = "returning"
            message = (
                f"{message_prefix}快速穿越已搜索区间 "
                f"{planner.searched_min:+.1f}~{planner.searched_max:+.1f} 圈，"
                f"前往 {planner.target:+.1f} 圈，档位 {gear}")
        else:
            gear = dynamic_search_gear
            state = "search_expanding"
            phase = "expanding"
            message = (
                f"{message_prefix}正在扩展搜索新区间，位置 "
                f"{planner.position:+.1f} 圈 / 目标 {planner.target:+.1f} 圈，"
                f"第 {planner.expansion_level + 1} 轮，档位 {gear}")
        commands = self._motion_commands(direction, gear)
        return CenterControlDecision(
            commands=commands, state=state, message=message,
            direction=direction, gear=gear,
            direction_mapping=self._mapping_text(),
            search_position_turns=planner.position,
            searched_min_turns=planner.searched_min,
            searched_max_turns=planner.searched_max,
            search_target_turns=planner.target,
            search_phase=phase,
            search_expansion_level=planner.expansion_level,
            search_span_turns=planner.span,
            search_center_turns=planner.center,
        )

    def _range_fields(self, phase: str) -> dict:
        """将持续积分的搜索范围附加到所有控制状态，包括中心定位阶段。"""
        if not self.search_planner_configured:
            return {}
        planner = self.search_planner
        return {
            "search_position_turns": planner.position,
            "searched_min_turns": planner.searched_min,
            "searched_max_turns": planner.searched_max,
            "search_target_turns": planner.target,
            "search_phase": phase,
            "search_expansion_level": planner.expansion_level,
            "search_span_turns": planner.span,
            "search_center_turns": planner.center,
        }

    def _dynamic_search_gear(self, base_gear: int, min_gear: int,
                             acceleration_step: int) -> int:
        """每完成一轮双侧搜索便提高速度，档位数字越小越快。"""
        fastest_allowed = min(self._gear(base_gear), self._gear(min_gear))
        return max(
            fastest_allowed,
            self._gear(base_gear)
            - self.search_planner.expansion_level * max(0, acceleration_step),
        )

    def _update_blur_state(
        self, *, blurred: bool, clear: bool,
        slowdown_frames: int, recovery_frames: int,
    ) -> None:
        """用一组计数器统一处理画面模糊与连续无视觉证据。"""
        if blurred:
            self.blur_frames += 1
            self.clear_visual_frames = 0
            if self.blur_frames >= slowdown_frames:
                self.blur_recovery_active = True
            return
        if not clear:
            return
        self.blur_frames = 0
        if not self.blur_recovery_active:
            self.clear_visual_frames = 0
            return
        self.clear_visual_frames += 1
        if self.clear_visual_frames >= recovery_frames:
            self.blur_recovery_active = False
            self.clear_visual_frames = 0

    def _learn_direction(self, center_x: float, minimum_delta: float) -> None:
        """根据已知运动方向和条纹位移学习正转对应的画面方向。"""
        if self.direction not in ("forward", "reverse"):
            self.learning_reference_x = center_x
            self.learning_direction = "stopped"
            return
        if (self.learning_reference_x is None
                or self.learning_direction != self.direction):
            self.learning_reference_x = center_x
            self.learning_direction = self.direction
            return
        delta = center_x - self.learning_reference_x
        if abs(delta) < minimum_delta:
            return
        observed_sign = 1 if delta > 0 else -1
        forward_sign = observed_sign if self.direction == "forward" else -observed_sign
        self.direction_learning_score += forward_sign
        self.direction_learning_score = max(-3, min(3, self.direction_learning_score))
        self.forward_x_sign = 1 if self.direction_learning_score > 0 else -1
        self.learning_reference_x = center_x

    def _advance_search_position(self, now: float) -> None:
        """按实际档位角速度积分虚拟位置，供扩展扫描规划使用。"""
        dt = max(0.0, float(now) - self.search_last_update_at)
        self.search_last_update_at = float(now)
        if dt <= 0 or self.direction not in ("forward", "reverse"):
            return
        gear = MOTOR_GEAR_TABLE.get(int(self.gear or 0))
        if not gear:
            return
        turns = dt / float(gear["turn_seconds"])
        signed_turns = turns if self.direction == "forward" else -turns
        self.search_planner.advance(signed_turns)

    def _mapping_text(self) -> str:
        if self.forward_x_sign > 0:
            return "正转使条纹向右"
        if self.forward_x_sign < 0:
            return "正转使条纹向左"
        return "正在学习方向"

    @staticmethod
    def _opposite_direction(current: str, fallback: str = "forward") -> str:
        if current == "forward":
            return "reverse"
        if current == "reverse":
            return "forward"
        return "reverse" if fallback == "forward" else "forward"

    def _motion_commands(
        self, direction: str, gear: int | None,
    ) -> tuple[tuple[str, int | None], ...]:
        commands: list[tuple[str, int | None]] = []
        if direction == "stopped":
            if self.direction != "stopped":
                commands.append(("stop", None))
            self.direction = "stopped"
            self.gear = None
            self.motion_updates = 0
            return tuple(commands)

        if self.direction not in ("stopped", direction):
            commands.append(("stop", None))
            self.direction = "stopped"
        if gear is not None and gear != self.gear:
            commands.append(("set_speed", gear))
            self.gear = gear
        if self.direction != direction:
            commands.append((f"start_{direction}", None))
            self.direction = direction
            self.motion_updates = 0
        else:
            self.motion_updates += 1
            if self.motion_updates >= self.command_refresh_frames:
                commands.append((f"start_{direction}", None))
                self.motion_updates = 0
        return tuple(commands)

    @staticmethod
    def _gear(value) -> int:
        return max(1, min(10, int(value)))

    @staticmethod
    def _direction(value) -> str:
        return "reverse" if str(value).lower() == "reverse" else "forward"

    @staticmethod
    def _direction_text(direction: str) -> str:
        return "反转" if direction == "reverse" else "正转"
