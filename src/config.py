"""配置管理：加载 config.yaml，提供统一访问接口"""
from pathlib import Path
import yaml
from src import PROJECT_ROOT


class ConfigError(ValueError):
    """配置文件格式或取值不安全。"""


class Config:
    """全局配置管理器"""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")
        self._path = Path(config_path)
        self._data: dict = {}
        self.reload()

    def reload(self):
        """重新加载配置文件"""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"无法读取配置 {self._path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("config.yaml 顶层必须是键值映射")
        self._data = data
        self.validate()

    def validate(self) -> None:
        """在启动阶段一次性报告配置错误，避免运行中隐式回退。"""
        errors: list[str] = []

        def number(path: tuple[str, ...], *, minimum=None, maximum=None,
                   integer=False):
            value = self.get(*path)
            if value is None:
                return
            valid_type = isinstance(value, int) and not isinstance(value, bool)
            if not integer:
                valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid_type:
                errors.append(f"{'.'.join(path)} 必须是{'整数' if integer else '数字'}")
                return
            if minimum is not None and value < minimum:
                errors.append(f"{'.'.join(path)} 不能小于 {minimum}")
            if maximum is not None and value > maximum:
                errors.append(f"{'.'.join(path)} 不能大于 {maximum}")

        number(("camera", "index"), minimum=0, integer=True)
        number(("camera", "fps"), minimum=1)
        number(("camera", "clarity_assist", "preview_exposure"), minimum=-20, maximum=0)
        number(("camera", "clarity_assist", "preview_gain"), minimum=0, maximum=255)
        number(("camera", "clarity_assist", "motion_exposure"), minimum=-20, maximum=0)
        number(("camera", "clarity_assist", "motion_gain"), minimum=0, maximum=255)
        number(("camera", "clarity_assist", "min_exposure"), minimum=-20, maximum=0)
        number(("camera", "clarity_assist", "max_gain"), minimum=0, maximum=255)
        number(("camera", "clarity_assist", "blur_ratio"), minimum=0.1, maximum=1)
        number(("camera", "clarity_assist", "check_frames"), minimum=1, maximum=120, integer=True)
        number(("camera", "clarity_assist", "trigger_checks"), minimum=1, maximum=30, integer=True)
        number(("camera", "clarity_assist", "min_brightness_for_shorter_exposure"), minimum=0, maximum=255)
        number(("camera", "clarity_assist", "software_enhancement", "sharpen_strength"), minimum=0, maximum=5)
        number(("camera", "clarity_assist", "software_enhancement", "max_sharpen_strength"), minimum=0, maximum=8)
        number(("camera", "clarity_assist", "software_enhancement", "stripe_contrast_strength"), minimum=0, maximum=8)
        number(("camera", "clarity_assist", "software_enhancement", "max_stripe_contrast_strength"), minimum=0, maximum=12)
        number(("camera", "clarity_assist", "software_enhancement", "color_gain"), minimum=1, maximum=5)
        number(("camera", "clarity_assist", "software_enhancement", "max_color_gain"), minimum=1, maximum=8)
        number(("camera", "clarity_assist", "software_enhancement", "original_mix"), minimum=0, maximum=1)
        number(("camera", "clarity_assist", "software_enhancement", "horizontal_kernel_size"), minimum=3, maximum=31, integer=True)
        number(("camera", "clarity_assist", "software_enhancement", "vertical_smooth_size"), minimum=1, maximum=31, integer=True)
        number(("camera", "clarity_assist", "software_enhancement", "background_kernel_size"), minimum=5, maximum=101, integer=True)
        number(("camera", "clarity_assist", "software_enhancement", "contrast_gain"), minimum=0.5, maximum=3)
        resolution = self.get("camera", "resolution")
        if resolution is not None and (not isinstance(resolution, list)
                                       or len(resolution) != 2
                                       or not all(isinstance(v, int) and v > 0 for v in resolution)):
            errors.append("camera.resolution 必须是两个正整数")
        number(("vision", "confidence_threshold"), minimum=0, maximum=1)
        number(("vision", "iou_threshold"), minimum=0, maximum=1)
        number(("vision", "imgsz"), minimum=32, integer=True)
        number(("vision", "center_search_expand_ratio"), minimum=1, maximum=6)
        number(("vision", "center_search_radius_ratio"), minimum=0.2, maximum=3)
        number(("vision", "center_search_margin_ratio"), minimum=0, maximum=1.5)
        number(("vision", "fringe_motion_window"), minimum=3, maximum=30, integer=True)
        number(("vision", "fringe_motion_threshold_px"), minimum=0.5, maximum=100)
        model_path = self.get("vision", "model_path")
        if "vision" in self._data:
            if not isinstance(model_path, str) or not model_path.strip():
                errors.append("vision.model_path 必须是非空路径")
            elif not self.resolve_path(model_path).is_file():
                errors.append(f"vision.model_path 文件不存在: {model_path}")
        number(("motor", "baudrate"), minimum=1, integer=True)
        number(("motor", "timeout"), minimum=0.01)
        temporary_enabled = self.get("temporary_measurement", "enabled")
        if (temporary_enabled is not None
                and not isinstance(temporary_enabled, bool)):
            errors.append("temporary_measurement.enabled 必须是布尔值")
        number(("temporary_measurement", "approach_gear"),
               minimum=1, maximum=10, integer=True)
        number(("temporary_measurement", "tolerance_mm"), minimum=0)
        number(("temporary_measurement", "max_duration_seconds"), minimum=1)
        number(("temporary_measurement", "poll_interval_ms"),
               minimum=20, integer=True)
        number(("temporary_measurement", "reading_timeout_seconds"),
               minimum=0.1)
        number(("temporary_measurement", "backlash_endpoint_tolerance_mm"),
               minimum=0.000001)
        number(("micrometer", "max_step_mm"), minimum=0.000001)
        number(("micrometer", "jump_required"),
               minimum=2, maximum=100, integer=True)
        number(("micrometer", "scale_ratio_tolerance"),
               minimum=0.001, maximum=0.2)
        number(("motor", "automatic", "search_gear"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "fast_gear"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "slow_gear"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "slow_zone_px"), minimum=1)
        number(("motor", "automatic", "tolerance_px"), minimum=1)
        number(("motor", "automatic", "stable_frames"), minimum=1, integer=True)
        search_mode = self.get("motor", "automatic", "search_mode")
        if (self.get("motor", "automatic") is not None
                and search_mode not in {
                    "bidirectional", "single_direction", "stop_and_detect"}):
            errors.append(
                "motor.automatic.search_mode 必须是 bidirectional、"
                "single_direction 或 stop_and_detect")
        number(("motor", "automatic", "dropout_hold_frames"), minimum=0, integer=True)
        number(("motor", "automatic", "center_confirm_frames"), minimum=1, maximum=30, integer=True)
        number(("motor", "automatic", "command_refresh_frames"), minimum=1, integer=True)
        number(("motor", "automatic", "learning_delta_px"), minimum=1)
        number(("motor", "automatic", "guide_min_confidence"), minimum=0, maximum=1)
        number(("motor", "automatic", "guide_loss_confirm_frames"), minimum=1, integer=True)
        number(("motor", "automatic", "search_initial_span_turns"), minimum=0.1, maximum=100)
        number(("motor", "automatic", "search_expansion_factor"), minimum=1.1, maximum=3)
        number(("motor", "automatic", "search_max_span_turns"), minimum=0, maximum=1000000000)
        number(("motor", "automatic", "search_min_gear"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "search_acceleration_step"), minimum=0, maximum=3, integer=True)
        number(("motor", "automatic", "blur_slowdown_frames"), minimum=1, maximum=60, integer=True)
        number(("motor", "automatic", "blur_safe_gear"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "blur_recovery_clear_frames"), minimum=1, maximum=60, integer=True)
        initial_span = self.get("motor", "automatic", "search_initial_span_turns")
        maximum_span = self.get("motor", "automatic", "search_max_span_turns")
        if (isinstance(initial_span, (int, float))
                and isinstance(maximum_span, (int, float))
                and maximum_span > 0
                and maximum_span < initial_span):
            errors.append(
                "motor.automatic.search_max_span_turns 不能小于初始搜索范围")
        search_gear = self.get("motor", "automatic", "search_gear")
        minimum_gear = self.get("motor", "automatic", "search_min_gear")
        if (isinstance(search_gear, int) and isinstance(minimum_gear, int)
                and minimum_gear > search_gear):
            errors.append(
                "motor.automatic.search_min_gear 必须小于或等于搜索档位")
        number(("motor", "automatic", "guide_worsening_px"), minimum=1)
        number(("motor", "automatic", "guide_trend_window"), minimum=6, maximum=30, integer=True)
        number(("motor", "automatic", "guide_focus_confirm_frames"), minimum=1, maximum=30, integer=True)
        number(("motor", "automatic", "guide_focus_shift_ratio"), minimum=0.05, maximum=2)
        number(("motor", "automatic", "guide_focus_min_shift_turns"), minimum=0.1, maximum=100)
        number(("motor", "automatic", "guide_focus_max_shift_turns"), minimum=0.1, maximum=1000)
        focus_min_shift = self.get("motor", "automatic", "guide_focus_min_shift_turns")
        focus_max_shift = self.get("motor", "automatic", "guide_focus_max_shift_turns")
        if (isinstance(focus_min_shift, (int, float))
                and isinstance(focus_max_shift, (int, float))
                and focus_max_shift < focus_min_shift):
            errors.append(
                "motor.automatic.guide_focus_max_shift_turns 不能小于最小平移量")
        number(("motor", "safety", "max_run_seconds"), minimum=0.1)
        number(("motor", "safety", "black_confirm_frames"), minimum=1, integer=True)
        number(("motor", "safety", "max_missing_frames"), minimum=1, integer=True)
        number(("micrometer", "scale_factor"), minimum=0.000000001)
        meter_model_path = self.get("micrometer", "model_path")
        if "micrometer" in self._data:
            if (not isinstance(meter_model_path, str)
                    or not meter_model_path.strip()):
                errors.append("micrometer.model_path 必须是非空路径")
            elif not self.resolve_path(meter_model_path).is_file():
                errors.append(
                    f"micrometer.model_path 文件不存在: {meter_model_path}")
        number(("micrometer", "camera_index"), minimum=0, integer=True)
        number(("micrometer", "fps"), minimum=1, maximum=120, integer=True)
        number(("micrometer", "interval_ms"), minimum=50, maximum=10000, integer=True)
        number(("micrometer", "min_score"), minimum=0, maximum=1)
        number(("micrometer", "decimal_places"), minimum=0, maximum=6, integer=True)
        number(("micrometer", "stable_window"), minimum=1, maximum=60, integer=True)
        number(("micrometer", "stable_required"), minimum=1, maximum=60, integer=True)
        meter_resolution = self.get("micrometer", "resolution")
        if meter_resolution is not None and (
                not isinstance(meter_resolution, list)
                or len(meter_resolution) != 2
                or not all(isinstance(v, int) and v > 0 for v in meter_resolution)):
            errors.append("micrometer.resolution 必须是两个正整数")
        meter_roi = self.get("micrometer", "roi")
        if meter_roi is not None and (
                not isinstance(meter_roi, list)
                or len(meter_roi) != 4
                or not all(isinstance(v, (int, float)) for v in meter_roi)):
            errors.append("micrometer.roi 必须是四个数字")
        stable_window = self.get("micrometer", "stable_window")
        stable_required = self.get("micrometer", "stable_required")
        if (isinstance(stable_window, int) and isinstance(stable_required, int)
                and stable_required > stable_window):
            errors.append("micrometer.stable_required 不能大于 stable_window")
        number(("experiment", "max_auto_seconds"), minimum=1)
        number(("experiment", "center_stable_frames"), minimum=1, integer=True)
        number(("experiment", "center_min_confidence"), minimum=0, maximum=1)
        number(("experiment", "center_max_jitter_px"), minimum=0.1)
        number(("agent", "llm", "timeout"), minimum=1)
        number(("agent", "llm", "max_tokens"), minimum=1, integer=True)
        number(("agent", "rag", "top_k"), minimum=1, maximum=20, integer=True)
        window_size = self.get("ui", "window_size")
        if window_size is not None and (not isinstance(window_size, list)
                                        or len(window_size) != 2
                                        or not all(isinstance(v, int) and v >= 400 for v in window_size)):
            errors.append("ui.window_size 必须是两个不小于 400 的整数")
        if errors:
            raise ConfigError("配置校验失败：\n- " + "\n- ".join(errors))

    def get(self, *keys, default=None):
        """用点路径获取配置值：config.get('camera', 'index')"""
        value = self._data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def resolve_path(self, relative_path: str) -> Path:
        """将相对路径转为项目根目录下的绝对路径"""
        return PROJECT_ROOT / relative_path

    @property
    def camera(self) -> dict:
        return self._data.get("camera", {})

    @property
    def vision(self) -> dict:
        return self._data.get("vision", {})

    @property
    def motor(self) -> dict:
        return self._data.get("motor", {})

    @property
    def agent(self) -> dict:
        return self._data.get("agent", {})

    @property
    def uncertainty(self) -> dict:
        return self._data.get("uncertainty", {})

    @property
    def ui(self) -> dict:
        return self._data.get("ui", {})


# 全局单例
config = Config()
