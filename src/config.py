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
        resolution = self.get("camera", "resolution")
        if resolution is not None and (not isinstance(resolution, list)
                                       or len(resolution) != 2
                                       or not all(isinstance(v, int) and v > 0 for v in resolution)):
            errors.append("camera.resolution 必须是两个正整数")
        number(("vision", "confidence_threshold"), minimum=0, maximum=1)
        number(("vision", "iou_threshold"), minimum=0, maximum=1)
        number(("vision", "imgsz"), minimum=32, integer=True)
        model_path = self.get("vision", "model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            errors.append("vision.model_path 必须是非空路径")
        elif not self.resolve_path(model_path).is_file():
            errors.append(f"vision.model_path 文件不存在: {model_path}")
        number(("motor", "baudrate"), minimum=1, integer=True)
        number(("motor", "timeout"), minimum=0.01)
        number(("motor", "automatic", "search_speed"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "color_speed"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "black_speed"), minimum=1, maximum=10, integer=True)
        number(("motor", "automatic", "black_threshold"), minimum=0, maximum=1)
        number(("motor", "safety", "max_run_seconds"), minimum=0.1)
        number(("motor", "safety", "black_confirm_frames"), minimum=1, integer=True)
        number(("motor", "safety", "max_missing_frames"), minimum=1, integer=True)
        number(("micrometer", "scale_factor"), minimum=0.000000001)
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
