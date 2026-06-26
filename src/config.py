"""配置管理：加载 config.yaml，提供统一访问接口"""
from pathlib import Path
import yaml
from src import PROJECT_ROOT


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
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

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
