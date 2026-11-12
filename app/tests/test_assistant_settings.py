import yaml
import pytest

from app.core.config import AppConfig
from app.services.assistant_settings import save_assistant_settings


def test_api_settings_preserve_device_configuration_and_blank_key(tmp_path):
    (tmp_path / "config.yaml").write_text("motor:\n  port: COM9\n", encoding="utf-8")
    cfg = AppConfig()
    save_assistant_settings(cfg, api_base="https://api.deepseek.com/", model="deepseek-chat",
                            api_key="test-only-key", root=tmp_path)
    save_assistant_settings(cfg, api_base="https://api.deepseek.com", model="another-model", root=tmp_path)
    data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert data["motor"]["port"] == "COM9"
    assert data["agent"]["llm"]["models"]["flash"] == "another-model"
    assert "test-only-key" not in (tmp_path / "config.yaml").read_text(encoding="utf-8")
    secrets = yaml.safe_load((tmp_path / "config" / "secrets.yaml").read_text(encoding="utf-8"))
    assert secrets["deepseek_api_key"] == "test-only-key"
    assert cfg.get("agent.llm.models.flash") == "another-model"


def test_invalid_endpoint_does_not_write_settings(tmp_path):
    with pytest.raises(ValueError):
        save_assistant_settings(AppConfig(), api_base="not a URL", model="test", root=tmp_path)
    assert not (tmp_path / "config.yaml").exists()
