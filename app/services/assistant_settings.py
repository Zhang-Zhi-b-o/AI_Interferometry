"""User-owned assistant connection settings, stored outside version control."""
from pathlib import Path
from urllib.parse import urlparse

import yaml

from app.core.config import ROOT


def save_assistant_settings(config, *, api_base, model, api_key="", root=None):
    api_base, model, api_key = api_base.strip().rstrip("/"), model.strip(), api_key.strip()
    url = urlparse(api_base)
    if url.scheme not in ("https", "http") or not url.netloc or url.username or url.password:
        raise ValueError("请填写有效的 API 地址，例如 https://api.deepseek.com")
    if not model:
        raise ValueError("请填写模型名称")
    root = Path(root) if root is not None else ROOT

    def read(path):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"配置文件格式不正确：{path.name}")
        return data or {}

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temporary.replace(path)

    path = root / "config.yaml"
    data = read(path)
    llm = data.setdefault("agent", {}).setdefault("llm", {})
    llm["api_base"] = api_base
    llm.setdefault("models", {})["flash"] = model
    if api_key:
        secrets = root / "config" / "secrets.yaml"
        saved = read(secrets)
        saved["deepseek_api_key"] = api_key
        write(secrets, saved)
    write(path, data)
    # Preserve the shared configuration object used by the other pages.
    config.data.setdefault("agent", {}).setdefault("llm", {}).update(llm)
