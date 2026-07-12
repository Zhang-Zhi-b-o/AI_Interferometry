"""OpenAI 兼容文本模型 Provider（DeepSeek 可直接使用）。"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


class DeepSeekProvider:
    def __init__(self, api_base: str, api_key: str, model: str,
                 timeout: float = 30.0, max_tokens: int = 600):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.available:
            raise ProviderError("未配置 API Key")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            raise ProviderError(f"模型调用失败: HTTP {exc.code} {detail}") from exc
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ProviderError(f"模型调用失败: {exc}") from exc
