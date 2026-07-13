"""OpenAI 兼容文本模型 Provider（DeepSeek 可直接使用）。"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


class ProviderCancelled(ProviderError):
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

    def chat(self, messages: list[dict[str, str]],
             cancel_event: threading.Event | None = None) -> str:
        if not self.available:
            raise ProviderError("未配置 API Key")
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderCancelled("请求已取消")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "stream": True,
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
                parts = []
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ProviderCancelled("请求已取消")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    data = json.loads(data_text)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        parts.append(content)
            answer = "".join(parts).strip()
            if not answer:
                raise ProviderError("模型返回了空响应")
            return answer
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            raise ProviderError(f"模型调用失败: HTTP {exc.code} {detail}") from exc
        except ProviderCancelled:
            raise
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ProviderError(f"模型调用失败: {exc}") from exc
