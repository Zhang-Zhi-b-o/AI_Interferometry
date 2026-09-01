"""OpenAI 兼容文本模型 Provider（DeepSeek 可直接使用）。

除纯文本 ``chat`` 外，还提供支持 function calling 的 ``chat_with_tools``，
用于把实验助手升级为可自主调用工具的智能体：模型通过 ``tools``/``tool_calls``
协议决定调用哪些测量/控制工具，返回结构化的 ``ChatResult`` 供智能体循环消费。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import threading
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


class ProviderCancelled(ProviderError):
    pass


@dataclass
class ChatResult:
    """一次模型调用返回的结构化结果。"""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class DeepSeekProvider:
    def __init__(self, api_base: str, api_key: str, model: str,
                 timeout: float = 30.0, max_tokens: int = 2000):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict],
             cancel_event: threading.Event | None = None,
             max_tokens: int | None = None,
             model: str | None = None,
             thinking: bool | None = None,
             reasoning_effort: str | None = None) -> str:
        result = self._stream(
            messages, tools=None, tool_choice=None,
            cancel_event=cancel_event, max_tokens=max_tokens, model=model,
            thinking=thinking, reasoning_effort=reasoning_effort)
        if not result.content:
            raise ProviderError("模型返回了空响应")
        return result.content

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str = "auto",
        cancel_event: threading.Event | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        """发起一次带工具定义的对话，返回文本与模型请求的 ``tool_calls``。"""
        result = self._stream(
            messages, tools=tools, tool_choice=tool_choice,
            cancel_event=cancel_event, max_tokens=max_tokens, model=model,
            thinking=thinking, reasoning_effort=reasoning_effort)
        if not result.content and not result.tool_calls:
            raise ProviderError("模型返回了空响应")
        return result

    def chat_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        system_prompt: str = "",
        mime_type: str = "image/jpeg",
        detail: str = "low",
        model: str | None = None,
        cancel_event: threading.Event | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """按 DeepSeek/OpenAI 兼容格式发送一张内联图像。"""
        if not image_bytes:
            raise ProviderError("图像数据为空")
        if len(image_bytes) > 32 * 1024 * 1024:
            raise ProviderError("内联图像超过 32 MiB 限制")
        if mime_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ProviderError(f"不支持的图像格式: {mime_type}")
        if detail not in {"low", "high", "original", "auto"}:
            raise ProviderError(f"不支持的图像细节级别: {detail}")
        data = base64.b64encode(image_bytes).decode("ascii")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{data}",
                    "detail": detail,
                }},
            ],
        })
        return self.chat(
            messages,
            cancel_event=cancel_event,
            max_tokens=max_tokens,
            model=model,
            thinking=False,
        )

    def _stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None,
        *,
        cancel_event: threading.Event | None,
        max_tokens: int | None,
        model: str | None,
        thinking: bool | None,
        reasoning_effort: str | None,
    ) -> ChatResult:
        if not self.available:
            raise ProviderError("未配置 API Key")
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderCancelled("请求已取消")
        body = {
            "model": str(model or self.model),
            "messages": messages,
            "max_tokens": int(max_tokens or self.max_tokens),
            "stream": True,
        }
        if thinking is not None:
            body["thinking"] = {
                "type": "enabled" if thinking else "disabled"}
        if thinking:
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
        else:
            body["temperature"] = 0.2
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
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
                parts: list[str] = []
                tool_slots: dict[int, dict[str, str]] = {}
                finish_reason = "stop"
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
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    content = delta.get("content", "")
                    if content:
                        parts.append(content)
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index", 0))
                        slot = tool_slots.setdefault(
                            index, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
            tool_calls = [
                self._finalise_tool_call(tool_slots[index])
                for index in sorted(tool_slots)
            ]
            return ChatResult(
                content="".join(parts).strip(),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
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

    @staticmethod
    def _finalise_tool_call(slot: dict[str, str]) -> dict:
        arguments = {}
        raw = slot.get("arguments", "")
        if raw:
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"_raw": raw}
        return {
            "id": slot.get("id", ""),
            "name": slot.get("name", ""),
            "arguments": arguments,
        }
