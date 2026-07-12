"""实验助手编排服务：只读上下文、检索、生成和离线降级。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import os

from src import PROJECT_ROOT
from src.agent.knowledge import KnowledgeBase, KnowledgeChunk
from src.agent.provider import DeepSeekProvider, ProviderError
from src.config import config


SYSTEM_PROMPT = """你是迈克尔逊干涉实验辅助智能体。请仅依据给定资料和实验状态回答。
规则：1. 不编造未提供的读数；2. 不声称已执行硬件操作；3. 涉及电机、激光或拆机时给出安全提醒；
4. 数值不足时明确说明缺少什么；5. 回答简洁，优先给判断依据和下一步；6. 使用[来源N]标注依据。"""


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    sources: tuple[KnowledgeChunk, ...]
    online: bool
    warning: str = ""


class AgentService:
    def __init__(self, context_provider: Callable[[], dict] | None = None,
                 knowledge_root: Path | None = None):
        agent_cfg = config.agent
        llm = agent_cfg.get("llm", {})
        rag = agent_cfg.get("rag", {})
        self.top_k = int(rag.get("top_k", 4))
        self.context_provider = context_provider
        self.knowledge = KnowledgeBase(
            knowledge_root or PROJECT_ROOT / "src" / "agent" / "knowledge_base")
        self.provider = DeepSeekProvider(
            api_base=llm.get("api_base", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", llm.get("api_key", "")),
            model=llm.get("model", "deepseek-chat"),
            timeout=float(llm.get("timeout", 30)),
            max_tokens=int(llm.get("max_tokens", 600)),
        )

    def ask(self, question: str, include_status: bool = True) -> AgentResponse:
        question = question.strip()
        if not question:
            return AgentResponse("请输入问题。", (), False)
        chunks = self.knowledge.search(question, self.top_k)
        context = self.context_provider() if include_status and self.context_provider else {}
        if not chunks:
            return AgentResponse(
                "本地知识库中没有找到足够相关的资料。请换一种表述，或补充实验现象和当前步骤。",
                (), False, "无检索结果")
        if not self.provider.available:
            return AgentResponse(self._offline_answer(chunks, context), tuple(chunks), False,
                                 "未配置 API Key，已使用本地检索回答")
        references = "\n\n".join(
            f"[来源{i}] {chunk.title}\n{chunk.text}"
            for i, chunk in enumerate(chunks, 1))
        status_text = f"\n当前实验状态：{context}" if context else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}{status_text}\n\n参考资料：\n{references}"},
        ]
        try:
            answer = self.provider.chat(messages)
            return AgentResponse(answer, tuple(chunks), True)
        except ProviderError as exc:
            return AgentResponse(self._offline_answer(chunks, context), tuple(chunks), False, str(exc))

    @staticmethod
    def _offline_answer(chunks: list[KnowledgeChunk], context: dict) -> str:
        lines = ["当前使用本地知识库回答："]
        if context:
            lines.append(f"实验状态摘要：{context}")
        for i, chunk in enumerate(chunks, 1):
            excerpt = chunk.text[:360].strip()
            lines.append(f"\n[来源{i}] {excerpt}")
        lines.append("\n如需综合推理，请在 config.yaml 或环境变量中配置 DeepSeek API Key。")
        return "\n".join(lines)
