"""实验助手编排服务：只读上下文、检索、生成和离线降级。"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Callable
import json
import os
import threading

from src import PROJECT_ROOT
from src.agent.knowledge import KnowledgeBase, KnowledgeChunk
from src.agent.provider import DeepSeekProvider, ProviderCancelled, ProviderError
from src.config import config
import yaml


SYSTEM_PROMPT = """你是“Michelson AI Lab”的实验台协作伙伴，正在与实验者共同完成迈克尔逊干涉实验。

你的目标不是像百科全书一样泛泛讲解，而是让用户感到你就在实验台旁观察仪器、分析现象并协助推进实验。
你会收到用户问题、经过整理的实验资料，以及可选的实时实验状态。请严格依据这些信息回答。

回答方式：
1. 优先结合当前相机、视觉识别、中心条纹和电机状态判断实验所处阶段。只要提示中出现“当前实验状态”，即使所有值为 false、0 或空，也表示状态已成功收到；此时应说“当前设备尚未启动”，不能说“没有收到实时状态”。只有完全没有“当前实验状态”字段时，才能说“我还没有收到实时状态”。
2. 先用一两句话给出现场判断，例如“目前光路可能尚未进入白光相干范围”或“模型已看到近场条纹，但零级条纹仍不稳定”。
3. 再给出最多三项按顺序执行的下一步，让用户像执行实验流程一样操作；一次不要堆砌大量步骤。
4. 操作过程中说明应该观察什么现象，以及什么现象意味着可以进入下一步。
5. 用户询问原理时，把解释与当前装置、条纹变化和可观察现象联系起来，避免只背诵定义。
6. 用户询问数据时，区分“已知读数”“计算结果”和“仍缺少的数据”；信息不足时直接列出缺少项。
7. 语气冷静、专注、友好，像可靠的实验搭档；可以使用“我们现在先……”“观察屏上如果出现……”等自然表达。
8. 默认使用简洁中文，不输出资料来源编号、链接、冗长前言或固定格式套话。

安全边界：
- 绝不编造相机画面、检测结果、位移、波长、置信度或不确定度。
- 绝不声称自己已经启动、停止或调节了电机；你只能提出建议，硬件动作由用户和确定性控制程序完成。
- 涉及激光、镜片、拆机、接线或电机移动时，先给出必要的安全提醒。
- 不把模型置信度当作测量不确定度，不把推测描述成已经证实的实验结论。
- 不凭空指定项目未提供的接口、连接方式、按钮名称或设备型号，例如不能擅自把 RS-232 说成 USB。
- 如果资料与实时状态冲突，以实时状态为观察依据，并明确指出冲突。

推荐的自然组织方式是“现场判断 → 下一步 → 观察标志”，但只在有助于当前问题时使用，不必机械地显示标题。"""


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
        local_key = self._load_local_api_key()
        self.top_k = int(rag.get("top_k", 4))
        self.context_provider = context_provider
        self._history: deque[tuple[str, str]] = deque(maxlen=4)
        self._history_lock = threading.Lock()
        self.knowledge = KnowledgeBase(
            knowledge_root or PROJECT_ROOT / "src" / "agent" / "knowledge_base")
        self.provider = DeepSeekProvider(
            api_base=llm.get("api_base", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", local_key),
            model=llm.get("model", "deepseek-chat"),
            timeout=float(llm.get("timeout", 30)),
            max_tokens=int(llm.get("max_tokens", 600)),
        )

    @staticmethod
    def _load_local_api_key() -> str:
        """读取被 Git 忽略的本地密钥文件。"""
        secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
        if not secrets_path.exists():
            return ""
        try:
            data = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
            return str(data.get("deepseek_api_key", "")).strip()
        except (OSError, yaml.YAMLError):
            return ""

    def ask(self, question: str, include_status: bool = True,
            context_override: dict | None = None,
            cancel_event: threading.Event | None = None) -> AgentResponse:
        question = question.strip()
        if not question:
            return AgentResponse("请输入问题。", (), False)
        chunks = self.knowledge.search(question, self.top_k)
        if include_status and context_override is not None:
            context = context_override
        else:
            context = self.context_provider() if include_status and self.context_provider else {}
        if not chunks and not self.provider.available:
            return AgentResponse(
                "本地知识库中没有找到足够相关的资料。请换一种表述，或补充实验现象和当前步骤。",
                (), False, "无检索结果")
        if not self.provider.available:
            answer = self._offline_answer(chunks, context)
            self._remember(question, answer)
            return AgentResponse(answer, tuple(chunks), False,
                                 "未配置 API Key，已使用本地检索回答")
        references = "\n\n".join(
            f"[来源{i}] {chunk.title}\n{chunk.text}"
            for i, chunk in enumerate(chunks, 1))
        status_text = ("\n当前实验状态：" + json.dumps(context, ensure_ascii=False,
                                                       separators=(",", ":"))) if context else ""
        if not references:
            references = "本地知识库未命中。只能回答一般性问题；涉及具体实验事实时应要求用户补充资料。"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        with self._history_lock:
            history = list(self._history)
        for old_question, old_answer in history:
            messages.extend((
                {"role": "user", "content": old_question[:600]},
                {"role": "assistant", "content": old_answer[:1000]},
            ))
        messages.append({
            "role": "user",
            "content": f"问题：{question}{status_text}\n\n参考资料：\n{references}",
        })
        try:
            answer = self.provider.chat(messages, cancel_event=cancel_event)
            self._remember(question, answer)
            return AgentResponse(answer, tuple(chunks), True)
        except ProviderCancelled:
            raise
        except ProviderError as exc:
            fallback = self._offline_answer(chunks, context) if chunks else (
                "在线模型调用失败，且本地知识库没有命中相关资料。请检查连接状态或补充资料。")
            return AgentResponse(fallback, tuple(chunks), False, str(exc))

    def test_connection(self, cancel_event: threading.Event | None = None) -> AgentResponse:
        if not self.provider.available:
            return AgentResponse("未读取到 DeepSeek API Key。", (), False, "请检查 config/secrets.yaml")
        try:
            text = self.provider.chat(
                [{"role": "user", "content": "只回复：连接成功"}],
                cancel_event=cancel_event)
            return AgentResponse(f"DeepSeek API 连接成功（模型：{self.provider.model}）。\n{text}", (), True)
        except ProviderError as exc:
            return AgentResponse("DeepSeek API 连接失败。", (), False, str(exc))

    @staticmethod
    def _offline_answer(chunks: list[KnowledgeChunk], context: dict) -> str:
        lines = ["当前使用本地知识库回答："]
        if context:
            lines.append("实验状态摘要：" + json.dumps(
                context, ensure_ascii=False, separators=(",", ":")))
        for chunk in chunks:
            excerpt = chunk.text[:360].strip()
            lines.append(f"\n{excerpt}")
        lines.append("\n如需综合推理，请在本地密钥文件或环境变量中配置 DeepSeek API Key。")
        return "\n".join(lines)

    def _remember(self, question: str, answer: str) -> None:
        with self._history_lock:
            self._history.append((question[:600], answer[:1000]))
