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


SYSTEM_PROMPT = """你是“Michelson AI Lab”的迈克尔逊干涉实验教学搭档。你陪伴实验者完成实验预习、实验过程指导、白光干涉条纹识别、数据处理、误差计算和实验报告整理。

视觉系统会直接提供带快照时间的实时状态，包括两台相机、画面矫正、ROI、YOLO逐目标识别、近场/零级条纹、中心条纹、电机、微分表OCR、测量记录、近期详细日志和 `experiment_progress`。把这些状态自然地融入实验指导，不要主动讨论软件功能边界、模块限制或“程序负责/不负责什么”，避免破坏共同做实验的沉浸感。

固定实验流程（不得跳步）：
1. 调整仪器，放置白光光源。
2. 打开两个摄像头，连接电机。
3. 矫正预览画面，标注 ROI。
4. 加载模型，开始预测，开始自动分析条纹。
5. 开始自动寻中。

你的任务：
1. 实验预习：讲清实验目的、核心原理、仪器作用、关键公式、安全事项、预期现象和容易混淆的概念；可以用简短问题帮助实验者自检。
2. 实验过程：优先读取 `experiment_progress` 的阶段、百分比、下一步和完成判据，再用设备与视觉状态核验。明确说出当前进度，并给出最多三项按顺序执行的操作；每项都说明观察标志。状态快照较旧或关键读数缺失时必须指出，不得用旧状态冒充实时状态。
3. 条纹分析：重点解释远场条纹、近场彩色条纹和零级黑条的特征，协助相机画面、ROI、模型识别、中心定位和电机寻零相关诊断。
4. 数据与误差：先列公式及物理量和单位，再代入用户提供的数据，保留合理有效数字；区分原始读数、计算结果、绝对误差、相对误差和不确定度。数据不足时列出缺少项，绝不补造数值。
5. 实验报告：用户要求生成报告时，必须使用下面的固定结构；已有信息直接整理，缺失内容写“[待补充：具体内容]”，不得虚构。

固定实验报告结构：
# 迈克尔逊干涉实验报告
## 1. 实验目的
## 2. 实验原理
## 3. 实验仪器
## 4. 实验步骤
## 5. 原始数据与实验现象
## 6. 数据处理与误差计算
## 7. 结果分析与讨论
## 8. 实验结论

回答规则：
- 只要提示中出现“当前实验状态”，即使所有值为 false、0 或空，也表示状态已成功收到；此时应说“当前设备尚未启动”，不能说“没有收到实时状态”。只有完全没有该字段时，才能说“我还没有收到实时状态”。
- 过程指导优先使用“现场判断 → 下一步 → 观察标志”的自然顺序；预习、计算和报告任务使用各自最合适的结构，不要机械套用现场格式。
- 当状态包含 `experiment_progress` 时，以其中的 `stage`、`progress_percent`、`next_action` 和 `completion_criterion` 为主，并结合设备与视觉状态解释原因。
- 回答“下一步”时必须按固定五步流程，只推进 `experiment_progress.step_number` 指定的当前步骤；先引用近期日志和实时状态核验，不因识别到后续数据而跳过未完成步骤。
- 日志只作为状态变化和故障诊断证据；若日志与最新快照冲突，以时间更新的快照为准。
- 原理解释要联系装置、光程差、条纹变化和实际可观察现象，避免只背诵定义。
- 默认使用简洁中文，语气冷静、专注、友好，像可靠的实验搭档；不输出资料来源编号、链接或冗长前言。

安全与真实性：
- 绝不编造相机画面、检测结果、实验读数、位移、波长、置信度、误差或不确定度。
- 绝不声称自己已经启动、停止或调节了电机；硬件动作由实验者和确定性控制程序完成。
- 涉及激光、镜片、拆机、接线或电机移动时，先给出必要的安全提醒。
- 不把模型置信度当作测量不确定度，不把像素位置直接当作光程差或镜面位移。
- 不凭空指定项目未提供的接口、连接方式、按钮名称或设备型号。
- 如果资料与实时状态冲突，以实时状态为观察依据，并明确指出冲突。"""


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
        self.history_question_chars = int(llm.get("history_question_chars", 1500))
        self.history_answer_chars = int(llm.get("history_answer_chars", 6000))
        self.context_max_chars = int(llm.get("context_max_chars", 60000))
        self._history: deque[tuple[str, str]] = deque(
            maxlen=max(1, int(llm.get("history_turns", 12))))
        self._history_lock = threading.Lock()
        self.knowledge = KnowledgeBase(
            knowledge_root or PROJECT_ROOT / "src" / "agent" / "knowledge_base")
        self.provider = DeepSeekProvider(
            api_base=llm.get("api_base", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", local_key),
            model=llm.get("model", "deepseek-v4-pro"),
            timeout=float(llm.get("timeout", 30)),
            max_tokens=int(llm.get("max_tokens", 2000)),
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
        if not chunks and not self.provider.available and not context:
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
        status_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(status_json) > self.context_max_chars:
            status_json = status_json[:self.context_max_chars] + "…[状态上下文已截断]"
        status_text = ("\n当前实验状态：" + status_json) if context else ""
        if not references:
            references = "本地知识库未命中。只能回答一般性问题；涉及具体实验事实时应要求用户补充资料。"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        with self._history_lock:
            history = list(self._history)
        for old_question, old_answer in history:
            messages.extend((
                {"role": "user", "content": old_question[:self.history_question_chars]},
                {"role": "assistant", "content": old_answer[:self.history_answer_chars]},
            ))
        messages.append({
            "role": "user",
            "content": f"问题：{question}{status_text}\n\n参考资料：\n{references}",
        })
        try:
            report_request = any(keyword in question.lower() for keyword in (
                "实验报告", "生成报告", "报告模板", "report"))
            output_budget = max(self.provider.max_tokens, 3000) if report_request else None
            answer = self.provider.chat(
                messages, cancel_event=cancel_event, max_tokens=output_budget)
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
        lines = ["当前使用本地实验指导："]
        progress = context.get("experiment_progress", {}) if context else {}
        if progress:
            lines.extend((
                f"现场判断：当前处于“{progress.get('stage', '未知阶段')}”阶段，"
                f"进度 {progress.get('progress_percent', 0)}%。",
                f"下一步：{progress.get('next_action', '请检查设备状态')}。",
                f"观察标志：{progress.get('completion_criterion', '--')}。",
            ))
        for chunk in chunks:
            excerpt = chunk.text[:360].strip()
            lines.append(f"\n{excerpt}")
        if not chunks:
            lines.append("\n当前建议来自实时状态判断；未使用未验证的实验数值。")
        return "\n".join(lines)

    def _remember(self, question: str, answer: str) -> None:
        with self._history_lock:
            self._history.append((
                question[:self.history_question_chars],
                answer[:self.history_answer_chars],
            ))
