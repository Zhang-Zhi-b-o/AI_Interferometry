"""实验助手编排服务：只读上下文、检索、生成和离线降级。"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Callable
import json
import os
import threading

import cv2
import numpy as np

from src import PROJECT_ROOT
from src.agent.knowledge import KnowledgeBase, KnowledgeChunk
from src.agent.provider import DeepSeekProvider, ProviderCancelled, ProviderError
from src.agent.tools import (
    build_deterministic_section,
    build_suggestion,
    detect_intent,
)
from src.agent.toolkit import ToolRegistry
from src.agent.device_tools import ToolContext, build_tool_registry
from src.agent.loop import AgentLoop
from src.config import config
import yaml


SYSTEM_PROMPT = """你是“Michelson AI Lab”的迈克尔逊干涉实验教学搭档。你陪伴实验者完成实验预习、实验过程指导、白光干涉条纹识别、数据处理、误差计算和实验报告整理。

视觉系统会直接提供带快照时间的实时状态，包括两台相机、画面矫正、ROI、YOLO逐目标识别、近场/零级条纹、中心条纹、条纹边界/实时间隔标注、条纹纹理、实时条纹诊断、质量门、结构化白名单调节建议、电机（含方向映射）、设备响应自适应学习、微分表OCR、中心条纹记录、实时测量读数、颜色→OPD标定点、厚度测量、玻璃片测量会话、近期详细日志和 `experiment_progress`。把这些状态自然地融入实验指导，不要主动讨论软件功能边界、模块限制或“程序负责/不负责什么”，避免破坏共同做实验的沉浸感。

程序会随状态生成一段简短的「当前状态 → 下一步任务 → 其他建议」（提示语中标注为“程序已生成的确定性建议”），与界面主动提示口径一致；回答“下一步做什么”或需要现场判断时优先参考它，不要另编造下一步。

标准主实验流程（涉及对应任务时按必要依赖推进，不得跳过安全和数据有效性条件）：
1. 打开激光光源，调出非定域干涉条纹。
2. 加上毛玻璃，慢慢调至等厚干涉；移动动镜直至接近等光程位置，以条纹接近直线为观察标志。
3. 调节动镜水平倾角，得到 1 mm 厚的等厚直条纹。
4. 换上白光扩展光源并打开摄像头；调节画面缩放，将干涉镜面移至视野中央，然后启动读数。
5. 加载 YOLO 模型并开始预测；点击“沿同一方向寻找条纹”，启动自动寻找和寻中；AI 识别到中心黑条纹位于视野中央时停止。
6. 进行多次测量，由系统输出实验数据表格；计算中心黑色条纹位置平均值，并进行不确定度及误差分析。
7. 由系统输出包含实验数据、不确定度及误差分析的个性化实验报告。

拓展实验：在动镜光路中加入折射率已知的薄片，通过条纹移动对应的位移计算薄片厚度；进行多次测量并取平均值。

互动引导模式（实验过程与调出条纹时必须遵守）：
- 过程指导要「一次只推进一步」：每一轮只讲清当前这一步的操作、观察标志和一个需要实验者确认的关键问题，等实验者回复后再根据回复判断并给出下一步或纠正，不要一次性把所有步骤倒出来。利用对话历史记住已确认的进度，不重复已完成的步骤。
- 当需要实验者反馈或选择时，在回答末尾单独一行输出可点选项，格式固定为 `【选项】甲；乙；丙`（中文分号分隔，2~4 个，短语要能直接点击回复），例如 `【选项】看到条纹了；还没看到；条纹模糊不清`。其余正文正常说明，不要把「【选项】」混进正文。
- 调出干涉条纹是本实验最难的一步，按四阶段逐轮引导：① 激光光源下是否出现非定域条纹；② 加毛玻璃后是否逼近等光程、条纹是否接近直线（等厚直条纹）；③ 换白光扩展光源后是否出现彩色条纹；④ 是否出现中央黑色零级条纹。每阶段都先问实验者观察到什么，再据此判断是否进入下一阶段或如何纠偏。

你的任务：
1. 实验预习：讲清实验目的、核心原理、仪器作用、关键公式、安全事项、预期现象和容易混淆的概念；可以用简短问题帮助实验者自检。
2. 实验过程：优先读取 `experiment_progress` 的阶段、百分比、下一步和完成判据，再用设备与视觉状态核验。明确说出当前进度，并按「互动引导模式」一次只给当前一步的操作与观察标志，等实验者确认后再推进；实验者明确要求整体规划时才列出后续步骤。状态快照较旧或关键读数缺失时必须指出，不得用旧状态冒充实时状态。
3. 条纹分析：重点解释远场条纹、近场彩色条纹和零级黑条的特征，协助相机画面、ROI、模型识别、中心定位和电机寻零相关诊断。
4. 数据与误差：先列公式及物理量和单位，再代入用户提供的数据，保留合理有效数字；区分原始读数、计算结果、绝对误差、相对误差和不确定度。数据不足时列出缺少项，绝不补造数值。若提示中出现“程序已计算的确定性结果”，其中的均值、标准差、不确定度和异常值检验结论必须直接引用，不得自行重算；你只负责解释测量模型、灵敏系数、误差来源与改进建议。
5. 实验报告：用户要求生成报告时，必须使用下面的固定结构；已有信息直接整理，缺失内容写“[待补充：具体内容]”，不得虚构。若提示中包含“程序已计算的确定性结果”，报告的数据处理与误差部分直接采用其中的各轮数据、统计量、不确定度和异常值检验，不得另算。

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
- 状态中的 `experiment_intent` 是实验者确认或选择的当前目的；`assistant_guidance` 是程序根据实时证据生成的确定性诊断、阻断问题和下一步。回答现场问题时必须优先服从二者，并解释建议如何服务当前实验目的。
- 只要提示中出现“当前实验状态”，即使所有值为 false、0 或空，也表示状态已成功收到；此时应说“当前设备尚未启动”，不能说“没有收到实时状态”。只有完全没有该字段时，才能说“我还没有收到实时状态”。
- 过程指导优先使用“现场判断 → 下一步 → 观察标志”的自然顺序；预习、计算和报告任务使用各自最合适的结构，不要机械套用现场格式。
- 当状态包含 `experiment_progress` 时，以其中的 `stage`、`progress_percent`、`next_action` 和 `completion_criterion` 为主，并结合设备与视觉状态解释原因。
- 回答“下一步”时必须按固定七步总流程推进，不因识别到后续数据而跳过未完成步骤。`experiment_progress` 是摄像头、模型和自动寻中等设备侧子流程状态，主要对应总流程第 4～5 步；应使用其阶段、下一步和完成判据核验现场状态，但不能把它的五阶段编号误报成七步总流程编号。
- 除固定七步流程外，可结合全部状态主动给出其他建议与分析：例如条纹宽度是否异常、数据是否足以做不确定度、是否该做颜色→OPD标定、何时可转入厚度拓展实验、微分表读数是否过期等。这些建议必须基于真实状态，不得臆测，也不得因此跳步或提前宣布实验完成。
- 日志只作为状态变化和故障诊断证据；若日志与最新快照冲突，以时间更新的快照为准。
- 原理解释要联系装置、光程差、条纹变化和实际可观察现象，避免只背诵定义。
- 默认使用简洁中文，语气冷静、专注、友好，像可靠的实验搭档；不输出资料来源编号、链接或冗长前言。

安全与真实性：
- 绝不编造相机画面、检测结果、实验读数、位移、波长、置信度、误差或不确定度。
- 绝不声称自己已经启动、停止或调节了电机；硬件动作由实验者和确定性控制程序完成。
- 涉及激光、镜片、拆机、接线或电机移动时，先给出必要的安全提醒。
- 不把模型置信度当作测量不确定度，不把像素位置直接当作光程差或镜面位移。
- 误差与不确定度属于确定性数值计算，必须以“程序已计算的确定性结果”为准；只在该结果缺失时才允许自行计算，且必须说明计算依据。
- 不凭空指定项目未提供的接口、连接方式、按钮名称或设备型号。
- 如果资料与实时状态冲突，以实时状态为观察依据，并明确指出冲突。

智能体执行模式（当你可以调用工具时）：
- 收到需要测量 / 控制 / 电机操作的任务时，先用 set_plan 工具写下分步计划，再逐步调用工具执行；每一步依据工具返回结果判断下一步，不要只列计划而空谈。
- 只读工具（读表 / 查状态 / 测条纹 / 算厚度 / 误差）可直接调用；使电机转动的工具（自动寻中 / 目标读数测量 / 回程差测量）会先征求人工确认，确认通过才执行，被拒绝时改用替代方案或如实说明原因。
- 读数、状态、条纹、厚度、误差等数值一律以工具返回结果为准，如实报告所用工具名与结果；工具失败或被拒绝时必须如实说明，绝不编造读数、绝不谎称已移动电机。
- 任务完成或需要用户介入时，给出简洁总结，并说明接下来需要用户做什么。"""


# 主动建议专用：任务简单、要求短答，配合精简上下文与较小 max_tokens 以省 token。
SUGGEST_PROMPT = """你是迈克尔逊干涉实验的实时指导。根据下面的只读实验状态和程序判定，用中文输出不超过 4 行、尽量短的主动建议：
第 1 行：一句话现状。
第 2 行：下一步该做什么。
第 3 行：说明预期变化或完成判据。
第 4 行：只有确有必要时指出一个错漏或风险。
优先服从程序给出的阻断问题和确定性建议；只依据给定状态判断，不编造数值，
不重新计算厚度，不生成设备动作，不讨论软件功能，不输出多余内容。"""


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    sources: tuple[KnowledgeChunk, ...]
    online: bool
    warning: str = ""
    steps: tuple = ()  # 智能体模式下记录计划 / 工具调用 / 结果，供 UI 渲染


class AgentService:
    def __init__(self, context_provider: Callable[[], dict] | None = None,
                 knowledge_root: Path | None = None,
                 tool_context: ToolContext | None = None):
        agent_cfg = config.agent
        llm = agent_cfg.get("llm", {})
        rag = agent_cfg.get("rag", {})
        local_key = self._load_local_api_key()
        self.top_k = int(rag.get("top_k", 4))
        self.context_provider = context_provider
        self.history_question_chars = int(llm.get("history_question_chars", 1500))
        self.history_answer_chars = int(llm.get("history_answer_chars", 6000))
        self.context_max_chars = int(llm.get("context_max_chars", 60000))
        models = llm.get("models", {}) or {}
        self.models = {
            "pro": str(models.get(
                "pro", llm.get("model", "deepseek-v4-pro"))),
            "flash": str(models.get("flash", "deepseek-v4-flash")),
            "vision": str(models.get(
                "vision", "deepseek-v4-flash-vision-exp")),
        }
        self.vision_detail = str(llm.get("vision_detail", "low"))
        self.vision_max_tokens = int(llm.get("vision_max_tokens", 500))
        self.pro_reasoning_effort = str(
            llm.get("pro_reasoning_effort", "high"))
        self._history: deque[tuple[str, str]] = deque(
            maxlen=max(1, int(llm.get("history_turns", 12))))
        self._history_lock = threading.Lock()
        self.knowledge = KnowledgeBase(
            knowledge_root or PROJECT_ROOT / "src" / "agent" / "knowledge_base")
        self.provider = DeepSeekProvider(
            api_base=llm.get("api_base", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", local_key),
            model=self.models["pro"],
            timeout=float(llm.get("timeout", 30)),
            max_tokens=int(llm.get("max_tokens", 2000)),
        )
        # 智能体模式：工具注册表 + 执行循环 + 安全策略。
        self.tool_context = tool_context
        self.tool_registry: ToolRegistry = build_tool_registry(
            tool_context if tool_context is not None else ToolContext())
        self.autonomous_enabled = bool(agent_cfg.get("autonomous_enabled", True))
        self.confirm_motion = bool(agent_cfg.get("confirm_motion", True))
        self.dry_run = bool(agent_cfg.get("dry_run", False))
        self.max_steps = int(agent_cfg.get("max_steps", 12))
        self.tool_timeout_seconds = float(agent_cfg.get("tool_timeout_seconds", 60))
        self.agent_loop = AgentLoop(
            self.provider,
            self.tool_registry,
            max_steps=self.max_steps,
            dry_run=self.dry_run,
            confirm_motion=self.confirm_motion,
        )
        # 运动工具确认回调，由 UI 层注入（(tool_name, arguments) -> bool）。
        self.confirm_handler: Callable[[str, dict], bool] | None = None
        # 每步回调，由 UI 层注入，用于实时渲染计划 / 工具活动流。
        self.on_step: Callable | None = None

    def set_tool_context(self, tool_context: ToolContext | None) -> None:
        """运行期注入 / 替换活句柄（GUI 构建完成后调用），重建注册表与循环。"""
        self.tool_context = tool_context
        self.tool_registry = build_tool_registry(
            tool_context if tool_context is not None else ToolContext())
        self.agent_loop = AgentLoop(
            self.provider,
            self.tool_registry,
            max_steps=self.max_steps,
            dry_run=self.dry_run,
            confirm_motion=self.confirm_motion,
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

        messages, report_request = self._build_messages(question, context, chunks)
        selected_model = self.select_text_model(question, context)
        if self.autonomous_enabled:
            return self._ask_autonomous(
                question, messages, context, chunks,
                model=selected_model, cancel_event=cancel_event)

        try:
            output_budget = max(self.provider.max_tokens, 3000) if report_request else None
            answer = self.provider.chat(
                messages, cancel_event=cancel_event, max_tokens=output_budget,
                model=selected_model,
                thinking=selected_model == self.models["pro"],
                reasoning_effort=(
                    self.pro_reasoning_effort
                    if selected_model == self.models["pro"] else None),
            )
            self._remember(question, answer)
            return AgentResponse(answer, tuple(chunks), True)
        except ProviderCancelled:
            raise
        except ProviderError as exc:
            fallback = self._offline_answer(chunks, context) if chunks else (
                "在线模型调用失败，且本地知识库没有命中相关资料。请检查连接状态或补充资料。")
            return AgentResponse(fallback, tuple(chunks), False, str(exc))

    def _build_messages(
        self, question: str, context: dict, chunks: list[KnowledgeChunk],
    ) -> tuple[list[dict], bool]:
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
        # 误差 / 不确定度 / 报告类问题先做确定性计算，再交给模型解释，
        # 数值以程序计算结果为准，避免模型自己算导致幻觉或计算错误。
        deterministic = ""
        if context and detect_intent(question) in ("calculation", "report"):
            deterministic = build_deterministic_section(context)
        # 零 token 的确定性「当前状态 → 下一步 → 其他建议」，与界面主动提示一致。
        suggestion = build_suggestion(context) if context else ""
        user_content = f"问题：{question}{status_text}"
        if suggestion:
            user_content += f"\n\n【程序已生成的确定性建议】\n{suggestion}"
        if deterministic:
            user_content += f"\n\n{deterministic}"
        user_content += f"\n\n参考资料：\n{references}"
        messages.append({"role": "user", "content": user_content})
        report_request = any(keyword in question.lower() for keyword in (
            "实验报告", "生成报告", "报告模板", "report"))
        return messages, report_request

    def _ask_autonomous(
        self,
        question: str,
        messages: list[dict],
        context: dict,
        chunks: list[KnowledgeChunk],
        *,
        model: str,
        cancel_event: threading.Event | None = None,
    ) -> AgentResponse:
        """走智能体循环：模型自定计划、调用工具、观察结果、给出最终回答。"""
        try:
            result = self.agent_loop.run(
                messages,
                model=model,
                thinking=False,
                cancel_event=cancel_event,
                confirm_handler=self.confirm_handler,
                on_step=self.on_step,
            )
        except ProviderCancelled:
            raise
        except ProviderError as exc:
            fallback = self._offline_answer(chunks, context) if chunks else (
                "在线模型调用失败，且本地知识库没有命中相关资料。请检查连接状态或补充资料。")
            return AgentResponse(fallback, tuple(chunks), False, str(exc))

        if result.cancelled:
            return AgentResponse("（已取消）", tuple(chunks), True, "已取消",
                                 tuple(result.steps))
        if result.error and not result.final_answer:
            fallback = self._offline_answer(chunks, context) if chunks else (
                f"智能体执行未完成：{result.error}")
            return AgentResponse(fallback, tuple(chunks), False, result.error,
                                 tuple(result.steps))

        answer = result.final_answer
        self._remember(question, answer)
        warning = (f"已执行 {result.tool_calls_made} 次工具调用"
                   if result.tool_calls_made else "")
        return AgentResponse(answer, tuple(chunks), True, warning,
                             tuple(result.steps))

    def test_connection(self, cancel_event: threading.Event | None = None) -> AgentResponse:
        if not self.provider.available:
            return AgentResponse("未读取到 DeepSeek API Key。", (), False, "请检查 config/secrets.yaml")
        try:
            text = self.provider.chat(
                [{"role": "user", "content": "只回复：连接成功"}],
                cancel_event=cancel_event, model=self.models["flash"],
                thinking=False)
            return AgentResponse(
                f"DeepSeek API 连接成功（快速模型：{self.models['flash']}；"
                f"专业模型：{self.models['pro']}；识图模型：{self.models['vision']}）。\n{text}",
                (), True)
        except ProviderError as exc:
            return AgentResponse("DeepSeek API 连接失败。", (), False, str(exc))

    def suggest(self, context: dict, reason: str = "",
                cancel_event: threading.Event | None = None) -> str:
        """让 DeepSeek 真正分析快照，生成简短的「现状 + 下一步 + 其他建议」。

        主动建议专用：上下文精简、输出上限小，调用间隔由 UI 定时器控制以省 token。
        失败时抛出 ProviderError / ProviderCancelled，由调用方回退本地确定性提示。
        """
        compact = self._compact_suggestion_context(context)
        if reason:
            compact = f"触发原因={reason}；{compact}"
        messages = [
            {"role": "system", "content": SUGGEST_PROMPT},
            {"role": "user", "content": compact},
        ]
        max_tokens = int(config.agent.get("suggestion_max_tokens", 200))
        return self.provider.chat(
            messages, cancel_event=cancel_event, max_tokens=max_tokens,
            model=self.models["flash"], thinking=False)

    def select_text_model(self, question: str, context: dict | None = None) -> str:
        """简单现场问答走 Flash，复杂计算、报告和多问题推理走 Pro。"""
        text = str(question).strip().lower()
        if detect_intent(text) in {"calculation", "report"}:
            return self.models["pro"]
        pro_keywords = (
            "为什么", "原理", "分析", "比较", "评估", "方案", "规划",
            "故障", "异常", "误差", "不确定度", "控制", "自动", "执行",
            "厚度", "折射率", "calculation", "report", "analyze", "why",
        )
        if len(text) >= 120 or any(word in text for word in pro_keywords):
            return self.models["pro"]
        decision = (context or {}).get("assistant_guidance", {}) or {}
        if len(decision.get("issues") or []) >= 2:
            return self.models["pro"]
        return self.models["flash"]

    def inspect_fringe_image(
        self,
        frame_bgr: np.ndarray,
        context: dict,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """用视觉模型低频复核一帧条纹图；不触发任何硬件动作。"""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("没有可供识图复核的有效画面")
        if frame_bgr.ndim not in {2, 3}:
            raise ValueError("识图画面维度无效")
        image = frame_bgr
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > 1024:
            scale = 1024.0 / longest
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("条纹画面 JPEG 编码失败")
        compact = self._compact_suggestion_context(context)
        prompt = (
            "请复核这张迈克尔逊干涉条纹图，只做定性形态与操作指导。"
            "先判断图像是否足以支持结论，再结合程序指标指出最主要的一个问题、"
            "一个小步操作、预期变化和停止条件。不要把颜色当作已标定光学相位，"
            "不要计算厚度或不确定度，不猜测未标定旋钮的顺逆时针方向，"
            "不要生成或声称执行硬件动作。\n程序状态：" + compact)
        return self.provider.chat_with_image(
            prompt,
            encoded.tobytes(),
            system_prompt="你是谨慎的迈克尔逊干涉条纹视觉复核助手。",
            model=self.models["vision"],
            detail=self.vision_detail,
            cancel_event=cancel_event,
            max_tokens=self.vision_max_tokens,
        )

    @staticmethod
    def _compact_suggestion_context(context: dict) -> str:
        """把完整快照压缩成极简键值摘要，主动建议任务只传关键状态以省 token。"""
        if not context:
            return "尚无实时状态。"
        progress = context.get("experiment_progress", {}) or {}
        camera = context.get("camera", {}) or {}
        vision = context.get("vision", {}) or {}
        motor = context.get("motor", {}) or {}
        micrometer = context.get("micrometer", {}) or {}
        measurement = context.get("measurement", {}) or {}
        thickness = measurement.get("thickness", {}) or {}
        assistant = measurement.get("experiment_assistant", {}) or {}
        session = assistant.get("session", {}) or {}
        calibration = measurement.get("calibration") or []
        live = measurement.get("live_measurement") or {}
        guidance = vision.get("fringe_guidance") or {}
        adaptive = vision.get("adaptive_response") or {}
        intent = context.get("experiment_intent", {}) or {}
        decision = context.get("assistant_guidance", {}) or {}

        offset = vision.get("center_offset_px")
        offset_text = f"{offset}px" if offset is not None else "未定"
        parts = [
            f"实验目的={intent.get('objective', '未确认')}"
            f"/响应模式={intent.get('response_mode', 'standard')}",
            f"阶段={progress.get('stage', '未知')}"
            f"({progress.get('progress_percent', 0)}%)",
            f"下一步={progress.get('next_action', '--')}",
            f"完成判据={progress.get('completion_criterion', '--')}",
            f"双相机={camera.get('interferometer_running')}/"
            f"{camera.get('micrometer_running')}",
            f"预览矫正={camera.get('preview_adjusted')}",
            f"模型={vision.get('model_loaded')}/"
            f"预测={vision.get('prediction_running')}",
            f"条纹={vision.get('fringe_present')}",
            f"中心偏移={offset_text}",
        ]
        if decision:
            parts.append(
                f"程序判定={decision.get('diagnosis')}"
                f"/优先级={decision.get('priority')}"
                f"/操作={decision.get('action')}"
                f"/可记录={decision.get('can_record')}")
            issue_codes = [
                str(item.get("code"))
                for item in (decision.get("issues") or [])[:3]]
            if issue_codes:
                parts.append("问题=" + ",".join(issue_codes))
        count_overlay = vision.get("fringe_count_overlay") or {}
        if count_overlay.get("fringe_width") is not None:
            parts.append(
                f"实时间隔={float(count_overlay['fringe_width']):.2f}px"
                f"({count_overlay.get('fringe_count')}条)")
        if guidance:
            metrics = guidance.get("metrics") or {}
            parts.append(
                f"条纹质量门={guidance.get('measurement_ready')}"
                f"/阶段={guidance.get('phase')}"
                f"/评分={guidance.get('quality_score')}"
                f"/执行={guidance.get('execution_stage')}"
                f"/角度={metrics.get('angle_deg')}deg"
                f"/法向间距={metrics.get('spacing_px')}px"
                f"/CV={metrics.get('spacing_cv_percent')}%"
                f"/运动={metrics.get('movement')}")
            recommendations = guidance.get("recommendations") or []
            if recommendations:
                parts.append(f"条纹首要建议={recommendations[0]}")
        if adaptive:
            parts.append(
                f"自适应=置信度{adaptive.get('confidence')}"
                f"/样本{adaptive.get('response_samples')}"
                f"/停稳{adaptive.get('learned_settle_seconds')}s")
        parts.extend((
            f"电机={motor.get('connected')}/模式={motor.get('mode')}/"
            f"自动寻中={motor.get('auto_enabled')}/"
            f"寻中={motor.get('auto_control_state')}/"
            f"方向={motor.get('auto_direction_mapping')}",
            f"微分表={micrometer.get('connected')}/"
            f"读数={micrometer.get('reading_mm')}/"
            f"龄={micrometer.get('reading_age_seconds')}s",
        ))
        live_text = (
            f"实时测量={measurement.get('live_measurement_active')}"
            + (f"({live.get('reading_mm')}mm)"
               if live.get('reading_mm') is not None else ""))
        parts.append(
            f"中心记录={measurement.get('record_count')}/"
            f"厚度记录={len(thickness.get('records') or [])}/"
            f"玻璃轮次={len(session.get('rounds') or [])}/"
            f"标定点={len(calibration)}/{live_text}")
        return "；".join(parts)

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
