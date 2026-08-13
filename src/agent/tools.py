"""实验助手的确定性数据分析工具。

把「计算与解释」分离：这里只做可追溯的数值计算（不确定度、异常值、
数据表格），结果以结构化文本交给大模型，由大模型负责解释与讨论，而不是
让模型自己算数值（避免幻觉与计算错误）。

只读取运行时快照中的测量数据，不接触硬件控制。
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.config import config
from src.constants import MICROMETER_ACCURACY
from src.measurement import GLASS_REFRACTIVE_INDEX
from src.measurement.uncertainty import (
    DEFAULT_REFRACTIVE_INDEX_TOLERANCE,
    analyze_glass_thickness,
)

# 意图关键词（与 service.py 的 SYSTEM_PROMPT 任务划分保持一致）。
_CALC_KEYWORDS = (
    "误差", "不确定度", "计算", "数据处理", "标准差", "平均值", "有效数字",
    "异常值", "uncertainty", "error",
)
_REPORT_KEYWORDS = (
    "实验报告", "生成报告", "报告模板", "整理报告", "report",
)


def detect_intent(question: str) -> str:
    """判断提问意图：``calculation`` / ``report`` / ``general``。"""
    lowered = question.lower()
    if any(k in lowered for k in _REPORT_KEYWORDS):
        return "report"
    if any(k in lowered for k in _CALC_KEYWORDS):
        return "calculation"
    return "general"


def extract_glass_rounds(context: dict[str, Any]) -> list[dict[str, Any]] | None:
    """从运行时快照中提取玻璃片厚度的多轮测量记录。"""
    if not context:
        return None
    assistant = context.get("measurement", {}).get("experiment_assistant", {})
    if not assistant:
        return None
    session = assistant.get("session", {})
    rounds = session.get("rounds") or []
    return rounds if isinstance(rounds, list) and rounds else None


def _config_value(key: str, default: float) -> float:
    """读取 uncertainty 配置节，缺省时回退到内置默认。"""
    raw = config.uncertainty.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _fmt_number(value: Any) -> str:
    """数值保留 6 位小数显示，非数值原样返回（如 '--'）。"""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def analyze_experiment_assistant(context: dict[str, Any]) -> dict | None:
    """对实验助手已记录的玻璃片厚度做确定性不确定度分析。

    没有测量数据时返回 None，调用方应跳过而不向模型注入结果。
    """
    rounds = extract_glass_rounds(context)
    if not rounds:
        return None
    assistant = context["measurement"]["experiment_assistant"]
    session = assistant.get("session", {})
    n_index = float(session.get(
        "refractive_index", GLASS_REFRACTIVE_INDEX))

    thicknesses = [float(r["thickness_mm"]) for r in rounds]
    d1_values = [float(r["d1_mm"]) for r in rounds]
    d2_values = [float(r["d2_mm"]) for r in rounds]

    return analyze_glass_thickness(
        thicknesses,
        d1_values=d1_values,
        d2_values=d2_values,
        refractive_index=n_index,
        micrometer_accuracy_mm=_config_value(
            "micrometer_accuracy_mm", MICROMETER_ACCURACY),
        refractive_index_tolerance=_config_value(
            "refractive_index_tolerance", DEFAULT_REFRACTIVE_INDEX_TOLERANCE),
    )


def build_deterministic_section(context: dict[str, Any]) -> str:
    """生成一段供大模型引用的确定性数据文本（玻璃片测量 + 不确定度）。

    只有能拿出真实测量数据时才返回非空字符串。
    """
    result = analyze_experiment_assistant(context)
    if not result:
        return ""

    assistant = context["measurement"]["experiment_assistant"]
    session = assistant.get("session", {})
    rounds = session.get("rounds") or []
    stats = assistant.get("statistics", {}) or {}

    lines = [
        "【程序已计算的确定性结果 —— 数值以此为准，不要自行重算】",
        f"测量模型：{result['formula']}，折射率 n = {result['refractive_index']:.4f}",
        f"测量次数：{result['count']}",
        "",
        "各轮测量（轮次 | d1 | d2 | 厚度 h，单位 mm）：",
    ]
    for r in rounds:
        lines.append(
            f"  第{r.get('sequence')}次 | {float(r['d1_mm']):.6f} | "
            f"{float(r['d2_mm']):.6f} | {float(r['thickness_mm']):.6f}"
            + (f"  [{r.get('note')}]" if r.get('note') else ""))
    lines.extend((
        "",
        f"平均值 h̄ = {result['thickness_mean_mm']:.6f} mm",
    ))
    if result["thickness_std_mm"] is not None:
        lines.append(f"样本标准差 s(h) = {result['thickness_std_mm']:.6f} mm")
    lines.extend((
        f"统计摘要：{stats.get('count', result['count'])} 次，"
        f"均值 {_fmt_number(stats.get('mean_mm'))}，"
        f"标准差 {_fmt_number(stats.get('std_mm'))}",
        "",
        "不确定度评定：",
        f"  A 类 u_A = s/√N = {result['type_a_mm']:.6f} mm",
        f"  B 类（微分表允差 {result['type_b_mm']['micrometer_accuracy_mm']:.6f} mm，"
        f"矩形分布）= {result['type_b_mm']['micrometer_contribution_mm']:.6f} mm",
        f"  B 类（折射率允差 {result['type_b_mm']['refractive_index_tolerance']:.4f}，"
        f"矩形分布）= {result['type_b_mm']['refractive_index_contribution_mm']:.6f} mm",
        f"  合成标准不确定度 u_c = {result['combined_uc_mm']:.6f} mm",
        f"  包含因子 k = {result['coverage_factor']:.3f}",
        f"  扩展不确定度 U = {result['expanded_U_mm']:.6f} mm",
        f"  相对不确定度 = {result['relative_uncertainty'] * 100:.2f}%"
        if result["relative_uncertainty"] is not None else "  相对不确定度：无法计算",
        f"  结果表达：h = {result['result_text']}",
    ))
    outlier = result.get("outlier")
    if outlier:
        lines.extend((
            "",
            f"异常值检验（Grubbs，α={outlier['alpha']}）：",
            f"  {outlier['note']}",
        ))
    for warning in result.get("warnings", []):
        lines.append(f"  ⚠ {warning}")
    return "\n".join(lines)


def deterministic_result_json(context: dict[str, Any]) -> str:
    """返回确定性分析结果的紧凑 JSON（供需要结构化数据的场景使用）。"""
    result = analyze_experiment_assistant(context)
    return json.dumps(result, ensure_ascii=False) if result else ""


def parse_options(text: str) -> tuple[str, list[str]]:
    """从助手回答中解析「可点选项」标记行。

    约定格式：回答末尾单独一行 ``【选项】A；B；C``（分隔符可为中文/英文
    分号、顿号或逗号）。返回 ``(去除标记行后的正文, 选项列表)``；无标记时
    原样返回正文、选项为空列表。
    """
    marker = "【选项】"
    index = text.find(marker)
    if index == -1:
        return text, []
    before = text[:index]
    tail = text[index + len(marker):]
    line_end = tail.find("\n")
    if line_end == -1:
        options_part = tail
        after = ""
    else:
        options_part = tail[:line_end]
        after = tail[line_end:]
    options = [part.strip() for part in re.split(r"[；;、,，]", options_part)
               if part.strip()]
    rendered = (before + after).strip()
    return rendered, options


def diagnose_context(context: dict[str, Any]) -> str:
    """基于只读快照用确定性规则给出一句话实时诊断（零 token、无幻觉）。

    只读判断，不触发任何硬件动作；供界面状态栏「AI 洞察」一行展示，
    随 500ms 快照刷新自然更新。字段名与
    ``src/ui/runtime_context.py:build_runtime_context`` 产出的结构一致。
    """
    if not context:
        return "等待实时状态：请先打开设备并连接仪器。"
    vision = context.get("vision", {}) or {}
    motor = context.get("motor", {}) or {}
    micrometer = context.get("micrometer", {}) or {}
    progress = context.get("experiment_progress", {}) or {}

    # 1) 微分表读数过期（连接中且超过 5s 未刷新）
    age = micrometer.get("reading_age_seconds")
    if age is not None and micrometer.get("connected") and float(age) > 5.0:
        return f"微分表读数已过期 {float(age):.1f}s，请等待新读数再记录。"

    # 2) 中心条纹已稳定
    if motor.get("auto_control_state") == "centered":
        return "中心条纹已稳定到达画面中心，可核对读数并记录数据。"

    # 3) 自动寻中运行中
    if motor.get("auto_enabled"):
        return "自动寻中运行中，请保持光路与设备稳定。"

    # 4) 中心条纹偏离画面中心较大
    offset = vision.get("center_offset_px")
    if offset is not None and abs(float(offset)) > 60:
        side = "右" if float(offset) > 0 else "左"
        return f"中心条纹偏离画面中心约 {abs(float(offset)):.0f} px（偏{side}），尚未居中。"

    # 5) 已识别到条纹
    if vision.get("fringe_present"):
        return "已识别到干涉条纹，可进入自动寻中。"

    # 6) 模型 / 预测未就绪
    if not vision.get("model_loaded") or not vision.get("prediction_running"):
        return "条纹尚未识别：模型或预测未就绪，请先完成第 4 步。"

    # 7) 回退到阶段指导
    stage = progress.get("stage") or "未知阶段"
    next_action = progress.get("next_action") or "等待实时状态"
    return f"当前阶段：{stage}；下一步：{next_action}。"
