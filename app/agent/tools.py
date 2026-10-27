"""v2 唯一工具注册表；不依赖 GUI，运动确认由调用者提供。"""
from __future__ import annotations

from dataclasses import dataclass
import math

from app.agent.context import snapshot


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: str = "READ"
    properties: dict | None = None
    required: tuple = ()

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.properties or {},
                           "required": list(self.required), "additionalProperties": False}}}


NUMBER = {"type": "number"}
TOOLS = (
    Tool("read_state", "读取当前设备与实验状态"),
    Tool("read_timeline", "读取本会话最近的操作时间线"),
    Tool("diagnose", "只读本地状态与趋势，输出确定性诊断，不运动"),
    Tool("search_knowledge", "检索本地资料，返回原文、文件路径、页码与出处编号",
         properties={"query": {"type": "string", "maxLength": 120}}, required=("query",)),
    Tool("generate_report", "由当前已记录数据生成 Word/DOCX、Markdown 与 JSON 报告", "WRITE"),
    Tool("read_micrometer", "读取最近的微分表读数、稳定性和时间，不启动设备"),
    Tool("measure_fringe", "在当前画面测量中心条纹位置、宽度、间距和倾角"),
    Tool("calculate_thickness", "由两次微分表读数 mm 计算玻璃厚度 mm",
         properties={"d1_mm": NUMBER, "d2_mm": NUMBER}, required=("d1_mm", "d2_mm")),
    Tool("evaluate_uncertainty", "对多轮厚度 mm 计算不确定度",
         properties={"values_mm": {"type": "array", "items": NUMBER}},
         required=("values_mm",)),
    Tool("analyze_thickness_map", "分析当前画面的相对厚度起伏，未经标定不能解释为绝对厚度"),
    Tool("start_search", "启动已知方向连续搜索与寻中；需人工确认方向", "MOTION"),
    Tool("stop_search", "停止自动寻中与目标读数闭环", "STOP"),
    Tool("emergency_stop", "停止闭环并立即请求电机停车", "STOP"),
)


class ToolRegistry:
    def __init__(self, ctx, *, report_provider=None, cancel_event=None):
        self.ctx = ctx
        self.report_provider = report_provider
        self.cancel_event = cancel_event
        self.tools = {tool.name: tool for tool in TOOLS}

    def schemas(self):
        return [tool.schema() for tool in TOOLS]

    def validate(self, name: str, arguments: dict) -> Tool:
        if name not in self.tools:
            raise ValueError(f"未知工具：{name}")
        tool = self.tools[name]
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是对象")
        props = tool.properties or {}
        if set(arguments) - props.keys() or set(tool.required) - arguments.keys():
            raise ValueError("工具参数缺失或包含未知字段")
        for key, value in arguments.items():
            if props[key]["type"] == "string":
                if not isinstance(value, str) or not value.strip() or len(value) > 120:
                    raise ValueError(f"{key} 必须是 1–120 字符的文本")
                continue
            values = value if props[key]["type"] == "array" else [value]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{key} 必须是非空数值列表")
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in values):
                raise ValueError(f"{key} 必须是有限数值")
        return tool

    def execute(self, name: str, arguments: dict, *, confirmed: bool = False):
        tool = self.validate(name, arguments)
        if tool.risk == "MOTION" and not confirmed:
            return {"error": "运动未获人工确认"}
        ctx = self.ctx
        if name == "read_state":
            return snapshot(ctx)
        if name == "read_timeline":
            return ctx.timeline.snapshot()
        if name == "diagnose":
            from app.services.diagnostics import diagnose

            return diagnose(snapshot(ctx))
        if name == "search_knowledge":
            from app.services.knowledge_index import default_index

            return default_index().search(arguments["query"], top_k=3)
        if name == "generate_report":
            from app.services.report import generate_report

            result = generate_report(ctx.report_snapshot(), ctx.record.out_dir,
                                     provider=self.report_provider() if self.report_provider else None,
                                     cancel_event=self.cancel_event() if self.cancel_event else None)
            ctx.timeline.add("report_generated", result=result["paths"])
            return result
        if name == "read_micrometer":
            return {"value_mm": ctx.reading.value_mm, "stable": ctx.reading.stable,
                    "connected": ctx.reading.connected,
                    "captured_at": ctx.state.reading_at}
        if name == "calculate_thickness":
            value = ctx.compute.thickness_mm(**arguments)
            if hasattr(ctx, "save_measurement"):
                ctx.save_measurement(arguments["d1_mm"], arguments["d2_mm"], value)
            return {"thickness_mm": value, "inputs": arguments,
                    "refractive_index": ctx.compute.refractive_index}
        if name == "evaluate_uncertainty":
            return ctx.compute.uncertainty(arguments["values_mm"])
        if name == "measure_fringe":
            from app.services.diagnostics import contrast_metric

            frame = self._frame()
            result = ctx.fringe.measure(frame, center_x=ctx.state.center_x).as_dict()
            roi = list(ctx.roi) if ctx.roi else None
            sample = frame
            if roi:
                x, y, w, h = roi
                sample = frame[y:y+h, x:x+w]
            ctx.timeline.add("fringe", result={**result, "contrast": contrast_metric(sample), "roi": roi})
            ctx.state.update(center_x=result["center_x"], fringe_width_px=result["width_px"],
                             fringe_spacing_px=result["spacing_px"], tilt_deg=result["tilt_deg"])
            return result
        if name == "analyze_thickness_map":
            result = ctx.thickness.analyze(
                self._frame(), refractive_index=ctx.config.refractive_index)
            ctx.last_map = {"mode": "relative", "metrics": result["metrics"]}
            ctx.timeline.add("thickness_map", result=ctx.last_map)
            return {"mode": "relative", "metrics": result["metrics"],
                    "warning": "未经标定的相对厚度起伏"}
        if name == "start_search":
            if ctx.search.running or ctx.target.running:
                return {"error": "已有闭环正在运行，请先停止"}
            ok, message = ctx.search.start()
            return {"ok": ok, "message": message}
        ctx.search.stop()
        ctx.target.stop()
        if name == "emergency_stop":
            ctx.motion.stop()
        return {"ok": True, "message": "已请求停止"}

    def _frame(self):
        frame = self.ctx.frame()
        if frame is None:
            raise ValueError("当前没有画面")
        return frame.copy()
