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
    def __init__(self, ctx):
        self.ctx = ctx
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
        if name == "read_micrometer":
            return {"value_mm": ctx.reading.value_mm, "stable": ctx.reading.stable,
                    "connected": ctx.reading.connected,
                    "captured_at": ctx.state.reading_at}
        if name == "calculate_thickness":
            return {"thickness_mm": ctx.compute.thickness_mm(**arguments)}
        if name == "evaluate_uncertainty":
            return ctx.compute.uncertainty(arguments["values_mm"])
        if name == "measure_fringe":
            return ctx.fringe.measure(self._frame(), center_x=ctx.state.center_x).as_dict()
        if name == "analyze_thickness_map":
            result = ctx.thickness.analyze(
                self._frame(), refractive_index=ctx.config.refractive_index)
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
