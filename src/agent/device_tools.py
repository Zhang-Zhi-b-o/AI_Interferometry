"""把「除参数设置以外」的测量 / 控制 / 电机能力包装成智能体工具。

工具分三类风险（见 :mod:`src.agent.toolkit`）：

- 只读（READ，自动执行）：读表、查状态、测条纹、算厚度、不确定度分析；
- 运动（MOTION，需人工确认）：自动寻中、目标读数测量、回程差测量；
- 停止（STOP，始终放行）：急停及各类停止。

关键设计：智能体只编排**高层操作**（开 / 停自动寻中、目标读数测量、回程差测量、
单帧分析），**不逐拍驱动电机**。低层 PID / 寻中 / 停车确认仍由 GUI 现有实时循环
安全执行。因此这里的运动工具通过 ``ToolContext`` 注入的高层回调触发，回调本身
线程安全（内部走 ``SerialCommandQueue`` 或 GUI 状态标志）。

``ToolContext`` 全部回调默认 ``None``，可在无 GUI（CLI / 测试）环境安全构造；
缺失句柄时运动工具返回「需在 GUI 内运行」而非抛异常。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.toolkit import Tool, ToolRegistry, ToolRisk
from src.constants import MICROMETER_ACCURACY
from src.measurement.thickness import GLASS_REFRACTIVE_INDEX, calculate_thickness_mm
from src.measurement.uncertainty import (
    DEFAULT_REFRACTIVE_INDEX_TOLERANCE,
    analyze_glass_thickness,
)
from src.vision.fringe_width import measure_center_fringe_width
from src.vision.thickness_distribution import (
    analyze_thickness_distribution,
    sample_colour,
)


@dataclass
class ToolContext:
    """GUI 注入的活句柄与回调集合；全部线程安全，缺失时安全降级。"""

    # 只读快照与当前画面
    get_snapshot: Callable[[], dict] | None = None          # 实时状态快照
    latest_frame: Callable[[], Any] | None = None           # 当前矫正帧（BGR ndarray）
    read_micrometer: Callable[[], float | None] | None = None
    query_motor: Callable[[], dict | None] | None = None
    center_line_x: Callable[[], float | None] | None = None
    frame_width: Callable[[], float | None] | None = None

    # 高层操作回调（运动 / 停止）
    start_auto_center: Callable[[], dict] | None = None
    stop_auto_center: Callable[[], dict] | None = None
    start_measurement: Callable[[float | None], dict] | None = None
    stop_measurement: Callable[[], dict] | None = None
    start_backlash: Callable[[float, float], dict] | None = None
    stop_backlash: Callable[[], dict] | None = None
    motor_emergency_stop: Callable[[], dict] | None = None

    # 把函数调度到 UI 主线程执行（root.after + Event），非 UI 环境为 None
    run_on_main: Callable[[Callable], Any] | None = None

    # 计划 / 记录回调，供 UI 展示与续接
    on_plan: Callable[[str], None] | None = None
    on_note: Callable[[str], None] | None = None

    def snapshot(self) -> dict:
        if self.get_snapshot is None:
            return {}
        try:
            value = self.get_snapshot()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def frame(self) -> Any:
        if self.latest_frame is None:
            return None
        try:
            return self.latest_frame()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _run(ctx: ToolContext, fn: Callable[[], Any]) -> Any:
    """有 UI 主线程调度器时经主线程执行，否则直接执行。"""
    if ctx.run_on_main is not None:
        return ctx.run_on_main(fn)
    return fn()


def _require_gui(callback: Any, name: str) -> str:
    if callback is None:
        return f"{name} 需要在 GUI 内运行（当前无硬件句柄）"
    return ""


# -- 只读 -----------------------------------------------------------------


def _get_context(args: dict, ctx: ToolContext) -> dict:
    return {"context": ctx.snapshot()}


def _micrometer_read(args: dict, ctx: ToolContext) -> dict:
    value = None
    if ctx.read_micrometer is not None:
        value = ctx.read_micrometer()
    if value is None:
        snap = ctx.snapshot()
        meter = (snap.get("micrometer") or {})
        value = meter.get("reading_mm")
    if value is None:
        return {"connected": bool(ctx.read_micrometer), "reading_mm": None,
                "note": "当前无稳定微分表读数，请先启动读数"}
    return {"reading_mm": float(value)}


def _motor_status(args: dict, ctx: ToolContext) -> dict:
    if ctx.query_motor is not None:
        status = ctx.query_motor()
        if isinstance(status, dict):
            return status
    snap = ctx.snapshot()
    motor = snap.get("motor") or {}
    return {
        "connected": bool(motor.get("connected")),
        "running": bool(motor.get("running")),
        "speed": motor.get("speed"),
        "direction": motor.get("direction"),
        "auto_enabled": bool(motor.get("auto_enabled")),
        "auto_control_state": motor.get("auto_control_state"),
        "note": "状态来自运行时快照" if ctx.query_motor is None else "状态来自电机串口",
    }


def _fringe_center_status(args: dict, ctx: ToolContext) -> dict:
    center_x = ctx.center_line_x() if ctx.center_line_x is not None else None
    width = ctx.frame_width() if ctx.frame_width is not None else None
    snap = ctx.snapshot()
    vision = snap.get("vision") or {}
    motor = snap.get("motor") or {}
    if center_x is None:
        center_x = vision.get("center_line_x")
    if width is None:
        width = vision.get("frame_width")
    return {
        "center_line_x": center_x,
        "frame_width": width,
        "fringe_present": bool(vision.get("fringe_present")),
        "auto_control_state": motor.get("auto_control_state"),
        "centered": motor.get("auto_control_state") == "centered",
    }


def _fringe_width(args: dict, ctx: ToolContext) -> dict:
    frame = ctx.frame()
    if frame is None:
        return {"error": "当前无可用画面，请先打开相机"}
    center_x = args.get("center_x")
    result = measure_center_fringe_width(frame, center_x=center_x)
    bands = result.get("bands") or []
    return {
        "period_px": result["period_px"],
        "num_bands": result["num_bands"],
        "num_bright": result["num_bright"],
        "num_dark": result["num_dark"],
        "reference_x": result["reference_x"],
        "center_band": result.get("center_band"),
        "frame_width": result["frame_width"],
    }


def _thickness_analyze(args: dict, ctx: ToolContext) -> dict:
    frame = ctx.frame()
    if frame is None:
        return {"error": "当前无可用画面，请先打开相机"}
    result = analyze_thickness_distribution(frame)
    return {
        "mode": result["mode"],
        "step_um": result["step_um"],
        "metrics": result["metrics"],
    }


def _sample_colour(args: dict, ctx: ToolContext) -> dict:
    frame = ctx.frame()
    if frame is None:
        return {"error": "当前无可用画面，请先打开相机"}
    r, g, b = sample_colour(frame)
    return {"r": r, "g": g, "b": b}


def _glass_thickness(args: dict, ctx: ToolContext) -> dict:
    d1 = float(args["d1_mm"])
    d2 = float(args["d2_mm"])
    n = float(args.get("refractive_index", GLASS_REFRACTIVE_INDEX))
    h = calculate_thickness_mm(d1, d2, n)
    return {
        "thickness_mm": h,
        "d1_mm": d1,
        "d2_mm": d2,
        "refractive_index": n,
        "formula": "h = (d2 - d1) / [20 × (n - 1)]",
    }


def _uncertainty(args: dict, ctx: ToolContext) -> dict:
    thickness = args.get("thickness_values")
    if thickness is None:
        # 从运行时快照的测量记录做确定性不确定度分析
        from src.agent.tools import analyze_experiment_assistant
        result = analyze_experiment_assistant(ctx.snapshot())
        if result is None:
            return {"error": "没有可分析的测量数据：请提供 thickness_values，或先完成玻璃片测量"}
        return result
    thickness = [float(v) for v in thickness]
    d1 = args.get("d1_values") or []
    d2 = args.get("d2_values") or []
    n = float(args.get("refractive_index", GLASS_REFRACTIVE_INDEX))
    return analyze_glass_thickness(
        thickness,
        d1_values=[float(v) for v in d1] if d1 else None,
        d2_values=[float(v) for v in d2] if d2 else None,
        refractive_index=n,
        micrometer_accuracy_mm=float(
            args.get("micrometer_accuracy_mm", MICROMETER_ACCURACY)),
        refractive_index_tolerance=float(
            args.get("refractive_index_tolerance",
                     DEFAULT_REFRACTIVE_INDEX_TOLERANCE)),
    )


def _set_plan(args: dict, ctx: ToolContext) -> dict:
    plan = str(args.get("plan", "")).strip()
    if not plan:
        return {"ok": False, "error": "plan 不能为空"}
    if ctx.on_plan is not None:
        ctx.on_plan(plan)
    return {"ok": True, "plan": plan}


def _record_note(args: dict, ctx: ToolContext) -> dict:
    note = str(args.get("note", "")).strip()
    if not note:
        return {"ok": False, "error": "note 不能为空"}
    if ctx.on_note is not None:
        ctx.on_note(note)
    return {"ok": True, "note": note}


def _session_stats(args: dict, ctx: ToolContext) -> dict:
    snap = ctx.snapshot()
    measurement = snap.get("measurement") or {}
    assistant = measurement.get("experiment_assistant") or {}
    session = assistant.get("session") or {}
    return {
        "rounds": len(session.get("rounds") or []),
        "statistics": assistant.get("statistics") or {},
        "has_reference": bool((measurement.get("reference") or {}).get("present")),
    }


# -- 运动（需确认） -------------------------------------------------------


def _auto_center_start(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.start_auto_center, "auto_center_start")
    if err:
        return {"error": err}
    return _run(ctx, ctx.start_auto_center)


def _measurement_start(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.start_measurement, "measurement_start")
    if err:
        return {"error": err}
    target_mm = args.get("target_mm")
    target_mm = float(target_mm) if target_mm is not None else None
    return _run(ctx, lambda: ctx.start_measurement(target_mm))


def _backlash_measure(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.start_backlash, "backlash_measure")
    if err:
        return {"error": err}
    start_mm = float(args["start_mm"])
    end_mm = float(args["end_mm"])
    return _run(ctx, lambda: ctx.start_backlash(start_mm, end_mm))


# -- 停止（始终放行） -----------------------------------------------------


def _motor_emergency_stop(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.motor_emergency_stop, "motor_emergency_stop")
    if err:
        return {"error": err}
    return _run(ctx, ctx.motor_emergency_stop)


def _auto_center_stop(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.stop_auto_center, "auto_center_stop")
    if err:
        return {"error": err}
    return _run(ctx, ctx.stop_auto_center)


def _measurement_stop(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.stop_measurement, "measurement_stop")
    if err:
        return {"error": err}
    return _run(ctx, ctx.stop_measurement)


def _backlash_stop(args: dict, ctx: ToolContext) -> dict:
    err = _require_gui(ctx.stop_backlash, "backlash_stop")
    if err:
        return {"error": err}
    return _run(ctx, ctx.stop_backlash)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

_OBJECT = {"type": "object", "properties": {}, "required": []}

_NUMBER = {"type": "number"}


def _schema(properties: dict[str, dict], required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


_TOOLS: list[tuple[str, str, ToolRisk, dict, Callable[[dict, ToolContext], Any]]] = [
    # -- 只读 --
    ("get_context", "读取当前实验运行时状态快照（电机/读表/视觉/进度/测量记录）",
     ToolRisk.READ, _OBJECT, _get_context),
    ("micrometer_read", "读取数显微分表当前稳定读数（mm）",
     ToolRisk.READ, _OBJECT, _micrometer_read),
    ("motor_status", "查询步进电机当前状态（是否运行/档位/方向/自动寻中状态）",
     ToolRisk.READ, _OBJECT, _motor_status),
    ("fringe_center_status", "查询中心条纹当前定位状态（中心线 x、视场宽度、是否已居中）",
     ToolRisk.READ, _OBJECT, _fringe_center_status),
    ("fringe_width_analyze", "分析当前画面中心条纹宽度（周期、明暗条纹数、中心条纹轮廓）",
     ToolRisk.READ, _schema({"center_x": _NUMBER}, []), _fringe_width),
    ("thickness_analyze", "对当前画面做单帧薄膜厚度分布估计，返回统计指标",
     ToolRisk.READ, _OBJECT, _thickness_analyze),
    ("sample_colour", "采样当前画面薄膜区域的中位颜色 (r,g,b)",
     ToolRisk.READ, _OBJECT, _sample_colour),
    ("glass_thickness_calculate", "按 h=(d2-d1)/[20×(n-1)] 由两次微分表读数计算玻璃片厚度（mm）",
     ToolRisk.READ,
     _schema({"d1_mm": _NUMBER, "d2_mm": _NUMBER,
              "refractive_index": _NUMBER}, ["d1_mm", "d2_mm"]),
     _glass_thickness),
    ("uncertainty_analyze", "对玻璃片厚度测量做 GUM 不确定度评定（A/B 类、合成、扩展、异常值）",
     ToolRisk.READ,
     _schema({"thickness_values": {"type": "array", "items": _NUMBER},
              "d1_values": {"type": "array", "items": _NUMBER},
              "d2_values": {"type": "array", "items": _NUMBER},
              "refractive_index": _NUMBER}, []),
     _uncertainty),
    ("experiment_session_stats", "查看实验会话统计（测量轮数、统计摘要、是否有无膜基准图）",
     ToolRisk.READ, _OBJECT, _session_stats),
    ("set_plan", "把当前任务的分步计划记录到界面，供人监督",
     ToolRisk.READ, _schema({"plan": {"type": "string"}}, ["plan"]), _set_plan),
    ("record_note", "记录一条实验备注到界面",
     ToolRisk.READ, _schema({"note": {"type": "string"}}, ["note"]), _record_note),

    # -- 运动（需确认） --
    ("auto_center_start", "启动自动寻中（把中心黑条纹移到画面中央）。会使电机转动，需人工确认",
     ToolRisk.MOTION, _OBJECT, _auto_center_start),
    ("measurement_start", "启动目标读数测量（可选 target_mm，电机移动到目标读数附近）",
     ToolRisk.MOTION,
     _schema({"target_mm": _NUMBER}, []), _measurement_start),
    ("backlash_measure", "启动回程差测量（电机在 start_mm 与 end_mm 之间往返）",
     ToolRisk.MOTION,
     _schema({"start_mm": _NUMBER, "end_mm": _NUMBER}, ["start_mm", "end_mm"]),
     _backlash_measure),

    # -- 停止（始终放行） --
    ("motor_emergency_stop", "立即停止电机（急停）",
     ToolRisk.STOP, _OBJECT, _motor_emergency_stop),
    ("auto_center_stop", "停止自动寻中",
     ToolRisk.STOP, _OBJECT, _auto_center_stop),
    ("measurement_stop", "停止目标读数测量",
     ToolRisk.STOP, _OBJECT, _measurement_stop),
    ("backlash_stop", "停止回程差测量",
     ToolRisk.STOP, _OBJECT, _backlash_stop),
]


def build_tool_registry(context: ToolContext) -> ToolRegistry:
    """用注入的 ``ToolContext`` 构造全部工具注册表。"""
    registry = ToolRegistry()
    for name, description, risk, parameters, impl in _TOOLS:
        registry.register(Tool(
            name=name,
            description=description,
            parameters=parameters,
            risk=risk,
            fn=lambda args, _impl=impl, _ctx=context: _impl(args, _ctx),
        ))
    return registry


def build_headless_registry(context: ToolContext | None = None) -> ToolRegistry:
    """无 GUI 环境构造注册表（CLI / 测试）；只读工具可用，运动工具安全降级。"""
    return build_tool_registry(context or ToolContext())
