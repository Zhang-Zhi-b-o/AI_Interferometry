"""``michelson`` 命令行：脱离 GUI 运行纯逻辑测量 / 分析 / 控制子命令。

与 GUI 智能体共用同一批纯逻辑能力（厚度计算、不确定度评定、条纹宽度分析、
单帧厚度分布、自动寻中状态机、工具注册表），供 AI 或人在终端便捷调用。硬件
子命令（运动 / 急停 / 读实时表）需在 GUI 内运行，此处不另建无头硬件连接。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.agent.device_tools import build_headless_registry
from src.agent.toolkit import ToolRisk
from src.constants import MICROMETER_ACCURACY
from src.measurement.thickness import GLASS_REFRACTIVE_INDEX, calculate_thickness_mm
from src.measurement.uncertainty import (
    DEFAULT_REFRACTIVE_INDEX_TOLERANCE,
    analyze_glass_thickness,
)
from src.vision.fringe_width import measure_center_fringe_width
from src.vision.thickness_distribution import analyze_thickness_distribution


def _load_image(path: str) -> np.ndarray:
    """OpenCV 不能读中文路径，改用 np.fromfile + imdecode。"""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图像：{path}")
    return image


def _print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _cmd_glass_thickness(args) -> None:
    h = calculate_thickness_mm(args.d1, args.d2, args.n)
    _print_json({
        "thickness_mm": h,
        "d1_mm": args.d1,
        "d2_mm": args.d2,
        "refractive_index": args.n,
        "formula": "h = (d2 - d1) / [20 × (n - 1)]",
    })


def _cmd_uncertainty(args) -> None:
    thickness = [float(v) for v in args.values]
    d1 = [float(v) for v in args.d1] if args.d1 else None
    d2 = [float(v) for v in args.d2] if args.d2 else None
    result = analyze_glass_thickness(
        thickness,
        d1_values=d1,
        d2_values=d2,
        refractive_index=args.n,
        micrometer_accuracy_mm=args.accuracy,
        refractive_index_tolerance=args.index_tolerance,
    )
    _print_json(result)


def _cmd_fringe_width(args) -> None:
    img = _load_image(args.image)
    result = measure_center_fringe_width(img, center_x=args.center_x)
    out = {k: v for k, v in result.items() if k != "bands"}
    out["center_band"] = result.get("center_band")
    _print_json(out)


def _cmd_thickness_distribution(args) -> None:
    img = _load_image(args.image)
    baseline = _load_image(args.baseline) if args.baseline else None
    result = analyze_thickness_distribution(
        img,
        wavelength_nm=args.wavelength,
        refractive_index=args.refractive,
        calibration=args.calibration,
        reference_image=baseline,
    )
    _print_json({
        "mode": result["mode"],
        "step_um": result["step_um"],
        "wavelength_nm": result["wavelength_nm"],
        "refractive_index": result["refractive_index"],
        "metrics": result["metrics"],
    })


def _cmd_center_search(args) -> None:
    from src.control.center_control import CenterControlStateMachine

    state_machine = CenterControlStateMachine()
    decision = state_machine.start(0.0)
    print(f"start -> {decision.state}: {decision.message}")
    params: dict = {"search_direction": args.direction}
    if args.mode == "single_direction":
        params["direction_mode"] = "single_direction"
    elif args.mode == "stop_and_detect":
        params["recognition_mode"] = "stop_and_detect"
    # 无相机时 center_x 传 None，纯状态机演示扩展搜索逻辑。
    for i in range(1, args.steps + 1):
        decision = state_machine.update(
            center_x=None, frame_width=1280.0, confidence=0.0,
            connected=True, params=params, safety={}, now=float(i) * 0.1)
        commands = [name for name, _ in decision.commands]
        print(f"[{i}] {decision.state}: {decision.message} (commands={commands})")
        if decision.completed or decision.state == "stopped":
            break


def _cmd_tools(args) -> None:
    registry = build_headless_registry()
    order = {ToolRisk.READ: 0, ToolRisk.MOTION: 1, ToolRisk.STOP: 2}
    rows = registry.describe()
    rows.sort(key=lambda r: order.get(ToolRisk(r["risk"]), 3))
    for row in rows:
        print(f"[{row['risk']:6}] {row['name']}")
        print(f"        {row['description']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="michelson",
        description="迈克尔逊干涉仪 CLI 工具（纯逻辑子命令）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="测量计算")
    measure_sub = measure.add_subparsers(dest="measure_command", required=True)

    glass = measure_sub.add_parser("glass-thickness", help="由两次微分表读数计算玻璃片厚度")
    glass.add_argument("d1", type=float, help="第一次中心条纹读数（mm）")
    glass.add_argument("d2", type=float, help="第二次中心条纹读数（mm）")
    glass.add_argument("--n", type=float, default=GLASS_REFRACTIVE_INDEX, dest="n",
                       help="折射率")
    glass.set_defaults(func=_cmd_glass_thickness)

    uncertainty = measure_sub.add_parser("uncertainty", help="玻璃片厚度不确定度评定")
    uncertainty.add_argument("values", type=float, nargs="+", help="厚度序列（mm）")
    uncertainty.add_argument("--d1", type=float, nargs="+", default=None,
                             help="各轮 d1 读数（mm）")
    uncertainty.add_argument("--d2", type=float, nargs="+", default=None,
                             help="各轮 d2 读数（mm）")
    uncertainty.add_argument("--n", type=float, default=GLASS_REFRACTIVE_INDEX, dest="n")
    uncertainty.add_argument("--accuracy", type=float, default=MICROMETER_ACCURACY)
    uncertainty.add_argument("--index-tolerance", type=float,
                             default=DEFAULT_REFRACTIVE_INDEX_TOLERANCE)
    uncertainty.set_defaults(func=_cmd_uncertainty)

    analyze = sub.add_parser("analyze", help="视觉分析")
    analyze_sub = analyze.add_subparsers(dest="analyze_command", required=True)

    fringe = analyze_sub.add_parser("fringe-width", help="分析图像中心条纹宽度")
    fringe.add_argument("image", help="图像路径")
    fringe.add_argument("--center-x", type=float, default=None)
    fringe.set_defaults(func=_cmd_fringe_width)

    thickness = analyze_sub.add_parser("thickness-distribution",
                                       help="单帧薄膜厚度分布估计")
    thickness.add_argument("image", help="图像路径")
    thickness.add_argument("--wavelength", type=float, default=589.3)
    thickness.add_argument("--refractive", type=float, default=1.523)
    thickness.add_argument("--calibration", default=None, help="颜色标定 CSV 路径")
    thickness.add_argument("--baseline", default=None, help="无膜基准图路径")
    thickness.set_defaults(func=_cmd_thickness_distribution)

    control = sub.add_parser("control", help="控制状态机")
    control_sub = control.add_subparsers(dest="control_command", required=True)
    center = control_sub.add_parser("center-search", help="自动寻中状态机（纯模拟）")
    center.add_argument("--steps", type=int, default=10)
    center.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    center.add_argument("--mode",
                        choices=["bidirectional", "single_direction", "stop_and_detect"],
                        default="bidirectional")
    center.set_defaults(func=_cmd_center_search)

    tools = sub.add_parser("tools", help="列出工具注册表（含风险分级）")
    tools.set_defaults(func=_cmd_tools)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
