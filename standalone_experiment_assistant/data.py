"""实验阶段、设备状态和本地答复数据。"""

STEPS = (
    {
        "title": "打开激光光源",
        "progress": 5,
        "next_action": "打开激光光源，观察并调出非定域干涉条纹。",
        "criterion": "视野中出现清晰、稳定的激光干涉条纹。",
        "reading": 0.000,
        "fringe": "laser",
        "devices": (False, False, False, False),
    },
    {
        "title": "调节等厚直条纹",
        "progress": 18,
        "next_action": "加入毛玻璃并缓慢移动动镜，使条纹接近直线。",
        "criterion": "得到宽度约 1 mm、方向稳定的等厚直条纹。",
        "reading": 0.006,
        "fringe": "laser",
        "devices": (False, False, False, False),
    },
    {
        "title": "换用白光并启动相机",
        "progress": 32,
        "next_action": "换上白光扩展光源，启动两台相机并调整视野。",
        "criterion": "条纹画面与数显读数画面均清晰可见。",
        "reading": 0.012,
        "fringe": "white",
        "devices": (True, True, False, False),
    },
    {
        "title": "连接设备并加载模型",
        "progress": 48,
        "next_action": "连接电机，加载 YOLO 模型并开始连续预测。",
        "criterion": "双相机、电机和模型均显示就绪。",
        "reading": 0.014,
        "fringe": "white",
        "devices": (True, True, True, True),
    },
    {
        "title": "沿既定方向寻找条纹",
        "progress": 65,
        "next_action": "选择正转方向，启动自动寻找条纹并保持光路稳定。",
        "criterion": "系统连续识别到稳定的中心条纹候选区域。",
        "reading": 0.021,
        "fringe": "searching",
        "devices": (True, True, True, True),
    },
    {
        "title": "中心条纹闭环定位",
        "progress": 82,
        "next_action": "观察中心偏差和电机减速过程，必要时准备停止。",
        "criterion": "中央黑条纹进入画面中心容差并保持 5 帧。",
        "reading": 0.028,
        "fringe": "centering",
        "devices": (True, True, True, True),
    },
    {
        "title": "记录位置并重复测量",
        "progress": 94,
        "next_action": "记录当前位置，恢复起点后继续下一次测量。",
        "criterion": "获得不少于 10 组完整读数和中心位置记录。",
        "reading": 0.030,
        "fringe": "centered",
        "devices": (True, True, True, True),
    },
    {
        "title": "误差分析与报告整理",
        "progress": 100,
        "next_action": "核对数据，计算平均值、标准差和不确定度并整理报告。",
        "criterion": "实验数据、计算过程、结果表达和原始图像均已归档。",
        "reading": 0.030,
        "fringe": "centered",
        "devices": (True, True, True, True),
    },
)

READINGS = (0.029, 0.030, 0.031, 0.030, 0.029, 0.030, 0.031, 0.030, 0.030, 0.029)


def answer_for(question: str, step_index: int) -> str:
    step = STEPS[step_index]
    lowered = question.lower()
    if any(word in lowered for word in ("误差", "不确定度", "计算", "数据")):
        mean = sum(READINGS) / len(READINGS)
        return (
            "当前记录包含 10 次中心位置读数：\n\n"
            f"- 平均值：{mean:.4f} mm\n"
            "- 极差：0.0020 mm\n"
            "- 建议继续计算样本标准差与均值的 A 类标准不确定度。\n"
            "- B 类分量应计入数显微分表分辨率、机械回程和视觉停车位置波动。\n\n"
            "最终结果应写成 `x = (平均值 ± U) mm，k = 2`，并说明覆盖因子。"
        )
    if any(word in lowered for word in ("报告", "整理")):
        return (
            "建议按“实验目的—实验原理—装置与方法—原始数据—"
            "误差与不确定度—结果讨论—结论”整理报告。\n\n"
            "当前实验进度和读数可以直接写入数据处理部分；装置照片、中心条纹画面、"
            "识别结果和重复测量曲线应分别配图并标注图号。"
        )
    if any(word in lowered for word in ("预习", "原理", "为什么")):
        return (
            "白光的相干长度很短，只有两臂光程差接近零时才能观察到清晰条纹。"
            "中央暗条纹对应零级干涉位置，彩色侧条纹在其两侧近似对称。\n\n"
            "实验的关键是先用激光完成光路预调，再换用白光精确寻找等光程位置。"
        )
    return (
        f"你当前处于第 {step_index + 1} 步：**{step['title']}**。\n\n"
        f"下一步：{step['next_action']}\n\n"
        f"完成标志：{step['criterion']}\n\n"
        "操作过程中请持续观察条纹画面、数显读数和电机状态；出现异常时立即停止。"
    )
