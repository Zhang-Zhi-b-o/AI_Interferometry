# 系统架构

## 架构概览

分层架构，上层依赖下层，下层不感知上层。

```
┌─────────────────────────────────────────┐
│                  UI                      │
│  主窗口 · 视频面板 · 控制面板 · 对话面板   │
├─────────────────────────────────────────┤
│     智能体              不确定度          │
│   工作流+RAG+LLM      A类/B类/合成       │
├─────────────────────────────────────────┤
│     视觉检测             硬件控制         │
│   YOLO·角度估计·校正    电机·Arduino      │
├─────────────────────────────────────────┤
│              相机采集                     │
│           OpenCV 摄像头管理               │
├─────────────────────────────────────────┤
│              报告导出                     │
│              PDF 生成                    │
├─────────────────────────────────────────┤
│           基础设施                        │
│       配置管理 · 日志 · 物理常数          │
└─────────────────────────────────────────┘
```

---

## 目录结构

```
AI_Interferometry/
├── run.py                        # 程序入口
├── config.yaml                   # 配置文件
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                 # 配置管理（加载 config.yaml）
│   ├── logging.py                # 日志配置
│   ├── constants.py              # 物理常数（λ_HeNe = 632.8nm 等）
│   │
│   ├── camera/                   # ===== 相机采集 =====
│   │   ├── __init__.py
│   │   └── manager.py            # 打开摄像头、采集帧、参数设置
│   │
│   ├── vision/                   # ===== 视觉检测 =====
│   │   ├── __init__.py
│   │   ├── detector.py           # YOLOv8 推理
│   │   ├── roi.py                # 有效区域搜索
│   │   ├── angle.py              # FFT 条纹角度估计
│   │   └── correct.py            # 画面旋转校正
│   │
│   ├── hardware/                 # ===== 硬件控制 =====
│   │   ├── __init__.py
│   │   ├── motor.py              # 步进电机（手动/连续/步进）
│   │   └── arduino.py            # Arduino 激光测距 + OLED
│   │
│   ├── agent/                    # ===== 智能体 =====
│   │   ├── __init__.py
│   │   ├── engine.py             # 工作流引擎
│   │   ├── steps/                # 工作流步骤
│   │   │   ├── __init__.py
│   │   │   ├── precheck.py       #   预习检查 + 评分
│   │   │   ├── alignment.py      #   调平引导
│   │   │   ├── measurement.py    #   测量指导
│   │   │   ├── analysis.py       #   数据分析
│   │   │   └── report.py         #   报告生成
│   │   ├── rag.py                # RAG 检索
│   │   ├── llm_client.py         # LLM 接口
│   │   ├── prompts.py            # Prompt 模板
│   │   └── knowledge/            # 知识库文档
│   │       ├── guide.md
│   │       ├── errors.md
│   │       ├── formulas.md
│   │       └── rubric.md
│   │
│   ├── uncertainty/              # ===== 不确定度 =====
│   │   ├── __init__.py
│   │   ├── type_a.py             # A 类评定
│   │   ├── type_b.py             # B 类评定
│   │   ├── combined.py           # 合成不确定度
│   │   ├── symmetry.py           # 正反行程对称处理
│   │   └── judge.py              # 结果判定
│   │
│   ├── report/                   # ===== 报告导出 =====
│   │   ├── __init__.py
│   │   ├── pdf_export.py         # PDF 生成
│   │   └── templates/
│   │       └── default.html      # 报告模板
│   │
│   └── ui/                       # ===== UI =====
│       ├── __init__.py
│       ├── app.py                # 主窗口
│       ├── widgets/
│       │   ├── __init__.py
│       │   ├── video_panel.py    #   实时视频 + YOLO 框叠加
│       │   ├── control_panel.py  #   电机控制按钮
│       │   ├── chart_panel.py    #   数据图表
│       │   ├── chat_panel.py     #   智能体对话
│       │   └── status_bar.py     #   状态栏
│       ├── themes.py             # 主题配色
│       └── layout.py             # 布局
│
├── models/                       # 模型权重文件
│   ├── yolov8_interference.pt
│   ├── resnet_stripe.pth
│   ├── cnn_counter.pth
│   └── opd_dnn.pth
│
├── data/                         # 数据集（.gitignore）
│   ├── raw/
│   ├── labeled/
│   └── synthetic/
│
├── tests/                        # 测试
│   ├── test_camera/
│   ├── test_vision/
│   ├── test_agent/
│   ├── test_uncertainty/
│   └── test_hardware/
│
└── resources/                    # 静态资源
    ├── icons/
    └── samples/
```

---

## 模块说明

### 1. 相机采集  `camera/`

封装 OpenCV，向上提供一个简单的取帧接口。

```python
from src.camera import CameraManager

cam = CameraManager(index=1, resolution=(1280, 1024), fps=60)
cam.start()
frame = cam.read()   # → np.ndarray
cam.stop()
```

---

### 2. 视觉检测  `vision/`

输入相机帧，输出检测结果。

```python
from src.vision import Detector, AngleEstimator, FrameCorrector

detector = Detector("models/yolov8_interference.pt")
result = detector.detect(frame)
# → {center, bbox, confidence}

angle = AngleEstimator.estimate(frame)
# → float (度)

corrected = FrameCorrector.apply(frame, angle)
# → np.ndarray (旋转校正后)
```

---

### 3. 硬件控制  `hardware/`

```python
from src.hardware import MotorController, ArduinoReader

motor = MotorController(port="COM3", baudrate=9600)
motor.manual("up")       # 手动模式：方向键
motor.continuous_scan()  # 连续模式：转→检测→变速
motor.step_scan()        # 步进模式：转→停→拍→分析

arduino = ArduinoReader(port="COM4")
data = arduino.read()
# → {turns, direction}
```

---

### 4. 智能体  `agent/`

工作流引擎按实验流程逐步推进，每步调用 LLM + RAG。

```python
from src.agent import WorkflowEngine

engine = WorkflowEngine(llm_client, rag, knowledge_base)
engine.start_session(student_name="张三")

# 逐步执行
engine.run_step("precheck")       # 预习检查
engine.run_step("alignment")      # 调平引导，传入 vision 检测结果
engine.run_step("measurement")    # 测量指导
engine.run_step("analysis")       # 数据分析，传入 uncertainty 结果
engine.run_step("report")         # 生成报告
```

```python
from src.agent import RagRetriever, LLMClient

# RAG 检索
rag = RagRetriever(knowledge_dir="agent/knowledge/")
docs = rag.search("迈克尔逊干涉条纹消失了怎么办")
# → [相关文档片段]

# LLM 调用
llm = LLMClient(provider="deepseek", model="deepseek-chat")
reply = llm.chat(messages=[...], context=docs)
```

**何时调用**：整个实验过程持续运行。UI 层在对话面板中调用，视觉和不确定度模块把结果传给智能体做决策。

---

### 5. 不确定度  `uncertainty/`

输入一组测量数据，输出完整误差分析。

```python
from src.uncertainty import (
    type_a, type_b, combined_uncertainty,
    symmetry_check, judge
)

# readings: [632.1, 632.5, 631.9, 632.3, 632.0, 632.2]  (nm)
# instrument_error: 0.0001 (mm, 千分尺精度)

u_a = type_a(readings)                    # A 类评定
u_b = type_b(instrument_error, ...)       # B 类评定
u_c = combined_uncertainty(u_a, u_b)      # 合成
sym = symmetry_check(forward, backward)   # 对称处理
ok = judge(u_c, tolerance=5.0)            # 是否达标
# → {u_a, u_b, u_c, symmetry, passed, report_text}
```

**何时调用**：一组测量完成后，UI 点击"分析"触发。

---

### 6. 报告导出  `report/`

```python
from src.report import PDFExporter

exporter = PDFExporter(template="report/templates/default.html")
exporter.export(
    student="张三",
    measurements=readings,
    uncertainty=u_result,
    agent_comments=engine.get_comments(),
    output="实验报告_张三_20260626.pdf"
)
```

**何时调用**：全部实验完成后，UI 点击"导出"触发。

---

### 7. UI  `ui/`

主窗口布局：

```
┌────────────────────────────────────────────┐
│  菜单栏：文件 | 实验 | 帮助                    │
├────────────────────┬───────────────────────┤
│                    │  控制面板               │
│   视频面板          │  ┌─────────────────┐  │
│  （实时画面         │  │ 电机模式切换     │  │
│   + YOLO框叠加）    │  │ [手动][连续][步进]│  │
│                    │  │ ▲ 上             │  │
│                    │  │◄左  右►          │  │
│                    │  │ ▼ 下             │  │
│                    │  ├─────────────────┤  │
│                    │  │ 开始测量         │  │
│                    │  │ 数据分析         │  │
│                    │  │ 导出报告         │  │
│                    │  └─────────────────┘  │
│                    │                       │
├────────────────────┤  图表面板              │
│   智能体对话面板    │  ┌─────────────────┐  │
│  ┌──────────────┐  │  │ 波长 (nm)       │  │
│  │ 助教：请调节   │  │  │ 633 ┤ ●──●──●   │  │
│  │ M2背后右下角   │  │  │ 632 ┤          │  │
│  │ 的螺钉…       │  │  │ 631 ┤          │  │
│  │              │  │  │     └─┬─┬─┬─   │  │
│  │ 学生：好的    │  │  │      1 2 3 4   │  │
│  │              │  │  │     测量次数    │  │
│  └──────────────┘  │  └─────────────────┘  │
│  ┌──────────────┐  │                       │
│  │ 输入消息...   │  │                       │
│  └──────────────┘  │                       │
├────────────────────┴───────────────────────┤
│  状态栏：相机: ✓ | YOLO: ✓ | 电机: COM3 |    │
└────────────────────────────────────────────┘
```

---

## 模块依赖关系

```
                   run.py
                      │
                   ui/app.py
                  /    |    \
                 /     |     \
           agent/   uncertainty/   report/
              |         |
         vision/    (纯计算，无依赖)
            |
        camera/
            |
        hardware/  (与 camera 同级，独立使用)
```

**规则**：
- 上层 import 下层，下层不 import 上层
- `ui/` 在最顶层，可以 import 所有模块
- `camera/` 在底层，不依赖任何业务模块
- `agent/` 通过函数参数接收 `vision` 和 `uncertainty` 的结果，不直接 import 它们

---

*2026-06-26*
