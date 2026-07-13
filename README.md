# AI Interferometry

迈克尔逊白光干涉实验辅助系统。软件通过 USB 相机采集干涉图像，使用
Ultralytics YOLO 识别彩色/黑色条纹，并可通过 RS-232 控制步进电机进行自动寻零。

## 环境与启动

要求 Python 3.10 或更新版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

程序优先使用 `models/current/` 中的第一个模型；该目录为空时使用 `config.yaml` 的
`vision.model_path`。模型类别名称必须能区分 `color/fringe` 与 `black/zero/order`；
类别 ID 不作固定假设。

## 配置

`config.yaml` 包含相机分辨率和帧率、YOLO 阈值与运行设备、串口参数及自动控制
安全限制。如果配置为 CUDA 但当前 PyTorch 无法使用 CUDA，程序会记录警告并回退 CPU。
程序会在启动时校验配置类型、数值范围和模型路径；发现问题会集中报告并拒绝进入
不确定的运行状态。

标定比例位于 `config/calibration.yaml`。

## 安全须知

- 首次联调应让电机空载，并使用低速、小范围运动。
- 自动控制要求相机、模型和电机均已连接，且模型包含黑条/零级类别。
- 自动控制包含最大运行时间、连续丢失条纹停止和多帧黑条确认，但不能替代机械限位。
- 在实际干涉仪上使用前，应验证串口协议、运动方向和紧急停止。

## 测试

无需硬件的基础测试：

```powershell
python -m unittest discover -s tests -v
```

硬件联调建议依次进行：相机单测、电机空载、离线视频推理、低速闭环、完整寻零。

## 实验辅助智能体 MVP

左侧“实验助手”插件提供迈克尔逊干涉原理、操作、故障诊断和不确定度问答。
它只读取紧凑的实验状态，不具备电机控制权限。知识库位于
`src/agent/knowledge_base/`，资料来源只用于内部检索，不追加到聊天回答。

未配置模型时，助手使用本地检索摘要，不依赖网络。需要启用 DeepSeek 综合回答时，
将示例文件 `config/secrets.example.yaml` 复制为 `config/secrets.yaml`，然后填写：

```yaml
deepseek_api_key: "你的密钥"
```

`config/secrets.yaml` 已被 Git 忽略。也可以通过环境变量临时提供密钥，环境变量的
优先级更高：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
python run.py
```

上下文控制策略：每次最多检索 4 个短知识块，只在用户勾选时附加结构化实验状态，
不上传相机图像或完整日志，仅保留最近 4 轮且截断后的短期对话。普通问答最多输出
2000 tokens，实验报告最多输出 3000 tokens。生成过程中可点击“停止”；关闭窗口也会
取消请求，在线超时不会阻止程序退出。

原始实验资料保存在 `resources/agent_sources/`，其中的 `README.md` 记录来源、页数、
用途和整理结论。助手面板提供“测试 DeepSeek”按钮；在线连接成功后会显示当前模型。
知识来源仅用于内部检索和约束回答，聊天面板默认不追加来源列表。插件管理栏支持
底部横向滚动条、鼠标滚轮和左右按钮，可查看超出宽度的插件开关。

助手回答会自动渲染常用 Markdown，包括多级标题、粗体、列表、引用、代码块和表格；
常见 LaTeX 实验公式会转换为数学字体和 Unicode 符号显示。面板中的“AI 状态”会根据
任务显示检索资料、读取实验状态、分析推理、误差计算、整理报告、回答完成或取消等阶段。

## 主要目录

- `src/camera`：OpenCV 相机采集
- `src/vision`：YOLO、ROI、条纹方向及中心定位
- `src/control`：连续/步进自动寻零状态机
- `src/hardware`：步进电机、Arduino 串口及后台命令队列
- `src/agent`：本地检索、DeepSeek Provider 和可取消会话
- `src/ui`：Tkinter 操作界面
- `tests`：无硬件安全测试

更详细的调用关系见 `ARCHITECTURE.md`，文件职责见 `PROJECT_STRUCTURE.md`。
