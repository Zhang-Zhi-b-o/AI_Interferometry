# 白光干涉测量工作台

可独立复制、安装和运行的新版 PySide6 程序。只需要本目录；旧版 `src/`、项目根配置、实验数据集及其他目录均不参与运行。

## 安装与启动

建议 Windows 10/11、64 位 Python 3.10 或 3.11。首次安装需联网下载 Python 依赖；默认支持 CPU，无需 CUDA。

在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py
```

安装后可双击 `start.bat`，它优先使用本目录的 `.venv`。如使用系统 Python，先执行 `python -m pip install -e .`，然后 `python run.py`。

```powershell
.\.venv\Scripts\python.exe run.py --demo
.\.venv\Scripts\python.exe run.py --theme dark
.\.venv\Scripts\python.exe run.py --image "图片.jpg"
.\.venv\Scripts\python.exe run.py --demo --screenshot "exports/demo.png"
```

`--demo` 无需设备及 API 密钥，使用合成条纹、模拟设备与离线助手。`--image` 仅替代主相机画面，其他功能按正常配置工作。安装后的命令入口是 `interferometry`；仓库根目录也可运行 `python -m app --demo`。

## 功能与设备

- 实时相机预览、画面校正、ROI、单帧/连续 YOLO 识别、图片与录像保存。
- 条纹中心、角度、宽度、间距分析与位置记录。
- 电机手动控制、自动寻中、目标读数及回程差。
- 第二相机微分表 OCR、稳定读数与测量记录。
- 厚度计算、不确定度、多轮统计、相对厚度分布、标定及三维曲面。
- 本地知识检索、在线助手、原文出处、Word 测量报告和性能评测报告。

实测需要兼容的干涉仪相机、拍摄微分表的第二相机，以及使用现有串口协议的电机控制器。默认主相机编号 1、读数相机编号 0，可在界面调整。自动运动前核对电机方向、行程及停止功能；`Esc` 请求停止运动。

`models/` 已包含全量 2954 张默认 YOLO 权重、人工确认 600 张备选权重、历史权重及 OCR 的识别/检测/分类权重。内置模型无需另行下载。来源见 `models/MODEL_SOURCES.md` 和 `models/micrometer/SOURCE.md`。

## 配置与助手

程序从本目录读取资源，不依赖启动命令所在目录。配置优先级：`config.v2.yaml` → `config/video_demo.yaml` 设备预设 → 本目录 `config.yaml` 用户覆盖。可复制 `config.example.yaml` 为 `config.yaml`，填写需要覆盖的项目；相对模型路径以本目录为基准。

在线助手需要用户自己的 API 密钥。在启动程序的 PowerShell 中设置：

```powershell
$env:DEEPSEEK_API_KEY = "填入你自己的密钥"
.\.venv\Scripts\python.exe run.py
```

也可在本目录的 `config/secrets.yaml` 设置 `deepseek_api_key`，此文件被 Git 忽略。默认接口为 `https://api.deepseek.com`，模型为 `deepseek-chat`。图像复核需在 `config.yaml` 的 `agent.llm.models.vision` 设置接口支持的视觉模型。请求费用由所用服务收取。

离线助手可设置 `agent.chat_mode: local` 或使用 `--demo`。正常在线模式缺少密钥或网络时提示错误；设备测量和本地报告可独立使用。

## 数据和目录

| 目录 | 内容 |
| --- | --- |
| `core/`、`services/`、`ui/`、`agent/` | 配置、服务、界面与助手 |
| `engine/` | 内置的原有测量、视觉、设备及控制实现 |
| `models/` | 推理权重及来源说明 |
| `knowledge/` | 论文、仪器说明、报告和操作指导 |
| `tests/` | 软件回归测试 |
| `exports/`、`logs/`、`data/` | 运行生成的导出、日志、检索缓存，不上传 Git |

测量报告导出 `.docx`、`.md`、`.json`，性能报告导出 `.docx`、`.json`。演示输出在 `exports/demo/`。知识索引缺失时从内置原文重建，出处可打开本目录中的文件。

## 检查与测量边界

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

软件测试与合成演示不证明物理精度。曝光、电机方向、OCR 识别率、传动比和标定仍需真机确认。未锚定厚度分布表示相对起伏，颜色坐标不能直接视为物理相位。
