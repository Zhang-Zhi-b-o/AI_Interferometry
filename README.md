# AI Interferometry

迈克尔逊白光干涉实验辅助系统。软件通过 USB 相机采集干涉图像，使用
Ultralytics YOLO 识别彩色/黑色条纹，并可通过 RS-232 控制步进电机进行自动寻零。

## 环境与启动

要求 Python 3.10 或更新版本：

```powershell
python -m pip install -r requirements.txt
python run.py
```

模型默认放在 `models/current/`。也可在 `config.yaml` 的 `vision.model_path`
中指定。模型类别名称必须能区分 `color` 与 `black/zero/order`；类别 ID 不作固定假设。

## 配置

`config.yaml` 包含相机分辨率和帧率、YOLO 阈值与运行设备、串口参数及自动控制
安全限制。如果配置为 CUDA 但当前 PyTorch 无法使用 CUDA，程序会记录警告并回退 CPU。

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

## 主要目录

- `src/camera`：OpenCV 相机采集
- `src/vision`：YOLO、ROI、条纹方向及中心定位
- `src/hardware`：步进电机与 Arduino 串口
- `src/ui`：Tkinter 操作界面
- `tests`：无硬件安全测试
