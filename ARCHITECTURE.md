# 系统架构

## 设计目标

本项目是迈克尔逊白光干涉实验辅助系统。它将相机采集、YOLO 条纹识别、中心条纹定位、步进电机寻零和只读实验助手组合在一个 Tkinter 桌面应用中。

核心约束：

- Tkinter 主线程只负责界面和状态消费，不执行可能阻塞的串口、模型推理或在线请求。
- 自动控制采用确定性状态机；实验助手没有硬件控制权限。
- 电机命令必须串行执行，停止命令优先于普通查询和调速。
- 配置错误在启动阶段集中报告，不在运行时静默使用危险默认值。

## 分层结构

```text
UI 层
  src/ui/app.py                  组装、事件路由、画面呈现
  src/ui/widgets/               独立 Tkinter 面板
        │
应用服务层
  src/agent/session.py          可取消的守护会话
  src/control/auto_control.py   纯自动控制状态机
  src/hardware/command_queue.py 串口后台命令队列
        │
领域与设备层
  src/agent/                    RAG、提示词、DeepSeek Provider
  src/vision/                   YOLO、ROI、角度、中心条纹
  src/hardware/                电机协议、Arduino 数据
  src/camera/                  OpenCV 相机
        │
基础设施
  src/config.py                 YAML 加载与范围校验
  src/logging.py                控制台和本地文件日志
```

`src/ui/app.py` 仍是唯一组装点，但串口调度、自动控制转换和助手生命周期已经拆成可独立测试的组件。后续新增功能应优先进入对应服务，不再继续扩张主控制器。

## 主要数据流

### 相机和视觉

```text
CameraManager.read
  → 旋转、缩放和平移
  → 后台 YOLODetector.detect
  → UI 消费检测结果
  → 中心条纹分析 / 自动控制状态机 / 画面显示
```

模型类别不能使用固定 ID。当前兼容层根据真实 `model.names` 将 `zero_order` 归为黑条/零级语义，将 `near_fringe`、`far_fringe` 归为彩条语义。

### 电机控制

```text
UI 参数 + 视觉置信度
  → AutoControlStateMachine.update
  → 产生 set_speed/start/stop 命令
  → SerialCommandQueue 单线程顺序执行
  → UI 定时领取 CommandResult
```

状态机不引用 Tkinter、串口或真实时间，因此测试可以注入虚拟时间，覆盖连续模式、步进模式、超时、条纹丢失和串口失联。

### 实验助手

```text
用户问题 + UI 主线程快照的实验状态
  → 本地 Markdown 检索（最多 top_k 块）
  → 最近 4 轮受限历史
  → DeepSeek 流式生成
  → AgentSession 将结果交回 UI
```

助手请求运行在守护线程中，支持停止生成；关闭窗口时会设置取消事件，不会因 30 秒网络超时阻止进程退出。知识来源只用于内部约束，聊天回答不追加来源列表。

## 生命周期与安全

- 关闭摄像头或模型插件时停止预测和自动控制。
- 自动控制达到最大运行时间、连续丢失条纹或串口失联时生成优先停止命令。
- 关闭窗口时依次停止预览、录制、预测、电机轮询和助手会话。
- 串口队列是守护线程；关闭时优先执行电机停止与断开操作。
- `config/secrets.yaml` 只保存在本地并由 Git 忽略。

## 验证边界

无硬件测试覆盖配置校验、本地检索、助手取消、类别映射、ROI 边界、中心条纹输入、电机协议、命令队列和自动控制状态转换。真实设备上线前仍需依次执行相机单测、电机空载、离线视频推理和低速闭环测试。
