# 系统架构

## 架构概览

分层架构，上层依赖下层，下层不感知上层。

```
┌─────────────────────────────────────────────┐
│                   UI 层                      │
│   app.py 主控制器 + 8 个 widget 面板          │
├─────────────────────────────────────────────┤
│   视觉层                 硬件层              │
│   YOLO检测·角度旋转·     步进电机·Arduino    │
│   画面校正·ROI搜索                           │
├─────────────────────────────────────────────┤
│              相机采集                        │
│           OpenCV 摄像头管理                  │
├─────────────────────────────────────────────┤
│             基础设施                         │
│      配置管理 · 日志 · 物理常数              │
└─────────────────────────────────────────────┘
```

---

## 目录结构（实际）

```
AI_Interferometry/
├── run.py                             # 入口：python run.py
├── config.yaml                        # 全局配置
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py                    # PROJECT_ROOT
│   ├── config.py                      # 配置加载（YAML）
│   ├── logging.py                     # 日志 → 控制台 + logs/app.log
│   ├── constants.py                   # 物理常数
│   │
│   ├── camera/                        # 相机层
│   │   └── manager.py                 #   CameraManager（OpenCV采集）
│   │
│   ├── vision/                        # 视觉层
│   │   ├── detector.py                #   YOLODetector（推理+画框）
│   │   ├── angle.py                   #   FFT角度估计 + rotate_expand
│   │   ├── correct.py                 #   FrameCorrector（旋转/缩放/平移）
│   │   ├── roi.py                     #   亮区搜索 + 掩膜
│   │   └── class_names.py             #   类别映射（0→color, 1→black）
│   │
│   ├── hardware/                      # 硬件层
│   │   ├── motor.py                   #   MotorController（RS-232步进电机）
│   │   └── arduino.py                 #   ArduinoReader（激光测距）
│   │
│   ├── ui/                            # UI层
│   │   ├── app.py                     #   YoloCamApp 主控制器（~750行）
│   │   └── widgets/
│   │       ├── collapsible.py         #   可折叠容器（带▲▼排序）
│   │       ├── plugin_toggles.py      #   插件开关栏（☑全开/全关/右键跳转）
│   │       ├── camera_plugin.py       #   摄像头+旋转/缩放/平移
│   │       ├── model_plugin.py        #   YOLO模型+ROI+预测结果
│   │       ├── status_panel.py        #   实时状态（FPS/电机/转速/档位）
│   │       ├── motor_control.py       #   电机控制（手动/连续/步进）
│   │       ├── video_recorder.py      #   视频录制
│   │       └── log_panel.py           #   运行日志
│   │
│   ├── agent/                         # 留空（智能体）
│   ├── uncertainty/                   # 留空（不确定度）
│   └── report/                        # 留空（报告导出）
│
├── models/
│   └── yolov8_interference.pt         # YOLO 模型权重
└── logs/
    └── app.log                        # 运行日志
```

---

## 依赖关系

```
app.py  ← 唯一组装点，import 一切
├── camera/manager.py          ← cv2, numpy
├── vision/
│   ├── detector.py            ← cv2, numpy, ultralytics.YOLO
│   ├── angle.py               ← cv2, numpy
│   ├── correct.py             ← cv2, numpy
│   ├── roi.py                 ← cv2, numpy
│   └── class_names.py         ← 无依赖
├── hardware/
│   ├── motor.py               ← pyserial
│   └── arduino.py             ← pyserial
├── ui/widgets/                ← 纯 tkinter，通过回调通信
│   ├── camera_plugin.py       ← tkinter
│   ├── model_plugin.py        ← tkinter
│   ├── status_panel.py        ← tkinter
│   ├── motor_control.py       ← tkinter
│   ├── video_recorder.py      ← tkinter
│   ├── log_panel.py           ← tkinter
│   ├── collapsible.py         ← tkinter
│   └── plugin_toggles.py      ← tkinter
├── src/config.py              ← yaml
└── src/logging.py             ← 标准库
```

---

## 调用流程

### 启动
```
python run.py
  → config.yaml 加载
  → logger 初始化
  → YoloCamApp.__init__()
      → 创建 CameraManager / YOLODetector / FrameCorrector / MotorController
      → _build_ui(): 构建全部 widget 面板
      → _wire_callbacks(): 绑定各面板按钮 → app 方法
      → root.mainloop(): Tkinter 事件循环
```

### 打开摄像头 → 预览
```
用户点 "打开摄像头"
  → camera_plugin._emit("open")
  → app._on_camera_cmd("open")
  → CameraManager.start()
  → _start_preview() → _preview_loop() [每30ms循环]
      → cam.read()
      → rotate_expand(frame, angle)     # 手动旋转
      → FrameCorrector.apply_zoom_pan() # 手动缩放
      → _show_frame() → Canvas 显示
```

### 加载模型 → 实时预测
```
用户点 "加载YOLO模型"
  → model_plugin._emit("load")
  → app._on_model_cmd("load")
  → 后台线程: YOLODetector.load()  # 不阻塞 UI

用户点 "开始预测"
  → model_plugin._emit("start")
  → _stop_preview()              # 停预览，避免画面覆盖
  → _predict_loop() [每90ms循环]
      → cam.read()
      → rotate_expand() + apply_zoom_pan()
      → detector.detect(frame, roi)    # ROI 内推理
      → 画框: class_names[0]=color, class_names[1]=black
      → _decide_motor_command()        # 根据框中心偏移决定电机方向
      → _auto_motor_control()          # 连续/步进模式自动调速
      → _show_frame(annotated)         # 显示带框画面
      → model_plugin.update_results()  # 更新预测结果面板
      → status.update_fps()            # 更新FPS

用户点 "停止预测"
  → _stop_predict()
  → _start_preview()              # 恢复预览画面
```

### 电机连接 → 控制
```
用户点 "连接"
  → motor_panel → app._on_motor_connect(port)
  → MotorController.connect()
  → _start_motor_poll() [每300ms]
      → query_status() → status.update_motor_speed/gear

手动模式:
  启动/加速/减速/停止/状态 → motor.start/speed_up/speed_down/stop

连续模式:
  _auto_motor_control():
    idle → 搜索速度(10) → 检测到color(>0.3)→减速(5) → 检测到black(>阈值)→锁定

步进模式:
  _auto_motor_control():
    idle → move(转首轮/循环ms) → pause(分析ms) → 检测到black→锁定 / 回到idle
```

### 插件管理
```
☑ 勾选/取消  → _toggle_plugin(key, enabled)
                  → 显示/隐藏面板 + 停止对应功能

▲/▼ 按钮    → _move_plugin(key, direction)
                  → 交换 _plugin_order → _reorder_shells()

右键插件名   → _jump_to_plugin(key)
                  → 自动展开面板 + 滚动到可见位置

点击标题栏   → CollapsibleFrame.toggle()
                  → 折叠/展开面板内容
```

---

## 事件通信方式

widget 面板 → app 主控制器：通过 `on_xxx` 回调函数

```
widget 定义回调属性:
    self.on_command = None       # 占位

用户操作 → widget._emit(cmd):
    if self.on_command:
        self.on_command(cmd)

app._wire_callbacks() 注入:
    widget.on_command = app._on_camera_cmd
```

---

*2026-06-26*
