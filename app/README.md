# app/ —— v2 重构版（PySide6）

> 旧程序一行没删，仍然可以用 `python run.py` 启动。v2 用 `python run_v2.py`。

## 1. 跑起来

```powershell
python -m pip install -e ".[v2]"    # 项目依赖与 PySide6，一次安装
python run_v2.py                     # 启动
python run_v2.py --demo              # 完整离线演示，不连接真实设备或请求模型
python run_v2.py --theme dark        # 深色主题
python run_v2.py --scale 1.1         # 全局字号放大 10%（嫌小就用它）
python run_v2.py --image 图片.jpg    # 演示模式：一张图代替相机
python run_v2.py --screenshot out.png --page compute   # 截图自检
```

## 2. 目录与职责（改代码前先看这个）

| 目录 | 职责 | 硬规矩 |
| --- | --- | --- |
| `app/core` | 配置、状态、后台任务 | 不 import Qt，不 import 硬件 |
| `app/services` | 相机 / YOLO / 条纹 / 读数 / 电机 / 计算 / 导出 / 标定 | **不 import Qt**，只靠回调交出结果 |
| `app/ui` | 主题、通用组件、7 个页面、主窗口 | **不 import 硬件**，只用 services |
| `app/agent` | 九工具、会话循环、流程建议、本地知识库 | 不 import `app.ui` |

数据流永远是单向的：

```
设备线程 → services 回调 → Bridge(Qt 信号) → MainWindow → AppState → 页面
```

## 3. 常见修改怎么下手

**改颜色 / 字号 / 圆角**
只改 `app/ui/theme.py`：`LIGHT` / `DARK` 两个字典、`SIZES` 字号阶梯、`stylesheet()` 里的 QSS。
不要在每个页面里写死颜色。

**加一个参数**
1. 在 `app/config.v2.yaml` 里加默认值 + 一行注释；
2. 需要语义化访问就在 `app/core/config.py` 的 `AppConfig` 上加 property；
3. 页面里用 `QSpinBox/QDoubleSpinBox` 绑定，并把值传给服务调用的关键字参数。

**换掉某个实现（例如相机采集、条纹算法）**
只改 `app/services/xxx.py`，保持类名与方法签名不变，界面一行都不用改。
例如把 `CameraService` 换成旧的 `src.camera.CameraManager`：在 `open()` 里换成
`CameraManager`，其余接口（`is_open` / `latest_frame` / `close`）保持不变。

**加一个页面**
1. 新建 `app/ui/pages/xxx.py`，继承 `Page`，设 `key` / `title` / `hint`，内容塞进 `self.add(...)`；
2. 在 `app/ui/main_window.py` 的 `PAGES` 元组里加一行；
3. 导航按钮、指标刷新、状态订阅都会自动接上。

**加一个「整幅流程」动作**（例如一键：寻中 → 测条纹 → 记录读数）
写到页面里，用 `ctx.navigate("next-page")` 引导用户，不要跨页面互相调用方法。

**永远不要在 Qt 回调里做阻塞 I/O**
耗时动作一律 `ctx.run_async(work, on_done=..., on_error=...)`。

## 4. 现在的功能边界（有意留的坑）

| 项目 | 现状 | 备注 |
| --- | --- | --- |
| 智能体 | 已接入九工具、会话、逐 token 文本流、取消、工具活动、运动确认及离线建议 | 复用现有 Provider/知识库；`michelson tools` 已输出同一注册表 |
| 自动寻中 | 复用旧 `CenterControlStateMachine`，含盲搜、双向扩展与四种模式 | 用当前零级框区域定位中心、近场框作路标；需真实相机的新画面和已加载模型 |
| 3D 形貌图 | 已接入可旋转曲面与 PNG 导出 | 保留无效区域和原像素坐标，未锚定结果标为相对起伏 |
| 离线演示 | `--demo` 注入独立的纯内存设备，复用实际寻中/计算/厚度分析 | 无真实硬件继承，助手强制离线，检测和条纹指标为合成参数 |
| 电机连接 | 搜索页顶部支持 auto/指定串口与波特率，后台连接/断开 | 取消/关窗丢弃迟到连接；串口 I/O 与 UI 状态锁分离；真机效果待验 |
| 视频录制 / 回程差 / 及时采样 / 激光引导 | 按需求去掉 | 源码仍在 `src/`，需要时可作为插件回归 |
| 显示帧率控制 | 相机线程直接推帧 | 若画面卡顿，在 `CameraService._loop` 里加节流 |

## 5. 界面设计说明

- 配色与字号仿 DeepSeek Harness：中性偏蓝的灰阶（`#f9fafb` / `#e1e5ee` / `#1b1b1c`）+
  DeepSeek 蓝（`#4176e6`）作唯一强调色，成功/警告/危险三色只用于状态。
- 字号阶梯：正文 15px、卡片标题 16px、页面标题 18px、指标数字 26px、主数值 34px。
- 布局：左侧导航（212px）+ 中间大画面（可拖动的 `QSplitter`）+ 右侧当前页操作卡；
  画面下方常驻 4 个指标卡，任何时候都能看到「位置 / 宽度 / 读数 / 厚度」。

## 6. 搜索条纹

搜索页分别选择“已知方向 / 双向扩展”“连续 / 转停识别”，并选实际电机的起始方向。
电机方向映射决定条纹居中后的纠偏方向，启动前应在真机核对。
高级参数可设置 1–10 档、最大运行时间、估算搜索半程以及转停节拍。
范围由标称转速积分估计，不是实测位移；0 表示不限范围，仍受超时限制。
画面或本次推理输入超过 1.5 秒、模型不可用、命令失败时停止；
目标读数与寻中不能同时运行。目标读数也拒绝超过 1.5 秒未更新的读数。
历史保留的 OCR 读数使用原时间，不会因为本次识别失败被视为新读数。

当前视觉适配复用零级区域内相干中心定位、短时重新定位和近场路标；
尚未迁入旧 UI 的完整二维视觉补偿、自适应调参与稳定框约束链路，真机效果待走查。

## 7. 助手配置

沿用 `agent.llm.api_base` 与 `agent.llm.models.flash`，密钥优先读取
`DEEPSEEK_API_KEY`，其次读取未跟踪的 `config/secrets.yaml` 中的
`deepseek_api_key`。没有密钥时仍可给出本地流程建议与知识库摘录。
每轮最多 6 次模型请求，每次最多输出 1200 tokens；历史最多保留 4 轮。
点击发送才会请求模型；运动每次单独确认，取消对话不会替代顶部“停止运动”。

## 8. 自检

```powershell
python -m pytest tests/test_app_v2_core.py -q          # 核心与服务层（无硬件、无 Qt）
python -m pytest tests/test_app_v2_ui_smoke.py -q      # 界面冒烟（offscreen）
python run_v2.py --screenshot exports/v2.png           # 人眼验收界面
```
