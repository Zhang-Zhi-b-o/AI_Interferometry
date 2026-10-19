# 现状分析：AI Interferometry（迈克尔逊白光干涉实验辅助系统）

> 阶段 1 产物。目的：把「现在有什么、乱在哪、哪些能直接搬走」讲清楚，供范围和保留决策使用。
> 分析范围：仓库根目录全部源码（`src/` 22,286 行 Python，不含 `Fringe width analysis/`、
> `Thickness_variation_analysis/`、`tmp/` 等原型脚本与报告素材）。

## 1. 一句话现状

一套**能力齐全但边界失控**的 Tkinter 单体桌面程序：核心算法（条纹定位/宽度/读数/厚度/不确定度）
质量不错且有测试覆盖，但**入口类 `src/ui/app.py` 5129 行、约 200 个方法**同时承担界面、相机、
YOLO、串口电机、自动搜索、OCR、录制、Agent 编排、报告导出；智能体能力被拆成 8 个模块 +
1553 行 Tk 面板，存在两套并行的"工具"概念和三个互相重叠的引导决策器。结果是：能跑，但改一处要读五处。

## 2. 技术栈与运行方式

| 项 | 现状 |
| --- | --- |
| 语言/运行时 | Python 3.10.11（本机 `C:\Users\36852\...\Python310\python.exe`） |
| GUI | **Tkinter**（纯 tk.Widget 手写布局，无第三方主题库；`src/ui/theme.py` 提供颜色/字体常量） |
| 视觉 | ultralytics YOLOv8 8.4.31、OpenCV 4.11.0、numpy 2.2.6、scipy |
| OCR | onnxruntime 1.20.1 + PP-OCRv6 ONNX（微分表读数） |
| 硬件 | pyserial 串口电机（`src/hardware/motor.py`）+ 后台命令队列 |
| LLM | DeepSeek OpenAI 兼容接口（`openai` 2.54.0），function calling |
| 入口 | `run.py` → `src.ui.app.run_app()`；另有 `python -m src.cli` / `michelson` |
| 测试 | `tests/` 41 个文件，约 320 KB，均为无硬件测试（领域层覆盖较好，UI 层仅少量 smoke） |
| 配置 | `config.yaml`（可提交）+ `config/video_demo.yaml`（录制预设）+ `config/secrets.yaml`（密钥，git 忽略） |
| 版本控制 | 工作区当前有 12 个文件**未提交修改**（README、app.py、agent_plugin.py 等），重构前需先提交或打 tag |

## 3. 模块清单与 S.U.P.E.R 评分

评分：✅ 达标 / ⚠️ 部分违规 / ❌ 明显违规（S=单一职责，U=不稳定依赖，P=模式一致，E=封装边界，R=可复用）

| 模块 | 行数 | 职责 | S | U | P | E | R | 结论 |
| --- | ---: | --- | :-: | :-: | :-: | :-: | :-: | --- |
| `src/ui/app.py` | 5129 | 组装 + 事件 + 相机 + YOLO + 电机 + 搜索 + OCR + 录制 + Agent + 导出 | ❌ | ✅ | ❌ | ❌ | ❌ | **重构主战场** |
| `src/ui/widgets/agent_plugin.py` | 1553 | 助手面板（现场总览/四阶段/对话/自主执行/确认框） | ❌ | ✅ | ⚠️ | ❌ | ❌ | 需重建 |
| `src/ui/widgets/temporary_measurement.py` | 1047 | 临时测量 5 个子功能塞一个面板 | ❌ | ✅ | ⚠️ | ⚠️ | ⚠️ | 拆分或裁掉 |
| `src/ui/widgets/auto_center_control.py` | 643 | 自动寻中参数+状态 | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | 参数过多，需精简 |
| `src/ui/widgets/` 其余 15 个 | ~2.0k | 相机/模型/电机/微分表/状态/日志/录制/浮窗 | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | 重写为薄面板 |
| `src/control/center_control.py` | 1594 | 双向自动寻中状态机（纯逻辑，可注入虚拟时间） | ⚠️ | ✅ | ✅ | ✅ | ✅ | **可直接复用** |
| `src/control/experiment_workflow.py` | 291 | 10 分钟全自动实验流程状态机 | ⚠️ | ✅ | ✅ | ✅ | ⚠️ | 建议裁掉 |
| `src/vision/fringe_width.py` | 1208 | 条纹宽度/间距/周期、中心带定位 | ⚠️ | ✅ | ✅ | ✅ | ✅ | **核心，复用** |
| `src/vision/fringe_center.py` | 409 | 零光程差中心条纹定位 | ✅ | ✅ | ✅ | ✅ | ✅ | **核心，复用** |
| `src/vision/fringe_guidance.py` | 548 | 实时调节指导（倾斜/清晰度/数量） | ⚠️ | ✅ | ⚠️ | ✅ | ⚠️ | 与 agent 的 guidance 重叠，合并 |
| `src/vision/` 其余（detector/strategy/orientation/angle/correct/motion/micrometer_ocr/thickness_distribution） | ~2.1k | YOLO 包装、策略、方向、校正、运动增强、OCR、厚度分布 | ⚠️ | ✅ | ✅ | ✅ | ✅ | 复用 |
| `src/measurement/thickness.py` | 94 | h=(d2-d1)/(20(n-1)) 厚度计算 | ✅ | ✅ | ✅ | ✅ | ✅ | **核心，复用** |
| `src/measurement/uncertainty.py` | 304 | GUM A/B 类、合成/扩展、Grubbs | ✅ | ✅ | ✅ | ✅ | ✅ | **核心，复用** |
| `src/measurement/experiment_assistant.py` | 271 | 多轮测量统计与持久化 | ✅ | ✅ | ✅ | ✅ | ✅ | 复用 |
| `src/hardware/*` | 771 | 电机协议、串口队列、Arduino、微分表 OCR 封装 | ✅ | ✅ | ✅ | ✅ | ✅ | 复用 |
| `src/camera/*` | 671 | OpenCV 双相机管理 | ✅ | ✅ | ✅ | ✅ | ✅ | 复用 |
| `src/agent/service.py` | 589 | 检索 + 生成 + 离线降级 + 上下文拼装（4 件事） | ❌ | ✅ | ⚠️ | ⚠️ | ⚠️ | 重建 |
| `src/agent/device_tools.py` + `toolkit.py` + `tools.py` | 952 | **两套工具概念**：设备工具(READ/MOTION/STOP) + 确定性计算工具 | ⚠️ | ✅ | ❌ | ✅ | ✅ | 合并为一套注册表 |
| `src/agent/loop.py` + `provider.py` + `session.py` | 472 | function calling 循环、Provider、可取消会话 | ✅ | ✅ | ✅ | ✅ | ✅ | 重建但思路可留 |
| `src/agent/experiment_guidance.py` + `proactive.py` + `laser_guidance_session.py` | 591 | 三个重叠的"该提示什么"决策器 | ❌ | ✅ | ❌ | ✅ | ⚠️ | **合并为 1 个** |
| `src/config.py` | 18k(B) | YAML 加载 + 范围校验 | ✅ | ✅ | ✅ | ✅ | ✅ | 复用（缩键） |
| `src/cli.py` | 10.7k(B) | 纯逻辑命令行 | ✅ | ✅ | ✅ | ✅ | ⚠️ | 保留并跟随新 API |

**健康度小结**：领域层（vision / measurement / hardware / camera / control）整体合格，是本项目最值钱的资产；
**违规热点集中在两个文件**（`ui/app.py`、`ui/widgets/agent_plugin.py`，合计 6,682 行 = 全部 UI 的 70%）
以及智能体侧的**概念重复**（2 套工具、3 个 guidance）。因此"整体重写 UI 与智能体、原样接管领域层"
是投入产出比最高的路线，不需要从头实现算法。

## 4. 具体"乱"在哪里（可按此验收重构成果）

1. **上帝对象**：`YoloCamApp.__init__` 到 `_build_ui` 就 600 行；`_backlash_step` 单个方法 199 行
   （4781–4980）；`_show_frame` 180 行；`_on_temporary_measurement_cmd` 151 行。
2. **UI 与逻辑双绑**：widget 直接读 `app.xxx` 属性、直接调 `app._agent_start_auto_center()`；
   改一个面板要动 app.py。无状态层，状态散落在 Tk 变量（`tk.StringVar`）里。
3. **导航冗余**：顶部有一个假导航（三个纯 Label，点了没反应）+ 插件开关栏 + 模块上下移
   (`_move_plugin`) + 跳转 (`_jump_to_plugin`) + 五层折叠 (`CollapsibleFrame`) + 浮动助手窗口。
   实际只有一页，用户要滚很久才能到"搜索条纹"。
4. **智能体双入口**：`tools.py`（确定性计算）与 `device_tools.py`（设备工具）两套 schema、
   两套风险分级，模型看到的工具表 18 个，其中 7 个是"停止/急停"类重复语义。
5. **引导决策三份**：`experiment_guidance.py`、`vision/fringe_guidance.py`、`proactive.py`
   各自算"现在该做什么"，结论可能互相矛盾。
6. **主流程被淹没**：用户要的"搜索条纹 → 定位置 → 读数 → 计算"四步，在界面上对应
   02 运动控制 / 01 中心条纹分析 / 03 微分表 / 03 厚度测量 四个折叠区，且顺序不是流程顺序。

## 5. 可复用资产清单（不建议重写）

- `src/vision/fringe_center.py`、`fringe_width.py`、`fringe_orientation.py`、`detector.py`、
  `detection_strategy.py`、`correct.py`、`micrometer_ocr.py`、`thickness_distribution.py`
- `src/measurement/*`（厚度、不确定度、多轮统计）
- `src/hardware/*`（电机协议 + 命令队列 + OCR 封装）、`src/camera/*`
- `src/control/center_control.py`（1594 行状态机，测试最厚）
- `tests/` 中对应的领域层测试（约 30 个文件可原样复用）
- `models/`（YOLO 权重、OCR ONNX）、`config/calibration.yaml`、`数据记录/`（标定数据）

## 6. 主要风险

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| 工作区 12 个文件未提交 | 重构中难以回退对比 | 动手前先 commit 或 `git tag pre-refactor` |
| 硬件不可常驻（相机/电机/微分表） | 新 UI 的真机联调断断续续 | 先把领域层做成"纯逻辑 + 可注入假设备"，用录像/截图回归 |
| 领域层被旧 UI 反向依赖 | 直接 import 会拖入 tkinter | 复用前先确认无 `import tkinter`；有则抽签名 |
| 无 UI 自动化测试 | 界面重写后回归靠肉眼 | 状态层（AppState）单测 + 关键页面 smoke 测试 |
| 竞赛/报告时间压力 | 重构可能拖过交付点 | 用绞杀者式：新程序可运行即替换入口，旧程序保留可回退 |

## 7. 结论（进入阶段 2 的前提）

- 重写范围：**UI 层（全部）+ 智能体层（全部）+ 应用编排层**，约 12,000 行。
- 复用范围：**领域层原样接管**，约 9,500 行，仅做接口收窄。
- 目标形态：单一主窗口、左侧一条流程（搜索 → 定位 → 读数 → 计算），右侧大画面；
  智能体退化为"右下角对话 + 可确认的动作"，不再与主流程争夺屏幕。
