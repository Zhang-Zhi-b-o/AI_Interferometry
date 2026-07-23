# 项目结构

```text
AI_Interferometry/
├── run.py                         程序入口
├── config.yaml                    可提交的运行配置
├── pyproject.toml                 唯一依赖与包配置
├── requirements.txt              兼容入口，安装当前项目
├── requirements-dev.txt          开发依赖
├── 快速使用.md                    快速上手指南
├── config/
│   ├── calibration.yaml           标定数据
│   └── secrets.yaml               本地密钥，Git 忽略
├── models/
│   ├── current/                   当前 YOLO 权重，Git 忽略
│   ├── candidates/                候选模型权重
│   ├── micrometer/                微分表 OCR ONNX 权重与来源说明
│   └── select.py                  模型选择辅助脚本
├── resources/
│   └── agent_sources/             本地实验资料与资料清单
├── scripts/                       辅助脚本预留目录
├── src/
│   ├── agent/                     检索、提示词、模型 Provider、可取消会话
│   │   └── knowledge_base/        本地 Markdown 知识库
│   ├── camera/                    OpenCV 相机管理
│   ├── control/                   双向自动寻中与实验进度状态机
│   ├── hardware/                  电机、视觉微分表、Arduino、串口后台队列
│   ├── report/                    实验报告功能预留目录（空）
│   ├── ui/                        Tkinter 主界面与功能面板
│   │   └── widgets/               可独立组合的界面插件
│   ├── uncertainty/               不确定度计算功能预留目录（空）
│   ├── vision/                    YOLO、ROI、校正、中心条纹、OCR 与运动检测
│   ├── config.py                  配置加载和校验
│   ├── constants.py               项目常量
│   └── logging.py                 日志配置
├── tests/                         无硬件自动测试
├── artifacts/                     流程图和设计图等制图产物
├── 数据记录/                      实际实验图片、标定过程和分析结果
└── 报告和PPT/                     项目报告与展示材料
```

## 开发约定

- 新的阻塞 I/O 不得直接放进 Tkinter 回调。
- 新的自动控制逻辑先加入 `src/control` 并编写无硬件测试。
- 不依赖固定 YOLO 类别 ID，必须读取模型类别名称并映射语义。
- 密钥、模型权重、原始 PDF 和运行日志不提交到 Git。
- 依赖版本只在 `pyproject.toml` 维护，`requirements.txt` 不复制版本列表。
