# 项目结构

```text
AI_Interferometry/
├── run.py                         程序入口
├── config.yaml                    可提交的运行配置
├── config/
│   ├── calibration.yaml           标定数据
│   ├── secrets.example.yaml       密钥格式示例
│   └── secrets.yaml               本地密钥，Git 忽略
├── pyproject.toml                 唯一依赖与包配置
├── requirements.txt              兼容入口，安装当前项目
├── models/current/                当前 YOLO 权重，Git 忽略
├── resources/agent_sources/       本地实验资料与资料清单
├── src/
│   ├── agent/                     检索、提示词、模型 Provider、可取消会话
│   ├── camera/                    OpenCV 相机管理
│   ├── control/                   可测试的自动控制状态机
│   ├── hardware/                  电机、Arduino、串口后台队列
│   ├── ui/                        Tkinter 主控制器与插件面板
│   ├── vision/                    YOLO、ROI、校正和中心条纹算法
│   ├── config.py                  配置加载和校验
│   └── logging.py                 日志配置
└── tests/
    ├── test_agent.py              检索、历史、取消与回答边界
    ├── test_config.py             配置类型和范围
    ├── test_control.py            自动状态机和命令队列
    └── test_safety.py             视觉输入、类别和串口协议
```

## 开发约定

- 新的阻塞 I/O 不得直接放进 Tkinter 回调。
- 新的自动控制逻辑先加入 `src/control` 并编写无硬件测试。
- 不依赖固定 YOLO 类别 ID，必须读取模型类别名称并映射语义。
- 密钥、模型权重、原始 PDF 和运行日志不提交到 Git。
- 依赖版本只在 `pyproject.toml` 维护，`requirements.txt` 不复制版本列表。
