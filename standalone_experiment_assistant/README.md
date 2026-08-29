# 实验助手兼容入口

实验助手已合并到主工作台，项目中只保留一套助手界面、
`AgentService`、对话会话和实时状态。

主入口：

```powershell
python run.py
```

旧命令仍然可用：

```powershell
python standalone_experiment_assistant/run.py
```

两个命令现在会启动同一个主工作台，不再打开另一个独立助手。
DeepSeek 密钥也只由主助手统一读取。
