# 独立实验助手

该目录包含一个不访问相机、电机或 YOLO 权重的独立实验助手程序。
它直接复用主程序的实验助手面板，使用目录内的阶段与读数数据展示完整实验过程，
并通过项目现有配置连接 DeepSeek。

启动：

```powershell
python standalone_experiment_assistant/run.py
```

左侧“实验进度控制”可以选择任意实验步骤，也可以使用“上一步”“下一步”逐步推进。
右侧助手的状态栏、快捷指令栏、对话栏、输入栏、折叠、尺寸调整和字体调整行为与主程序一致。

DeepSeek 密钥读取顺序与主程序一致：

1. 环境变量 `DEEPSEEK_API_KEY`；
2. `config/secrets.yaml` 中的 `deepseek_api_key`。

密钥文件已被 Git 忽略，不会随提交上传。
