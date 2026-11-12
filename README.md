# 白光干涉测量工作台

Windows 10/11 x64：**[下载最新免安装便携包](https://github.com/Zhang-Zhi-b-o/AI_Interferometry/releases/latest)**，选择 `AI_Interferometry_Windows_x64.zip`，完整解压后双击 **start.bat**。已包含 Python、CPU 依赖、YOLO/OCR/钠灯模型及微分表驱动文件，无需安装 Python、CUDA 或运行 pip。

- 无设备体验：双击 `demo.bat`。
- 实验助手：⑥实验助手 → **API 设置**，填写你自己的 API Key；发行包没有内置可用密钥。
- 微分表：④读数 → **开始信号读取与拦截**；支持 C216:010B USB HID，首次按提示允许管理员配置，必要时重启 Windows。其他型号可使用相机 OCR。
- 电机：②搜索条纹中选择串口与波特率（默认9600），先查询状态验证通信，再操作运动。

完整说明、设备连接与驱动撤销方法见 **[app/README.md](app/README.md)**。

GitHub **Code → Download ZIP 是源码**，不含 Python 运行时。源码方式需要先安装64位 Python 3.10/3.11，然后双击 `app/start.bat` 自动安装依赖。复制整个 `app/` 即可独立开发，不依赖旧项目。

功能包括实时识别、条纹位置分析、微分表信号/OCR读数、厚度计算、实验助手、钠灯拍频及平均波长自动计算。中心条纹仅显示一个框，非中心条纹可显示多个；各读数统一三位小数。

第三方组件保留各自许可证，来源见 `app/models/`、`app/engine/hardware/vendor/` 及便携包运行时。软件演示和自动计算不代替仪器校准与实机验证。
