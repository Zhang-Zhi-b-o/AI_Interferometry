# AI Interferometry — 白光干涉测量工作台

仓库当前版本仅保留可独立使用的新程序 `app/`。复制整个 `app` 文件夹即可安装、运行，不依赖旧版项目文件。

```powershell
git clone --depth 1 https://github.com/Zhang-Zhi-b-o/AI_Interferometry.git
cd AI_Interferometry/app
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py --demo
```

Windows 便捷启动：先安装 64 位 Python 3.10/3.11，再双击 `app/start.bat`，首次自动安装依赖并启动。更新后运行 `app/setup.bat` 更新依赖。无需硬件体验可运行 `app/start.bat --demo`。

内置 YOLO/OCR 权重、测量算法、设备控制和助手知识资料。实测需要兼容仪器；在线助手需自行配置 API 密钥。

完整使用说明：[app/README.md](app/README.md)。旧版本保留在 Git 历史中。
