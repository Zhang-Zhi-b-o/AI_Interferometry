# AI Interferometry — 白光干涉测量工作台

仓库当前版本仅保留可独立使用的新程序 `app/`。复制整个 `app` 文件夹即可安装、运行，不依赖旧版项目文件。

```powershell
git clone --depth 1 https://github.com/Zhang-Zhi-b-o/AI_Interferometry.git
cd AI_Interferometry/app
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py --demo
```

正常启动去掉 `--demo`，或安装后双击 `app/start.bat`。

内置 YOLO/OCR 权重、测量算法、设备控制和助手知识资料。实测需要兼容仪器；在线助手需自行配置 API 密钥。

完整使用说明：[app/README.md](app/README.md)。旧版本保留在 Git 历史中。
