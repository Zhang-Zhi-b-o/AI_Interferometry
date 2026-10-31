# 发布验证（2026-10-31）

- 在基于最新 GitHub 主分支的隔离工作区运行 `python -m pytest app/tests -q`：243 passed，3 subtests passed。
- 源码入口 `python run.py --demo --screenshot ...` 正常退出并生成界面截图。
- 使用 `pip wheel . --no-deps --no-build-isolation` 构建安装包成功；解包后通过 Python `-I` 隔离模式从安装包运行离线演示，正常生成截图。
- 安装包与 Git 文件清单包含 3 份 YOLO 权重、3 份 OCR 模型、11 份知识 PDF 及微分头信号模块。
- 发布文件扫描未发现常见格式的真实 API 密钥；未包含用户配置、密钥文件、日志或测量输出。
- Windows 首次安装脚本会创建虚拟环境并安装依赖；本次未在全新 Windows 系统上重新下载全部依赖。

软件测试和合成演示不证明物理测量精度；相机、电机、微分头驱动、标定和在线 API 仍需在使用者设备上配置与确认。
