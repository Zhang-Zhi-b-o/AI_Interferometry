# 发布验证（2026-10-31）

- 在基于最新 GitHub 主分支的隔离工作区运行 `python -m pytest app/tests -q`：243 passed，3 subtests passed。
- 源码入口 `python run.py --demo --screenshot ...` 正常退出并生成界面截图。
- 使用 `pip wheel . --no-deps --no-build-isolation` 构建安装包成功；解包后通过 Python `-I` 隔离模式从安装包运行离线演示，正常生成截图。
- 安装包与 Git 文件清单包含 3 份 YOLO 权重、3 份 OCR 模型、11 份知识 PDF 及微分头信号模块。
- 发布文件扫描未发现常见格式的真实 API 密钥；未包含用户配置、密钥文件、日志或测量输出。
- Windows 首次安装脚本会创建虚拟环境并安装依赖；本次未在全新 Windows 系统上重新下载全部依赖。

软件测试和合成演示不证明物理测量精度；相机、电机、微分头驱动、标定和在线 API 仍需在使用者设备上配置与确认。

## Windows portable release 2.1.0 (2026-11-12)

- 329 application regression tests and 5 subtests passed with the bundled CPU Python 3.11 runtime.
- Separate clean source checkout: white-light YOLO, sodium YOLO, all OCR models, local knowledge retrieval and demo/assistant UI loaded successfully.
- Repeated that offline smoke check after moving the complete runtime/application to a new directory containing Chinese characters and a space; passed.
- Signed micrometer driver verified; driver resource included without installing or changing any hardware during release QA.
- API settings save/preserve-key tests passed. Personal configuration, actual API keys, logs, exports and caches excluded from the published source and portable application.
- Fresh-machine driver installation, real camera/OCR accuracy, motor motion and metrological calibration remain dependent on the recipient's hardware; not validated by this software smoke check.

## Windows portable release 2.1.2 (2026-11-13)

- Rebuilt the portable bundle from the current main branch (version 2.1.2) with the same CPU Python 3.11 runtime and the pinned dependency list.
- `app/distribution/build_portable.py` now drops the `pip`/`setuptools` lines that were pinned to the build machine's local wheel paths, so the bundle can be rebuilt on any machine.
- Offline smoke check with the bundled runtime passed: white-light YOLO, sodium YOLO, OCR model, local knowledge index, micrometer driver resource, and demo/assistant UI; the window started on the offscreen platform.
- Thickness display precision confirmed in the bundle: `thickness_mm` renders as two decimals while stored records keep the raw numeric values; micrometer readings stay at three decimals.
- No personal `config.yaml`, `secrets.yaml`, logs, exports or caches are included, and no API key is embedded.
- Real camera/OCR accuracy, motor motion, driver installation and metrological calibration remain dependent on the recipient's hardware.
