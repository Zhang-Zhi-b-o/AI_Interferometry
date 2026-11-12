# 白光干涉测量工作台

迈克耳孙白光干涉实验的 Windows 桌面程序：相机实时预览、条纹识别与中心定位、电机自动寻中、微分表自动读数、玻璃片厚度与不确定度、钠灯拍频波长、可调用工具的 AI 实验助手。

**免安装使用：下载 → 完整解压 → 双击 `start.bat`。** 压缩包已内置 64 位 Python 运行时、CPU 版依赖、YOLO/OCR/钠灯模型、微分表驱动文件与知识库文献，**不需要**安装 Python、CUDA，也不用运行 pip。

[![最新版本](https://img.shields.io/github/v/release/Zhang-Zhi-b-o/AI_Interferometry?label=%E6%9C%80%E6%96%B0%E7%89%88%E6%9C%AC&color=2ea44f)](https://github.com/Zhang-Zhi-b-o/AI_Interferometry/releases/latest)
[![下载次数](https://img.shields.io/github/downloads/Zhang-Zhi-b-o/AI_Interferometry/total?label=%E4%B8%8B%E8%BD%BD%E6%AC%A1%E6%95%B0)](https://github.com/Zhang-Zhi-b-o/AI_Interferometry/releases/latest)

## 一、下载与安装（3 步）

1. **下载直链**（约 745 MB）：
   **[⬇ AI_Interferometry_Windows_x64.zip](https://github.com/Zhang-Zhi-b-o/AI_Interferometry/releases/latest/download/AI_Interferometry_Windows_x64.zip)**
   也可以在 [Releases 页面](https://github.com/Zhang-Zhi-b-o/AI_Interferometry/releases/latest) 里手动挑选版本，同页提供 `.sha256` 校验文件。
2. **完整解压**到可写目录，例如 `D:\AI_Interferometry`。
   不要在压缩软件的预览窗口里直接双击运行；路径含中文或空格不影响使用。
3. **双击 `start.bat`** 启动。没有连接仪器时双击 `demo.bat`，用虚拟设备先熟悉界面和流程。

### 电脑要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10 / 11，64 位（不支持 ARM 版 Windows） |
| 磁盘 | 压缩包约 745 MB，解压后约 2 GB，建议预留 5 GB |
| 内存 | 8 GB 起；同时开报告生成和在线助手时更吃内存 |
| 显卡 | 不需要。便携版固定使用 CPU，有独显只影响识别速度，不影响功能 |

### 可选：校验下载是否完整

```powershell
certutil -hashfile AI_Interferometry_Windows_x64.zip SHA256
```

将输出与 Release 页面同名 `.sha256` 文件里的值比对即可。

## 二、第一次打开要做什么

| 想做的事 | 操作 |
| --- | --- |
| 先看看界面 | 双击 `demo.bat`：虚拟相机、虚拟电机、合成读数，不碰硬件 |
| 接干涉相机 | 「① 实时画面」选择相机编号（默认 1），确认画面后加载条纹模型 |
| 找条纹 / 自动寻中 | 「② 搜索条纹」先选择串口与波特率（默认 9600）并查询状态，确认通信后再运动 |
| 拍频钠灯实验 | 「⑧ 钠灯拍频」先「加载钠灯模型」可预览识别框（只识别、不转电机），再选方向开始 |
| 读微分表 | 「④ 读数」→「开始信号读取与拦截」，首次按提示允许管理员配置；其他型号可用第二相机 OCR |
| 玻璃片厚度 | 「⑤ 计算」填两端读数；表格与报告里厚度显示两位小数，微分表读数保持三位小数 |
| AI 实验助手 | 「⑥ 实验助手」→「API 设置」，填**你自己的** API Key；发行包不含任何可用密钥，测量功能不依赖它 |

更细的设备连接、驱动安装与撤销、界面说明见 **[app/README.md](app/README.md)**。

## 三、常见问题

- **双击 `start.bat` 窗口一闪而过 / 打不开**：先确认是「完整解压」而不是在压缩包里运行；再查看 `app\logs\app.log` 末尾的报错。
- **Windows 提示"已保护你的电脑"或杀毒软件报警**：包内的 Python 运行时未做代码签名，点「更多信息 → 仍要运行」，或把解压目录加入信任。
- **微分表读不到数**：仅支持 C216:010B USB HID 信号设备；换 USB 接口后按提示重新配置，必要时重启 Windows。其他型号请走相机 OCR。
- **实验助手报错或不可用**：需要自己的 DeepSeek API Key，且本机可访问外网；设备测量、计算与报告不受影响。
- **升级到新版本**：解压新包到新目录，想保留原有配置时，把旧包里的 `app\config.yaml`、`app\config\secrets.yaml` 复制到新包对应位置。
- **彻底卸载**：直接删除解压目录即可；装卸过的微分表驱动如需撤销，见 app/README.md。

## 四、数据保存在哪里

- `app\logs\`：运行日志、信号配置日志（排查问题先看这里）。
- `app\exports\`：测量记录、Word 报告、图片与视频。

程序只在本机读写这些目录，不会上传实验数据。

## 五、从源码运行（开发者）

GitHub 的 **Code → Download ZIP 是源码**，不含 Python 运行时。源码方式先安装 64 位 Python 3.10/3.11，然后：

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py --demo
```

或直接双击 `app\start.bat`（首次联网安装依赖）。复制整个 `app/` 即可独立开发，不依赖旧项目；`app/start.bat` 会自动识别便携包里的 `runtime\`。

重新打便携包（需要 `uv`，产物放在仓库外）：

```powershell
cd app
uv python install 3.11
python distribution\build_portable.py --output D:\release
```

## 六、功能与设备

- 实时相机预览、画面校正、ROI、单帧/连续 YOLO 识别、图片与录像保存。
- 条纹中心、角度、宽度、间距分析与位置记录；中心条纹只显示一个框，非中心可多个。
- 电机手动控制、自动寻中（默认 10 分钟上限，`Esc` 请求停止）和目标读数。
- 微分头信号直接读取与键盘拦截、自动记录，也可用第二相机 OCR 读数。
- 厚度计算、不确定度和多轮统计；钠灯拍频双端测量及平均波长自动计算。
- 本地知识检索、在线助手、原文出处、Word 测量报告与性能评测报告。

实测需要兼容的干涉仪相机、拍摄微分表的第二相机，以及使用现有串口协议的电机控制器。默认主相机编号 1、读数相机编号 0，可在界面调整。自动运动前请核对电机方向、行程及停止功能。

## 七、说明

第三方组件保留各自许可证，来源见 `app/models/`、`app/engine/hardware/vendor/` 及便携包 `runtime/Lib/site-packages` 下的各 `dist-info`。软件测试和合成演示不替代仪器校准与实机验证：相机、电机、微分头驱动、标定和在线 API 仍需在使用者设备上配置与确认。
