"""迈克尔逊干涉仪智能辅助系统 — 程序入口（v2 / PySide6）。

旧 Tkinter 版本仍可从 ``src.ui.run_app`` 启动，但不再是默认入口。
"""
import faulthandler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from app.main import main as run_v2
    return run_v2()


if __name__ == "__main__":
    # 原生扩展（OpenCV / PyTorch / Tcl/Tk）异常时保留所有 Python 线程栈，
    # 避免只得到一个 Windows 退出码而没有诊断信息。
    faulthandler.enable(all_threads=True)
    main()
