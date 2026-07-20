"""迈克尔逊干涉仪智能辅助系统 — 程序入口"""
import faulthandler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging import logger
from src.config import config


def main():
    logger.info(f"启动 {config.get('app', 'name')} v{config.get('app', 'version')}")

    from src.ui import run_app
    run_app()


if __name__ == "__main__":
    # 原生扩展（OpenCV / PyTorch / Tcl/Tk）异常时保留所有 Python 线程栈，
    # 避免只得到一个 Windows 退出码而没有诊断信息。
    faulthandler.enable(all_threads=True)
    main()
