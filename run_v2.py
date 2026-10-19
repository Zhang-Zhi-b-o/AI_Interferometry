"""v2 启动脚本（旧入口 run.py 保持不动，随时可退回旧程序）。

    python run_v2.py
"""
from __future__ import annotations

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
