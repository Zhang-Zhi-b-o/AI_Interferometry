"""程序入口。

    python run_v2.py                      # 正常启动
    python run_v2.py --theme dark         # 深色主题
    python run_v2.py --scale 1.1          # 字号放大 10%
    python run_v2.py --image 图片.jpg     # 演示模式：用一张图代替相机
    python run_v2.py --screenshot a.png   # 截图后退出（自检 / 文档配图用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app import APP_NAME
from app.core.config import load_config
from app.ui import theme as theme_module
from app.ui.main_window import MainWindow


def _load_image(path: str) -> np.ndarray | None:
    import cv2

    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--theme", choices=["light", "dark"], default=None)
    parser.add_argument("--scale", type=float, default=None, help="字号缩放，1.0 为基准")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", default=None, help="用静止图片代替相机")
    source.add_argument("--demo", action="store_true", help="完整离线演示：虚拟设备与合成数据")
    parser.add_argument("--screenshot", default=None, help="渲染后截图到指定文件再退出")
    parser.add_argument("--page", default=None, help="启动后切到指定页面 key")
    parser.add_argument("--delay", type=int, default=1400, help="截图前等待毫秒")
    args = parser.parse_args(argv)

    config = load_config()
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    theme_name = args.theme or config.theme
    theme_module.apply_theme(app, theme_name, args.scale or config.font_scale)

    demo = _load_image(args.image) if args.image else None
    if args.image and demo is None:
        parser.error(f"无法读取演示图片：{args.image}")
    window = MainWindow(config, demo_image=demo, demo_mode=args.demo)
    window._theme_name = theme_name
    if args.page:
        QTimer.singleShot(0, lambda: window._navigate(args.page))
    window.show()

    if args.screenshot:
        target = Path(args.screenshot)

        def grab() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(target))
            print(f"screenshot saved: {target}")
            app.quit()

        QTimer.singleShot(max(200, args.delay), grab)

    try:
        return app.exec()
    finally:
        window.close()  # 截图退出也必须关闭相机、取消助手并停车


if __name__ == "__main__":
    raise SystemExit(main())
