"""Offline release smoke check. Does not open cameras, serial ports, or HID devices."""
import os
from pathlib import Path
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["YOLO_AUTOINSTALL"] = "false"
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import numpy as np
import torch
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication

from app.core.config import ROOT, load_config
from app.engine.vision.micrometer_ocr import MicrometerOCR
from app.services.knowledge_index import default_index
from app.ui.main_window import MainWindow


def main():
    assert ROOT == root / "app"
    assert not (ROOT / "config" / "secrets.yaml").exists(), "Release must not contain credentials"
    assert not torch.cuda.is_available(), "Portable release must use CPU PyTorch"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for path in (ROOT / "models/releases/yolov8n_full2954_20261027.pt", ROOT / "models/sodium/best.pt"):
        model = YOLO(str(path))
        model.predict(image, imgsz=320, device="cpu", verbose=False)
        print("MODEL OK", path.name, flush=True)
    MicrometerOCR().load()
    print("OCR MODEL OK", flush=True)
    index = default_index()
    assert index.search("白光干涉条纹如何调节", top_k=1)
    print("LOCAL KNOWLEDGE OK", flush=True)
    assert (ROOT / "engine/hardware/vendor/keyboard.sys").is_file()
    print("MICROMETER DRIVER RESOURCE OK (no installation performed)", flush=True)
    app = QApplication([])
    window = MainWindow(load_config(), demo_mode=True)
    try:
        assert len(window._pages) == 7
        page = window._pages["assistant"]
        assert page.settings_button.text() == "API 设置"
        page.transcript.appendPlainText("助手：普通实验助手可以协助分析读数。")
        page.transcript.render_now()
        for _ in range(5):
            app.processEvents()
            time.sleep(.05)
        print("DEMO AND ASSISTANT UI OK", flush=True)
    finally:
        window.close()
        app.processEvents()
    print("PORTABLE SMOKE CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
