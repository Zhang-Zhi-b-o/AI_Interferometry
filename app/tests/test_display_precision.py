import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.agent.runtime import make_provider
from app.core.config import AppConfig
from app.engine.vision.annotations import detection_style
from app.services.record import RecordService
from app.services.report import _reading_number
from app.ui.pages.compute import ComputePage


def test_annotation_labels_omit_scores():
    for name in ("zero_order", "near_fringe", "far_fringe", "custom"):
        for chinese in (True, False):
            for compact in (True, False):
                assert detection_style(name, .81, chinese=chinese, compact=compact) == detection_style(
                    name, chinese=chinese, compact=compact)


def test_reading_widgets_use_three_decimals():
    app = QApplication.instance() or QApplication([])
    spin = ComputePage._reading_spin()
    spin.setValue(1.23456)
    assert spin.decimals() == 3
    assert spin.value() == 1.235
    spin.close()


def test_report_and_csv_readings(tmp_path):
    assert _reading_number(1.2) == "1.200"
    assert _reading_number(1.23456) == "1.235"
    service = RecordService(tmp_path)
    path = service.export_rows([
        {"kind": "reading", "value_mm": 1.23456},
        {"kind": "calculation", "d1": 1.2, "d2": 2.34567, "value_mm": .123456},
    ])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["value_mm"] == "1.235"
    assert rows[1]["d1"] == "1.200"
    assert rows[1]["d2"] == "2.346"
    assert rows[1]["value_mm"] == "0.123456"


def test_provider_reads_standalone_credentials(tmp_path, monkeypatch):
    import app.agent.runtime as runtime

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "secrets.yaml").write_text(
        "deepseek_api_key: test-key\n", encoding="utf-8")
    assert make_provider(AppConfig()).api_key == "test-key"
