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
        {"厚度mm": .10347, "d1_mm": 0.0, "d2_mm": .949},
    ])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["value_mm"] == "1.235"
    assert rows[1]["d1"] == "1.200"
    assert rows[1]["d2"] == "2.346"
    assert rows[1]["value_mm"] == "0.12"
    assert rows[2]["厚度mm"] == "0.10"
    assert service.export_json({"thickness_mm": .123456}, tmp_path / "raw.json").read_text().find("0.123456") >= 0


def test_provider_reads_standalone_credentials(tmp_path, monkeypatch):
    import app.agent.runtime as runtime

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "secrets.yaml").write_text(
        "deepseek_api_key: test-key\n", encoding="utf-8")
    assert make_provider(AppConfig()).api_key == "test-key"


def test_single_thickness_display_keeps_two_decimals_and_reading_details():
    from types import SimpleNamespace
    from app.core.state import AppState
    from app.services.compute import ComputeService

    app = QApplication.instance() or QApplication([])
    saved = []
    ctx = SimpleNamespace(config=AppConfig(), state=AppState(), compute=ComputeService(),
                          save_measurement=lambda *args: saved.append(args), log=lambda _: None)
    page = ComputePage(ctx)
    try:
        page.d1.setValue(0)
        page.d2.setValue(.949)
        page._calculate()
        assert page.thickness_value.text() == "0.10 mm"
        assert "OPD" not in page.opd_label.text()
        assert "光程差" not in page.opd_label.text()
        assert "0.949 mm" in page.opd_label.text()
        assert "n = 1.4586" in page.opd_label.text()
        assert saved[0][2] == ctx.compute.thickness_mm(0, .949)
        assert saved[0][2] != .10
        page._rounds = [{"value_mm": saved[0][2], "d1": 0.0, "d2": .949}]
        page._refresh_rounds()
        assert page.rounds_table.item(0, 1).text() == "0.10"
        assert "厚度 0.10 mm" in page.stats_label.text()
        page._uncertainty()
        assert "平均厚度：0.10 mm" in page.unc_label.text()
    finally:
        page.close()
        app.processEvents()


def test_status_card_uses_the_shared_thickness_precision():
    from app.ui.widgets.status_bar import MetricsBar
    app = QApplication.instance() or QApplication([])
    bar = MetricsBar()
    try:
        bar.cards[3].set_value(.10347)
        assert bar.cards[3]._value.text() == "0.10"
        bar.cards[2].set_value(.949)
        assert bar.cards[2]._value.text() == "0.949"
    finally:
        bar.close()


def test_tool_activity_formats_a_copy_without_rounding_evidence():
    from app.core.formatting import thickness_display_data
    from app.agent.evidence import supported_answer
    raw = {"thickness_mm": .10347, "display_thickness_mm": "0.10", "inputs": {"d1": 0, "d2": .949}}
    displayed = thickness_display_data(raw)
    assert displayed["thickness_mm"] == "0.10"
    assert raw["thickness_mm"] == .10347
    assert displayed["inputs"]["d2"] == .949
    assert supported_answer("厚度为0.10 mm", [raw], [])
