import os
from types import SimpleNamespace as NS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from app.core.config import AppConfig
from app.engine.measurement.sodium import calculate_mean_wavelength_nm
from app.ui.pages.sodium import SodiumPage


@pytest.mark.parametrize("sign", [1, -1])
def test_two_period_travel_recovers_mean_wavelength(sign):
    # 589.3 nm and a fixed 0.6 nm splitting predict two-period mirror
    # travel of λ²/Δλ; the micrometer travels twenty times farther.
    travel_mm = 20 * (589.3**2 / .6) / 1e6
    assert calculate_mean_wavelength_nm(15, 15 + sign * travel_mm) == pytest.approx(589.3)


@pytest.mark.parametrize("first,second", [(1, 1), (float("nan"), 1), (1, float("inf"))])
def test_invalid_endpoints_do_not_produce_a_wavelength(first, second):
    with pytest.raises(ValueError):
        calculate_mean_wavelength_nm(first, second)


def test_page_shows_completed_result_and_clears_it_for_next_run():
    app = QApplication.instance() or QApplication([])

    class Bridge(QObject):
        sodium_update = Signal(dict)

    status = {"run_id": 1, "first_mm": 1.0, "second_mm": 12.57581633,
              "result": {"mean_wavelength_nm": 589.3, "reading_difference_mm": 11.57581633}}
    events = []
    ctx = NS(config=AppConfig(), bridge=Bridge(), log=lambda _: None,
             timeline=NS(add=lambda kind, **kw: events.append((kind, kw))),
             sodium=NS(snapshot=lambda: status, active=False, running=False, run_id=1))
    page = SodiumPage(ctx)
    try:
        assert page.wavelength._value.text() == "589.300"
        assert page.wavelength._unit.text() == "nm"
        assert page.first._value.text() == "1.000"
        assert page.second._value.text() == "12.576"
        ctx.bridge.sodium_update.emit(status)
        assert events[-1][1]["result"]["mean_wavelength_nm"] == 589.3
        status.update(run_id=2, first_mm=None, second_mm=None, result=None)
        ctx.sodium.run_id = 2
        page.refresh()
        assert page.wavelength._value.text() == "—"
    finally:
        page.timer.stop()
        page.close()
        app.processEvents()
