"""Signal bursts preserve records without repeatedly laying out the history table."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication
from app.core.config import AppConfig
from app.core.state import AppState
from app.core.timeline import Timeline
from app.ui.pages.reading import ReadingPage
from app.ui.widgets.common import StatusPill


class ReadingResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_signal_burst_keeps_every_record_and_coalesces_scrolling(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = NS(config=AppConfig(), state=AppState(), log=Mock(), history=[],
                     timeline=Timeline(Path(directory)),
                     reading=NS(connected=True, connecting=False, source='signal'))
            page = ReadingPage(ctx)
            try:
                with patch.object(page.table, 'scrollToBottom') as scroll:
                    for i in range(500):
                        page.on_reading(dict(source='signal', value_mm=i / 1000,
                                             at=time.time(), stable=True, text=str(i)))
                    scroll.assert_not_called()
                    self.assertTrue(page._history_scroll_timer.isActive())
                    with patch.object(page.table, 'isVisible', return_value=True):
                        page._scroll_history()
                    scroll.assert_called_once()
                self.assertEqual(page.table.rowCount(), 200)
                self.assertEqual(len(ctx.history), 200)
                self.assertAlmostEqual(ctx.history[-1]['value_mm'], .499)
                events = [json.loads(line) for line in ctx.timeline.path.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(events), 500)
                self.assertEqual(events[-1]['result']['value_mm'], .499)
                page._clear()
                self.assertFalse(page._history_scroll_timer.isActive())
            finally:
                page.close()

    def test_repeated_status_updates_do_not_reapply_styles(self):
        pill = StatusPill('ready', 'ok')
        with patch.object(pill, 'setStyleSheet') as style:
            for _ in range(100):
                pill.set_state('ok', 'ready')
            style.assert_not_called()
            pill.set_state('warn', 'waiting')
            style.assert_called_once()
        self.assertIn('waiting', pill.text())


if __name__ == '__main__':
    unittest.main()
