"""Phase 17a GUI: the Experiments tab."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.experiments_panel import ExperimentsPanel  # noqa: E402
from hcs_studio.synthetic import registry_runs  # noqa: E402


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_experiments_tab_scans_searches_and_shows_results(qtbot: QtBot, home: Path) -> None:
    runs = registry_runs(home / "runs", n_runs=3)
    panel = ExperimentsPanel()
    qtbot.addWidget(panel)
    logs: list[str] = []
    panel.log.connect(logs.append)
    panel.scan([home / "runs"])
    qtbot.waitUntil(lambda: panel.plate_table.rowCount() == 3, timeout=60000)
    assert "3 added" in logs[0]
    assert panel.compound_list.count() == 4 and panel.line_list.count() == 2

    panel.plate_table.selectRow(0)
    qtbot.waitUntil(lambda: bool(panel.page), timeout=20000)
    assert panel.page["plate"]["barcode"] == "REG-003" and "REG-003" in panel.plate_details.toHtml()
    with qtbot.waitSignal(panel.open_run, timeout=5000) as opened:
        panel._open_latest()
    assert Path(opened.args[0]) == runs[2]

    panel.pot_compound.setText("Drug A")
    panel.show_potency()
    qtbot.waitUntil(lambda: panel.potency_table.rowCount() == 2, timeout=20000)  # MCF7, HeLa
    assert set(panel.potency["n"]) == {3}

    panel.trend_readout.setCurrentText("nuclei_count")
    panel.trend_role.setCurrentText("negative_control")
    panel.show_trend()
    qtbot.waitUntil(lambda: len(panel.trend) == 3, timeout=20000)

    panel.compound_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: "Dose–response fits" in panel.reference.toHtml(), timeout=20000)

    panel.text.setText("REG-002")
    panel.save_search("just REG-002")
    qtbot.waitUntil(lambda: panel.saved.count() == 1, timeout=20000)
    qtbot.waitUntil(lambda: panel.plate_table.rowCount() == 1, timeout=20000)
    panel.text.setText("")
    panel.search()
    qtbot.waitUntil(lambda: panel.plate_table.rowCount() == 3, timeout=20000)
    panel.saved.setCurrentIndex(0)
    panel._apply_saved()
    qtbot.waitUntil(lambda: panel.plate_table.rowCount() == 1, timeout=20000)
    assert panel.text.text() == "REG-002"
