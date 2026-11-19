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


def test_normalization_tab_blocks_b_score_on_dose_response_and_compares_replicates(
    qtbot: QtBot, home: Path
) -> None:
    from hcs_studio.model.plate import well_name
    from hcs_studio.normalization.plate import WellInfo
    from hcs_studio.services import context
    from hcs_studio.services.registry import scan_folders
    from hcs_studio.synthetic import screening_plate, write_values_run

    registry_runs(home / "drc", n_runs=2, pharmacology=True)
    values, wells, _ = screening_plate(row_effect=-40, hits=8, seed=3)
    write_values_run(home / "runs" / "SCREEN-1__run", values, wells, barcode="SCREEN-1")
    vehicle = {well_name(r, c): 1000.0 - 80 * (r in (0, 15)) for r in range(16) for c in range(24)}
    write_values_run(
        home / "runs" / "UNIF-1__run",
        vehicle,
        [WellInfo(w, "negative_control") for w in vehicle],
        barcode="UNIF-1",
    )
    scan_folders(context(), [home / "drc", home / "runs"])
    panel = ExperimentsPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    qtbot.waitUntil(lambda: panel.norm_run.count() == 4, timeout=30000)

    panel.norm_run.setCurrentText("SCREEN-1__run")
    panel.norm_readout.setCurrentText("nuclei_count")
    panel.norm_method.setCurrentText("b_score")
    panel.normalize()
    qtbot.waitUntil(lambda: len(panel.normalized) == 384, timeout=20000)
    assert "B-scores are computed for sample wells" in panel.norm_text.toHtml()

    panel.norm_run.setCurrentText("REG-001__regression__run")
    panel.normalize()
    qtbot.waitUntil(lambda: "Not applied" in panel.norm_text.toHtml(), timeout=20000)
    assert "uniformity-plate correction" in panel.norm_text.toHtml()

    panel.norm_run_b.setCurrentText("REG-002__regression__run")
    panel.compare_replicates()
    qtbot.waitUntil(lambda: panel.agreement is not None, timeout=20000)
    assert panel.agreement.pearson > 0.95 and "concordant" in panel.norm_text.toHtml()

    panel.norm_run.setCurrentText("UNIF-1__run")
    panel.save_map("edge map")
    qtbot.waitUntil(lambda: panel.norm_map.count() == 1, timeout=20000)
    assert "Edge effect" in panel.norm_text.toHtml()

    panel.search()
    qtbot.waitUntil(lambda: panel.plate_table.rowCount() == 4, timeout=20000)
    panel.day_effects()
    qtbot.waitUntil(lambda: "R² day" in panel.norm_text.toHtml(), timeout=20000)
