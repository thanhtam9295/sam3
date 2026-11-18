"""Phase 13 GUI: the Pharmacology tab."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.pharmacology.analysis import CurveResult  # noqa: E402
from hcs_studio.pharmacology.combination import CombinationResult  # noqa: E402
from hcs_studio.pharmacology.synthetic import write_run  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def _items(panel: object, kind: type) -> list:  # type: ignore[type-arg]
    from PySide6.QtCore import Qt

    tree = panel.tree  # type: ignore[attr-defined]
    found = []
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        if isinstance(item.data(0, Qt.ItemDataRole.UserRole), kind):
            found.append(item)
        stack.extend(item.child(i) for i in range(item.childCount()))
    return found


def test_pharmacology_tab(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    run = write_run(tmp_path / "run")
    window.show_results(run)
    panel = window.pharmacology_panel
    assert panel.runs == [run]
    readouts = [panel.readouts.item(i).text() for i in range(panel.readouts.count())]
    assert {"nuclei_count", "pct_caspase_positive"} <= set(readouts)
    assert panel.options().readouts == ["nuclei_count"]  # a count is ticked by default
    window.tabs.setCurrentWidget(panel)

    with qtbot.waitSignal(panel.task.finished, timeout=60000):
        panel.analyse(write=False)
    curves = _items(panel, CurveResult)
    combos = _items(panel, CombinationResult)
    assert len(curves) == 6 and len(combos) == 2
    assert panel.fits_table.rowCount() == 6 and panel.synergy_table.rowCount() == 2
    assert panel.qc_table.rowCount() == 1

    # a curve: click a point → its well is selected and outlined on the Results plate
    curve_item = next(i for i in curves if "Drug D" in i.text(0) and i.parent().text(0) == "HeLa")
    panel.tree.setCurrentItem(curve_item)
    assert panel.views.currentIndex() == 1 and "EC50" in panel.curve_info.text()
    curve = panel._current
    assert isinstance(curve, CurveResult)
    plate, well = curve.points["plate"].iloc[0], curve.points["well"].iloc[0]
    with qtbot.waitSignal(panel.wells_selected) as blocker:
        panel.select([(run.name, plate, well)])
    assert blocker.args == [run.name, [well]]

    # exclude that well: the curve is refitted without it, the point stays (hollow)
    with qtbot.waitSignal(panel.task.finished, timeout=60000):
        panel.toggle_selected()
    current = panel._current
    assert isinstance(current, CurveResult) and current.compound == "Drug D"
    row = current.points[current.points["well"] == well]
    assert bool(row["excluded"].iloc[0]) and current.fit.n_points == len(current.points) - 1
    with qtbot.waitSignal(panel.task.finished, timeout=60000):
        panel.toggle_selected()  # and back in
    assert not bool(panel._current.points["excluded"].any())  # type: ignore[union-attr]

    # a combination: heat maps; a cell selects its replicate wells
    combo_item = next(i for i in _items(panel, CombinationResult) if "Drug B" in i.text(0))
    panel.tree.setCurrentItem(combo_item)
    assert panel.views.currentIndex() == 2
    assert "likely synergistic" in combo_item.text(0)
    assert "BLISS" in panel.matrix_info.text()
    with qtbot.waitSignal(panel.wells_selected) as blocker:
        panel.select_cell(3, 3)
    assert len(blocker.args[1]) == 3
    panel.matrix_model.setCurrentText("zip")
    assert "ZIP synergy" in panel.synergy_plot.titleLabel.text

    # write the report with the current exclusions
    with qtbot.waitSignal(panel.task.finished, timeout=90000):
        panel.analyse(write=True)
    assert (run / "pharmacology" / "pharmacology_report.html").exists()
    assert "wrote" in panel.status.text()
