"""Phase 14 GUI: the Compare tab."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.model.plate import parse_well_name  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.compare_builders import make_run  # noqa: E402

WELLS = [f"{r}{c:02d}" for r in "AB" for c in range(1, 7)]


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def test_compare_tab(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    run = make_run(tmp_path / "data")
    window.show_results(run)
    panel = window.compare_panel
    results = window.results_panel
    results.plate.set_selection([parse_well_name(w) for w in WELLS])
    with qtbot.waitSignal(panel.images_loaded, timeout=60000):
        results.compare_wells_button.click()
    assert window.tabs.currentWidget() is panel
    assert len(panel.items) == 12 and panel.wells == WELLS

    panel.arrangement.setCurrentText("by concentration × cell line")
    with qtbot.waitSignal(panel.images_loaded, timeout=60000):
        panel.rearrange()
    c = panel.comparison
    assert (c.rows, c.cols) == (2, 6) and c.row_headers == ["HeLa", "MCF7"]
    # zoom and pan are linked to the first panel
    from pyqtgraph import ViewBox

    assert panel.views[5].linkedView(ViewBox.XAxis) is panel.views[0]

    # linked contrast: one change, every panel redrawn with the same display
    panel.set_channel(0, high=600.0)
    for k in (0, 7, 11):
        shown = panel.items[k].image.transpose(1, 0, 2)
        expected = (panel.panel_rgb(k) * 255).astype(np.uint8)
        assert np.array_equal(shown, expected)
    assert all(p.display is None for p in panel.comparison.panels)

    panel.current = 3
    panel.unlink_current()
    assert "contrast unlinked" in panel.labels[3].text
    assert "contrast unlinked" not in panel.labels[2].text

    panel.z_max.setChecked(False)
    panel.z.setValue(2)
    assert {p.z for p in panel.comparison.panels} == {2}

    panel.views[0].setRange(xRange=(8, 40), yRange=(4, 36), padding=0)
    (vx0, vx1), (vy0, vy1) = panel.views[0].viewRange()  # the aspect lock may widen it
    panel.crop_to_view()
    crop = panel.comparison.crop
    assert crop is not None
    expected = (max(vy0, 0) * 0.5, max(vx0, 0) * 0.5, min(vy1, 64) * 0.5, min(vx1, 64) * 0.5)
    assert crop == pytest.approx(expected)  # µm (0.5 µm per pixel)
    assert crop[2] - crop[0] < 32  # smaller than the field

    out = panel.export_to(tmp_path / "figure.png")
    sidecar = json.loads((tmp_path / "figure.png.json").read_text())
    assert out.exists() and len(sidecar["comparison"]["panels"]) == 12
    assert sidecar["comparison"]["panels"][3]["display"] is not None

    # pharmacology / profiles selections reach the tab too
    window.compare_wells(run, ["A01", "B01"])
    qtbot.waitUntil(lambda: len(panel.items) == 2, timeout=60000)
    assert window.last_selection == (run, ["A01", "B01"])
