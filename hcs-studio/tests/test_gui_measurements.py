"""Phase 5 GUI: measurement tables after preview, cytofluorogram, map, Fiji export."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_imagexpress  # noqa: E402
from tests.test_segmentation import CENTRES, nuclei_image  # noqa: E402


def _image(row: int, col: int, site: int, wave: int) -> Any:
    return np.clip(nuclei_image(CENTRES, seed=site).data[wave, 0], 0, None).astype(np.uint16)


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    root = make_imagexpress(tmp_path / "ix", [(0, 0)], n_sites=1, image_fn=_image)
    with qtbot.waitSignal(win.loader.loaded, timeout=10000):
        win.open_path(root)
    win.open_field(0, 0, 0)
    yield win
    win.close()


def test_tables_cytofluorogram_map_and_export(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    pipeline = window.pipeline_panel
    pipeline.add_step("segment_classical", {"channel": 1})
    pipeline.add_step("expand_cells", {"distance_um": 3})
    pipeline.add_step("measure_objects", {"texture": False, "granularity": False})
    pipeline.add_step("colocalization", {"channel_a": 1, "channel_b": 2})
    with qtbot.waitSignal(window.preview.finished, timeout=30000):
        pipeline.request_preview()

    panel = window.measurements_panel
    names = [panel.table_choice.itemText(i) for i in range(panel.table_choice.count())]
    assert names == ["objects", "coloc"]
    assert panel.proxy.rowCount() == 10 and "objects: 10 rows" in panel.table_info.text()
    panel.table_choice.setCurrentText("coloc")
    assert panel.proxy.rowCount() == 5
    assert "Table 'objects': 10 row(s)" in window.log_view.toPlainText()
    panel.table.sortByColumn(0, panel.table.horizontalHeader().sortIndicatorOrder())

    assert [panel.region.itemText(i) for i in range(panel.region.count())] == [
        "(whole field)",
        "nuclei",
        "cells",
    ]
    panel.region.setCurrentText("cells")
    panel.channel_a.setCurrentIndex(0)
    panel.channel_b.setCurrentIndex(1)
    panel.analyse()
    assert panel.last_result is not None and "Pearson r = " in panel.result_text.text()
    assert len(panel.plot.getPlotItem().items) >= 3  # histogram image, thresholds, regression line

    panel.show_map()
    assert "colocalization map (nMDP)" in window.viewer_panel.viewer.layers

    panel.export(tmp_path / "fiji")
    saved = json.loads((tmp_path / "fiji" / "hcs_studio_results.json").read_text())
    assert saved["results"]["pearson"] == pytest.approx(panel.last_result.pearson)
    assert saved["channel_a"] == "DAPI" and "Run coloc2_compare.ijm" in panel.result_text.text()

    panel.channel_b.setCurrentIndex(0)
    panel.analyse()
    assert "two different channels" in panel.result_text.text()
