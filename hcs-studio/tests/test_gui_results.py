"""Phase 7 GUI: results plate, readouts, well panel, viewer and stage comparison."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.io.base import open_plate  # noqa: E402
from hcs_studio.model.layout import read_plate_layout  # noqa: E402
from hcs_studio.pipeline.executor import run_pipeline  # noqa: E402
from hcs_studio.pipeline.schema import Pipeline  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_imagexpress  # noqa: E402
from tests.test_results import LAYOUT, PIPELINE, WELLS, _image  # noqa: E402


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("gui_results")
    root = make_imagexpress(tmp / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)
    layout = tmp / "layout.csv"
    layout.write_text(LAYOUT, encoding="utf-8")
    pipeline = Pipeline.from_text(PIPELINE.replace("steps:\n", "steps:\n  - {module: contrast}\n"))
    return run_pipeline(
        open_plate(root),
        pipeline,
        output_root=tmp / "out",
        layout=read_plate_layout(layout),
        layout_path=layout,
        checksums=False,
    ).folder


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


def test_results_plate_readouts_and_well_panel(window: MainWindow, run_folder: Path) -> None:
    window.show_results(run_folder)
    panel = window.results_panel
    assert window.tabs.currentWidget() is panel
    assert "Plate1" in panel.folder_label.text() and "6 wells" in panel.folder_label.text()
    assert panel.readout.currentText() == "cells_count"
    plate = panel.plate
    assert plate.fill_of((0, 0)) != plate.fill_of((0, 2))  # coloured by the readout
    assert plate._items[(0, 0)].toolTip().startswith("A01: 2")
    assert panel.zprime.text().startswith("Z′ = ")
    panel.normalisation.setCurrentText("robust Z (negative controls)")
    assert "(Z " in plate._items[(0, 2)].toolTip()
    panel.normalisation.setCurrentText("values")

    panel.filter.setText("per_mm2")
    assert all("per_mm2" in panel.readout.itemText(i) for i in range(panel.readout.count()))
    panel.filter.setText("")

    panel.object_type.setCurrentText("nuclei")
    panel.feature.setCurrentText("dapi_mean")
    panel.threshold.setValue(1500)
    panel.show_percent_positive()
    assert panel.readout.currentText() == "% nuclei with dapi_mean > 1500"
    assert panel.raw_values["A03"] == 100 and panel.raw_values["A01"] == 0

    panel.plate.select_well(1, 1)  # B02, as a click does
    assert panel.title.text() == "Well B02" and "Drug X" in panel.annotation.text()
    assert panel.stats.rowCount() > 3 and panel.stats.model().index(0, 0).data().startswith(
        "% nuclei"
    )
    assert panel.fields.count() == 2 and sorted(panel.overlay_checks) == ["cells", "nuclei"]
    with_outlines = panel.large.pixmap().toImage()
    panel.overlay_checks["nuclei"].setChecked(False)
    panel.overlay_checks["cells"].setChecked(False)
    assert panel.large.pixmap().toImage() != with_outlines
    panel.stage.setCurrentText("Raw")
    assert not panel.large.pixmap().isNull()


def test_open_in_viewer_and_compare_stages(window: MainWindow, run_folder: Path) -> None:
    window.show_results(run_folder)
    panel = window.results_panel
    panel.select_well("B02")
    panel.fields.setCurrentRow(1)
    panel.stage.setCurrentText("Corrected")
    panel._open_in_viewer()
    viewer = window.viewer_panel.viewer
    names = [layer.name for layer in viewer.layers]
    assert "DAPI · corrected" in names and "nuclei · corrected" in names
    assert window.tabs.currentWidget() is window.viewer_panel.widget
    assert tuple(viewer.layers["DAPI · corrected"].scale) == pytest.approx((0.65, 0.65))

    panel._compare()
    names = [layer.name for layer in viewer.layers]
    assert {"DAPI · raw", "DAPI · corrected", "DAPI · enhanced"} <= set(names)
    assert "nuclei · enhanced" in names
    offsets = [viewer.layers[f"DAPI · {s}"].translate[-1] for s in ("raw", "corrected", "enhanced")]
    assert offsets[0] == 0 < offsets[1] < offsets[2]  # side by side
    assert "Comparing raw, corrected, enhanced for B02 field 2" in window.log_view.toPlainText()


def test_finished_job_opens_its_results(window: MainWindow, run_folder: Path) -> None:
    state = SimpleNamespace(
        status="completed", folder=str(run_folder), failed={}, job=SimpleNamespace(label="x")
    )
    window._on_job_finished(state)
    assert window.tabs.currentWidget() is window.results_panel
    assert "shareable report" in window.log_view.toPlainText()
    window.results_panel.load(run_folder / "missing")
    assert "not an HCS Studio run folder" in window.results_panel.folder_label.text()
