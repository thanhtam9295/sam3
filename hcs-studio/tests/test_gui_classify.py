"""Phase 8 GUI: gates on plots, classifier from clicked objects, colouring, mosaic."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.classify_panel import NO_Y, sample_objects  # noqa: E402
from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.io.base import open_plate  # noqa: E402
from hcs_studio.pipeline.executor import run_pipeline  # noqa: E402
from hcs_studio.pipeline.schema import Pipeline  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.test_analysis import _stitch_plate  # noqa: E402

PIPELINE = """
name: classify
steps:
  - {module: segment_classical, params: {channel: 1, min_distance_um: 4}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
  - {module: stitch_well}
  - {module: spatial_clusters, params: {simulations: 0, radii_um: [10]}}
"""


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("gui_classify")
    root, _ = _stitch_plate(tmp)
    return run_pipeline(
        open_plate(root), Pipeline.from_text(PIPELINE), output_root=tmp / "out", checksums=False
    ).folder


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


def test_sample_objects_streams_every_kth_row(run_folder: Path) -> None:
    everything = pd.read_parquet(run_folder / "tables" / "per_object_all.parquet")
    sample = sample_objects(run_folder, max_rows=20)
    assert 0 < len(sample) <= 25
    assert not sample["in_overlap_duplicate"].any()
    assert set(sample.columns) == set(everything.columns)


def test_gate_drawn_on_histogram_and_scatter(
    window: MainWindow, run_folder: Path, qtbot: QtBot
) -> None:
    window.show_results(run_folder)
    panel = window.classify_panel
    assert panel.isEnabled() and panel.object_type.currentText() == "nuclei"
    panel.feature_x.setCurrentText("area_um2")
    panel.feature_y.setCurrentText(NO_Y)
    areas = panel._values("area_um2")
    threshold = float(np.median(areas))
    panel.set_gate(threshold, float(areas.max()))
    assert "Inside the gate" in panel.gate_result.text()
    step = panel.gate_step()
    assert step["params"]["rules"] == [
        {"feature": "area_um2", "min": pytest.approx(threshold), "max": areas.max()}
    ]
    panel.feature_y.setCurrentText("perimeter_um")
    assert len(panel.gate_rules()) == 2
    panel.feature_y.setCurrentText(NO_Y)
    panel.set_gate(threshold, float(areas.max()))

    panel.add_steps.emit([panel.gate_step()])
    assert window.pipeline_panel.pipeline().steps[-1].module == "gate_objects"

    panel.gate_name.setText("large nuclei")
    panel._apply([panel.gate_step()])
    qtbot.waitUntil(lambda: "gate_large_nuclei_count" in _summary(run_folder), timeout=60000)
    window.analysis_task.wait()
    results = window.results_panel
    qtbot.waitUntil(
        lambda: results.results is not None and "gate_large_nuclei_count" in results.results.summary
    )
    assert "spatial_objects" in _summary(run_folder)  # the run's own table steps stay


def _summary(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")


def test_classifier_from_clicked_objects(
    window: MainWindow, run_folder: Path, qtbot: QtBot
) -> None:
    window.show_results(run_folder)
    panel = window.classify_panel
    window._classify_show_field(run_folder, "A01", 1)
    assert panel.shown == ("A01", 1)
    assert panel.mask_layer() == "nuclei · corrected"

    # Clicking an object in the viewer labels it with the current class.
    objects = pd.read_parquet(run_folder / "wells" / "A01" / "objects.parquet")
    nuclei = objects[objects.object_type == "nuclei"]
    first = nuclei[nuclei.field == 1].iloc[0]
    # The handler sits on the viewer, so it works whichever layer is selected or hidden.
    viewer, handler = window.viewer_panel._click_handlers[0]
    assert viewer is window.viewer_panel.viewer
    window.viewer_panel.viewer.layers["nuclei · corrected"].visible = False
    handler(viewer, SimpleNamespace(position=(first.centroid_y_um, first.centroid_x_um)))
    assert panel.examples == {("A01", 1, int(first.object_id)): "positive"}
    assert "examples" in window.viewer_panel.layer_names()
    panel.label_object(int(first.object_id))  # clicking again with the same class unlabels
    assert panel.examples == {}

    # Label large nuclei "positive", small "negative" across the well's fields.
    median = nuclei.area_um2.median()
    for record in nuclei.to_dict("records"):
        panel.field_shown("A01", int(record["field"]))
        panel.classes.setCurrentRow(0 if record["area_um2"] > median else 1)
        panel.label_object(int(record["object_id"]))
    assert panel.table.rowCount() == len(nuclei)
    panel.train()
    qtbot.waitUntil(lambda: panel.classifier is not None, timeout=60000)
    assert "cross-validated accuracy" in panel.accuracy.text()
    assert panel.confusion.rowCount() == 2
    cells = [panel.confusion.item(i, j) for i in range(2) for j in range(2)]
    total = sum(int(cell.text()) for cell in cells if cell is not None)
    assert total == len(nuclei)

    panel.suggest()
    assert panel.suggestions.count() == 0  # every A01 object is labelled already
    panel.examples = dict(list(panel.examples.items())[: len(nuclei) // 2])
    panel.suggest()
    assert panel.suggestions.count() > 0

    step = panel.classifier_step()
    assert step is not None and Path(step["params"]["model_path"]).exists()
    panel._apply([step])
    qtbot.waitUntil(lambda: "class_class_positive_fraction" in _summary(run_folder), timeout=60000)
    window.analysis_task.wait()

    # Colour the shown field by class.
    qtbot.waitUntil(lambda: panel.colour_by.findText("class_class") >= 0)
    window._classify_show_field(run_folder, "A01", 1)
    panel.colour_by.setCurrentText("class_class")
    panel.paint_field()
    assert "class_class" in window.viewer_panel.layer_names()
    assert "positive" in panel.legend.text() or "negative" in panel.legend.text()


def test_whole_well_mosaic(window: MainWindow, run_folder: Path) -> None:
    window.show_results(run_folder)
    window.results_panel.select_well("A01")
    assert window.results_panel.mosaic.isEnabled()
    window.results_panel._open_mosaic()
    assert window.viewer_panel.layer_names() == ["A01 mosaic"]
    layer = window.viewer_panel.viewer.layers["A01 mosaic"]
    assert layer.rgb and layer.scale[0] == pytest.approx(0.29901, rel=0.01)
