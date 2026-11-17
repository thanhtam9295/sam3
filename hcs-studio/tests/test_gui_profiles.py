"""Phase 9 GUI: Profiles tab linked to the results plate and the viewer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.analysis.profiling import ProfileOptions, run_profiling  # noqa: E402
from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.io.base import open_plate  # noqa: E402
from hcs_studio.model.layout import read_plate_layout  # noqa: E402
from hcs_studio.pipeline.executor import run_pipeline  # noqa: E402
from hcs_studio.pipeline.schema import Pipeline  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_imagexpress  # noqa: E402
from tests.test_profiling import make_profile_run  # noqa: E402
from tests.test_results import LAYOUT, PIPELINE, WELLS, _image  # noqa: E402


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("gui_profiles")
    root = make_imagexpress(tmp / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)
    layout = tmp / "layout.csv"
    layout.write_text(LAYOUT)
    return run_pipeline(
        open_plate(root),
        Pipeline.from_text(PIPELINE),
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


def test_profiles_linked_to_plate_and_viewer(
    window: MainWindow, run_folder: Path, qtbot: QtBot, tmp_path: Path
) -> None:
    window.show_results(run_folder)
    panel = window.profiles_panel
    assert "No profiles yet" in panel.status.text()
    assert panel.object_type.findText("nuclei") >= 0
    panel.object_type.setCurrentText("nuclei")
    panel.neighbors.setValue(4)
    panel.compute()
    qtbot.waitUntil(lambda: panel.result is not None, timeout=120000)
    assert "Saved in" in panel.status.text()
    points = panel.points()
    assert len(points) > 10 and {"object_id", "field", "centroid_y_um"} <= set(points.columns)
    assert [panel.space.itemText(i) for i in range(panel.space.count())] == ["PCA", "UMAP"]
    assert panel.colour_by.findText("compound") >= 0
    assert panel.loadings.rowCount() > 0

    # Lasso around everything: all wells are marked on the results plate.
    panel.space.setCurrentText("PCA")
    x, y = panel._coordinates()  # type: ignore[misc]
    lo, hi = (x.min() - 1, y.min() - 1), (x.max() + 1, y.max() + 1)
    panel.view_box.finish_lasso([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
    assert panel.selection.all()
    marked = {f"{'AB'[r]}{c + 1:02d}" for r, c in window.results_panel.plate.selected_wells()}
    assert marked == set(points["well"])
    # A lasso around nothing clears the plate marks.
    panel.view_box.finish_lasso(
        [(hi[0] + 5, hi[1] + 5), (hi[0] + 6, hi[1] + 5), (hi[0] + 6, hi[1] + 6)]
    )
    assert not panel.selection.any() and window.results_panel.plate.selected_wells() == []

    # Click a point: the object's crop is shown; open it in the viewer.
    panel.select_indices([0])
    panel.show_point(0)
    pixmap = panel.crop.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    panel._open_selected()
    layer = next(n for n in window.viewer_panel.labels_names() if n.startswith("nuclei"))
    labels = window.viewer_panel.viewer.layers[layer]
    assert labels.show_selected_label and labels.selected_label == int(points.object_id.iloc[0])

    # Plate → plot: selecting a well enlarges its points.
    window.results_panel.select_well("A03")
    assert panel.highlight == {"A03"}

    # The plate shows the distance to control as a readout.
    window.show_results(run_folder)
    assert window.results_panel.readout.findText("profile_distance_to_control") >= 0

    for name in ("plot.svg", "plot.png"):
        panel.export_figure(tmp_path / name)
        assert (tmp_path / name).stat().st_size > 1000


def test_dose_response_tab(window: MainWindow, tmp_path: Path) -> None:
    run = make_profile_run(tmp_path, "P1", seed=3)
    result = run_profiling(
        [run],
        ProfileOptions(umap=False, figures=False, max_cells=1500, dose_readouts=["readout_shift"]),
    )
    panel = window.profiles_panel
    panel.runs = [run]
    panel.show_result(result)
    panel.compound.setCurrentText("Drug X")
    panel.readout.setCurrentText("readout_shift")
    fit = panel.draw_dose()
    assert fit is not None and fit["fit"] and fit["ec50"] == pytest.approx(1.0, rel=1e-3)
    assert "EC50 1" in panel.dose_label.text()
    panel.readout.setCurrentText("profile_distance_to_control")
    fit = panel.draw_dose()
    assert fit is not None and fit["fit"] and fit["ec50"] == pytest.approx(1.0, rel=0.5)
    assert np.isfinite(fit["r2"])
