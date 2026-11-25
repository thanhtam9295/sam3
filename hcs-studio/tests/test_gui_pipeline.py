"""Phase 2 GUI: pipeline panel, parameter form, live preview, before/after, histograms."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.gui.param_form import ParamForm  # noqa: E402
from hcs_studio.gui.pipeline_panel import DISPLAY_ONLY_NOTICE, PipelinePanel  # noqa: E402
from hcs_studio.pipeline.schema import Pipeline  # noqa: E402
from hcs_studio.processing.base import MODULES, load_builtin_modules  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_imagexpress  # noqa: E402

SHAPE = (64, 80)


def _image(row: int, col: int, site: int, wave: int) -> Any:
    rng = np.random.default_rng(row * 1000 + col * 100 + site * 10 + wave)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    spot = 2000 * np.exp(-((yy - 30) ** 2 + (xx - 40) ** 2) / 30.0)
    return (300 + 2.0 * xx + spot + rng.normal(0, 3, SHAPE)).clip(0).astype(np.uint16)


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    root = make_imagexpress(tmp_path / "ix", [(0, 0), (0, 1)], n_sites=2, image_fn=_image)
    with qtbot.waitSignal(win.loader.loaded, timeout=10000):
        win.open_path(root)
    yield win
    win.close()


def test_param_form_builds_editors_and_validates(qtbot: QtBot) -> None:
    load_builtin_modules()
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_model(MODULES.get("background").Params, {"radius_um": 12})
    assert isinstance(form.editor("method"), QComboBox)
    radius = form.editor("radius_um")
    assert isinstance(radius, QLineEdit) and radius.text() == "12"
    values = form.values()
    assert {k: values[k] for k in ("channels", "method", "radius_um", "operation")} == {
        "channels": None,
        "method": "tophat",
        "radius_um": 12.0,
        "operation": "subtract",
    }

    channels = form.editor("channels")
    assert isinstance(channels, QLineEdit)
    channels.setText("[1, 2]")
    assert form.values()["channels"] == [1, 2]
    radius.setText("-3")
    with pytest.raises(ValueError, match="radius_um: Input should be greater than 0"):
        form.values()
    radius.setText("[unclosed")
    with pytest.raises(ValueError, match="radius_um: cannot read"):
        form.values()

    form.set_model(MODULES.get("af_channel").Params)
    with pytest.raises(ValueError, match="af_channel: Field required"):
        form.values()
    form.set_model(None)
    assert form.values() == {}
    assert not form.findChildren(QCheckBox)


def test_panel_edits_saves_and_loads_a_pipeline(qtbot: QtBot, tmp_path: Path) -> None:
    panel = PipelinePanel()
    qtbot.addWidget(panel)
    assert "measurements always use the data from before" in DISPLAY_ONLY_NOTICE
    panel.add_step("background", {"radius_um": 8})
    panel.add_step("clahe")
    panel.add_step("denoise")
    assert [panel.steps.item(i).text() for i in range(3)] == [
        "1. Background removal",
        "2. CLAHE (local contrast)  ◐ display only",
        "3. Denoise",
    ]
    panel.move_selected(-1)  # denoise before CLAHE
    panel.steps.item(0).setCheckState(Qt.CheckState.Unchecked)
    panel.name_edit.setText("demo")
    pipeline = panel.pipeline()
    assert [(s.module, s.enabled) for s in pipeline.steps] == [
        ("background", False),
        ("denoise", True),
        ("clahe", True),
    ]
    assert pipeline.steps[0].params["radius_um"] == 8

    panel.steps.setCurrentRow(1)
    sigma = panel.form.editor("sigma_px")
    assert isinstance(sigma, QLineEdit)
    sigma.setText("2.5")
    sigma.editingFinished.emit()
    assert panel.pipeline().steps[1].params["sigma_px"] == 2.5
    sigma.setText("0")
    sigma.editingFinished.emit()
    assert "greater than 0" in panel.error.text()
    assert panel.pipeline().steps[1].params["sigma_px"] == 2.5, "invalid input must not be stored"
    assert [s.module for s in panel.pipeline(upto_selected=True).steps] == ["background", "denoise"]

    panel.save_file(tmp_path / "demo.yaml")
    other = PipelinePanel()
    qtbot.addWidget(other)
    other.load_file(tmp_path / "demo.yaml")
    assert other.pipeline() == Pipeline.load(tmp_path / "demo.yaml") == panel.pipeline()
    (tmp_path / "bad.yaml").write_text("steps: [{module: nope}]", encoding="utf-8")
    other.load_file(tmp_path / "bad.yaml")
    assert "Could not load bad.yaml" in other.error.text()
    assert len(other.pipeline().steps) == 3, "a failed load must keep the current pipeline"

    panel.steps.setCurrentRow(2)
    panel.remove_selected()
    assert [s.module for s in panel.pipeline().steps] == ["background", "denoise"]


def test_preview_shows_before_and_after(window: MainWindow, qtbot: QtBot) -> None:
    panel = window.pipeline_panel
    panel.add_step("background", {"radius_um": 6})
    panel.request_preview()
    assert "Open a field" in panel.status.text()  # nothing in the viewer yet

    window.open_field(0, 1, 1)
    with qtbot.waitSignal(window.preview.finished, timeout=20000):
        panel.request_preview()
    viewer = window.viewer_panel.viewer
    names = [layer.name for layer in viewer.layers]
    assert names == ["DAPI", "FITC", "DAPI ▸ corrected", "FITC ▸ corrected"]
    assert [layer.visible for layer in viewer.layers] == [False, False, True, True]
    before, after = np.asarray(viewer.layers[0].data), np.asarray(viewer.layers[2].data)
    assert np.median(before) > 300 and np.median(after) < 10  # gradient removed
    assert after.max() == pytest.approx(2000, rel=0.15)  # spot kept
    assert "linear" in panel.status.text() and "background" in panel.status.text()
    assert [panel.histogram_channel.itemText(i) for i in range(2)] == ["DAPI", "FITC"]
    assert all(len(plot.listDataItems()) == 1 for plot in panel._plots)

    panel.compare.setCurrentText("Side by side")
    assert all(layer.visible for layer in viewer.layers)
    assert viewer.layers[2].translate[-1] == pytest.approx(80 * 0.65 * 1.05)
    assert viewer.layers[0].translate[-1] == 0
    # line profile, drawn on the "after" copy of side by side: mapped back to the field
    window.start_line_profile()
    lines = viewer.layers["line profile"]
    lines.add_lines(np.array([[40.0, 84.0], [40.0, 163.0]]))
    assert panel._profile_line == ((40.0, 0.0), (40.0, 79.0))
    curves = panel.profile_curves()
    assert set(curves) == {"before", "after"}
    distance, values = curves["before"]
    assert distance[-1] == pytest.approx(79 * 0.65)
    assert np.median(values) > 300 and np.median(curves["after"][1]) < 10
    assert len(panel.profile_plot.listDataItems()) == 2
    viewer.layers.remove(lines)

    panel.compare.setCurrentText("Before")
    assert [layer.visible for layer in viewer.layers] == [True, True, False, False]

    # A display-only step is labelled as such, in the layer names and the status line.
    panel.add_step("clahe", {"tile_um": 10})
    with qtbot.waitSignal(window.preview.finished, timeout=20000):
        panel.request_preview()
    assert viewer.layers[2].name == "DAPI ▸ enhanced (display only)"
    assert "DISPLAY ONLY" in panel.status.text()
    assert "DISPLAY ONLY" in window.log_view.toPlainText()


def test_live_preview_and_failures(window: MainWindow, qtbot: QtBot) -> None:
    panel = window.pipeline_panel
    window.open_field(0, 0, 0)
    panel.add_step("denoise")
    panel.live.setChecked(True)
    sigma = panel.form.editor("sigma_px")
    assert isinstance(sigma, QLineEdit)
    with qtbot.waitSignal(window.preview.finished, timeout=20000):
        sigma.setText("3")
        sigma.editingFinished.emit()  # debounced automatic preview
    panel.live.setChecked(False)

    # Flat-field needs more fields than this 4-field plate has: reported, not raised.
    panel.add_step("flatfield")
    with qtbot.waitSignal(window.preview.failed, timeout=20000):
        panel.request_preview()
    assert "Preview failed: Step 2 (Flat-field correction)" in panel.status.text()
    assert "at least 8" in panel.status.text()

    # Fitted estimates are cached between previews of the same plate.
    panel.remove_selected()
    panel.add_step("flatfield", {"min_fields": 2})
    with qtbot.waitSignal(window.preview.finished, timeout=20000):
        panel.request_preview()
    assert len(window._fit_cache) == 1
    with qtbot.waitSignal(window.preview.finished, timeout=20000):
        panel.request_preview()
    assert len(window._fit_cache) == 1
    window.metadata_panel.edits["pixel_size_um"].setText("0.5")
    window.metadata_panel.save()
    assert window._fit_cache == {}, "changed metadata must invalidate cached estimates"


def test_panel_keeps_the_image_qc_rules_of_a_pipeline(qtbot: QtBot, tmp_path: Path) -> None:
    from hcs_studio.gui.pipeline_panel import PipelinePanel
    from hcs_studio.qc.rules import default_rules, save_rules

    panel = PipelinePanel()
    qtbot.addWidget(panel)
    assert panel.pipeline().qc is None
    panel.qc_check.setChecked(True)
    assert panel.pipeline().qc == default_rules() and panel.qc_label.text() == "default v1"
    custom = default_rules(content_channel=2).model_copy(update={"name": "lab", "version": 3})
    path = save_rules(custom, tmp_path / "rules.yaml")
    panel.set_qc_rules(custom)
    panel.save_file(tmp_path / "p.yaml")
    panel.qc_check.setChecked(False)
    assert panel.pipeline().qc is None
    panel.load_file(tmp_path / "p.yaml")
    assert panel.qc_check.isChecked() and panel.pipeline().qc == custom
    panel.qc_check.setChecked(False)
    panel.qc_check.setChecked(True)  # the last rules come back
    assert panel.pipeline().qc == custom and path.exists()
