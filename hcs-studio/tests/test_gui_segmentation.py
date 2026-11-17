"""Phase 4 GUI: labels after preview, editing, SAM 3 prompts, training data, settings."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

import tifffile  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.gui.sam_settings import Sam3SettingsDialog  # noqa: E402
from hcs_studio.gui.segmentation_panel import SAM_BOXES, SAM_POINTS, SPLIT_SEEDS  # noqa: E402
from hcs_studio.segmentation import cellpose_backend  # noqa: E402
from hcs_studio.segmentation.sam3_backend import Sam3Settings, close_clients  # noqa: E402
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
    close_clients()


def _preview_segmentation(window: MainWindow, qtbot: QtBot) -> None:
    panel = window.pipeline_panel
    panel.add_step("segment_classical", {"channel": 1})
    panel.add_step("expand_cells", {"distance_um": 2})
    with qtbot.waitSignal(window.preview.finished, timeout=30000):
        panel.request_preview()


def test_preview_shows_editable_label_layers(window: MainWindow, qtbot: QtBot) -> None:
    _preview_segmentation(window, qtbot)
    viewer = window.viewer_panel
    assert viewer.labels_names() == ["nuclei", "cells"]
    nuclei = viewer.get_labels("nuclei")
    assert nuclei.shape == (1, 128, 128) and len(np.unique(nuclei)) == 6
    assert "Objects: nuclei: 5, cells: 5" in window.log_view.toPlainText()
    seg = window.segmentation_panel
    assert [seg.target.itemText(i) for i in range(seg.target.count())] == ["nuclei", "cells"]

    window.pipeline_panel.compare.setCurrentText("Side by side")
    shift = viewer.viewer.layers["nuclei"].translate[-1]
    assert shift == pytest.approx(viewer.viewer.layers[2].translate[-1]) and shift > 0


def test_merge_delete_split(window: MainWindow, qtbot: QtBot) -> None:
    _preview_segmentation(window, qtbot)
    seg, viewer = window.segmentation_panel, window.viewer_panel
    seg.target.setCurrentText("nuclei")
    labels = viewer.get_labels("nuclei")
    a, b = int(labels[0, 30, 30]), int(labels[0, 30, 90])
    seg.ids.setText(f"{a}, {b}")
    seg.merge()
    merged = viewer.get_labels("nuclei")
    assert merged[0, 30, 30] == merged[0, 30, 90] == min(a, b)
    assert "Merged" in seg.status.text()

    seg.split_id.setValue(min(a, b))
    seg._prompt_layer("points", SPLIT_SEEDS)
    viewer.viewer.layers[SPLIT_SEEDS].data = np.array([[30.0, 30.0], [30.0, 90.0]])
    seg.split()
    split = viewer.get_labels("nuclei")
    assert split[0, 30, 30] != split[0, 30, 90] and len(np.unique(split)) == 6

    seg.ids.setText(str(int(split[0, 90, 90])))
    seg.delete()
    assert viewer.get_labels("nuclei")[0, 90, 90] == 0
    seg.ids.setText("x")
    seg.merge()
    assert "whole numbers" in seg.status.text()


def test_sam_concept_and_prompts_with_the_stand_in_model(window: MainWindow, qtbot: QtBot) -> None:
    seg, viewer = window.segmentation_panel, window.viewer_panel
    seg.fake_sam = True
    seg.new_name.setText("sam")
    seg.new_layer()
    assert seg.target.currentText() == "sam" and not viewer.get_labels("sam").any()
    with qtbot.waitSignal(seg.task.finished, timeout=30000):
        seg.sam_concept()
    assert len(np.unique(viewer.get_labels("sam"))) - 1 == 5
    assert "SAM 3 found 5 'nucleus' object(s)" in seg.status.text()

    seg.new_name.setText("clicked")
    seg.new_layer()
    seg.add_sam_prompt_layers()
    viewer.viewer.layers[SAM_POINTS].data = np.array([[30.0, 30.0]])
    viewer.viewer.layers[SAM_BOXES].add_rectangles(np.array([[80.0, 80.0], [100.0, 100.0]]))
    with qtbot.waitSignal(seg.task.finished, timeout=30000):
        seg.sam_prompts()
    clicked = viewer.get_labels("clicked")
    assert clicked[0, 30, 30] > 0 and clicked[0, 90, 90] > 0 and clicked[0, 30, 90] == 0
    assert len(np.unique(clicked)) - 1 == 2
    assert len(viewer.viewer.layers[SAM_POINTS].data) == 0, "prompts are cleared after use"


def test_sam_needs_settings(window: MainWindow) -> None:
    seg = window.segmentation_panel
    seg.new_layer()
    seg.sam_concept()
    assert "Set up SAM 3 first: Tools → SAM 3 settings" in seg.status.text()


def test_training_pair_and_fine_tune_hand_off(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preview_segmentation(window, qtbot)
    seg = window.segmentation_panel
    seg.target.setCurrentText("nuclei")
    seg.training_folder.setText(str(tmp_path / "train"))
    seg.training_channels.setText("1, 2")
    seg.save_training("A01_f1")
    image = tifffile.imread(tmp_path / "train" / "A01_f1_img.tif")
    masks = tifffile.imread(tmp_path / "train" / "A01_f1_masks.tif")
    assert image.shape == (2, 128, 128) and masks.shape == (128, 128) and masks.max() == 5

    def fake_fine_tune(folder: Path, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        assert folder == tmp_path / "train" and kwargs["epochs"] == 7
        return folder / "models" / "mine", {"train_loss": [0.3], "images_train": 1}

    monkeypatch.setattr(cellpose_backend, "fine_tune", fake_fine_tune)
    seg.epochs.setValue(7)
    with qtbot.waitSignal(seg.model_trained, timeout=10000):
        seg.fine_tune()
    assert "Add a Cellpose step" in window.log_view.toPlainText()  # none in the pipeline yet

    window.pipeline_panel.add_step("segment_cellpose", {"channel": 1})
    with qtbot.waitSignal(seg.model_trained, timeout=10000):
        seg.fine_tune()
    steps = window.pipeline_panel.pipeline().steps
    cellpose = [s for s in steps if s.module == "segment_cellpose"]
    assert cellpose and cellpose[0].params["model"] == str(tmp_path / "train" / "models" / "mine")


def test_sam_settings_dialog(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    dialog = Sam3SettingsDialog()
    qtbot.addWidget(dialog)
    dialog.python.setText(sys.executable)  # this environment has no sam3
    assert not dialog.check() and "cannot be imported" in dialog.status.text()
    dialog.python.setText(str(tmp_path / "missing"))
    assert not dialog.check() and "Could not run" in dialog.status.text()
    sam_python = Path(__file__).resolve().parent.parent / ".venv-sam3" / "bin" / "python"
    if sam_python.exists():
        dialog.python.setText(str(sam_python))
        assert dialog.check() and "SAM 3 found" in dialog.status.text()
    dialog.checkpoint.setText("/models/sam3.pt")
    dialog.device.setCurrentText("cuda")
    dialog.save()
    saved = Sam3Settings.load()
    assert saved.checkpoint == "/models/sam3.pt" and saved.device == "cuda"
