"""Phase 3 GUI: PSF tool (theoretical and bead-measured), save → pipeline, decon preview."""

from __future__ import annotations

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
from scipy import signal  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.gui.psf_dialog import PsfDialog, projection_images  # noqa: E402
from hcs_studio.model.plate import ChannelInfo, ImageMetadata  # noqa: E402
from hcs_studio.processing import psf as psf_tools  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402

WIDEFIELD = ImageMetadata(
    channels=(ChannelInfo("GFP", excitation_nm=488, emission_nm=520),),
    pixel_size_um=0.1,
    z_step_um=0.3,
    objective_na=1.4,
    immersion_ri=1.515,
    modality="widefield",
    bit_depth=16,
)


def _bead_file(path: Path) -> Path:
    """OME-TIFF bead stack: 5 beads blurred by a Gaussian PSF, 0.1 µm × 0.3 µm voxels."""
    rng = np.random.default_rng(0)
    truth = np.zeros((21, 96, 96))
    for z, y, x in [(10, 20, 20), (10, 20, 70), (10, 70, 20), (10, 70, 70), (10, 46, 46)]:
        truth[z, y, x] = 50000
    grids = np.meshgrid(*[np.arange(n) - n // 2 for n in (13, 15, 15)], indexing="ij")
    kernel = np.exp(-0.5 * ((grids[0] / 1.8) ** 2 + (grids[1] / 2.0) ** 2 + (grids[2] / 2.0) ** 2))
    stack = rng.poisson(signal.fftconvolve(truth, kernel / kernel.sum(), mode="same") + 100)
    tifffile.imwrite(
        path,
        stack.astype(np.uint16),
        metadata={"axes": "ZYX", "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.3},
    )
    return path


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def test_projection_images_have_physical_aspect() -> None:
    psf = np.zeros((11, 31, 31), dtype=np.float32)
    psf[5, 15, 15] = 1
    xy, xz = projection_images(psf, 0.1, 0.3)
    assert xy.shape == (31, 31) and xy.dtype == np.uint8 and int(np.max(xy)) == 255
    assert xz is not None and xz.shape == (33, 31)  # 11 planes × 0.3/0.1
    assert projection_images(psf[5:6], 0.1, None)[1] is None


def test_theoretical_psf_in_the_dialog(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = PsfDialog(WIDEFIELD)
    qtbot.addWidget(dialog)
    assert not dialog.save_button.isEnabled()
    dialog.model.setCurrentText("scalar")
    with qtbot.waitSignal(dialog.task.finished, timeout=60000):
        dialog.compute()
    text = dialog.info.text()
    assert (
        "Theoretical widefield PSF (scalar), NA 1.4" in text
        and "FWHM: X 201 nm, Y 201 nm, Z 4" in text
    )
    assert dialog.xy_view.pixmap() is not None and not dialog.xy_view.pixmap().isNull()
    assert dialog.xz_view.pixmap() is not None and not dialog.xz_view.pixmap().isNull()
    assert dialog.psf_result is not None and dialog.psf_result.psf.ndim == 3

    with qtbot.waitSignal(dialog.psf_saved) as blocker:
        dialog.save(tmp_path / "theory.ome.tif")
    assert blocker.args == [tmp_path / "theory.ome.tif"]
    psf, pixel, z_step = psf_tools.load_psf(tmp_path / "theory.ome.tif")
    assert (pixel, z_step) == (0.1, 0.3) and psf.shape == dialog.psf_result.psf.shape


def test_dialog_explains_missing_metadata(qtbot: QtBot) -> None:
    no_plate = PsfDialog(None)
    qtbot.addWidget(no_plate)
    assert not no_plate.compute_button.isEnabled() and "Open a plate" in no_plate.info.text()

    unknown = ImageMetadata(**{**WIDEFIELD.__dict__, "modality": "unknown"})
    dialog = PsfDialog(unknown)
    qtbot.addWidget(dialog)
    dialog.compute()
    assert "set it to confocal or widefield" in dialog.info.text()
    dialog.source.setCurrentIndex(1)  # measured, but no field and no file
    dialog.compute()
    assert "Open the bead field in the viewer first" in dialog.info.text()


def test_measured_psf_from_the_viewer_field_feeds_deconvolution(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    beads = _bead_file(tmp_path / "beads.ome.tif")
    with qtbot.waitSignal(window.loader.loaded, timeout=10000):
        window.open_path(beads)
    window.open_field(0, 0, 0)

    dialog = window.open_psf_tool()
    qtbot.addWidget(dialog)
    dialog.source.setCurrentIndex(1)
    dialog.size_xy.setValue(15)
    dialog.size_z.setValue(13)
    with qtbot.waitSignal(dialog.task.finished, timeout=60000):
        dialog.compute()
    assert "Measured PSF: average of 5 beads" in dialog.info.text()
    assert dialog.psf_result is not None and dialog.psf_result.psf.shape == (13, 15, 15)
    fwhm = psf_tools.measure_fwhm(dialog.psf_result.psf, (0.3, 0.1, 0.1))
    assert fwhm["x"] == pytest.approx(2.3548 * 2.0 * 0.1, rel=0.1)

    dialog.save(tmp_path / "beads_psf.ome.tif")
    step = window.pipeline_panel.pipeline().steps[0]
    assert step.module == "deconvolution"
    assert step.params["psf_file"] == str(tmp_path / "beads_psf.ome.tif")
    assert "Deconvolution step now uses it" in window.log_view.toPlainText()

    # The preview reports iterations, residual and runtime, and sharpens the beads.
    window.pipeline_panel.set_module_param("deconvolution", "iterations", 15)
    with qtbot.waitSignal(window.preview.finished, timeout=60000):
        window.pipeline_panel.request_preview()
    status = window.pipeline_panel.status.text()
    assert "Deconvolution Channel 1: 3D, 15 iterations, residual" in status
    assert "1 tile(s) on cpu" in status
    layers = window.viewer_panel.viewer.layers
    before: Any = np.asarray(layers[0].data)
    after: Any = np.asarray(layers[1].data)
    assert layers[1].name.endswith("▸ corrected")
    assert after[10, 46, 46] > 3 * (before[10, 46, 46] - 100)
