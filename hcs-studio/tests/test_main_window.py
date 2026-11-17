"""Main window: docks, GPU status in the status bar, napari showing the test image."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402


@pytest.fixture
def window(qtbot: QtBot) -> Iterator[MainWindow]:
    status = GpuStatus(backend="cuda", device_name="Test GPU", vram_gb=8.0, torch_version="2.0")
    win = MainWindow(gpu_status=status)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def test_window_has_docks_and_gpu_status(window: MainWindow) -> None:
    assert window.windowTitle().startswith("HCS Studio")
    for dock in (window.browser_dock, window.metadata_dock, window.log_dock):
        assert dock.isVisible()
    assert window.gpu_label.text() == "GPU: Test GPU, 8.0 GB (PyTorch 2.0)"
    assert window.gpu_label.isVisible()
    assert "Test GPU" in window.log_view.toPlainText()


def test_test_image_is_shown_in_napari_with_micrometre_scale(
    window: MainWindow, qtbot: QtBot
) -> None:
    window.show_test_image()
    panel = window.viewer_panel
    assert panel.layer_names() == ["Nuclei (synthetic)", "Cytoplasm (synthetic)"]
    layer = panel.viewer.layers[0]
    assert tuple(layer.scale) == pytest.approx((0.325, 0.325))
    assert [str(u) for u in layer.units] == ["micrometer", "micrometer"]
    assert window.tabs.currentWidget() is panel.widget
    qtbot.wait(200)
    canvas = panel.viewer.screenshot(canvas_only=True, flash=False)
    assert np.asarray(canvas)[..., :3].std() > 5, "canvas rendered blank"


def test_viewer_rejects_mismatched_channels(window: MainWindow) -> None:
    with pytest.raises(ValueError, match="channel names"):
        window.viewer_panel.show_channels(np.zeros((2, 8, 8)), ["only one"], 1.0)
    with pytest.raises(ValueError, match="z_step_um"):
        window.viewer_panel.show_channels(np.zeros((1, 3, 8, 8)), ["a"], 1.0)
