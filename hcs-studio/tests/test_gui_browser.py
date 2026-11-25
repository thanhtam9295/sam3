"""Phase 1 GUI: open a plate from the browser, plate map, gallery, viewer, overrides."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

import tifffile  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.browser import BrowserPanel, load_favorites  # noqa: E402
from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.gui.pattern_dialog import PatternDialog  # noqa: E402
from hcs_studio.io.sidecar import load_overrides  # noqa: E402
from hcs_studio.utils.display import make_thumbnail  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import SHAPE, make_imagexpress, make_operetta, plane  # noqa: E402


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def _open(window: MainWindow, qtbot: QtBot, path: Path) -> None:
    with qtbot.waitSignal(window.loader.loaded, timeout=10000):
        window.open_path(path)


def test_open_operetta_plate_and_view_a_field(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    root = make_operetta(tmp_path / "Screen1", [(1, 2), (3, 10)])
    _open(window, qtbot, root)

    assert window.plate is not None and window.plate.vendor == "Operetta"
    assert "Screen1" in window.plate_panel.header.text()
    assert "384-well" in window.plate_panel.header.text()
    assert window.tabs.currentWidget() is window.plate_panel
    assert window.metadata_panel.edits["pixel_size_um"].text() == "0.29901"
    assert window.metadata_panel.modality.currentText() == "confocal"
    assert window.metadata_panel.channels.model().index(1, 0).data() == "Alexa 488"

    # Clicking an empty well does nothing; clicking a well with data lists its fields.
    window.plate_panel.plate_map.select_well(0, 0)
    assert window.plate_panel.gallery.count() == 0
    # ``ready`` is emitted once per field from worker threads: wait until all are delivered.
    # (Waiting for the first signal left the second one queued for a destroyed wait loop,
    # which crashed the interpreter in pytest-qt's event processing after the test.)
    window.plate_panel.plate_map.select_well(3, 10)
    thumbnails = window.plate_panel.thumbnails
    qtbot.waitUntil(lambda: thumbnails.idle(), timeout=10000)
    qtbot.wait(50)  # deliver the queued signals to the gallery
    assert window.plate_panel.gallery.count() == 2
    assert "D11" in window.plate_panel.gallery_label.text()

    window.plate_panel.gallery.itemClicked.emit(window.plate_panel.gallery.item(1))
    assert window.tabs.currentWidget() is window.viewer_panel.widget
    layers = window.viewer_panel.viewer.layers
    assert [layer.name for layer in layers] == ["HOECHST 33342", "Alexa 488"]
    assert tuple(layers[0].scale) == pytest.approx((2.0, 0.29901, 0.29901))  # z, y, x in µm
    assert [str(u) for u in layers[0].units] == ["micrometer"] * 3
    assert (layers[0].colormap.name, layers[1].colormap.name) == ("blue", "green")
    assert np.array_equal(np.asarray(layers[1].data[2]), plane(3, 10, 1, 1, 2))
    assert "D11 · field 2" in window.log_view.toPlainText()


def test_overrides_round_trip(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    root = make_imagexpress(tmp_path / "ix", [(0, 0)], n_sites=1)
    _open(window, qtbot, root)
    panel = window.metadata_panel
    assert window.plate is not None and window.plate.metadata.modality == "unknown"

    panel.edits["pixel_size_um"].setText("abc")
    panel.save()
    assert "not a number" in panel.status.text() and load_overrides(root) == {}

    panel.edits["pixel_size_um"].setText("0,325")  # decimal comma accepted
    panel.modality.setCurrentText("widefield")
    panel.channels.model().setData(panel.channels.model().index(0, 0), "Hoechst")
    panel.save()
    assert load_overrides(root) == {
        "pixel_size_um": 0.325,
        "modality": "widefield",
        "channel_names": ["Hoechst", "FITC"],
    }
    assert window.plate.metadata.pixel_size_um == 0.325
    assert "Overridden" in panel.status.text()
    assert not list(root.rglob("*.json")), "sidecar written into the raw folder"

    _open(window, qtbot, root)  # reopening re-applies the saved overrides
    assert window.plate.metadata.channels[0].name == "Hoechst"
    panel.reset()
    assert load_overrides(root) == {} and window.plate.metadata.pixel_size_um == 0.65


def test_uncalibrated_field_is_shown_in_pixels(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    for c in range(2):
        tifffile.imwrite(tmp_path / f"A01_f1_c{c + 1}.tif", plane(0, 0, 0, c))
    _open(window, qtbot, tmp_path)
    window.open_field(0, 0, 0)
    layer = window.viewer_panel.viewer.layers[0]
    assert tuple(layer.scale) == (1.0, 1.0)
    assert not window.viewer_panel.viewer.canvas.overlays.scale_bar.visible


def test_unrecognised_folder_asks_for_a_pattern(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for well in ("B02", "B03"):
        tifffile.imwrite(tmp_path / f"exp_{well}_dapi.tif", np.zeros(SHAPE, dtype=np.uint16))
    asked: list[Path] = []

    def fake_dialog(folder: Path) -> tuple[str, float | None]:
        asked.append(folder)
        return r"exp_(?P<well>[A-Z]\d\d)_(?P<channel>\w+)\.", 0.5

    monkeypatch.setattr(window, "_ask_pattern", fake_dialog)
    _open(window, qtbot, tmp_path)
    assert asked == [tmp_path]
    assert window.plate is not None and sorted(window.plate.wells) == [(1, 1), (1, 2)]
    assert window.plate.metadata.channels[0].name == "dapi"

    empty = tmp_path / "empty"
    empty.mkdir()
    window.open_path(empty)
    assert "no recognised plate data" in window.log_view.toPlainText()


def test_pattern_dialog_validates(qtbot: QtBot, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    tifffile.imwrite(tmp_path / "exp_B02_dapi.tif", np.zeros(SHAPE, dtype=np.uint16))
    dialog = PatternDialog(tmp_path)
    qtbot.addWidget(dialog)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled() and "no match" in dialog.preview.toPlainText()
    dialog.pattern.setText(r"exp_(?P<well>[A-Z]\d\d)_(?P<channel>\w+)\.")
    assert ok.isEnabled() and "well=B02, channel=dapi" in dialog.preview.toPlainText()
    dialog.pixel_size.setText("-1")
    assert not ok.isEnabled()
    dialog.pattern.setText("(unclosed")
    assert "Invalid pattern" in dialog.preview.toPlainText()


def test_layout_import_and_failed_open(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    root = make_operetta(tmp_path / "P", [(0, 0)], n_fields=1, n_planes=1)
    _open(window, qtbot, root)
    layout = tmp_path / "layout.csv"
    layout.write_text(
        "well,compound,control_type\nA01,DMSO,negative\nH12,Drug,\n", encoding="utf-8"
    )
    window.import_layout(layout)
    assert "DMSO" in window.plate_panel.plate_map._items[(0, 0)].toolTip()
    assert "no images on this plate: H12" in window.log_view.toPlainText()

    (root / "Images" / "Index.idx.xml").write_text("<broken", encoding="utf-8")
    with qtbot.waitSignal(window.loader.failed, timeout=10000):
        window.open_path(root)
    assert "Could not open" in window.log_view.toPlainText()
    assert window.plate is not None, "a failed open must keep the previous plate"


def test_browser_badges_and_favorites(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    root = make_operetta(tmp_path / "data" / "Screen1", [(0, 0)], n_fields=1, n_planes=1)
    (tmp_path / "data" / "other").mkdir()
    browser = BrowserPanel()
    qtbot.addWidget(browser)
    assert browser.model.vendor(str(root)) == "Operetta"
    assert browser.model.vendor(str(tmp_path / "data" / "other")) is None

    browser.reveal(root)
    assert browser.selected_path() == root
    browser.add_selected_favorite()
    browser.add_selected_favorite()  # no duplicate
    assert load_favorites() == [root] and browser.favorites.count() == 1
    with qtbot.waitSignal(browser.plate_requested) as blocker:
        browser.open_selected()
    assert blocker.args == [root]
    browser.favorites.setCurrentRow(0)
    browser.remove_selected_favorite()
    assert load_favorites() == []


def test_thumbnail_is_a_small_colour_composite(tmp_path: Path) -> None:
    import dask.array as da

    data = np.zeros((1, 2, 3, 400, 600), dtype=np.uint16)
    data[0, 0, 1, :200] = 1000  # channel 0, middle plane, top half
    data[0, 1, 1, :, :300] = 500  # channel 1, left half
    thumb = make_thumbnail(da.from_array(data), ["blue", "green"], max_size=100)
    assert thumb.shape == (67, 100, 3) and thumb.dtype == np.uint8
    assert thumb[5, 95].tolist() == [38, 89, 255]  # blue only
    assert thumb[60, 5].tolist() == [0, 255, 0]  # green only
    assert thumb[60, 95].tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        make_thumbnail(da.from_array(data), ["blue"])
