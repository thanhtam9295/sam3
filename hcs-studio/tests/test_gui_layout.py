"""Phase 11 GUI: the Layout tab (helpers, form, undo, copy/paste, validation, versions)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QTableWidgetItem  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.layout_editor import LayoutEditor  # noqa: E402
from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.io.layout_io import load_layout, save_layout  # noqa: E402
from hcs_studio.io.layout_store import LayoutStore  # noqa: E402
from hcs_studio.model.annotation import PlateLayout, Treatment  # noqa: E402
from hcs_studio.model.plate import parse_well_name, well_name  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_operetta  # noqa: E402


@pytest.fixture
def editor(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LayoutEditor:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    widget = LayoutEditor()
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    win = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def _wells(*names: str) -> list[tuple[int, int]]:
    return [parse_well_name(n) for n in names]


def _item(text: str) -> QTableWidgetItem:
    return QTableWidgetItem(text)


def _column(col: int, rows: int = 16) -> list[tuple[int, int]]:
    return [(r, col) for r in range(rows)]


def test_combo_and_drc_layouts_are_quick_to_build(editor: LayoutEditor) -> None:
    """Done when: a 6×6 combination plate and a 10-point DRC plate in well under 2 minutes."""
    started = time.perf_counter()
    actions = 0
    # 384-well 6×6 combination plate: three replicate matrices, controls in columns 1 and 24
    editor.map.set_selection(_column(0))
    editor.mark_controls(_column(0), "negative_control", "DMSO", 0.1, cell_line="MCF7")
    editor.mark_controls(_column(23), "positive_control", cell_line="MCF7")
    editor.apply_combination(
        (1, 1),
        drug_a="Drug A",
        top_a=10,
        unit_a="µM",
        drug_b="Drug B",
        top_b=1,
        unit_b="µM",
        replicates=3,
        replicate_direction="right",
        gap=1,
        cell_line="MCF7",
    )
    actions += 3
    combo = editor.doc
    assert sum(a.is_combination for a in combo.wells.values()) == 3 * 25
    matrix = {k: a for k, a in combo.wells.items() if 1 <= k[0] <= 6 and 1 <= k[1] <= 6}
    assert len(matrix) == 36
    assert {a.cell_line for a in matrix.values()} == {"MCF7"}
    assert combo.get((1, 1)).role == "negative_control"  # 0 × 0
    assert not [i for i in editor.run_validation() if i.level == "error"]

    # 10-point duplicate dilution series of 8 compounds on a fresh 384-well layout
    editor.new_layout(96)
    editor.new_layout(384)
    assert not editor.doc.wells
    editor.mark_controls(_column(0) + _column(1), "negative_control", "DMSO", 0.1, "HeLa")
    for n in range(8):
        editor.apply_dilution(
            (2 * n, 2),
            compound=f"C{n + 1}",
            top=30.0,
            unit="µM",
            factor=3.0,
            points=10,
            replicates=2,
            direction="right",
            cell_line="HeLa",
        )
        actions += 1
    drc = editor.doc
    assert set(drc.compounds()) == {f"C{n}" for n in range(1, 9)} | {"DMSO"}
    assert len({a.replicate_group for a in drc.wells.values() if a.role == "sample"}) == 80
    assert drc.get(parse_well_name("A03")).treatments[0].value == pytest.approx(30.0)
    assert drc.get(parse_well_name("A12")).treatments[0].value == pytest.approx(30 / 3**9)
    assert time.perf_counter() - started < 120
    assert actions <= 12, "a handful of helper actions, not well-by-well typing"


def test_form_undo_copy_paste_filter_and_validation(editor: LayoutEditor) -> None:
    wells = _wells("B02", "B03", "B04")
    editor.map.set_selection(wells)
    assert "3 well(s)" in editor.selection_label.text()
    editor.cell_line.setEditText("A549")
    assert editor.set_boxes["cell_line"].isChecked(), "editing a field ticks it"
    editor.treatments.setItem(0, 0, _item("Taxol"))
    editor.treatments.setItem(0, 1, _item("50"))
    editor.unit_combos[0].setCurrentText("nM")
    editor.set_boxes["treatments"].setChecked(True)
    editor.apply_form()
    a = editor.doc.get(parse_well_name("B03"))
    assert a.cell_line == "A549"
    assert a.treatments == (Treatment.of("Taxol", 50, "nM"),)
    assert a.treatments[0].concentration == pytest.approx(50e-9)

    # only ticked fields change: set the role of one well, keep its cell line
    editor.map.set_selection(_wells("B02"))
    editor.role.setCurrentText("positive_control")
    editor.set_boxes["role"].setChecked(True)
    editor.apply_form()
    assert editor.doc.get(parse_well_name("B02")).role == "positive_control"
    assert editor.doc.get(parse_well_name("B02")).cell_line == "A549"

    editor.undo()
    assert editor.doc.get(parse_well_name("B02")).role == "sample"
    editor.redo()
    assert editor.doc.get(parse_well_name("B02")).role == "positive_control"

    editor.map.set_selection(wells)
    editor.copy()
    editor.map.set_selection(_wells("D05"))
    editor.paste()
    assert editor.doc.get(parse_well_name("D06")) == editor.doc.get(parse_well_name("B03"))

    # header click selects a whole row
    editor.map.select_line("row", 1)
    assert {k[0] for k in editor.selection()} == {1}
    assert len(editor.selection()) == 24

    # filter bar: A549 wells with Taxol ≥ 10 nM
    bar = editor.filter_bar
    bar.cell_line.setCurrentText("A549")
    bar.compound.setCurrentText("Taxol")
    bar.minimum.setText("10")
    bar.unit.setCurrentText("nM")
    selected = bar.apply()
    assert sorted(well_name(*k) for k in selected) == ["B02", "B03", "B04", "D05", "D06", "D07"]
    assert sorted(editor.selection()) == sorted(selected)

    # validation: no vehicle controls → listed; clicking an issue selects its wells
    issues = editor.run_validation()
    codes = {i.code for i in issues}
    assert "no_vehicle_controls" in codes
    editor.map.set_selection(_wells("P24"))
    editor.cell_line.setEditText("")
    editor.treatments.setItem(0, 0, _item("Taxol"))
    editor.treatments.setItem(0, 1, _item("1"))
    editor.set_boxes["treatments"].setChecked(True)
    editor.set_boxes["cell_line"].setChecked(False)
    editor.apply_form()
    issues = editor.run_validation()
    missing = next(i for i in issues if i.code == "missing_cell_line")
    row = next(
        n for n in range(editor.issues.count()) if missing.message in editor.issues.item(n).text()
    )
    editor._select_issue(editor.issues.item(row))
    assert editor.selection() == list(missing.wells)

    # edge exclusion toggles and restores
    editor.exclude_edges.setChecked(True)
    assert editor.doc.get(parse_well_name("A01")).role == "excluded"
    assert editor.doc.get(parse_well_name("B02")).role == "positive_control"
    editor.exclude_edges.setChecked(False)
    assert editor.doc.get(parse_well_name("A01")).role != "excluded"


def test_layers_export_import_and_templates(editor: LayoutEditor, tmp_path: Path) -> None:
    editor.apply_dilution((0, 0), compound="X", top=10, unit="µM", points=6, cell_line="HeLa")
    for index in range(editor.layer.count()):
        editor.layer.setCurrentIndex(index)
    assert "µM" in editor.legend.text()  # concentration layer, last
    for kind, suffix in (
        ("csv", "csv"),
        ("xlsx", "xlsx"),
        ("platemap", "xlsx"),
        ("png", "png"),
        ("pdf", "pdf"),
        ("json", "json"),
    ):
        path = tmp_path / f"layout_{kind}.{suffix}"
        editor.export(kind, path)
        assert path.stat().st_size > 0
    for kind in ("csv", "xlsx", "platemap", "json"):
        suffix = "xlsx" if kind == "platemap" else kind
        assert load_layout(tmp_path / f"layout_{kind}.{suffix}").layout.wells == editor.doc.wells

    other = LayoutEditor()
    assert other.import_file(tmp_path / "layout_csv.csv")
    assert other.doc.wells == editor.doc.wells
    other.deleteLater()

    template = editor.doc
    editor.new_layout(96)
    editor.new_layout(384)
    editor.apply_template(template, cell_lines={"HeLa": "U2OS"}, compounds={"X": "Y"})
    assert editor.doc.cell_lines() == ["U2OS"]
    assert editor.doc.compounds() == ["Y"]


def test_layout_tab_in_the_main_window(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    root = make_operetta(tmp_path / "P", [(0, 0), (0, 1)], n_fields=1, n_planes=1)
    with qtbot.waitSignal(window.loader.loaded, timeout=10000):
        window.open_path(root)
    editor = window.layout_editor
    assert window.tabs.indexOf(editor) == 1
    assert editor.store is not None and editor.doc.rows == window.plate.rows  # type: ignore[union-attr]

    window.tabs.setCurrentWidget(editor)
    editor.mark_controls(_wells("A01"), "negative_control", "DMSO", 0.1, "HeLa")
    editor.apply_dilution((0, 1), compound="Drug", top=1, unit="µM", points=1, cell_line="HeLa")
    window.tabs.setCurrentWidget(window.plate_panel)
    assert "Drug" in window.plate_panel.plate_map._items[(0, 1)].toolTip()

    # a run uses a saved version: unsaved edits are saved first
    path = window.layout_for_run()
    assert path is not None and path.name == "v0001.json"
    assert load_layout(path).layout.wells == editor.doc.wells
    assert window.layout_for_run() == path, "no new version without edits"

    # reopening the plate restores the latest version
    with qtbot.waitSignal(window.loader.loaded, timeout=10000):
        window.open_path(root)
    assert window.layout_path == path
    assert window.layout_document is not None and window.layout_document.version == 1
    assert "Using layout version 1" in window.log_view.toPlainText()

    # importing an XLSX layout replaces the annotation (editable, undoable)
    imported = PlateLayout.for_plate_type(384).updated({(0, 0): window.layout_document.get((0, 1))})
    xlsx = tmp_path / "layout.xlsx"
    save_layout(imported, xlsx)
    window.import_layout(xlsx)
    assert window.layout_path == xlsx
    assert editor.doc.wells == imported.wells
    editor.undo()
    assert editor.doc.wells != imported.wells
    assert len(LayoutStore(editor.store.key).versions()) == 1
