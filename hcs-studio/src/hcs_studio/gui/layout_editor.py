"""Layout tab: annotate a plate (cell lines, compounds, doses, roles), with fill helpers,
templates, import/export, validation, undo/redo and copy/paste."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.analysis.profiling import PALETTE
from hcs_studio.gui.annotation_filter import AnnotationFilterBar, unit_combo
from hcs_studio.gui.plate_map import PlateMapView
from hcs_studio.io import layout_io
from hcs_studio.io.layout_store import (
    LayoutStore,
    list_templates,
    load_template,
    save_template,
)
from hcs_studio.model import units
from hcs_studio.model.annotation import (
    PLATE_TYPES,
    ROLES,
    PlateLayout,
    Treatment,
    WellAnnotation,
)
from hcs_studio.model.layout_tools import (
    DILUTIONS,
    Clipboard,
    Issue,
    apply_fields,
    clear,
    combination_matrix,
    copy_region,
    dilution_series,
    exclude_edges,
    paste,
    set_role,
    validate,
)
from hcs_studio.model.plate import well_name

VIRIDIS = (
    "#440154",
    "#482878",
    "#3e4a89",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
)
MAX_TREATMENTS = 3
Key = tuple[int, int]


def _viridis(t: float) -> QColor:
    t = min(1.0, max(0.0, t))
    x = t * (len(VIRIDIS) - 1)
    i = min(len(VIRIDIS) - 2, int(x))
    a, b = QColor(VIRIDIS[i]), QColor(VIRIDIS[i + 1])
    f = x - i
    return QColor(
        round(a.red() + (b.red() - a.red()) * f),
        round(a.green() + (b.green() - a.green()) * f),
        round(a.blue() + (b.blue() - a.blue()) * f),
    )


def _spin(value: float = 0.0, decimals: int = 4, maximum: float = 1e9) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


# -------------------------------------------------------------------------- dialogs


class DilutionDialog(QDialog):
    def __init__(self, parent: QWidget | None, start: str, cell_lines: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dilution series")
        self.compound = QLineEdit()
        self.top = _spin(10.0)
        self.unit = unit_combo()
        self.factor = QComboBox()
        self.factor.addItems([*DILUTIONS, "custom"])
        self.factor.setCurrentText("1:3")
        self.custom_factor = _spin(3.0, 3, 1000)
        self.points = QSpinBox()
        self.points.setRange(1, 48)
        self.points.setValue(10)
        self.direction = QComboBox()
        self.direction.addItems(["left → right", "top → bottom"])
        self.reverse = QCheckBox("Highest concentration last")
        self.replicates = QSpinBox()
        self.replicates.setRange(1, 32)
        self.cell_line = QComboBox()
        self.cell_line.setEditable(True)
        self.cell_line.addItems(["(keep)", *cell_lines])
        self.vehicle = QLineEdit()
        form = QFormLayout(self)
        form.addRow("Starts at", QLabel(start))
        form.addRow("Compound", self.compound)
        top = QHBoxLayout()
        top.addWidget(self.top)
        top.addWidget(self.unit)
        form.addRow("Top concentration", top)
        factor = QHBoxLayout()
        factor.addWidget(self.factor)
        factor.addWidget(self.custom_factor)
        form.addRow("Dilution", factor)
        form.addRow("Points", self.points)
        form.addRow("Direction", self.direction)
        form.addRow(self.reverse)
        form.addRow("Replicates (next rows / columns)", self.replicates)
        form.addRow("Cell line", self.cell_line)
        form.addRow("Vehicle", self.vehicle)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict[str, Any]:
        name = self.factor.currentText()
        line = self.cell_line.currentText()
        return {
            "compound": self.compound.text().strip(),
            "top": self.top.value(),
            "unit": self.unit.currentText(),
            "factor": self.custom_factor.value() if name == "custom" else DILUTIONS[name],
            "points": self.points.value(),
            "direction": "right" if self.direction.currentIndex() == 0 else "down",
            "reverse": self.reverse.isChecked(),
            "replicates": self.replicates.value(),
            "cell_line": None if line == "(keep)" else line,
            "vehicle": self.vehicle.text().strip(),
        }


class CombinationDialog(QDialog):
    def __init__(self, parent: QWidget | None, origin: str, cell_lines: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Combination matrix")
        self.drug_a, self.drug_b = QLineEdit(), QLineEdit()
        self.top_a, self.top_b = _spin(10.0), _spin(10.0)
        self.unit_a, self.unit_b = unit_combo(), unit_combo()
        self.factor_a, self.factor_b = QComboBox(), QComboBox()
        for combo in (self.factor_a, self.factor_b):
            combo.addItems(list(DILUTIONS))
            combo.setCurrentText("1:3")
        self.size_a, self.size_b = QSpinBox(), QSpinBox()
        for spin in (self.size_a, self.size_b):
            spin.setRange(2, 24)
            spin.setValue(6)
        self.include_zero = QCheckBox("Include 0 for each drug (single agents on the edges)")
        self.include_zero.setChecked(True)
        self.replicates = QSpinBox()
        self.replicates.setRange(1, 16)
        self.replicate_direction = QComboBox()
        self.replicate_direction.addItems(["to the right", "below"])
        self.gap = QSpinBox()
        self.gap.setRange(0, 10)
        self.cell_line = QComboBox()
        self.cell_line.setEditable(True)
        self.cell_line.addItems(["(keep)", *cell_lines])
        self.vehicle = QLineEdit()
        form = QFormLayout(self)
        form.addRow("Top-left well", QLabel(origin))
        for label, drug, top, unit, factor, size in (
            ("Drug A (rows)", self.drug_a, self.top_a, self.unit_a, self.factor_a, self.size_a),
            ("Drug B (columns)", self.drug_b, self.top_b, self.unit_b, self.factor_b, self.size_b),
        ):
            row = QHBoxLayout()
            for widget in (drug, top, unit, factor, QLabel("points"), size):
                row.addWidget(widget)
            form.addRow(label, row)
        form.addRow(self.include_zero)
        replicates = QHBoxLayout()
        for widget in (self.replicates, self.replicate_direction, QLabel("gap"), self.gap):
            replicates.addWidget(widget)
        form.addRow("Replicate matrices", replicates)
        form.addRow("Cell line", self.cell_line)
        form.addRow("Vehicle", self.vehicle)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict[str, Any]:
        line = self.cell_line.currentText()
        return {
            "drug_a": self.drug_a.text().strip(),
            "top_a": self.top_a.value(),
            "unit_a": self.unit_a.currentText(),
            "factor_a": DILUTIONS[self.factor_a.currentText()],
            "size_a": self.size_a.value(),
            "drug_b": self.drug_b.text().strip(),
            "top_b": self.top_b.value(),
            "unit_b": self.unit_b.currentText(),
            "factor_b": DILUTIONS[self.factor_b.currentText()],
            "size_b": self.size_b.value(),
            "include_zero": self.include_zero.isChecked(),
            "replicates": self.replicates.value(),
            "replicate_direction": "right"
            if self.replicate_direction.currentIndex() == 0
            else "down",
            "gap": self.gap.value(),
            "cell_line": None if line == "(keep)" else line,
            "vehicle": self.vehicle.text().strip(),
        }


class MappingDialog(QDialog):
    """Which field each column of an unfamiliar layout file holds."""

    FIELDS = (
        "well",
        "cell_line",
        "role",
        "passage",
        "seeding_density",
        "replicate_group",
        "time_point_h",
        "notes",
        *[
            f"{kind}_{n}"
            for n in range(1, MAX_TREATMENTS + 1)
            for kind in ("compound", "conc", "unit", "vehicle")
        ],
        "ignore",
    )

    def __init__(self, parent: QWidget | None, mapping: dict[str, str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Columns of the layout file")
        self.table = QTableWidget(len(mapping), 2)
        self.table.setHorizontalHeaderLabels(["Column in the file", "Holds"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.combos: dict[str, QComboBox] = {}
        for row, (column, target) in enumerate(mapping.items()):
            self.table.setItem(row, 0, QTableWidgetItem(column))
            combo = QComboBox()
            combo.addItems([*self.FIELDS, "custom field"])
            combo.setCurrentText("custom field" if target.startswith("custom:") else target)
            self.table.setCellWidget(row, 1, combo)
            self.combos[column] = combo
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Check what each column holds; unknown columns become custom fields.")
        )
        layout.addWidget(self.table)
        layout.addWidget(buttons)
        self.resize(520, 420)

    def mapping(self) -> dict[str, str]:
        return {
            column: f"custom:{column}"
            if combo.currentText() == "custom field"
            else combo.currentText()
            for column, combo in self.combos.items()
        }


# ---------------------------------------------------------------------- the editor


class LayoutEditor(QWidget):
    """``layout_saved(layout, path)``: a version was saved (to use for runs)."""

    layout_saved = Signal(object, object)
    layout_edited = Signal(object)

    def __init__(
        self, log: Callable[[str], None] | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.log = log or (lambda text: None)
        self.doc = PlateLayout.for_plate_type(384)
        self.history: list[PlateLayout] = [self.doc]
        self.position = 0
        self.clipboard: Clipboard | None = None
        self.store: LayoutStore | None = None
        self.saved_path: Path | None = None

        # plate side
        self.plate_type = QComboBox()
        self.plate_type.addItems([str(n) for n in (96, 384, 1536)])
        self.plate_type.setCurrentText("384")
        self.plate_type.activated.connect(
            lambda _: self.new_layout(int(self.plate_type.currentText()))
        )
        self.layer = QComboBox()
        self.layer.addItems([name.replace("_", " ") for name in layout_io.LAYERS])
        self.layer.currentIndexChanged.connect(lambda _: self.redraw())
        self.map = PlateMapView()
        self.map.selection_changed.connect(self._on_selection)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.filter_bar = AnnotationFilterBar(lambda: self.doc)
        self.filter_bar.selected.connect(self.map.set_selection)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Plate"))
        toolbar.addWidget(self.plate_type)
        toolbar.addWidget(QLabel("Colour by"))
        toolbar.addWidget(self.layer)
        for text, slot in (
            ("Import…", self._choose_import),
            ("Save version", self.save_version),
            ("Validate", self.run_validation),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        export = QToolButton()
        export.setText("Export")
        export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(export)
        for text, kind in (
            ("Long CSV…", "csv"),
            ("Long XLSX…", "xlsx"),
            ("Plate-map XLSX…", "platemap"),
            ("Figure (PNG)…", "png"),
            ("Figure (PDF)…", "pdf"),
            ("JSON…", "json"),
        ):
            menu.addAction(text, self._exporter(kind))
        export.setMenu(menu)
        toolbar.addWidget(export)
        templates = QToolButton()
        templates.setText("Templates")
        templates.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        template_menu = QMenu(templates)
        template_menu.addAction("Save as template…", self._save_template)
        template_menu.addAction("Apply template…", self._apply_template)
        templates.setMenu(template_menu)
        toolbar.addWidget(templates)
        for text, slot, keys in (
            ("Undo", self.undo, QKeySequence.StandardKey.Undo),
            ("Redo", self.redo, QKeySequence.StandardKey.Redo),
            ("Copy", self.copy, QKeySequence.StandardKey.Copy),
            ("Paste", self.paste, QKeySequence.StandardKey.Paste),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(slot)
        toolbar.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(toolbar)
        left_layout.addWidget(self.filter_bar)
        left_layout.addWidget(self.map, 1)
        left_layout.addWidget(self.legend)
        left_layout.addWidget(self.status)

        splitter = QSplitter()
        splitter.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_form())
        scroll.setMinimumWidth(470)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        outer = QVBoxLayout(self)
        outer.addWidget(splitter)
        self.redraw()

    # ------------------------------------------------------------------- form

    def _build_form(self) -> QWidget:
        self.selection_label = QLabel(
            "Select wells on the plate (click, Shift+click, drag, or a row/column header)."
        )
        self.selection_label.setWordWrap(True)
        self.set_boxes: dict[str, QCheckBox] = {}

        def guarded(name: str, widget: QWidget) -> QWidget:
            box = QCheckBox()
            box.setToolTip("Apply this field to the selected wells")
            self.set_boxes[name] = box
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(box)
            line.addWidget(widget, 1)
            return row

        self.cell_line = QComboBox()
        self.cell_line.setEditable(True)
        self.cell_line.editTextChanged.connect(
            lambda _: self.set_boxes["cell_line"].setChecked(True)
        )
        self.role = QComboBox()
        self.role.addItems(list(ROLES))
        self.role.activated.connect(lambda _: self.set_boxes["role"].setChecked(True))
        self.passage = QLineEdit()
        self.passage.textEdited.connect(lambda _: self.set_boxes["passage"].setChecked(True))
        self.density = _spin(0, 0, 1e9)
        self.density.valueChanged.connect(
            lambda _: self.set_boxes["seeding_density"].setChecked(True)
        )
        self.time_point = _spin(0, 2, 10000)
        self.time_point.valueChanged.connect(
            lambda _: self.set_boxes["time_point_h"].setChecked(True)
        )
        self.replicate = QLineEdit()
        self.replicate.setPlaceholderText("derived from cell line and treatments")
        self.replicate.textEdited.connect(
            lambda _: self.set_boxes["replicate_override"].setChecked(True)
        )
        self.notes = QLineEdit()
        self.notes.textEdited.connect(lambda _: self.set_boxes["notes"].setChecked(True))
        self.custom = QPlainTextEdit()
        self.custom.setPlaceholderText("one per line: field = value")
        self.custom.setMaximumHeight(70)
        self.custom.textChanged.connect(lambda: self.set_boxes["custom"].setChecked(True))
        self.treatments = QTableWidget(MAX_TREATMENTS, 4)
        self.treatments.setHorizontalHeaderLabels(["Compound", "Concentration", "Unit", "Vehicle"])
        header = self.treatments.horizontalHeader()
        for column, mode in enumerate(
            (
                QHeaderView.ResizeMode.Stretch,
                QHeaderView.ResizeMode.Interactive,
                QHeaderView.ResizeMode.ResizeToContents,
                QHeaderView.ResizeMode.Stretch,
            )
        ):
            header.setSectionResizeMode(column, mode)
        self.treatments.setColumnWidth(1, 90)
        self.treatments.verticalHeader().setVisible(False)
        self.treatments.setMinimumHeight(110)
        self.treatments.setMaximumHeight(130)
        self.unit_combos: list[QComboBox] = []
        for row in range(MAX_TREATMENTS):
            combo = unit_combo()
            combo.activated.connect(lambda _: self.set_boxes["treatments"].setChecked(True))
            self.treatments.setCellWidget(row, 2, combo)
            self.unit_combos.append(combo)
        self.treatments.itemChanged.connect(lambda _: self.set_boxes["treatments"].setChecked(True))

        form_widget = QGroupBox("Annotation of the selected wells")
        form = QFormLayout(form_widget)
        form.addRow(self.selection_label)
        form.addRow("Cell line", guarded("cell_line", self.cell_line))
        form.addRow("Role", guarded("role", self.role))
        form.addRow("Treatments", guarded("treatments", self.treatments))
        form.addRow("Passage", guarded("passage", self.passage))
        form.addRow("Cells per well", guarded("seeding_density", self.density))
        form.addRow("Time point (h)", guarded("time_point_h", self.time_point))
        form.addRow("Replicate group", guarded("replicate_override", self.replicate))
        form.addRow("Notes", guarded("notes", self.notes))
        form.addRow("Custom fields", guarded("custom", self.custom))
        apply = QPushButton("Apply ticked fields to selection")
        apply.clicked.connect(self.apply_form)
        erase = QPushButton("Clear selected wells")
        erase.clicked.connect(self.clear_selected)
        buttons = QHBoxLayout()
        buttons.addWidget(apply)
        buttons.addWidget(erase)
        form.addRow(buttons)

        helpers = QGroupBox("Fill helpers")
        helper_layout = QVBoxLayout(helpers)
        for text, slot in (
            ("Dilution series from the first selected well…", self._dilution_dialog),
            ("Combination matrix at the first selected well…", self._combination_dialog),
            ("Mark selection as vehicle controls…", lambda: self._controls("negative_control")),
            ("Mark selection as positive controls", lambda: self._controls("positive_control")),
            ("Mark selection as blank (no cells)", lambda: self._controls("blank")),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            helper_layout.addWidget(button)
        self.exclude_edges = QCheckBox("Exclude edge wells")
        self.exclude_edges.toggled.connect(self.toggle_edges)
        helper_layout.addWidget(self.exclude_edges)

        issues = QGroupBox("Validation")
        issues_layout = QVBoxLayout(issues)
        self.issues = QListWidget()
        self.issues.setMaximumHeight(140)
        self.issues.itemClicked.connect(self._select_issue)
        issues_layout.addWidget(self.issues)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(form_widget)
        layout.addWidget(helpers)
        layout.addWidget(issues)
        layout.addStretch(1)
        return panel

    # ---------------------------------------------------------------- history

    def commit(self, layout: PlateLayout, message: str = "") -> None:
        """Record an edit (undoable) and redraw."""
        if layout == self.doc:
            return
        self.history = [*self.history[: self.position + 1], layout]
        self.position = len(self.history) - 1
        self.doc = layout
        self.redraw()
        if message:
            self.status.setText(message)
        self.layout_edited.emit(layout)

    def undo(self) -> None:
        if self.position > 0:
            self.position -= 1
            self.doc = self.history[self.position]
            self.redraw()
            self.status.setText("Undone")

    def redo(self) -> None:
        if self.position < len(self.history) - 1:
            self.position += 1
            self.doc = self.history[self.position]
            self.redraw()
            self.status.setText("Redone")

    # ----------------------------------------------------------------- plate

    def set_plate(self, rows: int, cols: int, key: str | None = None) -> None:
        """Edit the layout of an opened plate: its latest saved version, or a new one."""
        self.saved_path = None
        self.store = LayoutStore(key) if key else None
        saved = self.store.load() if self.store else None
        if saved is not None and (saved.rows, saved.cols) == (rows, cols):
            self._reset(saved, f"Loaded layout version {saved.version}")
            self.saved_path = self.store.path(saved.version) if self.store else None
        else:
            self._reset(PlateLayout(rows=rows, cols=cols), "New layout")

    def new_layout(self, plate_type: int) -> None:
        rows, cols = PLATE_TYPES[plate_type]
        if (rows, cols) != (self.doc.rows, self.doc.cols):
            self.commit(PlateLayout(rows=rows, cols=cols), f"New {plate_type}-well layout")

    def set_layout(self, layout: PlateLayout, message: str = "") -> None:
        self.commit(layout, message)

    def _reset(self, layout: PlateLayout, message: str) -> None:
        self.doc = layout
        self.history, self.position = [layout], 0
        self.redraw()
        self.status.setText(message)

    def redraw(self) -> None:
        layout = self.doc
        plate_type = layout.plate_type
        if plate_type is not None:
            self.plate_type.setCurrentText(str(plate_type))
        keep = self.map.selected_wells()
        tooltips = {k: f"{well_name(*k)}\n{layout.get(k).label()}" for k in layout.all_wells()}
        self.map.set_grid(
            layout.rows, layout.cols, set(layout.all_wells()), tooltips, dict(layout.wells)
        )
        layer = layout_io.LAYERS[self.layer.currentIndex()]
        values = layout_io.layer_values(layout, layer)
        fills: dict[Key, QColor] = {k: QColor(240, 240, 240) for k in layout.all_wells()}
        legend = []
        if layer == "concentration":
            finite = [v for v in values.values() if v is not None]
            lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
            for key, value in values.items():
                if value is not None:
                    fills[key] = _viridis((value - lo) / ((hi - lo) or 1))
            if finite:
                legend.append(
                    "log scale: "
                    + " (dark) to ".join(
                        units.format_concentration(10**v, units.best_unit(10**v, "molar"))
                        for v in (lo, hi)
                    )
                    + " (yellow); single agents in molar units"
                )
        else:
            names = sorted({str(v) for v in values.values() if v is not None})
            for key, value in values.items():
                if value is not None:
                    fills[key] = QColor(PALETTE[names.index(str(value)) % len(PALETTE)])
            legend = [
                f'<span style="color:{PALETTE[i % len(PALETTE)]}">●</span> {n}'
                for i, n in enumerate(names[:30])
            ]
        self.map.set_fills(fills, tooltips)
        self.map.set_selection(keep)
        self.legend.setText("  ".join(legend))
        self.filter_bar.refresh()
        for combo, items in ((self.cell_line, layout.cell_lines()),):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(current)
            combo.blockSignals(False)
        counts = {
            "annotated": len(layout.wells),
            "combinations": sum(a.is_combination for a in layout.wells.values()),
            "controls": sum(a.role.endswith("control") for a in layout.wells.values()),
        }
        self.selection_label.setToolTip(", ".join(f"{v} {k}" for k, v in counts.items()))

    # ------------------------------------------------------------- selection

    def selection(self) -> list[Key]:
        return self.map.selected_wells()

    def _on_selection(self, wells: list[Key]) -> None:
        if not wells:
            self.selection_label.setText(
                "Select wells on the plate (click, Shift+click, drag, or a row/column header)."
            )
            return
        names = ", ".join(well_name(*k) for k in wells[:6]) + (" …" if len(wells) > 6 else "")
        self.selection_label.setText(f"{len(wells)} well(s): {names}")
        self.load_form(self.doc.get(wells[0]))

    def load_form(self, annotation: WellAnnotation) -> None:
        """Show an annotation in the form (nothing ticked until edited)."""
        widgets = [
            self.cell_line,
            self.role,
            self.passage,
            self.density,
            self.time_point,
            self.replicate,
            self.notes,
            self.custom,
            self.treatments,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.cell_line.setCurrentText(annotation.cell_line)
        self.role.setCurrentText(annotation.role)
        self.passage.setText(annotation.passage)
        self.density.setValue(annotation.seeding_density or 0)
        self.time_point.setValue(annotation.time_point_h or 0)
        self.replicate.setText(annotation.replicate_override)
        self.notes.setText(annotation.notes)
        self.custom.setPlainText(
            "\n".join(f"{k} = {v}" for k, v in annotation.custom if not k.startswith("_"))
        )
        for row in range(MAX_TREATMENTS):
            treatment = annotation.treatments[row] if row < len(annotation.treatments) else None
            self.treatments.setItem(
                row, 0, QTableWidgetItem(treatment.compound if treatment else "")
            )
            self.treatments.setItem(
                row, 1, QTableWidgetItem(units.format_value(treatment.value) if treatment else "")
            )
            self.unit_combos[row].setCurrentText(treatment.unit if treatment else "µM")
            self.treatments.setItem(
                row, 3, QTableWidgetItem(treatment.vehicle if treatment else "")
            )
        for widget in widgets:
            widget.blockSignals(False)
        for box in self.set_boxes.values():
            box.setChecked(False)

    def _cell(self, row: int, col: int) -> str:
        item = self.treatments.item(row, col)
        return item.text().strip() if item is not None else ""

    def form_treatments(self) -> tuple[Treatment, ...]:
        out = []
        for row in range(MAX_TREATMENTS):
            name = self._cell(row, 0)
            if not name:
                continue
            text = self._cell(row, 1)
            value = float(text.replace(",", ".")) if text else 0.0
            out.append(
                Treatment.of(name, value, self.unit_combos[row].currentText(), self._cell(row, 3))
            )
        return tuple(out)

    def apply_form(self) -> None:
        wells = self.selection()
        if not wells:
            self.status.setText("Select wells first.")
            return
        try:
            fields: dict[str, Any] = {}
            ticked = {name for name, box in self.set_boxes.items() if box.isChecked()}
            if "cell_line" in ticked:
                fields["cell_line"] = self.cell_line.currentText().strip()
            if "role" in ticked:
                fields["role"] = self.role.currentText()
            if "treatments" in ticked:
                fields["treatments"] = self.form_treatments()
            if "passage" in ticked:
                fields["passage"] = self.passage.text().strip()
            if "seeding_density" in ticked:
                fields["seeding_density"] = self.density.value() or None
            if "time_point_h" in ticked:
                fields["time_point_h"] = self.time_point.value() or None
            if "replicate_override" in ticked:
                fields["replicate_override"] = self.replicate.text().strip()
            if "notes" in ticked:
                fields["notes"] = self.notes.text().strip()
            if "custom" in ticked:
                pairs = [
                    line.split("=", 1)
                    for line in self.custom.toPlainText().splitlines()
                    if "=" in line
                ]
                fields["custom"] = tuple((k.strip(), v.strip()) for k, v in pairs if k.strip())
            if not fields:
                self.status.setText("Tick the fields to apply (they tick themselves when edited).")
                return
            layout = apply_fields(self.doc, wells, **fields)
        except ValueError as exc:
            self.status.setText(f"Not applied: {exc}")
            return
        self.commit(layout, f"Applied {', '.join(sorted(fields))} to {len(wells)} well(s)")

    def clear_selected(self) -> None:
        wells = self.selection()
        if wells:
            self.commit(clear(self.doc, wells), f"Cleared {len(wells)} well(s)")

    # --------------------------------------------------------------- helpers

    def apply_dilution(self, start: Key, **values: Any) -> None:
        if not values.get("compound"):
            self.status.setText("Give the compound's name.")
            return
        try:
            layout = dilution_series(self.doc, start, **values)
        except ValueError as exc:
            self.status.setText(f"Not applied: {exc}")
            return
        self.commit(layout, f"Dilution series of {values['compound']} from {well_name(*start)}")

    def apply_combination(self, origin: Key, **values: Any) -> None:
        if not values.get("drug_a") or not values.get("drug_b"):
            self.status.setText("Give both drugs' names.")
            return
        try:
            layout = combination_matrix(self.doc, origin, **values)
        except ValueError as exc:
            self.status.setText(f"Not applied: {exc}")
            return
        self.commit(
            layout,
            f"{values.get('size_a', 6)}×{values.get('size_b', 6)} matrix of "
            f"{values['drug_a']} × {values['drug_b']} at {well_name(*origin)}",
        )

    def _first(self) -> Key | None:
        wells = self.selection()
        if not wells:
            self.status.setText("Select the first well first.")
            return None
        return wells[0]

    def _dilution_dialog(self) -> None:
        start = self._first()
        if start is None:
            return
        dialog = DilutionDialog(self, well_name(*start), self.doc.cell_lines())
        if dialog.exec():
            self.apply_dilution(start, **dialog.values())

    def _combination_dialog(self) -> None:
        origin = self._first()
        if origin is None:
            return
        dialog = CombinationDialog(self, well_name(*origin), self.doc.cell_lines())
        if dialog.exec():
            self.apply_combination(origin, **dialog.values())

    def _controls(self, role: str) -> None:
        wells = self.selection()
        if not wells:
            self.status.setText("Select the control wells first.")
            return
        vehicle, percent = "", None
        if role == "negative_control":
            text, ok = QInputDialog.getText(
                self, "Vehicle", "Vehicle and concentration (e.g. DMSO 0.1 %), or empty:"
            )
            if not ok:
                return
            parts = text.replace("%", " ").split()
            if parts:
                vehicle = parts[0]
                percent = float(parts[1].replace(",", ".")) if len(parts) > 1 else None
        self.mark_controls(wells, role, vehicle, percent)

    def mark_controls(
        self,
        wells: list[Key],
        role: str,
        vehicle: str = "",
        percent: float | None = None,
        cell_line: str | None = None,
    ) -> None:
        """Set the role of ``wells`` (keeping their cell line unless ``cell_line`` is given)."""
        self.commit(
            set_role(
                self.doc, wells, role, vehicle=vehicle, vehicle_percent=percent, cell_line=cell_line
            ),
            f"{len(wells)} well(s) marked {role.replace('_', ' ')}",
        )

    def toggle_edges(self, on: bool) -> None:
        self.commit(
            exclude_edges(self.doc, on), "Edge wells excluded" if on else "Edge wells restored"
        )

    def copy(self) -> None:
        wells = self.selection()
        if wells:
            self.clipboard = copy_region(self.doc, wells)
            self.status.setText(f"Copied {len(wells)} well(s)")

    def paste(self) -> None:
        if self.clipboard is None:
            return
        at = self._first()
        if at is None:
            return
        try:
            self.commit(paste(self.doc, self.clipboard, at), f"Pasted at {well_name(*at)}")
        except ValueError as exc:
            self.status.setText(f"Not pasted: {exc}")

    # ----------------------------------------------------------- validation

    def run_validation(self) -> list[Issue]:
        issues = validate(self.doc)
        self.issues.clear()
        for issue in issues:
            item = QListWidgetItem(f"{'⚠' if issue.level == 'warning' else '✖'} {issue.text()}")
            item.setData(Qt.ItemDataRole.UserRole, list(issue.wells))
            self.issues.addItem(item)
        if not issues:
            self.issues.addItem("No problems found.")
        self.status.setText(
            f"{sum(i.level == 'error' for i in issues)} error(s), "
            f"{sum(i.level == 'warning' for i in issues)} warning(s)"
        )
        return issues

    def _select_issue(self, item: QListWidgetItem) -> None:
        wells = item.data(Qt.ItemDataRole.UserRole)
        if wells:
            self.map.set_selection([tuple(w) for w in wells])

    # ------------------------------------------------------- files, versions

    def save_version(self) -> Path | None:
        if self.store is None:
            self.status.setText(
                "Open a plate to save versions of its layout (or export the layout)."
            )
            return None
        saved = self.store.save(self.doc)
        self.doc = saved
        self.history[self.position] = saved
        self.saved_path = self.store.path(saved.version)
        errors = [i for i in validate(saved) if i.level == "error"]
        self.status.setText(
            f"Saved version {saved.version}"
            + (f" — {len(errors)} validation error(s); see Validate" if errors else "")
        )
        self.layout_saved.emit(saved, self.saved_path)
        return self.saved_path

    def import_file(self, path: Path, mapping: dict[str, str] | None = None) -> bool:
        try:
            imported = layout_io.load_layout(Path(path), mapping)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Not imported: {exc}")
            return False
        layout = imported.layout
        if (layout.rows, layout.cols) != (self.doc.rows, self.doc.cols) and self.doc.wells:
            self.status.setText(
                f"The file is a {layout.rows}×{layout.cols} layout; this plate is "
                f"{self.doc.rows}×{self.doc.cols}"
            )
            return False
        self.commit(
            layout,
            f"Imported {Path(path).name}: {len(layout.wells)} wells. " + " ".join(imported.notes),
        )
        return True

    def _choose_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a layout", "", "Layouts (*.csv *.xlsx *.xlsm *.json)"
        )
        if not path:
            return
        mapping = None
        if Path(path).suffix.lower() != ".json" and not layout_io.is_platemap_workbook(Path(path)):
            _, columns, _ = layout_io.read_records(Path(path))
            suggested = layout_io.suggest_mapping(columns)
            if "well" not in suggested.values() or any(
                t.startswith("custom:") for t in suggested.values()
            ):
                dialog = MappingDialog(self, suggested)
                if not dialog.exec():
                    return
                mapping = dialog.mapping()
        self.import_file(Path(path), mapping)

    def export(self, kind: str, path: Path) -> None:
        if kind == "platemap":
            layout_io.write_platemap_xlsx(self.doc, path)
        elif kind in ("png", "pdf"):
            layout_io.write_layout_figure(
                self.doc, path, layout_io.LAYERS[self.layer.currentIndex()]
            )
        else:
            layout_io.save_layout(self.doc, path)
        self.status.setText(f"Exported {Path(path).name}")

    def _exporter(self, kind: str) -> Callable[[], None]:
        return lambda: self._choose_export(kind)

    def _choose_export(self, kind: str) -> None:
        suffix = {"platemap": "xlsx"}.get(kind, kind)
        path, _ = QFileDialog.getSaveFileName(self, "Export the layout", f"plate_layout.{suffix}")
        if path:
            self.export(kind, Path(path))

    def _save_template(self) -> None:
        name, ok = QInputDialog.getText(self, "Save as template", "Template name:")
        if ok and name.strip():
            save_template(self.doc, name.strip())
            self.status.setText(f"Template {name.strip()!r} saved")

    def _apply_template(self) -> None:
        names = list_templates()
        if not names:
            self.status.setText("No templates saved yet.")
            return
        name, ok = QInputDialog.getItem(self, "Apply template", "Template:", names, 0, False)
        if not ok:
            return
        template = load_template(name)
        renames: dict[str, dict[str, str]] = {"cell_lines": {}, "compounds": {}}
        for kind, items in (
            ("cell_lines", template.cell_lines()),
            ("compounds", template.compounds()),
        ):
            for item in items:
                new, ok = QInputDialog.getText(
                    self,
                    "Apply template",
                    f"{kind.replace('_', ' ')[:-1]} {item!r} becomes:",
                    text=item,
                )
                if ok and new.strip() and new.strip() != item:
                    renames[kind][item] = new.strip()
        self.apply_template(template, **renames)

    def apply_template(self, template: PlateLayout, **renames: dict[str, str]) -> None:
        from hcs_studio.model.layout_tools import apply_template

        layout = apply_template(template, **renames)
        if (layout.rows, layout.cols) != (self.doc.rows, self.doc.cols):
            self.status.setText("The template is for a different plate size")
            return
        layout = replace(layout, barcode=self.doc.barcode, plate_id=self.doc.plate_id)
        self.commit(layout, "Template applied")
