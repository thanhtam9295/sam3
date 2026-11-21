"""Results tab: the clickable results plate of a finished run, with a well panel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.annotation_filter import AnnotationFilterBar
from hcs_studio.gui.plate_map import PlateMapView
from hcs_studio.model.plate import parse_well_name, well_name
from hcs_studio.output.results import RunResults, WellField

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
DIVERGING = (
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#f7f7f7",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
)
NORMALISATIONS = {
    "values": "none",
    "robust Z (plate)": "robust_z_plate",
    "robust Z (negative controls)": "robust_z_negative",
}
STAGES = ("Enhanced", "Corrected", "Raw")
_THUMB = 140


def colour_at(stops: tuple[str, ...], t: float) -> QColor:
    t = min(1.0, max(0.0, t))
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    a, b = QColor(stops[i]), QColor(stops[i + 1])
    f = x - i
    return QColor(
        round(a.red() + (b.red() - a.red()) * f),
        round(a.green() + (b.green() - a.green()) * f),
        round(a.blue() + (b.blue() - a.blue()) * f),
    )


def fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    if float(value).is_integer():
        return str(int(value))
    magnitude = abs(value)
    if magnitude and (magnitude < 0.01 or magnitude >= 1e6):
        return f"{value:.2e}"
    return (
        f"{value:.0f}"
        if magnitude >= 100
        else f"{value:.1f}"
        if magnitude >= 10
        else f"{value:.3f}"
    )


def compose(folder: Path, base: str | None, overlays: list[str]) -> QPixmap:
    """A preview image with outline overlays painted on top."""
    pixmap = QPixmap(str(folder / base)) if base else QPixmap()
    if pixmap.isNull():
        pixmap = QPixmap(_THUMB, _THUMB)
        pixmap.fill(Qt.GlobalColor.black)
        return pixmap
    painter = QPainter(pixmap)
    for overlay in overlays:
        layer = QPixmap(str(folder / overlay))
        if not layer.isNull():
            painter.drawPixmap(pixmap.rect(), layer)
    painter.end()
    return pixmap


class ResultsPanel(QWidget):
    """``open_field(folder, well, field index, stage)`` asks the main window to show a field."""

    open_field = Signal(object, str, int, str)
    compare_field = Signal(object, str, int)
    open_mosaic = Signal(object, str)
    well_selected = Signal(str)
    compare_wells = Signal(object, list)  # run folder, well names

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.results: RunResults | None = None
        self.values: dict[str, float] = {}
        self.raw_values: dict[str, float] = {}
        self.well: str | None = None
        self._fields: list[WellField] = []
        self._percent: tuple[str, str, float] | None = None

        self.folder_label = QLabel("No results open. Finish a run, or File → Open results…")
        self.folder_label.setWordWrap(True)
        open_button = QPushButton("Open run folder…")
        open_button.clicked.connect(self._choose)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter readouts")
        self.filter.textChanged.connect(self._fill_readouts)
        self.readout = QComboBox()
        self.readout.setMinimumWidth(240)
        self.readout.currentTextChanged.connect(self.redraw)
        self.normalisation = QComboBox()
        self.normalisation.addItems(list(NORMALISATIONS))
        self.normalisation.currentTextChanged.connect(self.redraw)

        self.object_type = QComboBox()
        self.object_type.currentTextChanged.connect(self._fill_features)
        self.feature = QComboBox()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e12, 1e12)
        self.threshold.setDecimals(3)
        positive = QPushButton("Show % positive")
        positive.clicked.connect(self.show_percent_positive)

        self.plate = PlateMapView()
        self.plate.setDragMode(PlateMapView.DragMode.NoDrag)
        self.annotation_filter = AnnotationFilterBar(
            lambda: self.results.layout_document if self.results is not None else None
        )
        self.annotation_filter.selected.connect(self.plate.set_selection)
        self.plate.well_clicked.connect(lambda r, c: self.select_well(well_name(r, c)))
        self.colorbar = QLabel()
        self.colorbar.setFixedHeight(16)
        self.cmin, self.cmax = QLabel(""), QLabel("")
        self.zprime = QLabel("")
        self.zprime.setWordWrap(True)

        self.title = QLabel("Click a well.")
        self.title.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.annotation = QLabel("")
        self.flags = QLabel("")
        self.flags.setStyleSheet("color: #b4232c;")
        self.flags.setWordWrap(True)
        self.stats = QTableWidget(0, 2)
        self.stats.setHorizontalHeaderLabels(["Readout", "Value"])
        self.stats.horizontalHeader().setStretchLastSection(True)
        self.stats.verticalHeader().setVisible(False)
        self.stage = QComboBox()
        self.stage.addItems(list(STAGES))
        self.stage.currentTextChanged.connect(self._show_fields)
        self.overlay_box = QHBoxLayout()
        self.overlay_checks: dict[str, QCheckBox] = {}
        self.fields = QListWidget()
        self.fields.setViewMode(QListView.ViewMode.IconMode)
        self.fields.setIconSize(QSize(_THUMB, _THUMB))
        self.fields.setResizeMode(QListView.ResizeMode.Adjust)
        self.fields.setMovement(QListView.Movement.Static)
        self.fields.currentRowChanged.connect(self._show_large)
        self.large = QLabel()
        self.large.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.large.setMinimumHeight(260)
        self.open_viewer = QPushButton("Open in viewer")
        self.open_viewer.clicked.connect(self._open_in_viewer)
        self.compare = QPushButton("Compare raw / corrected / enhanced")
        self.compare.clicked.connect(self._compare)
        self.open_folder = QPushButton("Open well folder")
        self.open_folder.clicked.connect(self._open_folder)
        self.mosaic = QPushButton("Whole-well mosaic")
        self.mosaic.setToolTip("The stitched well (needs the 'Stitch well' step)")
        self.mosaic.clicked.connect(self._open_mosaic)
        self.mosaic.setEnabled(False)
        self.compare_wells_button = QPushButton("Compare wells side by side")
        self.compare_wells_button.setToolTip(
            "Open the outlined wells (or the selected well) in the Compare tab"
        )
        self.compare_wells_button.clicked.connect(self._compare_wells)

        top = QHBoxLayout()
        top.addWidget(open_button)
        top.addWidget(self.folder_label, 1)
        controls = QHBoxLayout()
        for control in (self.filter, self.readout, QLabel("Scale"), self.normalisation):
            controls.addWidget(control)
        positive_box = QHBoxLayout()
        widget: QWidget
        for widget in (
            QLabel("% of"),
            self.object_type,
            QLabel("with"),
            self.feature,
            QLabel(">"),
            self.threshold,
            positive,
        ):
            positive_box.addWidget(widget)
        bar = QHBoxLayout()
        bar.addWidget(self.cmin)
        bar.addWidget(self.colorbar, 1)
        bar.addWidget(self.cmax)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(controls)
        left_layout.addLayout(positive_box)
        left_layout.addWidget(self.annotation_filter)
        left_layout.addWidget(self.plate, 1)
        left_layout.addLayout(bar)
        left_layout.addWidget(self.zprime)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.title)
        right_layout.addWidget(self.annotation)
        right_layout.addWidget(self.flags)
        right_layout.addWidget(self.stats, 1)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show"))
        view_row.addWidget(self.stage)
        view_row.addLayout(self.overlay_box)
        view_row.addStretch(1)
        right_layout.addLayout(view_row)
        right_layout.addWidget(self.fields, 1)
        right_layout.addWidget(self.large, 2)
        buttons = QHBoxLayout()
        for button in (
            self.open_viewer,
            self.compare,
            self.mosaic,
            self.compare_wells_button,
            self.open_folder,
        ):
            buttons.addWidget(button)
        right_layout.addLayout(buttons)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        self._enable(False)

    def _enable(self, on: bool) -> None:
        for widget in (
            self.readout,
            self.normalisation,
            self.filter,
            self.open_viewer,
            self.compare,
            self.open_folder,
        ):
            widget.setEnabled(on)

    # ----------------------------------------------------------------- open

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder")
        if folder:
            self.load(Path(folder))

    def load(self, folder: Path) -> None:
        try:
            results = RunResults(folder)
        except (ValueError, OSError) as exc:
            self.folder_label.setText(str(exc))
            return
        self.results, self.well, self._percent = results, None, None
        manifest = results.manifest
        self.folder_label.setText(
            f"<b>{results.plate_name}</b> · {manifest.get('pipeline', {}).get('name', '')} · "
            f"{manifest.get('status', '')} · {len(results.wells)} wells · {folder.name}"
        )
        layout = {parse_well_name(w): a for w, a in results.layout.items()}
        present = {parse_well_name(w) for w in results.wells}
        self.plate.set_grid(results.rows, results.cols, present, {}, layout)
        self.annotation_filter.refresh()
        self.object_type.clear()
        self._features = results.object_features()
        self.object_type.addItems(list(self._features))
        self._fill_readouts()
        self._enable(True)
        self.title.setText("Click a well.")
        self.fields.clear()
        self.large.clear()

    def _fill_features(self, object_type: str) -> None:
        self.feature.clear()
        self.feature.addItems(getattr(self, "_features", {}).get(object_type, []))

    def _fill_readouts(self) -> None:
        if self.results is None:
            return
        text = self.filter.text().strip().lower()
        keep = self.readout.currentText()
        self.readout.blockSignals(True)
        self.readout.clear()
        names = [c for c in self.results.readout_columns() if text in c.lower()]
        if self._percent is not None:
            names.insert(0, self._percent_name())
        self.readout.addItems(names)
        from PySide6.QtCore import Qt

        from hcs_studio.readouts import label

        for i, name in enumerate(names):  # the registry's name and unit as a tooltip
            self.readout.setItemData(i, label(name), Qt.ItemDataRole.ToolTipRole)
        if keep in names:
            self.readout.setCurrentText(keep)
        self.readout.blockSignals(False)
        self.redraw()

    def _percent_name(self) -> str:
        assert self._percent is not None
        kind, feature, threshold = self._percent
        return f"% {kind} with {feature} > {threshold:g}"

    def show_percent_positive(self) -> None:
        if self.results is None or not self.feature.currentText():
            return
        self._percent = (
            self.object_type.currentText(),
            self.feature.currentText(),
            self.threshold.value(),
        )
        self._percent_values = self.results.percent_positive(*self._percent)
        self._fill_readouts()
        self.readout.setCurrentText(self._percent_name())

    # ----------------------------------------------------------------- draw

    def current_values(self) -> dict[str, float]:
        if self.results is None or not self.readout.currentText():
            return {}
        name = self.readout.currentText()
        if self._percent is not None and name == self._percent_name():
            return dict(self._percent_values)
        return self.results.values(name)

    def redraw(self) -> None:
        if self.results is None:
            return
        self.raw_values = self.current_values()
        method = NORMALISATIONS[self.normalisation.currentText()]
        try:
            self.values = self.results.normalise(self.raw_values, method)  # type: ignore[arg-type]
            note = ""
        except ValueError as exc:
            self.values, note = {w: float("nan") for w in self.raw_values}, str(exc)
        finite = [v for v in self.values.values() if math.isfinite(v)]
        z = method != "none"
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        if z:
            span = min(6.0, max(3.0, abs(lo), abs(hi)))
            lo, hi = -span, span
        if hi == lo:
            hi = lo + 1
        stops = DIVERGING if z else VIRIDIS
        fills, tips = {}, {}
        image_qc = self.results.image_qc_wells()
        failed = (
            {str(w): str(r) for w, r in image_qc.loc[image_qc["qc_failed"], "qc_reasons"].items()}
            if image_qc is not None
            else {}
        )
        self.plate.set_hatched({parse_well_name(w) for w in failed})
        for well, value in self.values.items():
            key = parse_well_name(well)
            if math.isfinite(value):
                fills[key] = colour_at(stops, (value - lo) / (hi - lo))
            raw = self.raw_values.get(well, float("nan"))
            annotation = self.results.layout.get(well)
            extra = ""
            if annotation:
                extra = "\n" + annotation.label()
            if well in failed:
                extra += f"\nfailed image QC: {failed[well]}"
            tips[key] = f"{well}: {fmt(raw)}" + (f" (Z {fmt(value)})" if z else "") + extra
        self.plate.set_fills(fills, tips)
        self._draw_colorbar(stops)
        self.cmin.setText(fmt(lo))
        self.cmax.setText(fmt(hi))
        zp = self.results.z_prime(self.raw_values)
        verdict = (
            ""
            if zp is None
            else (
                " (excellent)" if zp >= 0.5 else " (acceptable)" if zp > 0 else " (no separation)"
            )
        )
        self.zprime.setText(
            (
                f"Z′ = {zp:.2f}{verdict}"
                if zp is not None
                else "Z′ needs two positive and two negative control wells in the layout."
            )
            + (f"  {note}" if note else "")
        )
        if self.well:
            self._show_stats()

    def _draw_colorbar(self, stops: tuple[str, ...]) -> None:
        width = max(200, self.colorbar.width())
        pixmap = QPixmap(width, 16)
        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, width, 0)
        for i, stop in enumerate(stops):
            gradient.setColorAt(i / (len(stops) - 1), QColor(stop))
        painter.fillRect(pixmap.rect(), gradient)
        painter.end()
        self.colorbar.setPixmap(pixmap)

    # ----------------------------------------------------------------- well

    def select_well(self, well: str) -> None:
        if self.results is None or well not in self.results.wells:
            return
        self.well = well
        self.plate.selected = parse_well_name(well)
        self.plate._restyle()
        annotation = self.results.layout.get(well)
        self.title.setText(f"Well {well}")
        self.annotation.setText(annotation.label() if annotation else "")
        flags = self.results.well_flags(well)
        self.flags.setText("QC: " + "; ".join(flags) if flags else "")
        self._fields = self.results.well_fields(well)
        self.mosaic.setEnabled(
            (self.results.folder / "wells" / well / "previews" / "mosaic.png").exists()
        )
        for check in self.overlay_checks.values():
            self.overlay_box.removeWidget(check)
            check.deleteLater()
        self.overlay_checks = {}
        for name in self._fields[0].overlays if self._fields else {}:
            check = QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(self._show_fields)
            self.overlay_box.addWidget(check)
            self.overlay_checks[name] = check
        self._show_stats()
        self._show_fields()
        self.well_selected.emit(well)

    def _show_stats(self) -> None:
        assert self.results is not None and self.well is not None
        current = self.readout.currentText()
        keys = [
            k
            for k in self.results.readout_columns()
            if any(m in k for m in ("_count", "_per_mm2", "fields", "coloc_"))
            and not k.endswith(("_std", "_mean"))
        ][:14]
        rows = [(current, self.raw_values.get(self.well, float("nan")))]
        rows += [
            (k, self.results.values(k).get(self.well, float("nan"))) for k in keys if k != current
        ]
        self.stats.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.stats.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(fmt(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.stats.setItem(row, 1, item)

    def _stage_path(self, field: WellField) -> str | None:
        stage = self.stage.currentText()
        return {"Enhanced": field.enhanced, "Corrected": field.corrected, "Raw": field.raw}[
            stage
        ] or field.enhanced

    def _overlays(self, field: WellField) -> list[str]:
        return [
            path
            for name, path in field.overlays.items()
            if self.overlay_checks.get(name) and self.overlay_checks[name].isChecked()
        ]

    def _show_fields(self) -> None:
        if self.results is None:
            return
        current = max(0, self.fields.currentRow())
        self.fields.clear()
        for field in self._fields:
            pixmap = compose(self.results.folder, self._stage_path(field), self._overlays(field))
            counts = ", ".join(f"{n} {k}" for k, n in field.objects.items())
            item = QListWidgetItem(
                QIcon(pixmap.scaled(_THUMB, _THUMB, Qt.AspectRatioMode.KeepAspectRatio)),
                f"Field {field.index}",
            )
            item.setToolTip(counts)
            self.fields.addItem(item)
        if self._fields:
            self.fields.setCurrentRow(min(current, len(self._fields) - 1))

    def _show_large(self, row: int) -> None:
        if self.results is None or not 0 <= row < len(self._fields):
            self.large.clear()
            return
        field = self._fields[row]
        pixmap = compose(self.results.folder, self._stage_path(field), self._overlays(field))
        size = self.large.size()
        self.large.setPixmap(
            pixmap.scaled(
                max(200, size.width()),
                max(200, size.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def selected_field(self) -> WellField | None:
        row = self.fields.currentRow()
        return self._fields[row] if 0 <= row < len(self._fields) else None

    def _open_in_viewer(self) -> None:
        field = self.selected_field()
        if self.results is not None and self.well and field:
            self.open_field.emit(
                self.results.folder, self.well, field.index, self.stage.currentText()
            )

    def _compare(self) -> None:
        field = self.selected_field()
        if self.results is not None and self.well and field:
            self.compare_field.emit(self.results.folder, self.well, field.index)

    def mark_wells(self, wells: list[str]) -> None:
        """Outline wells chosen elsewhere (e.g. a lasso in the Profiles tab)."""
        if self.results is None:
            return
        self.plate.set_selection([parse_well_name(w) for w in wells if w in self.results.wells])

    def selected_wells(self) -> list[str]:
        if self.results is None:
            return []
        wells = [well_name(*k) for k in self.plate.selected_wells()]
        return [w for w in wells if w in self.results.wells] or ([self.well] if self.well else [])

    def _compare_wells(self) -> None:
        wells = self.selected_wells()
        if self.results is not None and wells:
            self.compare_wells.emit(self.results.folder, wells)

    def _open_mosaic(self) -> None:
        if self.results is not None and self.well:
            self.open_mosaic.emit(self.results.folder, self.well)

    def _open_folder(self) -> None:
        if self.results is not None and self.well:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.results.folder / "wells" / self.well))
            )


def image_from_rgb(rgb: Any) -> QImage:
    array = np.ascontiguousarray(rgb)
    height, width = array.shape[:2]
    return QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
