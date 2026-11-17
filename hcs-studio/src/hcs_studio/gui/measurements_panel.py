"""Measurements dock: measurement tables of the preview, and interactive colocalization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.analysis import coloc
from hcs_studio.analysis.coloc_export import export_for_fiji
from hcs_studio.analysis.measure import coloc_inputs, colocalization_map
from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.processing.base import ImageData, LabelSet, ModuleError

MAX_ROWS = 5000
WHOLE_FIELD = "(whole field)"


class DataFrameModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__()
        self.frame = frame

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return len(self.frame)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return len(self.frame.columns)

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, float | np.floating):
                return "" if np.isnan(value) else f"{value:.4g}"
            return str(value)
        if role == Qt.ItemDataRole.UserRole:  # for sorting
            return value.item() if isinstance(value, np.generic) else value
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return (
            str(self.frame.columns[section])
            if orientation == Qt.Orientation.Horizontal
            else str(section + 1)
        )


class MeasurementsPanel(QWidget):
    def __init__(
        self, viewer: ViewerPanel, log: Callable[[str], None], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._log = log
        self._tables: dict[str, pd.DataFrame] = {}
        self._image: ImageData | None = None
        self._labels: tuple[LabelSet, ...] = ()
        self.last_result: coloc.ColocResult | None = None

        self.table_choice = QComboBox()
        self.table_choice.currentTextChanged.connect(self._show_table)
        self.table_info = QLabel("Preview a pipeline with measurement steps to see its tables.")
        self.table_info.setWordWrap(True)
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.table.setModel(self.proxy)

        self.channel_a = QComboBox()
        self.channel_b = QComboBox()
        self.region = QComboBox()
        self.object_id = QSpinBox()
        self.object_id.setRange(0, 10_000_000)
        self.object_id.setSpecialValueText("all objects")
        self.randomizations = QSpinBox()
        self.randomizations.setRange(0, 1000)
        self.randomizations.setValue(0)
        self.block_px = QSpinBox()
        self.block_px.setRange(2, 64)
        self.block_px.setValue(3)
        analyse = QPushButton("Cytofluorogram")
        analyse.clicked.connect(self.analyse)
        show_map = QPushButton("Colocalization map")
        show_map.clicked.connect(self.show_map)
        export = QPushButton("Export for Fiji…")
        export.clicked.connect(self._choose_export)
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        self.result_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.setLabel("bottom", "channel A")
        self.plot.setLabel("left", "channel B")

        form = QFormLayout()
        form.addRow("Channel A", self.channel_a)
        form.addRow("Channel B", self.channel_b)
        form.addRow("Region", self.region)
        form.addRow("Object", self.object_id)
        form.addRow("Randomizations", self.randomizations)
        form.addRow("Block (px ≈ PSF)", self.block_px)
        buttons = QHBoxLayout()
        for button in (analyse, show_map, export):
            buttons.addWidget(button)
        coloc_box = QGroupBox("Colocalization (current plane, linear data)")
        coloc_layout = QVBoxLayout(coloc_box)
        coloc_layout.addLayout(form)
        coloc_layout.addLayout(buttons)
        coloc_layout.addWidget(self.plot)
        coloc_layout.addWidget(self.result_text)
        table_box = QGroupBox("Tables")
        table_layout = QVBoxLayout(table_box)
        table_layout.addWidget(self.table_choice)
        table_layout.addWidget(self.table_info)
        table_layout.addWidget(self.table, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(table_box, 1)
        layout.addWidget(coloc_box, 1)

    # ------------------------------------------------------------------ data

    def set_data(
        self, image: ImageData | None, labels: tuple[LabelSet, ...], tables: dict[str, Any]
    ) -> None:
        """Linear image and segmentations of the field in the viewer, and its tables."""
        self._image, self._labels, self._tables = image, labels, dict(tables)
        for combo in (self.channel_a, self.channel_b):
            current = combo.currentIndex()
            combo.clear()
            if image is not None:
                combo.addItems([c.name for c in image.metadata.channels])
            combo.setCurrentIndex(max(0, current))
        if (
            image is not None
            and self.channel_b.count() > 1
            and self.channel_b.currentIndex() == self.channel_a.currentIndex()
        ):
            self.channel_b.setCurrentIndex(1)
        region = self.region.currentText()
        self.region.clear()
        self.region.addItems([WHOLE_FIELD, *[ls.name for ls in labels]])
        if region:
            self.region.setCurrentText(region)
        self.table_choice.blockSignals(True)
        self.table_choice.clear()
        self.table_choice.addItems(list(self._tables))
        self.table_choice.blockSignals(False)
        self._show_table(self.table_choice.currentText())

    def _show_table(self, name: str) -> None:
        frame = self._tables.get(name)
        if frame is None:
            self.proxy.setSourceModel(DataFrameModel(pd.DataFrame()))
            if not self._tables:
                self.table_info.setText(
                    "Preview a pipeline with measurement steps to see its tables."
                )
            return
        shown = frame.head(MAX_ROWS)
        self.proxy.setSourceModel(DataFrameModel(shown))
        more = f" (first {MAX_ROWS} shown)" if len(frame) > MAX_ROWS else ""
        self.table_info.setText(f"{name}: {len(frame)} rows × {len(frame.columns)} columns{more}")

    # ---------------------------------------------------------------- coloc

    def _inputs(self) -> tuple[Any, Any, Any]:
        if self._image is None:
            raise ModuleError("Open a field and preview it first")
        a, b = self.channel_a.currentIndex(), self.channel_b.currentIndex()
        if a == b:
            raise ModuleError("Choose two different channels")
        region = self.region.currentText()
        label = None if region in ("", WHOLE_FIELD) else region
        z = 0
        names = self.viewer.layer_names()
        if names and self._image.data.shape[1] > 1:
            z = min(self.viewer.current_z(names[0]), self._image.data.shape[1] - 1)
        return coloc_inputs(self._image, self._labels, a, b, label, self.object_id.value(), z)

    def analyse(self) -> None:
        try:
            a, b, mask = self._inputs()
        except ModuleError as exc:
            self.result_text.setText(str(exc))
            return
        n = self.randomizations.value()
        block = (self.block_px.value(),) * 2
        result = coloc.colocalize(a, b, mask, randomizations=n, block=block)
        self.last_result = result
        self._draw(a[mask], b[mask], result)
        p = f", Costes p = {result.p_value:.3g}" if n else ""
        warnings = f"\nWarnings: {'; '.join(result.warnings)}" if result.warnings else ""
        self.result_text.setText(
            f"Pearson r = {result.pearson:.3f}, Spearman = {result.spearman:.3f}, "
            f"ICQ = {result.icq:.3f}{p}\n"
            f"M1 = {result.m1:.3f}, M2 = {result.m2:.3f}; "
            f"tM1 = {result.tm1:.3f}, tM2 = {result.tm2:.3f} "
            f"(Costes thresholds {result.threshold_a:.4g}, {result.threshold_b:.4g}); "
            f"{result.n_pixels} pixels{warnings}"
        )
        pair = f"{self.channel_a.currentText()}/{self.channel_b.currentText()}"
        self._log(f"Colocalization {pair}: r = {result.pearson:.3f}")

    def _draw(self, a: Any, b: Any, result: coloc.ColocResult) -> None:
        self.plot.clear()
        if a.size == 0:
            return
        bins = 128
        top_a, top_b = float(np.percentile(a, 99.9)) or 1.0, float(np.percentile(b, 99.9)) or 1.0
        histogram, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, top_a], [0, top_b]])
        image = pg.ImageItem(np.log1p(histogram))
        image.setRect(0, 0, top_a, top_b)
        image.setColorMap(pg.colormap.get("viridis"))
        self.plot.addItem(image)
        if np.isfinite(result.costes_slope):
            x = np.array([0, top_a])
            intercept = result.threshold_b - result.costes_slope * result.threshold_a
            self.plot.plot(x, result.costes_slope * x + intercept, pen=pg.mkPen("w", width=1))
        self.plot.addItem(
            pg.InfiniteLine(
                result.threshold_a, angle=90, pen=pg.mkPen("r", style=Qt.PenStyle.DashLine)
            )
        )
        self.plot.addItem(
            pg.InfiniteLine(
                result.threshold_b, angle=0, pen=pg.mkPen("r", style=Qt.PenStyle.DashLine)
            )
        )
        self.plot.setXRange(0, top_a, padding=0.02)
        self.plot.setYRange(0, top_b, padding=0.02)
        self.plot.setLabel("bottom", self.channel_a.currentText())
        self.plot.setLabel("left", self.channel_b.currentText())

    def show_map(self) -> None:
        try:
            a, b, mask = self._inputs()
        except ModuleError as exc:
            self.result_text.setText(str(exc))
            return
        assert self._image is not None
        nmdp = colocalization_map(a, b, mask)
        pixel = self._image.metadata.pixel_size_um or 1.0
        name = "colocalization map (nMDP)"
        if name in self.viewer.viewer.layers:
            self.viewer.viewer.layers.remove(name)
        unit = "µm" if self._image.metadata.pixel_size_um is not None else "pixel"
        positive = np.clip(nmdp, 0, None)
        top = float(np.percentile(positive[positive > 0], 99)) if (positive > 0).any() else 1.0
        # Positive nMDP only (both channels above their mean), as an additive heat overlay.
        self.viewer.viewer.add_image(
            positive,
            name=name,
            colormap="magma",
            blending="additive",
            contrast_limits=(0, max(top, 1e-6)),
            scale=(pixel, pixel),
            units=(unit, unit),
        )
        self._log("Added the colocalization map (nMDP > 0: both channels above their mean)")

    def _choose_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export folder for Fiji Coloc 2")
        if folder:
            self.export(Path(folder))

    def export(self, folder: Path) -> None:
        try:
            a, b, mask = self._inputs()
        except ModuleError as exc:
            self.result_text.setText(str(exc))
            return
        result = export_for_fiji(
            folder,
            a,
            b,
            mask,
            names=(self.channel_a.currentText(), self.channel_b.currentText()),
            psf_px=self.block_px.value(),
            randomizations=max(self.randomizations.value(), 100),
            description=f"region {self.region.currentText()}, object {self.object_id.value()}",
        )
        self.result_text.setText(
            f"Exported to {folder} (r = {result.pearson:.3f}). Run coloc2_compare.ijm in Fiji."
        )
        self._log(f"Exported colocalization inputs for Fiji to {folder}")
