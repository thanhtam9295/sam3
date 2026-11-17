"""Plate tab: header, plate map, hover preview and the field gallery of the selected well."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.plate_map import PlateMapView
from hcs_studio.gui.workers import ThumbnailLoader
from hcs_studio.model.layout import WellAnnotation
from hcs_studio.model.plate import Plate, well_name

_THUMB = 128


def _pixmap(rgb: Any) -> QPixmap:
    array = np.ascontiguousarray(rgb)
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class PlatePanel(QWidget):
    """Shows one plate; emits ``field_requested(row, col, field_position)`` to open a field."""

    field_requested = Signal(int, int, int)
    well_selected = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plate: Plate | None = None
        self.header = QLabel("No plate open. Choose a plate folder in the data browser.")
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #a05a00;")
        self.notes.setVisible(False)

        self.plate_map = PlateMapView()
        self.preview = QLabel("Hover over a well for a preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedSize(_THUMB + 60, _THUMB + 40)
        self.preview.setWordWrap(True)
        self._hover_key: tuple[int, int, int] | None = None

        self.gallery_label = QLabel("Click a well to list its fields.")
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(_THUMB, _THUMB))
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setSpacing(8)
        self.gallery.itemClicked.connect(self._open_item)
        self._gallery_items: dict[tuple[int, int, int], QListWidgetItem] = {}

        self.thumbnails = ThumbnailLoader(self)
        self.thumbnails.ready.connect(self._on_thumbnail)
        self.plate_map.well_clicked.connect(self._on_well_clicked)
        self.plate_map.well_hovered.connect(self._on_well_hovered)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.plate_map, 1)
        top_layout.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignTop)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.gallery_label)
        bottom_layout.addWidget(self.gallery)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.notes)
        layout.addWidget(splitter, 1)

    def set_plate(
        self, plate: Plate | None, layout: dict[tuple[int, int], WellAnnotation] | None = None
    ) -> None:
        keep = (
            self.plate_map.selected
            if plate and self.plate and plate.path == self.plate.path
            else None
        )
        self.plate = plate
        self.thumbnails.set_plate(plate)
        self.gallery.clear()
        self._gallery_items.clear()
        self._hover_key = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Hover over a well for a preview")
        self.plate_map.set_plate(plate, layout)
        if plate is None:
            self.header.setText("No plate open. Choose a plate folder in the data browser.")
            self.notes.setVisible(False)
            self.gallery_label.setText("Click a well to list its fields.")
            return
        dims = ""
        if plate.shape_tczyx is not None:
            t, c, z, y, x = plate.shape_tczyx
            dims = f" · T{t} C{c} Z{z} · {x}×{y} px"
        barcode = f" · barcode {plate.barcode}" if plate.barcode else ""
        self.header.setText(
            f"<b>{plate.name}</b> · {plate.vendor} · {plate.rows * plate.cols}-well layout · "
            f"{len(plate.wells)} wells, {plate.n_fields} fields{dims}{barcode}"
        )
        self.notes.setText("\n".join(f"⚠ {n}" for n in plate.notes))
        self.notes.setVisible(bool(plate.notes))
        self.gallery_label.setText("Click a well to list its fields.")
        if keep is not None and keep in plate.wells:
            self.plate_map.select_well(*keep)

    def _on_well_clicked(self, row: int, col: int) -> None:
        if self.plate is None:
            return
        well = self.plate.wells[(row, col)]
        self.gallery.clear()
        self._gallery_items.clear()
        label = f" ({well.label})" if well.label else ""
        self.gallery_label.setText(
            f"Well {well_name(row, col)}{label}: {len(well.fields)} field(s). "
            "Click a field to open it."
        )
        placeholder = QPixmap(_THUMB, _THUMB)
        placeholder.fill(Qt.GlobalColor.black)
        for position, field in enumerate(well.fields):
            item = QListWidgetItem(QIcon(placeholder), f"Field {field.index}")
            item.setData(Qt.ItemDataRole.UserRole, (row, col, position))
            self.gallery.addItem(item)
            key = (row, col, field.index)
            self._gallery_items[key] = item
            self.thumbnails.request(key, field)
        self.well_selected.emit(row, col)

    def _on_well_hovered(self, row: int, col: int) -> None:
        if self.plate is None:
            return
        field = self.plate.wells[(row, col)].fields[0]
        self._hover_key = (row, col, field.index)
        self.preview.setPixmap(QPixmap())
        self.preview.setText(f"{well_name(row, col)}\nloading preview…")
        self.thumbnails.request(self._hover_key, field)

    def _on_thumbnail(self, key: tuple[int, int, int], rgb: Any) -> None:
        pixmap = _pixmap(rgb).scaled(
            _THUMB,
            _THUMB,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        item = self._gallery_items.get(key)
        if item is not None:
            item.setIcon(QIcon(pixmap))
        if key == self._hover_key:
            self.preview.setPixmap(pixmap)

    def _open_item(self, item: QListWidgetItem) -> None:
        row, col, position = item.data(Qt.ItemDataRole.UserRole)
        self.field_requested.emit(row, col, position)
