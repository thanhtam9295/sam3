"""Clickable plate map drawn with Qt Graphics View."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QResizeEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from hcs_studio.model.layout import WellAnnotation
from hcs_studio.model.plate import Plate, row_label, well_name

_PITCH = 30.0
_EMPTY = QColor(225, 225, 225)
_FILLED = QColor(70, 130, 200)
_SELECTED = QColor(240, 150, 30)
_CONTROL = {"negative": QColor(120, 120, 120), "positive": QColor(200, 70, 70)}


class _WellItem(QGraphicsEllipseItem):
    def __init__(self, view: PlateMapView, row: int, col: int, has_data: bool) -> None:
        margin = _PITCH * 0.08
        super().__init__(
            col * _PITCH + margin, row * _PITCH + margin, _PITCH - 2 * margin, _PITCH - 2 * margin
        )
        self._view = view
        self.row, self.col, self.has_data = row, col, has_data
        self.setAcceptHoverEvents(has_data)
        if has_data:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._view.well_hovered.emit(self.row, self.col)
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.has_data and event.button() == Qt.MouseButton.LeftButton:
            self._view.select_well(self.row, self.col)
            event.accept()
        else:
            super().mousePressEvent(event)


class PlateMapView(QGraphicsView):
    """Plate grid with the correct geometry; wells that contain images are clickable."""

    well_clicked = Signal(int, int)
    well_hovered = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumHeight(220)
        self._items: dict[tuple[int, int], _WellItem] = {}
        self._plate: Plate | None = None
        self._layout: dict[tuple[int, int], WellAnnotation] = {}
        self.selected: tuple[int, int] | None = None

    def set_plate(
        self, plate: Plate | None, layout: dict[tuple[int, int], WellAnnotation] | None = None
    ) -> None:
        self._scene.clear()
        self._items.clear()
        self._plate, self._layout, self.selected = plate, layout or {}, None
        if plate is None:
            return
        font = QFont()
        font.setPixelSize(int(_PITCH * 0.42))
        for col in range(plate.cols):
            self._label(str(col + 1), col * _PITCH, -_PITCH * 0.75, font)
        for row in range(plate.rows):
            self._label(row_label(row), -_PITCH * 0.95, row * _PITCH, font)
        for row in range(plate.rows):
            for col in range(plate.cols):
                well = plate.wells.get((row, col))
                item = _WellItem(self, row, col, well is not None)
                if well is not None:
                    item.setToolTip(self._tooltip(row, col))
                self._scene.addItem(item)
                self._items[(row, col)] = item
        self._restyle()
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-6, -6, 6, 6))
        self._fit()

    def _label(self, text: str, x: float, y: float, font: QFont) -> None:
        item = QGraphicsSimpleTextItem(text)
        item.setFont(font)
        rect = item.boundingRect()
        item.setPos(x + (_PITCH - rect.width()) / 2, y + (_PITCH - rect.height()) / 2)
        self._scene.addItem(item)

    def _tooltip(self, row: int, col: int) -> str:
        assert self._plate is not None
        well = self._plate.wells[(row, col)]
        lines = [f"{well_name(row, col)} — {len(well.fields)} field(s)"]
        if well.label:
            lines.append(well.label)
        note = self._layout.get((row, col))
        if note is not None:
            parts = [note.compound, note.treatment, note.control_type]
            if note.concentration is not None:
                parts.insert(1, f"{note.concentration:g}")
            lines.append(" · ".join(p for p in parts if p))
        return "\n".join(lines)

    def _restyle(self) -> None:
        for key, item in self._items.items():
            if not item.has_data:
                item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                item.setPen(QPen(_EMPTY, 1.0))
                continue
            note = self._layout.get(key)
            colour = _CONTROL.get(note.control_type, _FILLED) if note else _FILLED
            item.setBrush(QBrush(colour))
            selected = key == self.selected
            item.setPen(QPen(_SELECTED, 3.0) if selected else QPen(colour.darker(130), 1.0))

    def select_well(self, row: int, col: int) -> None:
        """Select a well that has data and announce it; other positions are ignored."""
        item = self._items.get((row, col))
        if item is None or not item.has_data:
            return
        self.selected = (row, col)
        self._restyle()
        self.well_clicked.emit(row, col)

    def _fit(self) -> None:
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(QRectF(rect), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit()
