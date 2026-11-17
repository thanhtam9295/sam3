"""Clickable plate map drawn with Qt Graphics View."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QResizeEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QStyle,
    QWidget,
)

from hcs_studio.model.layout import WellAnnotation
from hcs_studio.model.plate import Plate, row_label, well_name

_PITCH = 30.0
_EMPTY = QColor(225, 225, 225)
_FILLED = QColor(70, 130, 200)
_SELECTED = QColor(240, 150, 30)
_NAN = QColor(200, 204, 210)
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
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._view.well_hovered.emit(self.row, self.col)
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not (self.has_data and event.button() == Qt.MouseButton.LeftButton):
            super().mousePressEvent(event)
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._view.select_range(self.row, self.col)
            event.accept()
            return
        super().mousePressEvent(event)  # Qt handles click and Ctrl+click selection
        if not modifiers & Qt.KeyboardModifier.ControlModifier:
            self._view.select_well(self.row, self.col)

    def paint(self, painter: QPainter, option: Any, widget: QWidget | None = None) -> None:
        # Selection is drawn by the pen (see _restyle), not Qt's dashed rectangle.
        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)


class PlateMapView(QGraphicsView):
    """Plate grid with the correct geometry; wells that contain images are clickable."""

    well_clicked = Signal(int, int)
    well_hovered = Signal(int, int)
    #: Wells selected for a run (click, Ctrl+click, Shift+click range, or drag a rectangle).
    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._anchor: tuple[int, int] | None = None
        self.setMinimumHeight(220)
        self._items: dict[tuple[int, int], _WellItem] = {}
        self._plate: Plate | None = None
        self._layout: dict[tuple[int, int], WellAnnotation] = {}
        self._fills: dict[tuple[int, int], QColor] | None = None
        self.selected: tuple[int, int] | None = None

    def set_plate(
        self, plate: Plate | None, layout: dict[tuple[int, int], WellAnnotation] | None = None
    ) -> None:
        self._plate = plate
        if plate is None:
            self.set_grid(0, 0, set(), {}, layout)
            return
        self._layout = layout or {}  # tooltips include the layout's annotations
        tooltips = {key: self._tooltip(*key) for key in plate.wells}
        self.set_grid(plate.rows, plate.cols, set(plate.wells), tooltips, layout)

    def set_grid(
        self,
        rows: int,
        cols: int,
        present: set[tuple[int, int]],
        tooltips: dict[tuple[int, int], str],
        layout: dict[tuple[int, int], WellAnnotation] | None = None,
    ) -> None:
        """Draw a rows × cols plate; wells in ``present`` have data and can be clicked."""
        self._scene.clear()
        self._items.clear()
        self._layout, self.selected = layout or {}, None
        self._fills = None
        self._anchor = None
        if rows == 0:
            return
        font = QFont()
        font.setPixelSize(int(_PITCH * 0.42))
        for col in range(cols):
            self._label(str(col + 1), col * _PITCH, -_PITCH * 0.75, font)
        for row in range(rows):
            self._label(row_label(row), -_PITCH * 0.95, row * _PITCH, font)
        for row in range(rows):
            for col in range(cols):
                has_data = (row, col) in present
                item = _WellItem(self, row, col, has_data)
                if has_data:
                    item.setToolTip(tooltips.get((row, col), well_name(row, col)))
                self._scene.addItem(item)
                self._items[(row, col)] = item
        self._restyle()
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-6, -6, 6, 6))
        self._fit()

    def set_fills(
        self,
        fills: dict[tuple[int, int], QColor] | None,
        tooltips: dict[tuple[int, int], str] | None = None,
    ) -> None:
        """Colour wells by value (a heatmap); None returns to the data/control colours."""
        self._fills = fills
        for key, text in (tooltips or {}).items():
            item = self._items.get(key)
            if item is not None and item.has_data:
                item.setToolTip(text)
        self._restyle()

    def fill_of(self, key: tuple[int, int]) -> QColor:
        return self._items[key].brush().color()

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
            if self._fills is not None:
                colour = self._fills.get(key, _NAN)
            item.setBrush(QBrush(colour))
            if key == self.selected:
                item.setPen(QPen(_SELECTED, 3.5))
            elif item.isSelected():
                item.setPen(QPen(_SELECTED.darker(120), 2.5))
            else:
                item.setPen(QPen(colour.darker(130), 1.0))

    def select_well(self, row: int, col: int) -> None:
        """Make a well with data the current well and announce it; other positions are ignored."""
        item = self._items.get((row, col))
        if item is None or not item.has_data:
            return
        self.selected = (row, col)
        self._anchor = (row, col)
        if not item.isSelected():
            self._scene.blockSignals(True)
            self._scene.clearSelection()
            item.setSelected(True)
            self._scene.blockSignals(False)
            self._on_selection_changed()
        self._restyle()
        self.well_clicked.emit(row, col)

    def select_range(self, row: int, col: int) -> None:
        """Select the rectangle of wells between the last clicked well and (row, col)."""
        anchor = self._anchor or (row, col)
        rows = range(min(anchor[0], row), max(anchor[0], row) + 1)
        cols = range(min(anchor[1], col), max(anchor[1], col) + 1)
        self._scene.blockSignals(True)
        self._scene.clearSelection()
        for key, item in self._items.items():
            if item.has_data and key[0] in rows and key[1] in cols:
                item.setSelected(True)
        self._scene.blockSignals(False)
        self._on_selection_changed()

    def set_selection(self, wells: list[tuple[int, int]]) -> None:
        self._scene.blockSignals(True)
        self._scene.clearSelection()
        for key in wells:
            item = self._items.get(key)
            if item is not None and item.has_data:
                item.setSelected(True)
        self._scene.blockSignals(False)
        self._on_selection_changed()

    def selected_wells(self) -> list[tuple[int, int]]:
        return sorted(key for key, item in self._items.items() if item.isSelected())

    def _on_selection_changed(self) -> None:
        self._restyle()
        self.selection_changed.emit(self.selected_wells())

    def _fit(self) -> None:
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(QRectF(rect), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit()
