"""Select wells by annotation (cell line, compound, concentration range, role)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from hcs_studio.model import units
from hcs_studio.model.annotation import ROLES, AnnotationFilter, PlateLayout

UNIT_CHOICES = [u for kind in ("molar", "mass", "percent") for u in units.CHOICES[kind]]
Key = tuple[int, int]


def unit_combo(value: str = "µM") -> QComboBox:
    combo = QComboBox()
    combo.addItems(UNIT_CHOICES)
    combo.setCurrentText(value)
    return combo


class AnnotationFilterBar(QWidget):
    """Select wells by annotation, e.g. MCF7 wells with drug X ≥ 100 nM."""

    selected = Signal(list)  # wells (row, col)

    def __init__(self, layout_getter: Callable[[], PlateLayout | None]) -> None:
        super().__init__()
        self._layout = layout_getter
        self.cell_line = QComboBox()
        self.cell_line.setEditable(True)
        self.compound = QComboBox()
        self.compound.setEditable(True)
        self.minimum = QLineEdit()
        self.minimum.setPlaceholderText("≥")
        self.minimum.setMaximumWidth(70)
        self.maximum = QLineEdit()
        self.maximum.setPlaceholderText("≤")
        self.maximum.setMaximumWidth(70)
        self.unit = unit_combo()
        self.role = QComboBox()
        self.role.addItems(["any role", *ROLES])
        button = QPushButton("Select")
        button.clicked.connect(self.apply)
        self.result = QLabel("")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for widget in (
            QLabel("Select"),
            self.cell_line,
            self.compound,
            self.minimum,
            self.maximum,
            self.unit,
            self.role,
            button,
            self.result,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        layout = self._layout()
        for combo, items, blank in (
            (self.cell_line, layout.cell_lines() if layout else [], "any cell line"),
            (self.compound, layout.compounds() if layout else [], "any compound"),
        ):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([blank, *items])
            if keep and keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)

    def current_filter(self) -> AnnotationFilter:
        def number(text: str) -> float | None:
            text = text.strip().replace(",", ".")
            return float(text) if text else None

        line = self.cell_line.currentText()
        compound = self.compound.currentText()
        return AnnotationFilter(
            cell_line="" if line == "any cell line" else line,
            compound="" if compound == "any compound" else compound,
            min_concentration=number(self.minimum.text()),
            max_concentration=number(self.maximum.text()),
            unit=self.unit.currentText(),
            roles=() if self.role.currentIndex() == 0 else (self.role.currentText(),),
        )

    def apply(self) -> list[Key]:
        layout = self._layout()
        if layout is None:
            return []
        try:
            wells = self.current_filter().select(layout)
        except ValueError as exc:
            self.result.setText(str(exc))
            return []
        self.result.setText(f"{len(wells)} well(s)")
        self.selected.emit(wells)
        return wells
