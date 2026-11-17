"""Metadata dock: optics, channels and dimensions, with user overrides."""

from __future__ import annotations

from typing import Any, get_args

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.model.plate import ImageMetadata, Modality, Plate

_NUMERIC = (
    ("pixel_size_um", "Pixel size XY (µm)"),
    ("z_step_um", "Z step (µm)"),
    ("objective_na", "Objective NA"),
    ("magnification", "Magnification"),
    ("immersion_ri", "Immersion RI"),
    ("pinhole_um", "Pinhole (µm)"),
)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


class MetadataPanel(QWidget):
    """Shows a plate's metadata. ``overrides_saved(dict)`` carries the fields the user changed."""

    overrides_saved = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file_metadata: ImageMetadata | None = None
        self.source = QLabel("No plate open.")
        self.source.setWordWrap(True)
        self.dimensions = QLabel()
        self.objective = QLabel()
        self.bit_depth = QLabel()
        self.edits: dict[str, QLineEdit] = {}

        form = QFormLayout()
        form.addRow("Dimensions", self.dimensions)
        form.addRow("Objective", self.objective)
        form.addRow("Bit depth", self.bit_depth)
        for key, label in _NUMERIC:
            edit = QLineEdit()
            edit.setPlaceholderText("unknown")
            self.edits[key] = edit
            form.addRow(label, edit)
        self.modality = QComboBox()
        self.modality.addItems(list(get_args(Modality)))
        form.addRow("Modality", self.modality)

        self.channels = QTableWidget(0, 3)
        self.channels.setHorizontalHeaderLabels(["Channel", "Ex (nm)", "Em (nm)"])
        self.channels.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.channels.verticalHeader().setVisible(False)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.save_button = QPushButton("Save overrides")
        self.save_button.clicked.connect(self.save)
        self.reset_button = QPushButton("Reset to file values")
        self.reset_button.clicked.connect(self.reset)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.source)
        layout.addLayout(form)
        layout.addWidget(self.channels, 1)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        hint = QLabel("Overrides are stored in your HCS Studio settings, never in the raw data.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout.addWidget(hint)
        self.setEnabled(False)

    def set_plate(self, plate: Plate | None, file_metadata: ImageMetadata | None) -> None:
        """Show ``plate`` (overrides applied); ``file_metadata`` is what the files say."""
        self._file_metadata = file_metadata
        self.setEnabled(plate is not None)
        if plate is None or file_metadata is None:
            self.source.setText("No plate open.")
            return
        self.source.setText(f"<b>{plate.name}</b> ({plate.vendor})")
        if plate.shape_tczyx is not None:
            t, c, z, y, x = plate.shape_tczyx
            self.dimensions.setText(f"T {t} · C {c} · Z {z} · Y {y} · X {x}")
        else:
            self.dimensions.setText("unknown")
        self.objective.setText(file_metadata.objective_name or "unknown")
        self.bit_depth.setText(
            str(file_metadata.bit_depth) if file_metadata.bit_depth else "unknown"
        )
        self._show(plate.metadata)
        changed = self._diff(plate.metadata)
        self.status.setText(f"Overridden: {', '.join(sorted(changed))}" if changed else "")

    def _show(self, metadata: ImageMetadata) -> None:
        for key, _ in _NUMERIC:
            self.edits[key].setText(_fmt(getattr(metadata, key)))
        self.modality.setCurrentText(metadata.modality)
        self.channels.setRowCount(len(metadata.channels))
        for row, channel in enumerate(metadata.channels):
            self.channels.setItem(row, 0, QTableWidgetItem(channel.name))
            for column, value in ((1, channel.excitation_nm), (2, channel.emission_nm)):
                item = QTableWidgetItem(_fmt(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.channels.setItem(row, column, item)

    def _diff(self, metadata: ImageMetadata) -> dict[str, Any]:
        """Fields where ``metadata`` differs from the file values."""
        assert self._file_metadata is not None
        base = self._file_metadata
        changed: dict[str, Any] = {
            key: getattr(metadata, key)
            for key in (*[k for k, _ in _NUMERIC], "modality")
            if getattr(metadata, key) != getattr(base, key)
        }
        names = [c.name for c in metadata.channels]
        if names != [c.name for c in base.channels]:
            changed["channel_names"] = names
        return changed

    def current_overrides(self) -> dict[str, Any]:
        """Overrides implied by the widgets. Raises ``ValueError`` on invalid input."""
        if self._file_metadata is None:
            return {}
        values: dict[str, Any] = {}
        for key, label in _NUMERIC:
            text = self.edits[key].text().strip().replace(",", ".")
            if not text:
                values[key] = None
                continue
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number") from None
            if number <= 0:
                raise ValueError(f"{label} must be positive")
            values[key] = number
        values["modality"] = self.modality.currentText()
        names = []
        for row in range(self.channels.rowCount()):
            item = self.channels.item(row, 0)
            name = item.text().strip() if item is not None else ""
            if not name:
                raise ValueError(f"Channel {row + 1} needs a name")
            names.append(name)
        values["channel_names"] = names
        return self._diff(self._file_metadata.with_overrides(values))

    def save(self) -> None:
        try:
            overrides = self.current_overrides()
        except ValueError as exc:
            self.status.setText(f"Not saved: {exc}")
            return
        self.overrides_saved.emit(overrides)

    def reset(self) -> None:
        if self._file_metadata is not None:
            self._show(self._file_metadata)
            self.overrides_saved.emit({})
