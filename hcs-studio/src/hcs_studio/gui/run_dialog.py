"""Run dialog: choose where a pipeline runs (field, well, selected wells, plate, plates)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.model.plate import Plate, well_name
from hcs_studio.pipeline.jobs import Job
from hcs_studio.pipeline.schema import Pipeline

SCOPES = ("field", "well", "selected", "plate", "plates")


class RunDialog(QDialog):
    def __init__(
        self,
        plate: Plate,
        pipeline: Pipeline,
        *,
        current_field: tuple[int, int, int] | None,
        selected_wells: list[tuple[int, int]],
        layout_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run pipeline")
        self.resize(560, 460)
        self.plate, self.pipeline = plate, pipeline
        self.current_field, self.selected = current_field, selected_wells
        self.layout_path = layout_path

        steps = len(pipeline.enabled_steps())
        header = QLabel(f"<b>{pipeline.name}</b> ({steps} steps) on <b>{plate.name}</b>")
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QRadioButton] = {}
        texts = {
            "field": "Current field" + (self._field_text() if current_field else " (none open)"),
            "well": "Current well"
            + (f" ({well_name(*current_field[:2])})" if current_field else " (none open)"),
            "selected": f"Selected wells ({len(selected_wells)})",
            "plate": f"Whole plate ({len(plate.wells)} wells, {plate.n_fields} fields)",
            "plates": "Whole plate, and these other plates:",
        }
        for scope in SCOPES:
            button = QRadioButton(texts[scope])
            self.group.addButton(button)
            self.buttons[scope] = button
        self.buttons["field"].setEnabled(current_field is not None)
        self.buttons["well"].setEnabled(current_field is not None)
        self.buttons["selected"].setEnabled(bool(selected_wells))
        default = "selected" if len(selected_wells) > 1 else "plate"
        self.buttons[default].setChecked(True)

        self.other_plates = QListWidget()
        self.other_plates.setMaximumHeight(90)
        add = QPushButton("Add plate…")
        add.clicked.connect(self._add_plate)
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self.other_plates.takeItem(self.other_plates.currentRow()))

        self.output = QLineEdit()
        self.output.setPlaceholderText("next to the raw data")
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._choose_output)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("auto")
        self.workers.setValue(0)
        self.checksums = QCheckBox("Record SHA-256 checksums of the input files (slower)")
        self.checksums.setChecked(True)
        self.message = QLabel("")
        self.message.setStyleSheet("color: #b00020;")

        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        output_row.addWidget(browse)
        plates_row = QHBoxLayout()
        plates_row.addWidget(add)
        plates_row.addWidget(remove)
        form = QFormLayout()
        form.addRow("Output folder", output_row)
        form.addRow("Worker processes", self.workers)
        form.addRow(self.checksums)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Add to job queue")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        for scope in SCOPES:
            layout.addWidget(self.buttons[scope])
        layout.addWidget(self.other_plates)
        layout.addLayout(plates_row)
        layout.addLayout(form)
        layout.addWidget(self.message)
        layout.addWidget(box)
        self.jobs: list[Job] = []

    def _field_text(self) -> str:
        assert self.current_field is not None
        row, col, position = self.current_field
        index = self.plate.wells[(row, col)].fields[position].index
        return f" ({well_name(row, col)}, field {index})"

    def scope(self) -> str:
        return next(s for s, b in self.buttons.items() if b.isChecked())

    def add_other_plate(self, path: Path) -> None:
        self.other_plates.addItem(str(path))
        self.buttons["plates"].setChecked(True)

    def _add_plate(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Another plate folder")
        if folder:
            self.add_other_plate(Path(folder))

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder")
        if folder:
            self.output.setText(folder)

    def build_jobs(self) -> list[Job]:
        """Jobs for the chosen scope. Raises ValueError when the choice is incomplete."""
        scope = self.scope()
        common: dict[str, Any] = {
            "pipeline": self.pipeline.to_dict(),
            "output_root": self.output.text().strip() or None,
            "workers": self.workers.value() or None,
            "checksums": self.checksums.isChecked(),
        }
        plate = self.plate
        base = Job(
            plate_source=plate.source,
            metadata=plate.metadata,
            layout_path=str(self.layout_path) if self.layout_path else None,
            **common,
        )
        if not plate.source:
            raise ValueError("This plate cannot be reopened by a background job")
        if scope == "field":
            assert self.current_field is not None
            row, col, position = self.current_field
            index = plate.wells[(row, col)].fields[position].index
            return [
                _with(
                    base,
                    wells=[(row, col)],
                    fields={(row, col): [index]},
                    label=f"{plate.name} {well_name(row, col)} field {index}",
                )
            ]
        if scope == "well":
            assert self.current_field is not None
            key = self.current_field[:2]
            return [_with(base, wells=[key], label=f"{plate.name} {well_name(*key)}")]
        if scope == "selected":
            if not self.selected:
                raise ValueError("No wells are selected on the plate map")
            return [
                _with(
                    base,
                    wells=list(self.selected),
                    label=f"{plate.name} ({len(self.selected)} wells)",
                )
            ]
        jobs = [_with(base, label=f"{plate.name} (whole plate)")]
        if scope == "plates":
            paths = [
                Path(self.other_plates.item(i).text()) for i in range(self.other_plates.count())
            ]
            if not paths:
                raise ValueError("Add at least one other plate, or choose 'Whole plate'")
            for path in paths:
                jobs.append(
                    Job(
                        plate_source={"path": str(path)},
                        label=f"{path.name} (whole plate)",
                        **common,
                    )
                )
        return jobs

    def _accept(self) -> None:
        try:
            self.jobs = self.build_jobs()
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.accept()


def _with(job: Job, **changes: Any) -> Job:
    from dataclasses import replace

    return replace(job, **changes)
