"""Assay wizard (Phase 20): choose an assay, map its channels, check the control wells,
preview the calibration on the controls, then apply it to the run (or add it to the pipeline).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask


class AssayWizard(QDialog):
    #: (module name, params) to add to the pipeline panel.
    add_to_pipeline = Signal(str, object)
    log = Signal(str)

    def __init__(self, folder: Path | None, parent: QWidget | None = None) -> None:
        from hcs_studio.assays.base import assay_names

        super().__init__(parent)
        self.setWindowTitle("Assay wizard")
        self.folder = Path(folder) if folder else None
        self.channels = self._channels()
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._pending: Callable[[Any], None] | None = None
        self.calibration: dict[str, Any] | None = None

        self.assay = QComboBox()
        self.assay.addItems(assay_names())
        self.assay.currentTextChanged.connect(self._build_roles)
        self.roles_form = QFormLayout()
        self.role_combos: dict[str, QComboBox] = {}
        self.controls = QLabel("")
        self.controls.setWordWrap(True)
        self.preview_button = QPushButton("Preview calibration on the controls")
        self.preview_button.clicked.connect(self.preview)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(180)
        self.plot.setLabel("bottom", "DNA content (N)")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Reason (recorded with the analysis)")
        self.apply_button = QPushButton("Apply to this run")
        self.apply_button.clicked.connect(self.apply)
        self.pipeline_button = QPushButton("Add to the pipeline")
        self.pipeline_button.clicked.connect(self._add)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        top = QFormLayout()
        top.addRow("Assay", self.assay)
        layout.addLayout(top)
        layout.addWidget(QLabel("<b>Channels</b>"))
        layout.addLayout(self.roles_form)
        layout.addWidget(QLabel("<b>Control wells</b> (from the plate layout)"))
        layout.addWidget(self.controls)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.plot)
        layout.addWidget(self.summary)
        layout.addWidget(self.reason)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.pipeline_button)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        self._build_roles(self.assay.currentText())
        self._show_controls()
        has_run = self.folder is not None and (self.folder / "wells").is_dir()
        self.preview_button.setEnabled(has_run)
        self.apply_button.setEnabled(has_run)

    # ---------------------------------------------------------------- setup

    def _channels(self) -> list[str]:
        if self.folder is None:
            return []
        try:
            manifest = json.loads((self.folder / "run_manifest.json").read_text("utf-8"))
        except (OSError, ValueError):
            return []
        channels = ((manifest.get("plate") or {}).get("metadata") or {}).get("channels") or []
        return [str(c.get("name", "")) for c in channels if c.get("name")]

    def _module(self) -> Any:
        from hcs_studio.processing.base import MODULES

        return MODULES.get(self.assay.currentText())

    def _build_roles(self, name: str) -> None:
        while self.roles_form.rowCount():
            self.roles_form.removeRow(0)
        self.role_combos = {}
        if not name:
            return
        for role in self._module().roles:
            combo = QComboBox()
            combo.setEditable(True)
            if not role.required:
                combo.addItem("")
            combo.addItems(self.channels)
            guess = next((c for c in self.channels if role.title.lower() in c.lower()), None)
            if role.key == "dna" and guess is None:
                guess = next((c for c in self.channels if c.lower() in ("dapi", "hoechst")), None)
            if guess:
                combo.setCurrentText(guess)
            combo.setToolTip(role.description)
            self.role_combos[role.key] = combo
            self.roles_form.addRow(f"{role.title}{'' if role.required else ' (optional)'}", combo)

    def _show_controls(self) -> None:
        if self.folder is None:
            self.controls.setText("No run open: the step can be added to the pipeline.")
            return
        module = self._module()(self._module().Params())
        module.run_folder = self.folder
        roles = module.well_roles()
        vehicle = sorted(w for w, r in roles.items() if r == "negative_control")
        positive = sorted(w for w, r in roles.items() if r == "positive_control")
        if not roles:
            self.controls.setText(
                "The run has no plate layout: the calibration will use all wells."
            )
        else:
            self.controls.setText(
                f"Vehicle: {', '.join(vehicle) or 'none'}. Positive control: "
                f"{', '.join(positive) or 'none'}. Change them in the Layout tab if wrong."
            )

    def params(self) -> dict[str, Any]:
        return {
            f"{key}_channel": combo.currentText().strip() for key, combo in self.role_combos.items()
        }

    # --------------------------------------------------------------- actions

    def preview(self) -> None:
        folder = self.folder
        cls = self._module()
        params = self.params()

        def work() -> tuple[dict[str, Any], Any]:
            import pandas as pd

            module = cls(cls.Params.model_validate(params))
            module.run_folder = folder
            state = module.fit_table(pd.DataFrame(columns=["object_type"]))
            vehicle = module.control_objects(("negative_control",), [state["dna_column"]])
            values = (
                vehicle[state["dna_column"]].to_numpy(float) * state["scale_to_n"]
                if not vehicle.empty
                else np.array([])
            )
            return module.describe_state(state), values

        def done(result: tuple[dict[str, Any], Any]) -> None:
            state, values = result
            self.calibration = state
            self.plot.clear()
            if len(values):
                counts, edges = np.histogram(values, np.linspace(0, 8, 161))
                self.plot.plot(
                    edges, counts, stepMode="center", fillLevel=0, brush=(80, 120, 200, 120)
                )
                for x in (2, 4):
                    self.plot.addItem(
                        pg.InfiniteLine(
                            x, angle=90, pen=pg.mkPen("#999", style=Qt.PenStyle.DashLine)
                        )
                    )
            fractions = state.get("vehicle_fractions_pct", {})
            source, nuclei = state.get("calibrated_on"), state.get("vehicle_nuclei")
            text = f"Calibrated on {source} ({nuclei} nuclei). Vehicle: " + ", ".join(
                f"{k} {v:.1f} %" for k, v in fractions.items()
            )
            if state.get("warnings"):
                text += (
                    "<br><span style='color:#b00020'>" + "<br>".join(state["warnings"]) + "</span>"
                )
            self.summary.setText(text)
            self.status.setText("Calibration preview ready.")

        self._run(work, done, "Calibrating on the control wells…")

    def apply(self) -> None:
        from hcs_studio.pipeline.schema import PipelineStep
        from hcs_studio.services import context
        from hcs_studio.services.operations import analyze_run

        if self.folder is None:
            return
        folder = self.folder
        step = PipelineStep(module=self.assay.currentText(), params=self.params())
        ctx = context(self.reason.text().strip())

        def work() -> Any:
            return analyze_run(ctx, folder, [step])

        def done(_: Any) -> None:
            self.status.setText(f"Applied to {folder.name}: see assays/{step.module}/report.html")
            self.log.emit(f"Assay {step.module} applied to {folder.name}")

        self._run(work, done, "Applying the assay to the run…")

    def _add(self) -> None:
        self.add_to_pipeline.emit(self.assay.currentText(), self.params())
        self.status.setText("Added to the pipeline panel.")

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], message: str) -> None:
        self._pending = done
        self.status.setText(message)
        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.status.setText(f"Failed: {message}")

    def _on_done(self, result: Any) -> None:
        done, self._pending = self._pending, None
        if done is not None:
            done(result)
