"""Pipeline dock: ordered steps, parameters, live preview controls and histograms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pydantic import ValidationError
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.param_form import ParamForm, format_errors
from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.processing.base import MODULES, ImageData, load_builtin_modules

DISPLAY_ONLY_NOTICE = (
    "Steps marked ◐ change intensities non-linearly. Their output is for viewing and "
    "segmentation only. Intensity measurements always use the data from before the first ◐ step."
)
COMPARE_MODES = ("After", "Before", "Side by side")


class PipelinePanel(QWidget):
    """Edits a pipeline; ``preview_requested`` carries the steps up to the selected one."""

    profile_requested = Signal()
    preview_requested = Signal(object)
    compare_mode_changed = Signal(str)
    #: The user wants to run the whole pipeline on a plate (opens the run dialog).
    run_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        load_builtin_modules()
        self._steps: list[dict[str, Any]] = []  # {"module", "enabled", "params"}
        self._loading = False

        self.name_edit = QLineEdit("pipeline")
        self.steps = QListWidget()
        self.steps.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.steps.setToolTip("Drag steps to reorder them; untick a step to skip it.")
        self.steps.currentRowChanged.connect(self._on_row_changed)
        self.steps.itemChanged.connect(self._on_item_changed)
        self.steps.model().rowsMoved.connect(self._on_rows_moved)

        self.module_combo = QComboBox()
        for name in MODULES:
            self.module_combo.addItem(MODULES.get(name).title, name)
        self.add_button = QPushButton("Add step")
        self.add_button.clicked.connect(lambda: self.add_step(self.module_combo.currentData()))
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        self.up_button = QPushButton("▲")
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button = QPushButton("▼")
        self.down_button.clicked.connect(lambda: self.move_selected(1))

        self.form = ParamForm()
        self.form.changed.connect(self._on_params_changed)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #b00020;")
        notice = QLabel(DISPLAY_ONLY_NOTICE)
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #a05a00;")

        self.preview_button = QPushButton("Preview up to selected step")
        self.preview_button.clicked.connect(self.request_preview)
        self.preview_all_button = QPushButton("Preview all steps")
        self.preview_all_button.clicked.connect(self.request_preview_all)
        self.run_button = QPushButton("Run on plate…")
        self.run_button.setToolTip("Run the whole pipeline on wells or plates as a background job.")
        self.run_button.clicked.connect(self.request_run)
        self.live = QCheckBox("Live")
        self.live.setToolTip("Re-run the preview whenever a step or parameter changes.")
        self.compare = QComboBox()
        self.compare.addItems(COMPARE_MODES)
        self.compare.currentTextChanged.connect(self.compare_mode_changed)
        self.status = QLabel("Open a field, then preview the selected step.")
        self.status.setWordWrap(True)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self.request_preview)

        self.histogram_channel = QComboBox()
        self.histogram_channel.currentIndexChanged.connect(self._draw_histograms)
        self.histograms = pg.GraphicsLayoutWidget()
        self.histograms.setMinimumHeight(170)
        self._plots = []
        for index, title in enumerate(("Before", "After")):
            plot = self.histograms.addPlot(row=index, col=0, title=title)
            plot.setLogMode(y=True)
            plot.setMouseEnabled(x=False, y=False)
            self._plots.append(plot)
        self._before: ImageData | None = None
        self._after: ImageData | None = None

        self.profile_button = QPushButton("Line profile…")
        self.profile_button.setToolTip(
            "Draw a line in the viewer to plot the intensity along it, before and after "
            "the previewed steps (channel of the histogram)."
        )
        self.profile_button.clicked.connect(self.profile_requested)
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setMinimumHeight(140)
        self.profile_plot.addLegend(offset=(5, 5))
        self.profile_plot.setLabel("left", "intensity")
        self.profile_plot.setLabel("bottom", "distance")
        self._profile_line: tuple[tuple[float, float], tuple[float, float]] | None = None

        self.load_button = QPushButton("Load…")
        self.load_button.clicked.connect(self._choose_load)
        self.save_button = QPushButton("Save…")
        self.save_button.clicked.connect(self._choose_save)

        def row(*widgets: QWidget) -> QHBoxLayout:
            layout = QHBoxLayout()
            for widget in widgets:
                layout.addWidget(widget)
            return layout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(row(QLabel("Name"), self.name_edit, self.load_button, self.save_button))
        layout.addWidget(self.steps, 2)
        layout.addLayout(row(self.module_combo, self.add_button))
        layout.addLayout(row(self.remove_button, self.up_button, self.down_button))
        layout.addWidget(self.form)
        layout.addWidget(self.error)
        layout.addWidget(notice)
        layout.addLayout(row(self.preview_button, self.preview_all_button, self.live))
        layout.addLayout(row(QLabel("Show"), self.compare, self.run_button))
        layout.addWidget(self.status)
        layout.addLayout(row(QLabel("Histogram of"), self.histogram_channel, self.profile_button))
        layout.addWidget(self.histograms, 2)
        layout.addWidget(self.profile_plot, 1)

    # ------------------------------------------------------------------ steps

    def _label(self, index: int) -> str:
        module = MODULES.get(self._steps[index]["module"])
        mark = "" if module.preserves_linearity else "  ◐ display only"
        return f"{index + 1}. {module.title}{mark}"

    def _rebuild(self, select: int | None = None) -> None:
        self._loading = True
        self.steps.clear()
        for index, step in enumerate(self._steps):
            item = QListWidgetItem(self._label(index))
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if step["enabled"] else Qt.CheckState.Unchecked
            )
            self.steps.addItem(item)
        self._loading = False
        if self._steps:
            self.steps.setCurrentRow(min(select if select is not None else 0, len(self._steps) - 1))
        else:
            self.form.set_model(None)

    def add_step(self, module: str, params: dict[str, Any] | None = None) -> None:
        self._steps.append({"module": module, "enabled": True, "params": dict(params or {})})
        self._rebuild(select=len(self._steps) - 1)
        self._changed()

    def remove_selected(self) -> None:
        row = self.steps.currentRow()
        if row >= 0:
            del self._steps[row]
            self._rebuild(select=max(0, row - 1))
            self._changed()

    def move_selected(self, delta: int) -> None:
        row = self.steps.currentRow()
        target = row + delta
        if row >= 0 and 0 <= target < len(self._steps):
            self._steps[row], self._steps[target] = self._steps[target], self._steps[row]
            self._rebuild(select=target)
            self._changed()

    def _on_rows_moved(self, *args: Any) -> None:
        """Follow a drag-and-drop reorder of the list."""
        order = [
            self.steps.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.steps.count())
        ]
        if sorted(order) != list(range(len(self._steps))):
            return
        current = self.steps.currentRow()
        self._steps = [self._steps[i] for i in order]
        self._rebuild(select=current)
        self._changed()

    def _on_row_changed(self, row: int) -> None:
        if self._loading or row < 0:
            return
        step = self._steps[row]
        self.form.set_model(MODULES.get(step["module"]).Params, step["params"])

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        self._steps[self.steps.row(item)]["enabled"] = item.checkState() == Qt.CheckState.Checked
        self._changed()

    def _on_params_changed(self) -> None:
        row = self.steps.currentRow()
        if row < 0:
            return
        try:
            self._steps[row]["params"] = self.form.values()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self._changed()

    def _changed(self) -> None:
        self.error.setText("")
        if self.live.isChecked():
            self._timer.start()

    def has_module(self, module: str) -> bool:
        return any(step["module"] == module for step in self._steps)

    def set_module_param(self, module: str, key: str, value: Any) -> None:
        """Set a parameter on the selected step of ``module`` (else the first; else add one)."""
        row = self.steps.currentRow()
        if not (0 <= row < len(self._steps) and self._steps[row]["module"] == module):
            matches = [i for i, step in enumerate(self._steps) if step["module"] == module]
            if not matches:
                self.add_step(module)
                matches = [len(self._steps) - 1]
            row = matches[0]
        self._steps[row]["params"] = {**self._steps[row]["params"], key: value}
        self._rebuild(select=row)
        self._changed()

    # --------------------------------------------------------------- pipeline

    def pipeline(self, upto_selected: bool = False) -> Pipeline:
        """The pipeline in the panel. Raises ``ValueError`` when a step is invalid."""
        steps = self._steps
        if upto_selected and self.steps.currentRow() >= 0:
            steps = steps[: self.steps.currentRow() + 1]
        try:
            return Pipeline(
                name=self.name_edit.text().strip() or "pipeline",
                steps=[PipelineStep(**step) for step in steps],
            )
        except ValidationError as exc:
            raise ValueError(format_errors(exc)) from None

    def set_pipeline(self, pipeline: Pipeline) -> None:
        self.name_edit.setText(pipeline.name)
        self._steps = [step.model_dump() for step in pipeline.steps]
        self._rebuild()
        self.error.setText("")

    def load_file(self, path: Path) -> None:
        try:
            self.set_pipeline(Pipeline.load(path))
        except (OSError, ValueError) as exc:
            self.error.setText(f"Could not load {path.name}: {exc}")

    def save_file(self, path: Path) -> None:
        try:
            self.pipeline().save(path)
        except (OSError, ValueError) as exc:
            self.error.setText(f"Not saved: {exc}")

    def _choose_load(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "Load pipeline", "", "Pipelines (*.yaml *.yml *.json)"
        )
        if file:
            self.load_file(Path(file))

    def _choose_save(self) -> None:
        file, _ = QFileDialog.getSaveFileName(
            self, "Save pipeline", "pipeline.yaml", "YAML (*.yaml);;JSON (*.json)"
        )
        if file:
            self.save_file(Path(file))

    # ---------------------------------------------------------------- preview

    def request_preview(self) -> None:
        if not self._steps:
            self.status.setText("Add a step first.")
            return
        try:
            self._on_params_pending()
            pipeline = self.pipeline(upto_selected=True)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.preview_requested.emit(pipeline)

    def request_preview_all(self) -> None:
        if not self._steps:
            self.status.setText("Add a step first.")
            return
        try:
            self._on_params_pending()
            pipeline = self.pipeline()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.preview_requested.emit(pipeline)

    def request_run(self) -> None:
        try:
            self._on_params_pending()
            pipeline = self.pipeline()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        if not pipeline.enabled_steps():
            self.error.setText("The pipeline has no enabled steps.")
            return
        self.run_requested.emit(pipeline)

    def _on_params_pending(self) -> None:
        """Commit the form in case the user clicked Preview without leaving a text field."""
        row = self.steps.currentRow()
        if row >= 0:
            self._steps[row]["params"] = self.form.values()

    def set_histograms(self, before: ImageData, after: ImageData) -> None:
        self._before, self._after = before, after
        names = [c.name for c in after.metadata.channels]
        self.histogram_channel.blockSignals(True)
        current = self.histogram_channel.currentIndex()
        self.histogram_channel.clear()
        self.histogram_channel.addItems(names)
        self.histogram_channel.setCurrentIndex(current if 0 <= current < len(names) else 0)
        self.histogram_channel.blockSignals(False)
        self._draw_histograms()

    def set_profile_line(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """Plot the intensity along ``start`` → ``end`` ((y, x) pixels of the field)."""
        self._profile_line = (start, end)
        self._draw_profile()

    def profile_curves(self) -> dict[str, tuple[Any, Any]]:
        """Distance and values along the drawn line, before and after the preview."""
        from hcs_studio.analysis.line_profile import line_profile

        channel = self.histogram_channel.currentIndex()
        if self._profile_line is None:
            return {}
        curves: dict[str, tuple[Any, Any]] = {}
        for label, image in (("before", self._before), ("after", self._after)):
            if image is None or not 0 <= channel < image.data.shape[0]:
                continue
            stack = image.data[channel]
            name = f"{label} (max over Z)" if stack.shape[0] > 1 else label
            curves[name] = line_profile(
                stack.max(axis=0), *self._profile_line, image.metadata.pixel_size_um
            )
        return curves

    def _draw_profile(self) -> None:
        self.profile_plot.clear()
        unit = "µm" if self._after is not None and self._after.metadata.pixel_size_um else "px"
        self.profile_plot.setLabel("bottom", f"distance ({unit})")
        for (name, (distance, values)), colour in zip(
            self.profile_curves().items(), ((150, 150, 150), (230, 120, 20)), strict=False
        ):
            self.profile_plot.plot(distance, values, pen=pg.mkPen(colour, width=2), name=name)

    def _draw_histograms(self) -> None:
        if self._profile_line is not None:
            self._draw_profile()
        channel = self.histogram_channel.currentIndex()
        for plot, image in zip(self._plots, (self._before, self._after), strict=True):
            plot.clear()
            if image is None or not 0 <= channel < image.data.shape[0]:
                continue
            values = image.data[channel].ravel()
            counts, edges = np.histogram(values, bins=256)
            plot.plot(
                edges,
                np.maximum(counts, 1),
                stepMode="center",
                fillLevel=0,
                brush=(90, 140, 210, 150),
            )
