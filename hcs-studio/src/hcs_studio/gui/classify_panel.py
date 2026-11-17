"""Classify dock: gates drawn on plots, a classifier trained on clicked example objects,
and colouring of objects by class or cluster.

Works on a finished run (the one open in the Results tab). Gates and classifiers become
table steps: added to the pipeline, or applied to the run straight away (``analyze_run``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.analysis.classifier import (
    TrainedClassifier,
    labelled_examples,
    robust_stats,
    save_classifier,
    train_classifier,
    uncertain_objects,
)
from hcs_studio.analysis.table_steps import (
    ANNOTATION_PREFIXES,
    feature_columns,
    read_well_objects,
)
from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.gui.workers import BackgroundTask

#: Distinct colours for classes and clusters (colour-blind friendly first).
PALETTE = (
    "#4477aa",
    "#ee6677",
    "#228833",
    "#ccbb44",
    "#66ccee",
    "#aa3377",
    "#bbbbbb",
    "#ee8866",
    "#44bb99",
    "#99ddff",
)
#: Rows read from the plate table for plots and normalisation statistics.
SAMPLE_ROWS = 50_000
NO_Y = "(histogram)"


def sample_objects(folder: Path, max_rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """Every k-th object of a run's plate table, at most about ``max_rows`` (streamed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = folder / "tables" / "per_object_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    file = pq.ParquetFile(path)
    step = max(1, math.ceil(file.metadata.num_rows / max_rows))
    parts, seen = [], 0
    for batch in file.iter_batches(batch_size=65536):
        start = (-seen) % step
        indices = np.arange(start, len(batch), step)
        seen += len(batch)
        if len(indices):
            parts.append(pa.Table.from_batches([batch]).take(indices))
    if not parts:
        return pd.DataFrame()
    frame: pd.DataFrame = pa.concat_tables(parts, promote_options="permissive").to_pandas()
    if "in_overlap_duplicate" in frame:
        frame = frame[~frame["in_overlap_duplicate"].fillna(False).astype(bool)]
    return frame.reset_index(drop=True)


def well_objects(folder: Path, well: str) -> pd.DataFrame:
    """All objects of one well, with any table-step annotations."""
    return read_well_objects(folder / "wells" / well)


class ClassifyPanel(QWidget):
    """``add_steps(list of {module, params})``: add table steps to the pipeline.
    ``apply_steps(folder, list of PipelineStep dicts)``: apply table steps to the run.
    ``show_field(folder, well, field)``: show a field's corrected image and masks."""

    add_steps = Signal(list)
    apply_steps = Signal(object, list)
    show_field = Signal(object, str, int)

    def __init__(
        self, viewer: ViewerPanel, log: Callable[[str], None], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.log = log
        self.folder: Path | None = None
        self.sample = pd.DataFrame()
        self.classifier: TrainedClassifier | None = None
        self.model_path: Path | None = None
        #: Labelled examples: (well, field, object_id) → class.
        self.examples: dict[tuple[str, int, int], str] = {}
        self.shown: tuple[str, int] | None = None
        self._well_cache: dict[str, pd.DataFrame] = {}
        self._suggestions: list[tuple[str, int, int]] = []
        self.trainer = BackgroundTask(self)
        self.trainer.finished.connect(self._on_trained)
        self.trainer.failed.connect(self._on_train_failed)

        self.run_label = QLabel("Open a run in the Results tab.")
        self.run_label.setWordWrap(True)
        self.object_type = QComboBox()
        self.object_type.currentTextChanged.connect(self._on_object_type)
        top = QFormLayout()
        top.addRow(self.run_label)
        top.addRow("Objects", self.object_type)

        self.tabs = QTabWidget()
        for page, title in (
            (self._build_gate_tab(), "Gate"),
            (self._build_classifier_tab(), "Classifier"),
            (self._build_view_tab(), "Colour"),
        ):
            scroll = QScrollArea()  # the dock must fit small screens
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        self.setEnabled(False)

    # ------------------------------------------------------------------ gate tab

    def _build_gate_tab(self) -> QWidget:
        self.feature_x = QComboBox()
        self.feature_y = QComboBox()
        self.feature_x.currentTextChanged.connect(self.redraw_gate)
        self.feature_y.currentTextChanged.connect(self.redraw_gate)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMinimumHeight(220)
        self.region = pg.LinearRegionItem(brush=(68, 119, 170, 50))
        self.region.sigRegionChanged.connect(self._on_region_moved)
        self.roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#4477aa", width=2))
        self.roi.sigRegionChanged.connect(self._on_region_moved)
        self.x_min, self.x_max = self._spin(), self._spin()
        self.y_min, self.y_max = self._spin(), self._spin()
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.editingFinished.connect(self._on_spins_edited)
        self.gate_name = QLineEdit("positive")
        self.gate_result = QLabel("")
        add = QPushButton("Add gate to pipeline")
        add.clicked.connect(lambda: self.add_steps.emit([self.gate_step()]))
        apply = QPushButton("Apply gate to run")
        apply.clicked.connect(lambda: self._apply([self.gate_step()]))

        widget = QWidget()
        form = QFormLayout(widget)
        form.addRow("X feature", self.feature_x)
        form.addRow("Y feature", self.feature_y)
        form.addRow(self.plot)
        ranges = QHBoxLayout()
        for label, spin in (
            ("X ≥", self.x_min),
            ("≤", self.x_max),
            ("Y ≥", self.y_min),
            ("≤", self.y_max),
        ):
            ranges.addWidget(QLabel(label))
            ranges.addWidget(spin)
        form.addRow(ranges)
        form.addRow("Gate name", self.gate_name)
        form.addRow(self.gate_result)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(apply)
        form.addRow(buttons)
        return widget

    @staticmethod
    def _spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e15, 1e15)
        spin.setDecimals(4)
        spin.setKeyboardTracking(False)
        return spin

    def _typed(self) -> pd.DataFrame:
        if self.sample.empty:
            return self.sample
        return self.sample[self.sample["object_type"] == self.object_type.currentText()]

    def _values(self, column: str) -> np.ndarray:
        frame = self._typed()
        if column not in frame:
            return np.zeros(0)
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)

    def redraw_gate(self) -> None:
        self.plot.clear()
        x_name, y_name = self.feature_x.currentText(), self.feature_y.currentText()
        x = self._values(x_name)
        finite = x[np.isfinite(x)]
        if not len(finite):
            self.gate_result.setText("")
            return
        lo, hi = np.percentile(finite, [50, 100])
        if y_name == NO_Y:
            counts, edges = np.histogram(finite, bins=60)
            self.plot.plot(edges, counts, stepMode="center", fillLevel=0, brush=(150, 150, 150))
            self.plot.addItem(self.region)
            self.region.setRegion((lo, hi))
            self.plot.setLabel("bottom", x_name)
            self.plot.setLabel("left", "objects")
            for spin in (self.y_min, self.y_max):
                spin.setEnabled(False)
        else:
            y = self._values(y_name)
            both = np.isfinite(x) & np.isfinite(y)
            keep = np.flatnonzero(both)[:20000]
            self.plot.addItem(
                pg.ScatterPlotItem(
                    x[keep], y[keep], size=3, pen=None, brush=pg.mkBrush(80, 80, 80, 90)
                )
            )
            y_lo, y_hi = np.percentile(y[both], [50, 100]) if both.any() else (0.0, 1.0)
            self.plot.addItem(self.roi)
            self.roi.setPos((lo, y_lo))
            self.roi.setSize((max(hi - lo, 1e-9), max(y_hi - y_lo, 1e-9)))
            self.plot.setLabel("bottom", x_name)
            self.plot.setLabel("left", y_name)
            for spin in (self.y_min, self.y_max):
                spin.setEnabled(True)
        self._on_region_moved()

    def _on_region_moved(self) -> None:
        if self.feature_y.currentText() == NO_Y:
            lo, hi = self.region.getRegion()
            values = [(self.x_min, lo), (self.x_max, hi)]
        else:
            (x0, y0), (w, h) = self.roi.pos(), self.roi.size()
            values = [
                (self.x_min, x0),
                (self.x_max, x0 + w),
                (self.y_min, y0),
                (self.y_max, y0 + h),
            ]
        #: Exact gate bounds (the spin boxes show them rounded).
        self._bounds = {id(spin): float(value) for spin, value in values}
        for spin, value in values:
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._update_gate_result()

    def _on_spins_edited(self) -> None:
        self.set_gate(
            self.x_min.value(), self.x_max.value(), self.y_min.value(), self.y_max.value()
        )

    def set_gate(self, x_min: float, x_max: float, y_min: float = 0, y_max: float = 0) -> None:
        if self.feature_y.currentText() == NO_Y:
            self.region.setRegion((x_min, x_max))
        else:
            self.roi.setPos((x_min, y_min))
            self.roi.setSize((max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)))
        self._on_region_moved()

    def _bound(self, spin: QDoubleSpinBox) -> float:
        return float(getattr(self, "_bounds", {}).get(id(spin), spin.value()))

    def gate_rules(self) -> list[dict[str, Any]]:
        rules = [
            {
                "feature": self.feature_x.currentText(),
                "min": self._bound(self.x_min),
                "max": self._bound(self.x_max),
            }
        ]
        if self.feature_y.currentText() != NO_Y:
            rules.append(
                {
                    "feature": self.feature_y.currentText(),
                    "min": self._bound(self.y_min),
                    "max": self._bound(self.y_max),
                }
            )
        return rules

    def gate_step(self) -> dict[str, Any]:
        name = "".join(c if c.isalnum() else "_" for c in self.gate_name.text().strip())
        return {
            "module": "gate_objects",
            "params": {
                "object_type": self.object_type.currentText(),
                "name": name or "positive",
                "rules": self.gate_rules(),
            },
        }

    def _update_gate_result(self) -> None:
        frame = self._typed()
        if frame.empty:
            return
        inside = np.ones(len(frame), dtype=bool)
        for rule in self.gate_rules():
            values = pd.to_numeric(frame[rule["feature"]], errors="coerce").to_numpy(float)
            inside &= (values >= rule["min"]) & (values <= rule["max"])
        self.gate_result.setText(
            f"Inside the gate: {int(inside.sum())} of {len(frame)} sampled objects "
            f"({100 * inside.mean():.1f} %)"
        )

    # ------------------------------------------------------------ classifier tab

    def _build_classifier_tab(self) -> QWidget:
        self.classes = QListWidget()
        self.classes.setMaximumHeight(90)
        for name in ("positive", "negative"):
            self._add_class(name)
        self.classes.setCurrentRow(0)
        self.new_class = QLineEdit()
        self.new_class.setPlaceholderText("new class")
        add_class = QPushButton("Add class")
        add_class.clicked.connect(lambda: self._add_class(self.new_class.text().strip()))
        self.well = QComboBox()
        self.field = QSpinBox()
        self.field.setRange(0, 100000)
        show = QPushButton("Show field")
        show.clicked.connect(self._request_field)
        self.click_to_label = QCheckBox("Click objects in the viewer to label them")
        self.click_to_label.setChecked(True)
        self.click_to_label.toggled.connect(lambda _: self._install_click())
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Well", "Field", "Object", "Class"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(120)
        self.table.setMaximumHeight(180)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        self.algorithm = QComboBox()
        self.algorithm.addItems(["random_forest", "hist_gradient_boosting"])
        self.normalisation = QComboBox()
        self.normalisation.addItems(["plate_robust_z", "none"])
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train)
        self.accuracy = QLabel("")
        self.accuracy.setWordWrap(True)
        self.confusion = QTableWidget(0, 0)
        self.confusion.setMaximumHeight(120)
        suggest = QPushButton("Suggest uncertain objects")
        suggest.clicked.connect(self.suggest)
        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(110)
        self.suggestions.itemDoubleClicked.connect(self._open_suggestion)
        save = QPushButton("Save classifier…")
        save.clicked.connect(self._choose_save)
        add = QPushButton("Add to pipeline")
        add.clicked.connect(self._add_classifier_step)
        apply = QPushButton("Apply to run")
        apply.clicked.connect(self._apply_classifier)
        self.class_name = QLineEdit("class")

        widget = QWidget()
        form = QFormLayout(widget)
        form.addRow("Classes", self.classes)
        row = QHBoxLayout()
        row.addWidget(self.new_class)
        row.addWidget(add_class)
        form.addRow(row)
        where = QHBoxLayout()
        where.addWidget(self.well)
        where.addWidget(QLabel("field"))
        where.addWidget(self.field)
        where.addWidget(show)
        form.addRow("Well", where)
        form.addRow(self.click_to_label)
        form.addRow(self.table)
        form.addRow(remove)
        form.addRow("Algorithm", self.algorithm)
        form.addRow("Normalisation", self.normalisation)
        form.addRow(self.train_button)
        form.addRow(self.accuracy)
        form.addRow(self.confusion)
        form.addRow(suggest)
        form.addRow(self.suggestions)
        form.addRow("Column name", self.class_name)
        buttons = QHBoxLayout()
        for button in (save, add, apply):
            buttons.addWidget(button)
        form.addRow(buttons)
        return widget

    def _add_class(self, name: str) -> None:
        existing = [self.classes.item(i).text() for i in range(self.classes.count())]
        if not name or name in existing:
            return
        item = QListWidgetItem(name)
        item.setForeground(pg.mkColor(PALETTE[len(existing) % len(PALETTE)]))
        self.classes.addItem(item)
        self.classes.setCurrentItem(item)

    def class_names(self) -> list[str]:
        return [self.classes.item(i).text() for i in range(self.classes.count())]

    def current_class(self) -> str:
        item = self.classes.currentItem()
        return item.text() if item else ""

    def _request_field(self) -> None:
        if self.folder is not None and self.well.currentText():
            self.show_field.emit(self.folder, self.well.currentText(), self.field.value())

    def field_shown(self, well: str, field: int) -> None:
        """The main window shows this field of the run: enable labelling and paint examples."""
        self.shown = (well, field)
        self.well.setCurrentText(well)
        self.field.setValue(field)
        self._install_click()
        self.paint_examples()

    def mask_layer(self) -> str:
        """The viewer's labels layer of the chosen object type ('' if none is shown)."""
        kind = self.object_type.currentText()
        for name in self.viewer.labels_names():
            if name == kind or name.startswith(f"{kind} · "):
                return name
        return ""

    def _install_click(self) -> None:
        layer = self.mask_layer()
        if self.shown is None or not self.click_to_label.isChecked():
            self.viewer.on_object_click(layer, None)
            return
        self.viewer.on_object_click(layer, self.label_object)

    def label_object(self, object_id: int) -> None:
        """Label an object of the shown field with the current class (again: unlabel)."""
        if self.shown is None or not self.current_class():
            return
        key = (self.shown[0], self.shown[1], object_id)
        if self.examples.get(key) == self.current_class():
            del self.examples[key]
        else:
            self.examples[key] = self.current_class()
        self._fill_table()
        self.paint_examples()

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.examples))
        for row, ((well, field, object_id), label) in enumerate(sorted(self.examples.items())):
            for column, value in enumerate((well, field, object_id, label)):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        counts = pd.Series(list(self.examples.values()), dtype=str).value_counts()
        self.accuracy.setText(
            "Examples: " + ", ".join(f"{k} {v}" for k, v in counts.items()) if len(counts) else ""
        )

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        keys = sorted(self.examples)
        for row in rows:
            self.examples.pop(keys[row], None)
        self._fill_table()
        self.paint_examples()

    def paint_examples(self) -> None:
        layer = self.mask_layer()
        if self.shown is None or not layer:
            return
        names = self.class_names()
        categories = {
            object_id: names.index(label)
            for (well, field, object_id), label in self.examples.items()
            if (well, field) == self.shown and label in names
        }
        self.viewer.paint_categories(layer, categories, PALETTE, name="examples")

    def labels_frame(self) -> pd.DataFrame:
        rows = [
            {"well": w, "field": f, "timepoint": 0, "object_id": o, "label": label}
            for (w, f, o), label in self.examples.items()
        ]
        return pd.DataFrame(rows, columns=["well", "field", "timepoint", "object_id", "label"])

    def _objects_for(self, wells: set[str]) -> pd.DataFrame:
        assert self.folder is not None
        parts = []
        for well in sorted(wells):
            if well not in self._well_cache:
                self._well_cache[well] = well_objects(self.folder, well)
            parts.append(self._well_cache[well])
        frame = pd.concat(parts, ignore_index=True)
        return frame[frame["object_type"] == self.object_type.currentText()]

    def train(self) -> None:
        if self.folder is None:
            return
        labels = self.labels_frame()
        if labels["label"].nunique() < 2:
            self.accuracy.setText("Label examples of at least two classes first.")
            return
        objects = self._objects_for(set(labels.well))
        examples = labelled_examples(objects, labels)
        features = feature_columns(self._typed())
        stats = robust_stats(self._typed(), features)
        options: dict[str, Any] = {
            "features": features,
            "algorithm": self.algorithm.currentText(),
            "normalisation": self.normalisation.currentText(),
            "plate_stats": stats,
            "object_type": self.object_type.currentText(),
        }
        self.train_button.setEnabled(False)
        self.accuracy.setText("Training…")
        self.trainer.request(
            lambda: train_classifier(examples, examples["label"].astype(str).tolist(), **options)
        )

    def _on_trained(self, classifier: TrainedClassifier) -> None:
        self.train_button.setEnabled(True)
        self.classifier = classifier
        self.model_path = None
        if classifier.cv_accuracy is None:
            text = "Trained. Too few examples per class for cross-validation (need 2+ each)."
        else:
            text = (
                f"{classifier.cv_folds}-fold cross-validated accuracy "
                f"{classifier.cv_accuracy:.1%} (balanced {classifier.cv_balanced_accuracy:.1%}) "
                f"on {sum(classifier.n_examples.values())} examples."
            )
        self.accuracy.setText(text)
        n = len(classifier.classes)
        self.confusion.setRowCount(n)
        self.confusion.setColumnCount(n)
        self.confusion.setHorizontalHeaderLabels([f"→ {c}" for c in classifier.classes])
        self.confusion.setVerticalHeaderLabels(classifier.classes)
        for i, row in enumerate(classifier.confusion):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.confusion.setItem(i, j, item)
        self.log(f"Classifier trained: {text}")

    def _on_train_failed(self, message: str) -> None:
        self.train_button.setEnabled(True)
        self.accuracy.setText(f"Training failed: {message}")

    def suggest(self) -> None:
        """Active learning: objects of the labelled wells the classifier is least sure of."""
        if self.classifier is None or self.folder is None:
            self.accuracy.setText("Train a classifier first.")
            return
        wells = {w for w, _, _ in self.examples} or {self.well.currentText()}
        objects = self._objects_for(wells)
        stats = robust_stats(self._typed(), self.classifier.features)
        found = uncertain_objects(
            self.classifier, objects, stats, n=15, exclude=self.labels_frame()
        )
        self.suggestions.clear()
        self._suggestions = []
        for record in found.to_dict("records"):
            key = (str(record["well"]), int(record["field"]), int(record["object_id"]))
            self._suggestions.append(key)
            self.suggestions.addItem(
                f"{key[0]} field {key[1]} object {key[2]}: {record['predicted']} "
                f"(margin {record['margin']:.2f})"
            )

    def _open_suggestion(self, item: QListWidgetItem) -> None:
        well, field, object_id = self._suggestions[self.suggestions.row(item)]
        if self.folder is not None:
            self.show_field.emit(self.folder, well, field)
            self.viewer.highlight_object(self.mask_layer(), object_id)

    def save_to(self, path: Path) -> None:
        if self.classifier is None:
            raise ValueError("Train a classifier first")
        save_classifier(self.classifier, path)
        self.model_path = Path(path)
        self.log(f"Classifier saved: {path}")

    def _choose_save(self) -> None:
        if self.classifier is None:
            self.accuracy.setText("Train a classifier first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save classifier", "classifier.joblib", "Classifier (*.joblib)"
        )
        if path:
            self.save_to(Path(path))

    def classifier_step(self) -> dict[str, Any] | None:
        if self.model_path is None:
            if self.classifier is None or self.folder is None:
                self.accuracy.setText("Train a classifier first.")
                return None
            # Keep an unsaved classifier with the run it was trained on.
            target = self.folder / "classifiers" / f"{self.class_name.text() or 'class'}.joblib"
            target.parent.mkdir(exist_ok=True)
            self.save_to(target)
        name = "".join(c if c.isalnum() else "_" for c in self.class_name.text().strip())
        return {
            "module": "classify_objects",
            "params": {
                "object_type": self.object_type.currentText(),
                "name": name or "class",
                "model_path": str(self.model_path),
            },
        }

    def _add_classifier_step(self) -> None:
        step = self.classifier_step()
        if step:
            self.add_steps.emit([step])

    def _apply_classifier(self) -> None:
        step = self.classifier_step()
        if step:
            self._apply([step])

    # ------------------------------------------------------------------ view tab

    def _build_view_tab(self) -> QWidget:
        self.colour_by = QComboBox()
        paint = QPushButton("Colour the shown field")
        paint.clicked.connect(self.paint_field)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        widget = QWidget()
        form = QFormLayout(widget)
        form.addRow("Colour by", self.colour_by)
        form.addRow(paint)
        form.addRow(self.legend)
        hint = QLabel(
            "Classes, gates, clusters and colonies come from table steps of the run. "
            "Show a field (Classifier tab, or Results → Open in viewer) first."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return widget

    def paint_field(self) -> None:
        column = self.colour_by.currentText()
        layer = self.mask_layer()
        if self.folder is None or self.shown is None or not column:
            self.legend.setText("Show a field first.")
            return
        if not layer:
            self.legend.setText(f"The viewer shows no '{self.object_type.currentText()}' masks.")
            return
        well, field = self.shown
        objects = self._objects_for({well})
        objects = objects[objects["field"] == field]
        values = objects[column]
        if values.dropna().empty:
            self.legend.setText(f"No values of {column} in this field.")
            return
        names = sorted({str(v) for v in values.dropna()}, key=_natural)
        categories = {
            int(o): names.index(str(v))
            for o, v in zip(objects["object_id"], values, strict=True)
            if pd.notna(v)
        }
        colours = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        self.viewer.paint_categories(layer, categories, colours, name=column)
        counts = pd.Series([names[c] for c in categories.values()]).value_counts()
        self.legend.setText(
            " ".join(
                f'<span style="color:{colours[i]}">■</span> {n} ({counts.get(n, 0)})'
                for i, n in enumerate(names)
            )
        )

    # ------------------------------------------------------------------- common

    def set_run(self, folder: Path) -> None:
        """Use the objects of a run folder (called when the Results tab opens one)."""
        self.folder = Path(folder)
        self._well_cache.clear()
        self.examples.clear()
        self.classifier, self.model_path, self.shown = None, None, None
        self.sample = sample_objects(self.folder)
        if self.sample.empty:
            self.run_label.setText(f"{self.folder.name}: no object measurements in this run.")
            self.setEnabled(False)
            return
        self.setEnabled(True)
        self.run_label.setText(f"Run: {self.folder.name} ({len(self.sample)} objects sampled)")
        keep = self.object_type.currentText()
        self.object_type.blockSignals(True)
        self.object_type.clear()
        self.object_type.addItems(sorted(self.sample["object_type"].dropna().unique()))
        if keep:
            self.object_type.setCurrentText(keep)
        self.object_type.blockSignals(False)
        self.well.clear()
        self.well.addItems(sorted(self.sample["well"].dropna().unique()))
        self._fill_table()
        self._on_object_type(self.object_type.currentText())

    def refresh(self) -> None:
        """Reload the run's tables (after table steps were applied)."""
        if self.folder is None:
            return
        examples, classifier, path, shown = (
            dict(self.examples),
            self.classifier,
            self.model_path,
            self.shown,
        )
        self.set_run(self.folder)
        self.examples, self.classifier, self.model_path, self.shown = (
            examples,
            classifier,
            path,
            shown,
        )
        self._fill_table()

    def _on_object_type(self, object_type: str) -> None:
        typed = self._typed()
        features = feature_columns(typed) if not typed.empty else []
        for combo, extra in ((self.feature_x, []), (self.feature_y, [NO_Y])):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + features)
            if keep in extra + features:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        annotations = [
            c
            for c in typed.columns
            if c.startswith(ANNOTATION_PREFIXES)
            and not c.endswith(("_probability", "_um", "_um2", "_per_mm2", "_size", "_neighbours"))
            and "_p_" not in c
        ]
        self.colour_by.clear()
        self.colour_by.addItems(annotations)
        self.redraw_gate()
        self._install_click()

    def _apply(self, steps: list[dict[str, Any]]) -> None:
        if self.folder is None:
            return
        self.apply_steps.emit(self.folder, steps)


def _natural(text: str) -> tuple[int, float, str]:
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, 0.0, text)
