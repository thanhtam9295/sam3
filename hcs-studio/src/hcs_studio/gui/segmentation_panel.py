"""Segmentation dock: edit labels, SAM 3 prompts, training pairs and Cellpose fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.processing.base import ImageData, ModuleError
from hcs_studio.segmentation import objects
from hcs_studio.segmentation.sam3_backend import Sam3Settings, get_client, to_rgb
from hcs_studio.segmentation.training import save_training_pair

SPLIT_SEEDS = "split seeds"
SAM_POINTS = "SAM points (+)"
SAM_NEGATIVE = "SAM points (−)"
SAM_BOXES = "SAM boxes"
EDIT_HINT = (
    "Paint, erase and fill with the napari tools in the Viewer (select the labels layer). "
    "To split an object, add 2+ points inside it to the 'split seeds' layer."
)


def parse_ids(text: str) -> list[int]:
    try:
        return [int(part) for part in text.replace(";", ",").replace(" ", ",").split(",") if part]
    except ValueError:
        raise ModuleError(f"Object IDs must be whole numbers: {text!r}") from None


class SegmentationPanel(QWidget):
    """``model_trained(path)`` is emitted when fine-tuning has written a model."""

    model_trained = Signal(object)

    def __init__(
        self,
        viewer: ViewerPanel,
        current_image: Callable[[], ImageData | None],
        log: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._current_image = current_image
        self._log = log
        #: Set by tests to use the SAM worker's stand-in model.
        self.fake_sam = False
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_task_finished)
        self.task.failed.connect(lambda message: self.set_status(f"Failed: {message}"))
        self._on_done: Callable[[Any], None] | None = None

        self.target = QComboBox()
        self.target.setToolTip("Labels layer that edits and SAM results go into.")
        refresh = QPushButton("↻")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self.refresh_layers)
        self.new_name = QLineEdit("objects")
        new_button = QPushButton("New empty layer")
        new_button.clicked.connect(self.new_layer)

        # Editing
        self.ids = QLineEdit()
        self.ids.setPlaceholderText("object IDs, e.g. 4, 7")
        merge = QPushButton("Merge")
        merge.clicked.connect(self.merge)
        delete = QPushButton("Delete")
        delete.clicked.connect(self.delete)
        self.split_id = QSpinBox()
        self.split_id.setRange(1, 10_000_000)
        seeds = QPushButton("Add seed layer")
        seeds.clicked.connect(lambda: self._prompt_layer("points", SPLIT_SEEDS))
        split = QPushButton("Split")
        split.clicked.connect(self.split)
        hint = QLabel(EDIT_HINT)
        hint.setWordWrap(True)
        hint.setEnabled(False)

        # SAM 3
        self.prompt = QLineEdit("nucleus")
        self.sam_channels = QLineEdit("1")
        self.sam_channels.setToolTip("1-3 channel numbers shown to SAM 3 (grey or RGB).")
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.5)
        concept = QPushButton("Segment all instances")
        concept.clicked.connect(self.sam_concept)
        prompt_layers = QPushButton("Add prompt layers")
        prompt_layers.clicked.connect(self.add_sam_prompt_layers)
        from_prompts = QPushButton("Segment from points/boxes")
        from_prompts.clicked.connect(self.sam_prompts)

        # Training
        self.training_folder = QLineEdit(str(Path.home() / "hcs_training"))
        choose = QPushButton("…")
        choose.setFixedWidth(32)
        choose.clicked.connect(self._choose_folder)
        self.training_channels = QLineEdit("1")
        save_pair = QPushButton("Save as training pair")
        save_pair.clicked.connect(self.save_training)
        self.base_model = QComboBox()
        self.base_model.addItems(["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"])
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2000)
        self.epochs.setValue(100)
        train = QPushButton("Fine-tune Cellpose")
        train.clicked.connect(self.fine_tune)

        self.status = QLabel("Preview a segmentation step, or create an empty labels layer.")
        self.status.setWordWrap(True)

        def hbox(*widgets: QWidget) -> QHBoxLayout:
            layout = QHBoxLayout()
            for widget in widgets:
                layout.addWidget(widget)
            return layout

        layers = QFormLayout()
        layers.addRow("Labels layer", hbox(self.target, refresh))
        layers.addRow("New layer", hbox(self.new_name, new_button))
        edit_box = QGroupBox("Edit objects")
        edit = QFormLayout(edit_box)
        edit.addRow(hbox(self.ids, merge, delete))
        edit.addRow("Split object", hbox(self.split_id, seeds, split))
        edit.addRow(hint)
        sam_box = QGroupBox("SAM 3 (interactive)")
        sam = QFormLayout(sam_box)
        sam.addRow("Concept", self.prompt)
        sam.addRow("Channels", self.sam_channels)
        sam.addRow("Min. score", self.confidence)
        sam.addRow(hbox(concept))
        sam.addRow(hbox(prompt_layers, from_prompts))
        train_box = QGroupBox("Training data and fine-tuning")
        training = QFormLayout(train_box)
        training.addRow("Folder", hbox(self.training_folder, choose))
        training.addRow("Image channels", self.training_channels)
        training.addRow(hbox(save_pair))
        training.addRow("Base model", self.base_model)
        training.addRow("Epochs", self.epochs)
        training.addRow(hbox(train))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(layers)
        layout.addWidget(edit_box)
        layout.addWidget(sam_box)
        layout.addWidget(train_box)
        layout.addWidget(self.status)
        layout.addStretch(1)

    # ---------------------------------------------------------------- helpers

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def refresh_layers(self) -> None:
        current = self.target.currentText()
        self.target.clear()
        self.target.addItems(self.viewer.labels_names())
        if current:
            self.target.setCurrentText(current)

    def _layer(self) -> str:
        name = self.target.currentText()
        if not name or name not in self.viewer.labels_names():
            raise ModuleError("Choose a labels layer first")
        return name

    def _run(self, action: Callable[[], None]) -> None:
        try:
            action()
        except ModuleError as exc:
            self.set_status(str(exc))

    def _channels(self, text: str) -> list[int]:
        return [i - 1 for i in parse_ids(text)]

    def _prompt_layer(self, kind: str, name: str) -> Any:
        try:
            return self.viewer.prompt_layer(kind, name, self._layer())
        except ModuleError as exc:
            self.set_status(str(exc))
            return None

    def _image(self) -> ImageData:
        image = self._current_image()
        if image is None:
            raise ModuleError("Open a field in the viewer first")
        return image

    # ---------------------------------------------------------------- editing

    def new_layer(self) -> None:
        def action() -> None:
            image = self._image()
            name = self.new_name.text().strip() or "objects"
            self.viewer.add_empty_labels(name, image.data.shape[1:], image.metadata)
            self.refresh_layers()
            self.target.setCurrentText(name)
            self.set_status(f"Created an empty labels layer '{name}'.")

        self._run(action)

    def merge(self) -> None:
        def action() -> None:
            name = self._layer()
            ids = parse_ids(self.ids.text())
            self.viewer.set_labels(name, objects.merge_labels(self.viewer.get_labels(name), ids))
            self.set_status(f"Merged objects {ids} in '{name}'.")

        self._run(action)

    def delete(self) -> None:
        def action() -> None:
            name = self._layer()
            ids = parse_ids(self.ids.text())
            self.viewer.set_labels(name, objects.delete_labels(self.viewer.get_labels(name), ids))
            self.set_status(f"Deleted objects {ids} from '{name}'.")

        self._run(action)

    def split(self) -> None:
        def action() -> None:
            name = self._layer()
            if SPLIT_SEEDS not in self.viewer.viewer.layers:
                raise ModuleError("Add a seed layer and place 2+ points inside the object")
            points = np.asarray(self.viewer.viewer.layers[SPLIT_SEEDS].data)
            labels = self.viewer.get_labels(name)
            seeds = (
                points if points.shape[1] == 3 else np.column_stack([np.zeros(len(points)), points])
            )
            label_id = int(self.split_id.value())
            self.viewer.set_labels(name, objects.split_label(labels, label_id, seeds))
            self.viewer.viewer.layers[SPLIT_SEEDS].data = np.zeros((0, points.shape[1]))
            self.set_status(f"Split object {label_id} in '{name}'.")

        self._run(action)

    # ------------------------------------------------------------------ SAM 3

    def _sam_client(self) -> Any:
        import sys

        settings = Sam3Settings.load()
        if self.fake_sam:
            return get_client(sys.executable, "", "cpu", fake=True)
        if not settings.configured:
            raise ModuleError("Set up SAM 3 first: Tools → SAM 3 settings")
        device = settings.device or "auto"
        return get_client(settings.python, settings.checkpoint, device)

    def _sam_inputs(self) -> tuple[str, int, Any]:
        if not self.fake_sam and not Sam3Settings.load().configured:
            raise ModuleError("Set up SAM 3 first: Tools → SAM 3 settings")
        name = self._layer()
        image = self._image()
        z = self.viewer.current_z(name)
        rgb = to_rgb(image.data[:, z], self._channels(self.sam_channels.text()))
        return name, z, rgb

    def _add_objects(self, name: str, z: int, new: Any, replace_layer: bool) -> int:
        """Paint SAM objects into the layer as new IDs (existing objects are kept)."""
        labels = self.viewer.get_labels(name)
        plane = np.zeros_like(labels[z]) if replace_layer else labels[z].copy()
        start = int(np.max(labels))
        added = 0
        for value in np.unique(new):
            if value == 0:
                continue
            free = (new == value) & (plane == 0)
            if free.any():
                added += 1
                plane[free] = start + added
        labels[z] = plane
        self.viewer.set_labels(name, labels)
        return added

    def sam_concept(self) -> None:
        def action() -> None:
            name, z, rgb = self._sam_inputs()
            prompt, threshold = self.prompt.text().strip(), self.confidence.value()
            if not prompt:
                raise ModuleError("Type a concept, e.g. 'nucleus' or 'mitotic cell'")

            def work() -> Any:
                client = self._sam_client()
                client.set_image(
                    rgb, key=rgb.tobytes()[:0] + bytes(str(hash(rgb.tobytes())), "ascii")
                )
                return client.text(prompt, threshold)

            def done(result: Any) -> None:
                added = self._add_objects(name, z, result.labels, replace_layer=True)
                self.set_status(f"SAM 3 found {added} '{prompt}' object(s) on plane {z}.")
                self._log(f"SAM 3 concept '{prompt}': {added} object(s) in '{name}'")

            self._start(work, done, f"SAM 3: segmenting every '{prompt}' …")

        self._run(action)

    def add_sam_prompt_layers(self) -> None:
        for kind, layer in (
            ("points", SAM_POINTS),
            ("points", SAM_NEGATIVE),
            ("shapes", SAM_BOXES),
        ):
            self._prompt_layer(kind, layer)
        self.set_status(
            "Click positive/negative points, or draw rectangles, then 'Segment from points/boxes'."
        )

    def sam_prompts(self) -> None:
        def action() -> None:
            name, z, rgb = self._sam_inputs()
            layers = self.viewer.viewer.layers

            def yx(layer_name: str) -> list[list[float]]:
                if layer_name not in layers:
                    return []
                data = np.asarray(layers[layer_name].data)
                return [[float(p[-1]), float(p[-2])] for p in data]  # (x, y) for SAM

            positive, negative = yx(SAM_POINTS), yx(SAM_NEGATIVE)
            boxes = []
            if SAM_BOXES in layers:
                for shape in layers[SAM_BOXES].data:
                    corners = np.asarray(shape)[:, -2:]
                    (y0, x0), (y1, x1) = corners.min(axis=0), corners.max(axis=0)
                    boxes.append([float(x0), float(y0), float(x1), float(y1)])
            if not positive and not boxes:
                raise ModuleError("Add at least one positive point or one box")

            def work() -> Any:
                client = self._sam_client()
                client.set_image(rgb)
                results = []
                if positive:
                    client.reset()
                    results.append(
                        client.points(
                            positive + negative, [1] * len(positive) + [0] * len(negative)
                        ).labels
                    )
                for box in boxes:
                    client.reset()
                    results.append(client.box(box).labels)
                return results

            def done(results: Any) -> None:
                added = 0
                for result in results:
                    added += self._add_objects(name, z, result, replace_layer=False)
                for layer_name in (SAM_POINTS, SAM_NEGATIVE, SAM_BOXES):
                    if layer_name in layers:
                        layers[layer_name].data = (
                            []
                            if layer_name == SAM_BOXES
                            else np.zeros((0, layers[layer_name].ndim))
                        )
                self.set_status(f"SAM 3 added {added} object(s) to '{name}' on plane {z}.")
                self._log(f"SAM 3 prompts: added {added} object(s) to '{name}'")

            self._start(work, done, "SAM 3: segmenting from prompts …")

        self._run(action)

    # --------------------------------------------------------------- training

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Training folder", self.training_folder.text()
        )
        if folder:
            self.training_folder.setText(folder)

    def save_training(self, field_name: str | None = None) -> None:
        def action() -> None:
            name = self._layer()
            image = self._image()
            files = save_training_pair(
                Path(self.training_folder.text()),
                field_name
                or f"{name}_{len(list(Path(self.training_folder.text()).glob('*_img.tif'))):04d}",
                image.data,
                self.viewer.get_labels(name),
                self._channels(self.training_channels.text()),
            )
            self.set_status(
                f"Saved {len(files)} training pair(s) to {self.training_folder.text()}."
            )
            self._log(f"Training data: {', '.join(f.name for f in files)}")

        self._run(action)

    def fine_tune(self) -> None:
        from hcs_studio.segmentation.cellpose_backend import fine_tune

        folder = Path(self.training_folder.text())
        base, epochs = self.base_model.currentText(), self.epochs.value()

        def done(result: Any) -> None:
            path, history = result
            loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
            self.set_status(f"Model saved to {path} (final training loss {loss:.3f}).")
            self._log(f"Fine-tuned {base} on {history['images_train']} image(s): {path}")
            self.model_trained.emit(path)

        self._start(
            lambda: fine_tune(folder, base_model=base, epochs=epochs),
            done,
            f"Fine-tuning {base} for {epochs} epochs …",
        )

    # ---------------------------------------------------------------- tasks

    def _start(self, work: Callable[[], Any], done: Callable[[Any], None], message: str) -> None:
        self._on_done = done
        self.set_status(message)
        self.task.request(work)

    def _on_task_finished(self, result: Any) -> None:
        if self._on_done is not None:
            self._on_done(result)
