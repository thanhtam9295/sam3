"""ilastik dock (PROMPT_ILASTIK.md §6, §9): choose a project (file or the model registry), see
whether it fits the current field, preview it, put its preset into the pipeline, export a Fiji
macro, and the "Train in ilastik" round trip (training set → ilastik's window → register)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.processing.base import ImageData

PREVIEW_STEPS = {"pixel": "ilastik_pixel", "autocontext": "ilastik_autocontext"}


def _ask_yes(parent: QWidget, title: str, text: str) -> bool:
    answer = QMessageBox.question(parent, title, text)
    return answer == QMessageBox.StandardButton.Yes


class IlastikPanel(QWidget):
    """``preset_ready(pipeline)`` asks the main window to load a pipeline into the builder."""

    preset_ready = Signal(object)

    def __init__(
        self,
        viewer: ViewerPanel,
        current_image: Callable[[], ImageData | None],
        log: Callable[[str], None],
        plate: Callable[[], Any] | None = None,
        selected_wells: Callable[[], list[tuple[int, int]]] | None = None,
        layout: Callable[[], dict[tuple[int, int], Any] | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._current_image = current_image
        self._log = log
        self._plate = plate or (lambda: None)
        self._selected = selected_wells or (lambda: [])
        self._layout = layout or (lambda: None)
        # injectable for tests
        self.choose_project: Callable[[], str] = lambda: QFileDialog.getOpenFileName(
            self, "ilastik project", "", "ilastik projects (*.ilp)"
        )[0]
        self.choose_folder: Callable[[str], str] = lambda title: QFileDialog.getExistingDirectory(
            self, title
        )
        self.choose_save: Callable[[str, str], str] = lambda title, pattern: (
            QFileDialog.getSaveFileName(self, title, "", pattern)[0]
        )
        self.confirm: Callable[[str, str], bool] = lambda title, text: _ask_yes(self, title, text)
        self.launch: Callable[[list[str]], Any] = lambda command: subprocess.Popen(command)

        self.info: Any = None
        self.last_preview: ImageData | None = None
        self.training_project: Path | None = None
        self.training_set: Path | None = None
        self.ilastik_process: Any = None
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._after: Callable[[Any], None] | None = None
        self._watch = QTimer(self)
        self._watch.setInterval(1000)
        self._watch.timeout.connect(self._check_ilastik)

        self.project = QLineEdit()
        self.project.setPlaceholderText("project file, or registry:NAME@VERSION")
        self.project.editingFinished.connect(self.load_project)
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        self.library = QComboBox()
        self.library.setToolTip("ilastik projects in the model registry (Models tab)")
        self.library.activated.connect(self._from_library)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self.refresh_library)
        self.details = QLabel("Choose a trained ilastik project (.ilp).")
        self.details.setWordWrap(True)
        self.badge = QLabel("")
        self.badge.setWordWrap(True)
        self.export = QComboBox()
        self.export.addItems(["probabilities", "segmentation"])
        self.channels = QLineEdit("1")
        self.channels.setToolTip("Channels sent to ilastik (from 1), as the project was trained")

        preview = QPushButton("Preview on this field")
        preview.clicked.connect(self.preview)
        preset = QPushButton("Use in pipeline")
        preset.setToolTip("Load this project's preset pipeline into the pipeline builder")
        preset.clicked.connect(self.use_preset)
        macro = QPushButton("Export Fiji macro…")
        macro.clicked.connect(self.export_macro)
        register = QPushButton("Add to library…")
        register.setToolTip("Register this project in the model registry (kind ilastik)")
        register.clicked.connect(lambda: self.register_project())

        self.new_workflow = QComboBox()
        self.new_workflow.addItems(["pixel", "autocontext", "objects", "multicut", "tracking"])
        self.new_workflow.setToolTip("Workflow of a new project")
        train = QPushButton("Train in ilastik…")
        train.setToolTip(
            "Write representative fields of the selected wells (all wells if none) as HDF5, "
            "then open ilastik on the project (a new one if the field above is empty)"
        )
        train.clicked.connect(self.train_in_ilastik)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        def hbox(*widgets: QWidget) -> QHBoxLayout:
            box = QHBoxLayout()
            for widget in widgets:
                box.addWidget(widget)
            return box

        project_box = QGroupBox("Project")
        form = QFormLayout(project_box)
        form.addRow("File", hbox(self.project, browse))
        form.addRow("Library", hbox(self.library, refresh))
        form.addRow(self.details)
        form.addRow(self.badge)
        form.addRow("Output", self.export)
        form.addRow("Channels", self.channels)
        form.addRow(hbox(preview, preset))
        form.addRow(hbox(macro, register))
        train_box = QGroupBox("Train in ilastik")
        train_form = QFormLayout(train_box)
        train_form.addRow("New project workflow", self.new_workflow)
        train_form.addRow(train)
        column = QVBoxLayout(self)
        column.addWidget(project_box)
        column.addWidget(train_box)
        column.addWidget(self.status)
        column.addStretch(1)
        self.refresh_library()

    # ------------------------------------------------------------------ project

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        if text:
            self._log(f"ilastik: {text.splitlines()[0]}")

    def _browse(self) -> None:
        chosen = self.choose_project()
        if chosen:
            self.project.setText(chosen)
            self.load_project()

    def refresh_library(self) -> None:
        self.library.clear()
        self.library.addItem("—")
        try:
            from hcs_studio.training.registry import list_models

            for model in list_models(kind="ilastik"):
                if model.status != "retired":
                    self.library.addItem(f"{model.ref} ({model.status})", model.ref)
        except Exception as exc:  # no registry yet: the list stays empty
            self.library.setToolTip(f"Model registry not available: {exc}")

    def _from_library(self, index: int) -> None:
        ref = self.library.itemData(index)
        if ref:
            self.project.setText(str(ref))
            self.load_project()

    def project_path(self) -> Path | None:
        text = self.project.text().strip()
        if not text:
            return None
        if text.startswith("registry:"):
            from hcs_studio.training.registry import resolve_path

            return Path(resolve_path(text, "ilastik"))
        return Path(text)

    def load_project(self) -> None:
        from hcs_studio.segmentation.ilastik_backend.project import read_project

        self.info = None
        self.badge.setText("")
        try:
            path = self.project_path()
            if path is None:
                self.details.setText("Choose a trained ilastik project (.ilp).")
                return
            self.info = read_project(path)
        except (OSError, ValueError) as exc:
            self.details.setText(f"Cannot read the project: {exc}")
            return
        info = self.info
        classes = ", ".join(info.label_names) or "?"
        self.details.setText(
            f"{info.workflow_name or info.workflow} · classes: {classes} · saved with ilastik "
            f"{info.ilastik_version or '?'} · training data: {info.axes or '?'}"
            + (f", {info.channels} channel(s)" if info.channels else "")
        )
        if info.channels:
            self.channels.setText(" ".join(str(c) for c in range(1, info.channels + 1)))
        self.check_fit()

    def selected_channels(self) -> list[int]:
        try:
            return [int(c) for c in self.channels.text().replace(",", " ").split()]
        except ValueError:
            return []

    def check_fit(self) -> bool:
        """The compatibility badge: does the project fit the field in the viewer?"""
        from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes
        from hcs_studio.segmentation.ilastik_backend.project import check_compatible

        image = self._current_image()
        if self.info is None or image is None:
            self.badge.setText("")
            return False
        channels = self.selected_channels() or list(range(1, image.data.shape[0] + 1))
        if any(not 1 <= c <= image.data.shape[0] for c in channels):
            self.badge.setText("✗ channel numbers do not exist in this image")
            return False
        _, z, y, x = image.data.shape
        shape = (1, len(channels), z, y, x)
        problems = check_compatible(self.info, ilastik_axes(shape), len(channels))
        if problems:
            self.badge.setText("✗ " + "; ".join(problems))
            return False
        note = "" if self.info.confirmed else " (training axes not confirmed on real projects)"
        self.badge.setText("✓ fits this field" + note)
        return True

    # ------------------------------------------------------------------ actions

    def _run(self, work: Callable[[], Any], after: Callable[[Any], None], message: str) -> None:
        if self._after is not None:
            self.set_status("Busy: wait for the current ilastik task")
            return
        self._after = after
        self.set_status(message)
        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self._after = None
        self.set_status(f"Failed: {message}")

    def _on_done(self, value: Any) -> None:
        after, self._after = self._after, None
        if after is not None:
            after(value)

    def preview(self) -> None:
        from hcs_studio.processing.base import create_module

        image = self._current_image()
        if image is None or self.info is None:
            self.set_status("Open a field and choose a project first")
            return
        module_name = PREVIEW_STEPS.get(self.info.workflow)
        if module_name is None:
            self.set_status(
                f"Preview is for pixel/autocontext projects; run {self.info.workflow} projects "
                "in a pipeline"
            )
            return
        params = {
            "project": self.project.text().strip(),
            "export": self.export.currentText(),
            "channels": self.selected_channels() or None,
        }
        step = create_module(module_name, params)

        def work() -> ImageData:
            step.validate(image.metadata)
            return step.apply(image, step.fit(None))  # type: ignore[arg-type]

        self._run(work, self._show_preview, "Running ilastik on this field…")

    def _show_preview(self, result: ImageData) -> None:
        self.last_preview = result
        self.viewer.show_field(result.data[None], result.metadata)  # (T=1, C, Z, Y, X)
        names = [c.name for c in result.metadata.channels if c.name.startswith("ilastik_")]
        self.set_status("Preview: " + ", ".join(names))

    def use_preset(self) -> None:
        from hcs_studio.segmentation.ilastik_backend.presets import pixel_preset

        if self.info is None:
            self.set_status("Choose a project first")
            return
        try:
            pipeline = pixel_preset(
                Path(self.project.text().strip()),
                self.info,
                channels=self.selected_channels() or None,
            )
        except ValueError as exc:
            self.set_status(str(exc))
            return
        self.preset_ready.emit(pipeline)
        self.set_status(f"Preset with {len(pipeline.steps)} steps loaded into the pipeline")

    def export_macro(self) -> Path | None:
        from hcs_studio.segmentation.ilastik_backend.job import IlastikError
        from hcs_studio.segmentation.ilastik_backend.runners.fiji import (
            EXPORT_CHOICES,
            project_macro,
        )
        from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings

        path = self.project_path() if self.info is not None else None
        if path is None or self.info is None:
            self.set_status("Choose a project first")
            return None
        target = self.choose_save("Fiji macro", "ImageJ macros (*.ijm)")
        if not target:
            return None
        choices = EXPORT_CHOICES.get(self.info.workflow, {})
        export = self.export.currentText() if self.export.currentText() in choices else None
        settings = IlastikSettings.load()
        try:
            text = project_macro(
                path,
                export=export,
                ilastik=settings.ilastik or None,
                threads=settings.threads,
                ram_mb=settings.ram_mb,
            )
        except IlastikError as exc:
            self.set_status(str(exc))
            return None
        Path(target).write_text(text, encoding="utf-8")
        self.set_status(f"Fiji macro written to {target}")
        return Path(target)

    def register_project(self, path: Path | None = None) -> Any:
        """Add a project to the model registry (kind ilastik), with its metadata."""
        from hcs_studio.segmentation.ilastik_backend.training_set import project_metadata
        from hcs_studio.services import context
        from hcs_studio.services.training import register_model

        path = path or self.project_path()
        if path is None or not path.exists():
            self.set_status("No project file to register")
            return None
        try:
            record = register_model(
                context(),
                path.stem,
                "ilastik",
                path,
                hyperparameters=project_metadata(path, self.training_set),
            )
        except (OSError, ValueError) as exc:
            self.set_status(f"Not registered: {exc}")
            return None
        self.refresh_library()
        self.project.setText(record.ref)
        self.load_project()
        self.set_status(f"Registered as {record.ref} (candidate; promote it in the Models tab)")
        return record

    # ------------------------------------------------------------------ training round trip

    def train_in_ilastik(self) -> None:
        from hcs_studio.segmentation.ilastik_backend import training_set as ts
        from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings, detect

        plate = self._plate()
        if plate is None:
            self.set_status("Open a plate first")
            return
        folder = self.choose_folder("Folder for the training set (not in the raw data)")
        if not folder:
            return
        wells = self._selected() or sorted(plate.wells)
        fields = ts.representative_fields(wells, ts.fields_per_well(plate), self._layout())
        channels = self.selected_channels() or None
        existing = self.project_path() if self.project.text().strip() else None
        project = existing or Path(folder) / "project.ilp"
        workflow = self.info.workflow if self.info is not None else self.new_workflow.currentText()
        settings = IlastikSettings.load()

        def work() -> tuple[Path, list[str]]:
            target = ts.export_training_set(plate.path, fields, Path(folder), channels=channels)
            found = detect(settings)
            if not found.ilastik:
                raise ValueError("ilastik was not found: set it in Tools → ilastik settings")
            return target, ts.ilastik_gui_command(found.ilastik, project, workflow)

        def after(value: tuple[Path, list[str]]) -> None:
            target, command = value
            self.training_set, self.training_project = target, project
            self.ilastik_process = self.launch(command)
            self._watch.start()
            self.set_status(
                f"{len(fields)} field(s) written to {target.parent}. ilastik is open: add these "
                "files as raw data, train, save the project and close ilastik."
            )

        self._run(work, after, f"Writing {len(fields)} training field(s)…")

    def _check_ilastik(self) -> None:
        process = self.ilastik_process
        if process is None or process.poll() is None:
            return
        self._watch.stop()
        self.ilastik_process = None
        self.ilastik_closed()

    def ilastik_closed(self) -> None:
        project = self.training_project
        if project is None or not project.exists():
            self.set_status("ilastik closed without a saved project")
            return
        if self.confirm(
            "Register the project?",
            f"Add {project.name} to the ilastik project library (model registry)?",
        ):
            self.register_project(project)
        else:
            self.project.setText(str(project))
            self.load_project()
