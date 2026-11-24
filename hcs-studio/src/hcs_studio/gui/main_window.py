"""Main application window: browser, plate map, viewer, metadata and log."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QWidget,
)

from hcs_studio import __version__
from hcs_studio.gui.browser import BrowserPanel
from hcs_studio.gui.classify_panel import ClassifyPanel
from hcs_studio.gui.compare_panel import ComparePanel
from hcs_studio.gui.experiments_panel import ExperimentsPanel
from hcs_studio.gui.ilastik_settings import IlastikSettingsDialog
from hcs_studio.gui.jobs_panel import JobsPanel
from hcs_studio.gui.layout_editor import LayoutEditor
from hcs_studio.gui.measurements_panel import MeasurementsPanel
from hcs_studio.gui.metadata_panel import MetadataPanel
from hcs_studio.gui.models_panel import ModelsPanel
from hcs_studio.gui.pattern_dialog import PatternDialog
from hcs_studio.gui.pharmacology_panel import PharmacologyPanel
from hcs_studio.gui.pipeline_panel import PipelinePanel
from hcs_studio.gui.plate_panel import PlatePanel
from hcs_studio.gui.profiles_panel import ProfilesPanel
from hcs_studio.gui.psf_dialog import PsfDialog
from hcs_studio.gui.qc_panel import QcPanel
from hcs_studio.gui.results_panel import ResultsPanel
from hcs_studio.gui.run_dialog import RunDialog
from hcs_studio.gui.sam_settings import Sam3SettingsDialog
from hcs_studio.gui.screen_panel import ScreenPanel
from hcs_studio.gui.segmentation_panel import SegmentationPanel
from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.gui.workers import BackgroundTask, PlateLoader, PreviewWorker
from hcs_studio.io.base import detect_reader
from hcs_studio.io.generic import TiffFolderReader
from hcs_studio.io.layout_io import load_layout
from hcs_studio.io.layout_store import plate_key
from hcs_studio.io.sidecar import load_overrides, save_overrides
from hcs_studio.model.annotation import PlateLayout, WellAnnotation
from hcs_studio.model.plate import Plate, parse_well_name, well_name
from hcs_studio.pipeline.executor import PipelineRunner, ProcessResult
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import ImageData
from hcs_studio.utils.gpu import GpuStatus, detect_gpu
from hcs_studio.utils.synthetic import make_cell_field


def manifest_steps(folder: Path) -> list[str]:
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    return [
        s["module"] for s in manifest.get("pipeline", {}).get("steps", []) if s.get("enabled", True)
    ]


class MainWindow(QMainWindow):
    """Data browser (left), plate map and viewer (centre), metadata (right), log (bottom)."""

    def __init__(self, gpu_status: GpuStatus | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"HCS Studio {__version__}")
        self.resize(1500, 950)

        self.gpu_status = gpu_status if gpu_status is not None else detect_gpu()
        #: The open plate with overrides applied, and as read from the files.
        self.plate: Plate | None = None
        self._file_plate: Plate | None = None
        self.layout_annotations: dict[tuple[int, int], WellAnnotation] = {}
        #: The layout runs use (saved version or imported file) and its file.
        self.layout_document: PlateLayout | None = None
        #: The layout stored in ``layout_path``.
        self._run_layout: PlateLayout | None = None
        #: (row, col, field position) of the field shown in the viewer.
        self.current_field: tuple[int, int, int] | None = None
        #: Last preview of the current field, and the raw field once loaded.
        self._preview: ProcessResult | None = None
        self._raw_field: ImageData | None = None
        self._fit_cache: dict[str, Any] = {}

        self.viewer_panel = ViewerPanel()
        self.plate_panel = PlatePanel()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.plate_panel, "Plate")
        self.layout_editor = LayoutEditor(self.log)
        self.tabs.addTab(self.layout_editor, "Layout")
        self.tabs.addTab(self.viewer_panel.widget, "Viewer")
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.results_panel, "Results")
        self.profiles_panel = ProfilesPanel()
        self.tabs.addTab(self.profiles_panel, "Profiles")
        self.pharmacology_panel = PharmacologyPanel(self.log)
        self.tabs.addTab(self.pharmacology_panel, "Pharmacology")
        self.qc_panel = QcPanel()
        self.qc_panel.log.connect(self.log)
        self.qc_panel.reviewed.connect(lambda folder: self.results_panel.load(Path(folder)))
        self.tabs.addTab(self.qc_panel, "QC")
        self.screen_panel = ScreenPanel()
        self.screen_panel.log.connect(self.log)
        self.tabs.addTab(self.screen_panel, "Screen")
        self.models_panel = ModelsPanel()
        self.models_panel.log.connect(self.log)
        self.tabs.addTab(self.models_panel, "Models")
        #: The last wells selected anywhere (run folder, wells), for the Compare tab.
        self.last_selection: tuple[Path, list[str]] | None = None
        self.compare_panel = ComparePanel(self.log, lambda: self.last_selection)
        self.tabs.addTab(self.compare_panel, "Compare")
        self.experiments_panel = ExperimentsPanel()
        self.experiments_panel.open_run.connect(self.show_results)
        self.experiments_panel.log.connect(self.log)
        self.tabs.addTab(self.experiments_panel, "Experiments")
        self.setCentralWidget(self.tabs)

        self.browser = BrowserPanel()
        self.metadata_panel = MetadataPanel()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.browser_dock = self._add_dock(
            "Data browser", "browserDock", self.browser, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.metadata_dock = self._add_dock(
            "Metadata", "metadataDock", self.metadata_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.log_dock = self._add_dock(
            "Log", "logDock", self.log_view, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.pipeline_panel = PipelinePanel()
        self.pipeline_dock = self._add_dock(
            "Pipeline", "pipelineDock", self.pipeline_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.segmentation_panel = SegmentationPanel(self.viewer_panel, self.current_image, self.log)
        self.segmentation_dock = self._add_dock(
            "Segmentation",
            "segmentationDock",
            self.segmentation_panel,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.measurements_panel = MeasurementsPanel(self.viewer_panel, self.log)
        self.measurements_dock = self._add_dock(
            "Measurements",
            "measurementsDock",
            self.measurements_panel,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.classify_panel = ClassifyPanel(self.viewer_panel, self.log)
        self.classify_dock = self._add_dock(
            "Classify", "classifyDock", self.classify_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.analysis_task = BackgroundTask(self)
        self.analysis_task.finished.connect(self._on_analysis_finished)
        self.analysis_task.failed.connect(self._on_analysis_failed)
        self.report_task = BackgroundTask(self)
        self.report_task.finished.connect(self._on_report_finished)
        self.report_task.failed.connect(lambda m: self.log(f"Experiment report failed: {m}"))
        self.jobs_panel = JobsPanel()
        self.jobs_dock = self._add_dock(
            "Jobs", "jobsDock", self.jobs_panel, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.tabifyDockWidget(self.log_dock, self.jobs_dock)
        self.log_dock.raise_()
        self.tabifyDockWidget(self.metadata_dock, self.pipeline_dock)
        self.tabifyDockWidget(self.pipeline_dock, self.segmentation_dock)
        self.tabifyDockWidget(self.segmentation_dock, self.measurements_dock)
        self.tabifyDockWidget(self.measurements_dock, self.classify_dock)
        self.metadata_dock.raise_()
        self.resizeDocks(
            [self.browser_dock, self.metadata_dock], [300, 380], Qt.Orientation.Horizontal
        )
        self.resizeDocks([self.log_dock], [130], Qt.Orientation.Vertical)

        self.gpu_label = QLabel(self.gpu_status.summary)
        self.gpu_label.setObjectName("gpuStatusLabel")
        self.statusBar().addPermanentWidget(self.gpu_label)

        self.loader = PlateLoader(self)
        self.loader.loaded.connect(self._on_plate_loaded)
        self.loader.failed.connect(self._on_plate_failed)
        self.browser.plate_requested.connect(self.open_path)
        self.plate_panel.field_requested.connect(self.open_field)
        self.metadata_panel.overrides_saved.connect(self._on_overrides_saved)
        self.preview = PreviewWorker(self)
        self.preview.finished.connect(self._on_preview_finished)
        self.preview.failed.connect(self._on_preview_failed)
        self.pipeline_panel.preview_requested.connect(self.preview_pipeline)
        self.pipeline_panel.compare_mode_changed.connect(self.viewer_panel.set_compare_mode)
        self.pipeline_panel.profile_requested.connect(self.start_line_profile)
        self.segmentation_panel.model_trained.connect(self._on_model_trained)
        self.pipeline_panel.run_requested.connect(self.open_run_dialog)
        self.results_panel.open_field.connect(self.open_result_field)
        self.results_panel.compare_field.connect(self.compare_result_field)
        self.results_panel.open_mosaic.connect(self.open_mosaic)
        self.results_panel.well_selected.connect(
            lambda well: self.profiles_panel.highlight_wells([well])
        )
        self.profiles_panel.wells_selected.connect(self._on_profile_wells)
        self.pharmacology_panel.wells_selected.connect(self._on_pharmacology_wells)
        self.pharmacology_panel.show_well.connect(self._show_pharmacology_well)
        self.pharmacology_panel.compare_requested.connect(self.compare_wells)
        self.results_panel.compare_wells.connect(self.compare_wells)
        self.profiles_panel.open_object.connect(self.open_result_object)
        self.classify_panel.show_field.connect(self._classify_show_field)
        self.classify_panel.add_steps.connect(self._add_table_steps)
        self.classify_panel.apply_steps.connect(self.apply_table_steps)
        self.jobs_panel.job_finished.connect(self._on_job_finished)
        self.jobs_panel.event_received.connect(self._on_job_event)
        self.plate_panel.plate_map.selection_changed.connect(self._on_selection_changed)
        #: Layout file of the open plate (passed to runs).
        self.layout_path: Path | None = None
        self.layout_editor.layout_saved.connect(self._on_layout_saved)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_menus()
        self.log(f"HCS Studio {__version__} started. {self.gpu_status.summary}")
        for note in self.gpu_status.notes:
            self.log(f"GPU detection: {note}")
        self._fitted = False

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._fitted:  # sizes set before the first show do not stick
            self._fitted = True
            self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Open at 1500 × 950, never larger than the screen (docks may have grown it)."""
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width = min(1500, available.width()) if available else 1500
        height = min(950, available.height()) if available else 950
        self.resize(width, height)

    # ------------------------------------------------------------------ layout

    def _add_dock(
        self, title: str, object_name: str, widget: QWidget, area: Qt.DockWidgetArea
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _action(self, text: str, slot: Any, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._action("Open plate &folder…", self._choose_folder, "Ctrl+O"))
        file_menu.addAction(self._action("Open &image file…", self._choose_file))
        file_menu.addAction(self._action("Import plate &layout…", self._choose_layout))
        file_menu.addAction(self._action("Open &results…", self._choose_results))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Quit", self.close, "Ctrl+Q"))

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self._action("&PSF tool…", self.open_psf_tool))
        tools_menu.addAction(self._action("&SAM 3 settings…", self.open_sam_settings))
        tools_menu.addAction(self._action("&ilastik settings…", self.open_ilastik_settings))
        tools_menu.addAction(
            self._action("&Experiment report for the run in Results", self.experiment_report)
        )
        tools_menu.addAction(self._action("&Assay wizard…", self.open_assay_wizard))
        tools_menu.addAction(self._action("A&udit trail…", self.open_audit_trail))
        tools_menu.addAction(self._action("Si&gn the run in Results…", self.sign_run))

        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self.browser_dock,
            self.metadata_dock,
            self.pipeline_dock,
            self.segmentation_dock,
            self.measurements_dock,
            self.log_dock,
            self.jobs_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._action("Show &test image", self.show_test_image))
        help_menu.addAction(self._action("&About", self._show_about))

    # ----------------------------------------------------------- opening data

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open plate folder")
        if folder:
            self.open_path(Path(folder))

    def _choose_file(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "Open image file", "", "Images (*.nd2 *.ome.tif *.ome.tiff *.htd *.HTD)"
        )
        if file:
            self.open_path(Path(file))

    def _choose_layout(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "Import plate layout", "", "Layouts (*.csv *.xlsx *.xlsm *.json)"
        )
        if file:
            self.import_layout(Path(file))

    def open_path(self, path: Path) -> None:
        """Open a plate folder or image file in the background."""
        path = Path(path)
        opener = None
        try:
            unrecognised = path.is_dir() and detect_reader(path) is None
        except OSError as exc:
            self._on_plate_failed(str(path), str(exc))
            return
        if unrecognised:
            has_tiffs = any(p.suffix.lower() in (".tif", ".tiff") for p in path.iterdir())
            if not has_tiffs:
                self._on_plate_failed(str(path), "no recognised plate data in this folder")
                return
            choice = self._ask_pattern(path)
            if choice is None:
                return
            pattern, pixel_size = choice

            def opener(p: Path) -> Plate:
                return TiffFolderReader.read_with_pattern(p, pattern, pixel_size)

        self.statusBar().showMessage(f"Opening {path} …")
        self.log(f"Opening {path}")
        if opener is None:
            self.loader.request(path)
        else:
            self.loader.request(path, opener)

    def _ask_pattern(self, folder: Path) -> tuple[str, float | None] | None:
        dialog = PatternDialog(folder, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.pattern.text(), dialog.pixel_size_um()

    def _on_plate_loaded(self, plate: Plate) -> None:
        self._file_plate = plate
        self.plate = plate
        try:
            overrides = load_overrides(plate.path)
            if overrides:
                self.plate = replace(plate, metadata=plate.metadata.with_overrides(overrides))
                self.log(f"Applied saved metadata overrides: {', '.join(sorted(overrides))}")
        except (OSError, ValueError) as exc:
            self.log(f"Saved metadata overrides could not be applied: {exc}")
        self.layout_annotations = {}
        self.layout_document = self._run_layout = None
        self.layout_path = None
        self.current_field = None
        self._fit_cache.clear()
        self._restore_layout(plate)
        self._refresh_plate()
        self.tabs.setCurrentWidget(self.plate_panel)
        self.statusBar().showMessage(f"Opened {plate.name}", 5000)
        self.log(
            f"Opened {plate.vendor} plate '{plate.name}': "
            f"{len(plate.wells)} wells, {plate.n_fields} fields"
        )
        for note in plate.notes:
            self.log(f"Note: {note}")

    def _on_plate_failed(self, path: str, message: str) -> None:
        self.statusBar().showMessage("Could not open the selected data", 8000)
        self.log(f"Could not open {path}: {message}")

    def _refresh_plate(self) -> None:
        self.plate_panel.layout_document = self.layout_document
        self.plate_panel.set_plate(self.plate, self.layout_annotations)
        self.metadata_panel.set_plate(
            self.plate, self._file_plate.metadata if self._file_plate else None
        )

    def open_field(self, row: int, col: int, position: int) -> None:
        """Show one field of the open plate in the viewer."""
        if self.plate is None:
            return
        field = self.plate.wells[(row, col)].fields[position]
        try:
            self.viewer_panel.show_field(field.load(), self.plate.metadata)
        except Exception as exc:
            self.log(f"Could not open {well_name(row, col)} field {field.index}: {exc}")
            return
        self.current_field = (row, col, position)
        self._preview, self._raw_field = None, None
        self.segmentation_panel.refresh_layers()
        self.measurements_panel.set_data(self.current_image(), (), {})
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(f"Showing {self.plate.name} · {well_name(row, col)} · field {field.index}")

    # ---------------------------------------------------------- segmentation

    def current_image(self) -> ImageData | None:
        """The field in the viewer as the segmentation input: the last preview, else raw."""
        if self._preview is not None:
            return self._preview.final
        if self.plate is None or self.current_field is None:
            return None
        if self._raw_field is None:
            row, col, position = self.current_field
            field = self.plate.wells[(row, col)].fields[position]
            data = np.asarray(field.load()[0], dtype=np.float32)
            self._raw_field = ImageData(data, self.plate.metadata)
        return self._raw_field

    # ------------------------------------------------------------------ jobs

    def open_run_dialog(self, pipeline: Pipeline) -> RunDialog | None:
        if self.plate is None:
            self.log("Open a plate before running a pipeline.")
            return None
        dialog = RunDialog(
            self.plate,
            pipeline,
            current_field=self.current_field,
            selected_wells=self.plate_panel.plate_map.selected_wells(),
            layout_path=self.layout_for_run(),
            parent=self,
        )
        dialog.accepted.connect(lambda: self.queue_jobs(dialog.jobs))
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def queue_jobs(self, jobs: list[Any]) -> None:
        self.jobs_panel.add_jobs(jobs)
        self.jobs_dock.raise_()
        for job in jobs:
            self.log(f"Queued job: {job.label}")

    def _on_job_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "log" and event.get("level") in ("warning", "error"):
            first = str(event["text"]).strip().splitlines()[0]
            self.log(f"Job {event['level']}: {first}")

    def _on_job_finished(self, state: Any) -> None:
        failed = f", {len(state.failed)} failed well(s)" if state.failed else ""
        self.log(f"Job '{state.job.label}' {state.status}{failed}. Output: {state.folder or '-'}")
        self.statusBar().showMessage(f"Job {state.status}: {state.job.label}", 10000)
        if state.status in ("completed", "completed_with_failures") and state.folder:
            self.show_results(Path(state.folder))

    # --------------------------------------------------------------- results

    def _choose_results(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder")
        if folder:
            self.show_results(Path(folder))

    def open_audit_trail(self) -> None:
        from hcs_studio.gui.security_dialogs import AuditDialog

        dialog = AuditDialog(self)
        dialog.search()
        dialog.open()  # non-blocking

    def sign_run(self) -> None:
        from hcs_studio.gui.security_dialogs import SignRunDialog

        results = self.results_panel.results
        if results is None:
            self.log("Open a run in Results first, then sign it.")
            return
        dialog = SignRunDialog(results.folder, self)
        dialog.signed.connect(lambda _: self.results_panel.load(results.folder))
        dialog.open()

    def open_assay_wizard(self) -> None:
        """Choose an assay, map channels, preview the calibration, apply it to the run open in
        Results (or add it to the pipeline)."""
        from hcs_studio.gui.assay_wizard import AssayWizard

        results = self.results_panel.results
        wizard = AssayWizard(results.folder if results is not None else None, self)
        wizard.log.connect(self.log)
        wizard.add_to_pipeline.connect(self.pipeline_panel.add_step)
        if results is not None:
            wizard.log.connect(lambda _: self.results_panel.load(results.folder))
        self.assay_wizard = wizard
        wizard.show()

    def show_results(self, folder: Path) -> None:
        self.results_panel.load(folder)
        self.classify_panel.set_run(folder)
        self.profiles_panel.set_run(folder)
        self.pharmacology_panel.set_run(folder)
        self.qc_panel.open(folder)
        self.tabs.setCurrentWidget(self.results_panel)
        report = folder / "index.html"
        if report.exists():
            self.log(f"Results of {folder.name}; shareable report: {report}")

    def open_result_field(self, folder: Path, well: str, index: int, stage: str) -> None:
        """Show a field's output image (corrected or enhanced) with its masks in the viewer."""
        from hcs_studio.output.load import load_field_output

        try:
            image = load_field_output(folder, well, index, stage)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            self.log(f"Could not open {well} field {index}: {exc}")
            return
        self.viewer_panel.show_stages([(stage.lower(), image)])
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(f"Showing {stage.lower()} {well} field {index} from {folder.name}")

    def compare_wells(self, folder: Path, wells: list[str]) -> None:
        """Open wells of a run side by side in the Compare tab."""
        self.last_selection = (Path(folder), list(wells))
        self.compare_panel.set_wells(Path(folder), list(wells))
        self.tabs.setCurrentWidget(self.compare_panel)

    def _on_profile_wells(self, folder: Path, wells: list[str]) -> None:
        self.last_selection = (Path(folder), list(wells))
        results = self.results_panel.results
        if results is not None and results.folder == Path(folder):
            self.results_panel.mark_wells(wells)

    def _on_pharmacology_wells(self, run: str, wells: list[str]) -> None:
        """Wells of a clicked curve point or matrix cell: outlined on the Results plate."""
        folder = self.pharmacology_panel.run_folder(run)
        if folder is not None:
            self.last_selection = (folder, list(wells))
        results = self.results_panel.results
        if results is not None and results.folder.name == run:
            self.results_panel.mark_wells(wells)

    def _show_pharmacology_well(self, run: str, well: str) -> None:
        results = self.results_panel.results
        if results is None or results.folder.name != run:
            self.log(f"Open run {run} in the Results tab to see {well}")
            return
        self.results_panel.select_well(well)
        self.tabs.setCurrentWidget(self.results_panel)

    def open_result_object(
        self, folder: Path, well: str, field: int, object_id: int, object_type: str
    ) -> None:
        """Show the field of an object in the viewer with that object highlighted."""
        self.open_result_field(folder, well, field, "Corrected")
        for name in self.viewer_panel.labels_names():
            if name == object_type or name.startswith(f"{object_type} · "):
                self.viewer_panel.highlight_object(name, object_id)
                self.log(f"Object {object_id} of {well} field {field} highlighted")
                return

    def _classify_show_field(self, folder: Path, well: str, index: int) -> None:
        self.open_result_field(folder, well, index, "Corrected")
        self.classify_panel.field_shown(well, index)

    def open_mosaic(self, folder: Path, well: str) -> None:
        """Show a well's stitched mosaic (preview resolution) in the viewer."""
        from PIL import Image

        well_dir = folder / "wells" / well
        try:
            rgb = np.asarray(Image.open(well_dir / "previews" / "mosaic.png").convert("RGB"))
            summary = json.loads((well_dir / "well_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.log(f"Could not open the mosaic of {well}: {exc}")
            return
        stitching = summary.get("stitching") or {}
        self.viewer_panel.show_rgb(f"{well} mosaic", rgb, stitching.get("mosaic_um_per_px"))
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        notes = "; ".join(stitching.get("notes", []))
        self.log(
            f"Mosaic of {well}: {stitching.get('refined_pairs', 0)} overlaps refined"
            + (f" ({notes})" if notes else "")
        )

    def _add_table_steps(self, steps: list[dict[str, Any]]) -> None:
        for step in steps:
            self.pipeline_panel.add_step(step["module"], step["params"])
        self.pipeline_dock.raise_()
        self.log(f"Added {len(steps)} table step(s) to the pipeline (they run after all wells)")

    def apply_table_steps(self, folder: Path, steps: list[dict[str, Any]]) -> None:
        """Apply gates / classifiers to a finished run in the background."""
        from hcs_studio.pipeline.schema import PipelineStep
        from hcs_studio.services import context
        from hcs_studio.services.operations import analyze_run

        try:
            parsed = [PipelineStep.model_validate(s) for s in steps]
        except ValueError as exc:
            self.log(f"Cannot apply: {exc}")
            return
        self.log(f"Applying {len(parsed)} table step(s) to {folder.name}…")
        self.statusBar().showMessage("Applying table steps to the run…")
        ctx = context()
        self.analysis_task.request(lambda: (folder, analyze_run(ctx, folder, parsed)))

    def _on_analysis_finished(self, outcome: tuple[Path, dict[str, Any]]) -> None:
        folder, record = outcome
        self.log(f"Applied {len(record['steps'])} table step(s) to {folder.name}")
        self.statusBar().showMessage("Analysis applied", 5000)
        self.results_panel.load(folder)
        self.classify_panel.refresh()

    def _on_analysis_failed(self, message: str) -> None:
        self.log(f"Applying table steps failed: {message}")
        self.statusBar().showMessage("Analysis failed", 5000)

    def experiment_report(self) -> None:
        """Write the experiment report (HTML + PDF) of the run open in Results."""
        results = self.results_panel.results
        if results is None:
            self.log("Open a run in the Results tab first.")
            return
        folder = results.folder

        def work() -> tuple[Path, list[Path]]:
            from hcs_studio.output.experiment import write_experiment_report

            return folder, write_experiment_report(folder)

        self.statusBar().showMessage("Writing the experiment report…")
        self.report_task.request(work)

    def _on_report_finished(self, outcome: tuple[Path, list[Path]]) -> None:
        folder, files = outcome
        report = next((f for f in files if f.suffix == ".html"), None)
        self.log(f"Experiment report of {folder.name}: {report}")
        self.statusBar().showMessage("Experiment report written", 5000)
        if self.results_panel.results is not None and self.results_panel.results.folder == folder:
            self.results_panel.load(folder)

    def compare_result_field(self, folder: Path, well: str, index: int) -> None:
        """Raw, corrected (incl. deconvolution) and enhanced versions of a field, side by side."""
        from dataclasses import replace as replace_metadata

        from hcs_studio.io.base import reopen_plate
        from hcs_studio.output.load import load_field_output, run_metadata

        stages: list[tuple[str, ImageData]] = []
        try:
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            source = manifest["plate"].get("source") or {"path": manifest["plate"]["path"]}
            plate = reopen_plate(source)
            item = next(f for f in plate.wells[parse_well_name(well)].fields if f.index == index)
            metadata = run_metadata(folder)
            raw = np.asarray(item.load()[0], dtype=np.float32)
            stages.append(("raw", ImageData(raw, replace_metadata(metadata))))
        except Exception as exc:  # raw data moved or unavailable: compare what the run holds
            self.log(f"Raw data not available for comparison: {exc}")
        try:
            corrected = load_field_output(folder, well, index, "Corrected")
            steps = manifest_steps(folder)
            stages.append(
                ("corrected + deconvolved" if "deconvolution" in steps else "corrected", corrected)
            )
            enhanced = load_field_output(folder, well, index, "Enhanced")
            if not enhanced.linear:
                stages.append(("enhanced", enhanced))
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            self.log(f"Could not read the run outputs of {well} field {index}: {exc}")
            return
        self.viewer_panel.show_stages(stages)
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(f"Comparing {', '.join(t for t, _ in stages)} for {well} field {index}")

    def _on_selection_changed(self, wells: list[tuple[int, int]]) -> None:
        if len(wells) > 1:
            self.statusBar().showMessage(f"{len(wells)} wells selected", 4000)

    def open_ilastik_settings(self) -> IlastikSettingsDialog:
        dialog = IlastikSettingsDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def open_sam_settings(self) -> Sam3SettingsDialog:
        dialog = Sam3SettingsDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def _on_model_trained(self, path: Path) -> None:
        if self.pipeline_panel.has_module("segment_cellpose"):
            self.pipeline_panel.set_module_param("segment_cellpose", "model", str(path))
            self.log(f"The Cellpose step now uses the fine-tuned model {path}")
        else:
            self.log(
                f"Fine-tuned model saved: {path}. Add a Cellpose step and set its model to it."
            )

    # ------------------------------------------------------------------- PSF

    def open_psf_tool(self) -> PsfDialog:
        """Open the PSF tool for the open plate; the field in the viewer can be a bead stack."""
        field_data = None
        if self.plate is not None and self.current_field is not None:
            row, col, position = self.current_field
            field_data = self.plate.wells[(row, col)].fields[position].load()
        dialog = PsfDialog(self.plate.metadata if self.plate else None, field_data, self)
        dialog.psf_saved.connect(self._on_psf_saved)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def _on_psf_saved(self, path: Path) -> None:
        self.pipeline_panel.set_module_param("deconvolution", "psf_file", str(path))
        self.log(f"Saved PSF to {path}; the Deconvolution step now uses it.")

    # ---------------------------------------------------------------- preview

    #: Fields sampled per plate-level estimate during previews (full runs use the step setting).
    PREVIEW_FIT_FIELDS = 24

    def preview_pipeline(self, pipeline: Pipeline) -> None:
        """Run ``pipeline`` on the field in the viewer, in the background."""
        panel = self.pipeline_panel
        if self.plate is None or self.current_field is None:
            panel.status.setText("Open a field from the plate map first.")
            return
        plate, layout, cache = self.plate, self.layout_annotations, self._fit_cache
        row, col, position = self.current_field
        field = plate.wells[(row, col)].fields[position]
        limit = self.PREVIEW_FIT_FIELDS

        def work() -> tuple[ImageData, ProcessResult]:
            runner = PipelineRunner(plate, pipeline, layout, fit_field_limit=limit, fit_cache=cache)
            raw = runner.load(field)
            return raw, runner.process(raw)

        panel.status.setText(f"Computing preview for {well_name(row, col)} field {field.index} …")
        self.preview.request(work)

    def _on_preview_finished(self, raw: ImageData, result: ProcessResult) -> None:
        panel = self.pipeline_panel
        self._preview = result
        self.viewer_panel.show_comparison(raw, result.final, panel.compare.currentText())
        if result.final.labels:
            self.viewer_panel.show_labels(result.final.labels, result.final.metadata)
            self.viewer_panel.set_compare_mode(panel.compare.currentText())
            counts = ", ".join(f"{ls.name}: {ls.count}" for ls in result.final.labels)
            self.log(f"Objects: {counts}")
        self.measurements_panel.set_data(result.corrected, result.final.labels, result.tables)
        for table_name, table in result.tables.items():
            self.log(f"Table '{table_name}': {len(table)} row(s)")
        self.segmentation_panel.refresh_layers()
        panel.set_histograms(raw, result.final)
        steps = " → ".join(result.final.history) or "no steps"
        if result.final.linear:
            note = "intensities remain linear (usable for measurement)"
        else:
            note = "DISPLAY ONLY: not for intensity measurement"
        details = [
            f"Deconvolution {c['channel']}: {c['mode']}, {c['iterations']} iterations, residual "
            f"{c['residual']:.3f}, {c['seconds']:.1f} s, {c['tiles']} tile(s) on {r['backend']}"
            for r in result.final.reports
            if r.get("module") == "deconvolution"
            for c in r["channels"]
        ]
        for report in result.final.reports:
            if report.get("module") == "projection":
                details.append(f"projection {report['method']}: planes {report['planes']}")
            if report.get("module") == "background":
                for c in report["channels"]:
                    if "object_mean_change_pct" in c:
                        details.append(
                            f"background ({report['method']}) channel {c['channel']}: mean "
                            f"inside objects {c['object_mean_change_pct']:+.1f} %, background "
                            f"{c['background_mean_change_pct']:+.1f} %"
                        )
                details.extend(f"WARNING {w}" for w in report.get("warnings", []))
        panel.status.setText(
            f"Preview: {steps}. {note}. Plate-level estimates use up to "
            f"{self.PREVIEW_FIT_FIELDS} fields in previews; first timepoint shown."
            + "".join(f"\n{d}" for d in details)
        )
        for line in details:
            self.log(line)
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(f"Preview: {steps} ({note})")

    def start_line_profile(self) -> None:
        """Draw a line in the viewer; the Pipeline dock plots before/after along it."""
        if self._preview is None:
            self.pipeline_panel.status.setText("Preview a step first, then draw the line.")
            return
        pixel = self._preview.final.metadata.pixel_size_um
        self.viewer_panel.start_line_profile(self.pipeline_panel.set_profile_line, pixel)
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log("Line profile: draw a line in the viewer (click the start, double-click the end).")

    def _on_preview_failed(self, message: str) -> None:
        self.pipeline_panel.status.setText(f"Preview failed: {message}")
        self.log(f"Preview failed: {message}")

    def _on_overrides_saved(self, overrides: dict[str, Any]) -> None:
        if self._file_plate is None:
            return
        try:
            metadata = self._file_plate.metadata.with_overrides(overrides)
            where = save_overrides(self._file_plate.path, overrides)
        except (OSError, ValueError) as exc:
            self.log(f"Overrides not saved: {exc}")
            return
        self.plate = replace(self._file_plate, metadata=metadata)
        self._fit_cache.clear()
        self._refresh_plate()
        if overrides:
            self.log(f"Saved metadata overrides ({', '.join(sorted(overrides))}) to {where}")
        else:
            self.log("Metadata reset to the values in the files.")

    def import_layout(self, path: Path) -> None:
        """Attach a layout file (CSV, XLSX, plate map or JSON) to the open plate."""
        if self.plate is None:
            self.log("Open a plate before importing a layout.")
            return
        try:
            imported = load_layout(Path(path))
        except (OSError, ValueError, KeyError) as exc:
            self.log(f"Layout not imported: {exc}")
            return
        layout = imported.layout
        outside = [k for k in layout.wells if k[0] >= self.plate.rows or k[1] >= self.plate.cols]
        if outside:
            self.log(
                f"Layout not imported: it is for a {layout.rows}×{layout.cols} plate; "
                f"this plate is {self.plate.rows}×{self.plate.cols}"
            )
            return
        if (layout.rows, layout.cols) != (self.plate.rows, self.plate.cols):
            layout = PlateLayout.from_dict(
                {**layout.to_dict(), "rows": self.plate.rows, "cols": self.plate.cols}
            )
        self.layout_editor.set_layout(layout, f"Imported {Path(path).name}")
        self._use_layout(layout, Path(path))
        self._refresh_plate()
        missing = sorted(well_name(*k) for k in layout.wells if k not in self.plate.wells)
        self.log(f"Imported layout for {len(layout.wells)} wells from {Path(path).name}")
        for note in imported.notes:
            self.log(f"Layout note: {note}")
        if missing:
            self.log(f"Layout wells with no images on this plate: {', '.join(missing)}")

    def _restore_layout(self, plate: Plate) -> None:
        """Load the latest saved layout version of this plate into the editor and runs."""
        try:
            key = plate_key(plate.path, plate.barcode or "")
            self.layout_editor.set_plate(plate.rows, plate.cols, key)
        except (OSError, ValueError) as exc:
            self.log(f"Saved layout not loaded: {exc}")
            self.layout_editor.set_plate(plate.rows, plate.cols)
            return
        path = self.layout_editor.saved_path
        if path is not None:
            self._use_layout(self.layout_editor.doc, path)
            self.log(f"Using layout version {self.layout_editor.doc.version} ({path})")

    def _use_layout(self, layout: PlateLayout, path: Path) -> None:
        self.layout_document = self._run_layout = layout
        self.layout_annotations = dict(layout.wells)
        self.layout_path = path

    def _on_layout_saved(self, layout: PlateLayout, path: Path) -> None:
        self._use_layout(layout, path)
        self._refresh_plate()
        self.log(f"Saved layout version {layout.version}; new runs use it ({path})")

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.experiments_panel:
            self.experiments_panel.refresh()  # the registry may have changed meanwhile
        # show the editor's (possibly unsaved) annotation on the plate map
        edited = dict(self.layout_editor.doc.wells) != self.layout_annotations
        if self.tabs.widget(index) is self.plate_panel and self.plate is not None and edited:
            self.layout_annotations = dict(self.layout_editor.doc.wells)
            self.layout_document = self.layout_editor.doc
            self._refresh_plate()

    def layout_for_run(self) -> Path | None:
        """The layout file a new run uses; unsaved edits are saved as a new version first."""
        editor = self.layout_editor
        if self.plate is None or not editor.doc.wells:
            return self.layout_path
        unsaved = self.layout_path is None or editor.doc != self._run_layout
        if unsaved and editor.store is not None:
            editor.save_version()
        return self.layout_path

    # ------------------------------------------------------------------- misc

    def show_test_image(self, switch_tab: bool = True) -> None:
        """Display a synthetic two-channel field, to confirm the viewer works."""
        field = make_cell_field()
        self.viewer_panel.show_channels(field.image, field.channel_names, field.pixel_size_um)
        if switch_tab is not False:
            self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(
            f"Showing synthetic test image: {field.image.shape[0]} channels, "
            f"{field.image.shape[1]}×{field.image.shape[2]} px, {field.pixel_size_um} µm/px"
        )

    def log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About HCS Studio",
            f"HCS Studio {__version__}\n\n{self.gpu_status.summary}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.jobs_panel.shutdown()
        self.plate_panel.thumbnails.set_plate(None)
        self.loader.wait()
        self.preview.wait()
        self.plate_panel.thumbnails.wait()
        self.viewer_panel.close()
        super().closeEvent(event)
