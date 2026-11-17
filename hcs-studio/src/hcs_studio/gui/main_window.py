"""Main application window: browser, plate map, viewer, metadata and log."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
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
from hcs_studio.gui.metadata_panel import MetadataPanel
from hcs_studio.gui.pattern_dialog import PatternDialog
from hcs_studio.gui.pipeline_panel import PipelinePanel
from hcs_studio.gui.plate_panel import PlatePanel
from hcs_studio.gui.psf_dialog import PsfDialog
from hcs_studio.gui.sam_settings import Sam3SettingsDialog
from hcs_studio.gui.segmentation_panel import SegmentationPanel
from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.gui.workers import PlateLoader, PreviewWorker
from hcs_studio.io.base import detect_reader
from hcs_studio.io.generic import TiffFolderReader
from hcs_studio.io.sidecar import load_overrides, save_overrides
from hcs_studio.model.layout import WellAnnotation, read_plate_layout
from hcs_studio.model.plate import Plate, well_name
from hcs_studio.pipeline.executor import PipelineRunner, ProcessResult
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import ImageData
from hcs_studio.utils.gpu import GpuStatus, detect_gpu
from hcs_studio.utils.synthetic import make_cell_field


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
        self.tabs.addTab(self.viewer_panel.widget, "Viewer")
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
        self.tabifyDockWidget(self.metadata_dock, self.pipeline_dock)
        self.tabifyDockWidget(self.pipeline_dock, self.segmentation_dock)
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
        self.segmentation_panel.model_trained.connect(self._on_model_trained)

        self._build_menus()
        self.log(f"HCS Studio {__version__} started. {self.gpu_status.summary}")
        for note in self.gpu_status.notes:
            self.log(f"GPU detection: {note}")

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
        file_menu.addAction(self._action("Import plate &layout (CSV)…", self._choose_layout))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Quit", self.close, "Ctrl+Q"))

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self._action("&PSF tool…", self.open_psf_tool))
        tools_menu.addAction(self._action("&SAM 3 settings…", self.open_sam_settings))

        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self.browser_dock,
            self.metadata_dock,
            self.pipeline_dock,
            self.segmentation_dock,
            self.log_dock,
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
        file, _ = QFileDialog.getOpenFileName(self, "Import plate layout", "", "CSV (*.csv)")
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
        self.current_field = None
        self._fit_cache.clear()
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
        self.segmentation_panel.refresh_layers()
        panel.set_histograms(raw, result.final)
        steps = " → ".join(result.final.history) or "no steps"
        if result.final.linear:
            note = "intensities remain linear (usable for measurement)"
        else:
            note = "DISPLAY ONLY: not for intensity measurement"
        details = [
            f"{c['channel']}: {c['mode']}, {c['iterations']} iterations, residual "
            f"{c['residual']:.3f}, {c['seconds']:.1f} s, {c['tiles']} tile(s) on {r['backend']}"
            for r in result.final.reports
            if r.get("module") == "deconvolution"
            for c in r["channels"]
        ]
        panel.status.setText(
            f"Preview: {steps}. {note}. Plate-level estimates use up to "
            f"{self.PREVIEW_FIT_FIELDS} fields in previews; first timepoint shown."
            + "".join(f"\nDeconvolution {d}" for d in details)
        )
        for line in details:
            self.log(f"Deconvolution {line}")
        self.tabs.setCurrentWidget(self.viewer_panel.widget)
        self.log(f"Preview: {steps} ({note})")

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
        """Attach a plate layout CSV to the open plate (tooltips and control colours)."""
        if self.plate is None:
            self.log("Open a plate before importing a layout.")
            return
        try:
            layout = read_plate_layout(path)
        except (OSError, ValueError) as exc:
            self.log(f"Layout not imported: {exc}")
            return
        missing = sorted(well_name(*k) for k in layout if k not in self.plate.wells)
        self.layout_annotations = layout
        self._refresh_plate()
        self.log(f"Imported layout for {len(layout)} wells from {path.name}")
        if missing:
            self.log(f"Layout wells with no images on this plate: {', '.join(missing)}")

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
        self.plate_panel.thumbnails.set_plate(None)
        self.loader.wait()
        self.preview.wait()
        self.plate_panel.thumbnails.wait()
        self.viewer_panel.close()
        super().closeEvent(event)
