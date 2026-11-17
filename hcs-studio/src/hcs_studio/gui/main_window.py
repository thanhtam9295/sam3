"""Main application window: docks around an embedded napari viewer."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QWidget,
)

from hcs_studio import __version__
from hcs_studio.gui.viewer import ViewerPanel
from hcs_studio.utils.gpu import GpuStatus, detect_gpu
from hcs_studio.utils.synthetic import make_cell_field


def _placeholder(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    label.setMargin(12)
    label.setEnabled(False)
    return label


class MainWindow(QMainWindow):
    """Data browser (left), viewer (centre), metadata (right), log (bottom)."""

    def __init__(self, gpu_status: GpuStatus | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"HCS Studio {__version__}")
        self.resize(1500, 950)

        self.gpu_status = gpu_status if gpu_status is not None else detect_gpu()
        self.viewer_panel = ViewerPanel()
        self.setCentralWidget(self.viewer_panel.widget)

        self.browser_dock = self._add_dock(
            "Data browser",
            "browserDock",
            _placeholder("Open a plate folder to browse wells.\n(Readers arrive in Phase 1.)"),
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.metadata_dock = self._add_dock(
            "Metadata",
            "metadataDock",
            _placeholder("Optics, channels and dimensions of the selected image appear here."),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_dock = self._add_dock(
            "Log", "logDock", self.log_view, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.resizeDocks(
            [self.browser_dock, self.metadata_dock], [280, 280], Qt.Orientation.Horizontal
        )
        self.resizeDocks([self.log_dock], [130], Qt.Orientation.Vertical)

        self.gpu_label = QLabel(self.gpu_status.summary)
        self.gpu_label.setObjectName("gpuStatusLabel")
        self.statusBar().addPermanentWidget(self.gpu_label)

        self._build_menus()
        self.log(f"HCS Studio {__version__} started. {self.gpu_status.summary}")
        for note in self.gpu_status.notes:
            self.log(f"GPU detection: {note}")

    def _add_dock(
        self, title: str, object_name: str, widget: QWidget, area: Qt.DockWidgetArea
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        for dock in (self.browser_dock, self.metadata_dock, self.log_dock):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        test_action = QAction("Show &test image", self)
        test_action.triggered.connect(self.show_test_image)
        help_menu.addAction(test_action)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def show_test_image(self) -> None:
        """Display a synthetic two-channel field, to confirm the viewer works."""
        field = make_cell_field()
        self.viewer_panel.show_channels(field.image, field.channel_names, field.pixel_size_um)
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
        self.viewer_panel.close()
        super().closeEvent(event)
