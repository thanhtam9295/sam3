"""Start the Qt application, with an application log and an error handler that reports
unexpected errors instead of closing the window."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import traceback
from pathlib import Path
from types import TracebackType

from PySide6.QtWidgets import QApplication, QMessageBox

from hcs_studio import __version__
from hcs_studio.gui.main_window import MainWindow
from hcs_studio.io.sidecar import config_dir

log = logging.getLogger("hcs_studio")


def app_log_path() -> Path:
    return config_dir() / "logs" / "hcs-studio.log"


def setup_logging() -> Path:
    """Log to ``<settings>/logs/hcs-studio.log`` (rotated at 5 MB, 3 files kept)."""
    path = app_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in log.handlers):
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024**2, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return path


def install_error_handler(window: MainWindow | None) -> None:
    """Unexpected exceptions (e.g. in a slot) are logged and shown; the app keeps running."""

    def handle(kind: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        text = "".join(traceback.format_exception(kind, value, tb))
        log.error("Unexpected error:\n%s", text)
        if window is not None:
            window.log(f"Unexpected error: {kind.__name__}: {value} (details in {app_log_path()})")
            box = QMessageBox(window)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("HCS Studio: unexpected error")
            box.setText(f"{kind.__name__}: {value}")
            box.setInformativeText(
                "The application keeps running; your data was not changed. Details are in "
                f"{app_log_path()} — please include that file when reporting the problem."
            )
            box.setDetailedText(text)
            box.open()  # non-blocking

    sys.excepthook = handle


def run_gui(smoke_test: bool = False) -> int:
    """Run the application. ``smoke_test``: show the window and the test image, then quit
    (exit code 0 when everything loaded)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("HCS Studio")
    setup_logging()
    log.info("HCS Studio %s starting (Python %s)", __version__, sys.version.split()[0])
    window = MainWindow()
    install_error_handler(window)
    window.show()
    window.show_test_image(switch_tab=False)
    if smoke_test:
        from PySide6.QtCore import QTimer

        def finish() -> None:
            layers = window.viewer_panel.layer_names()
            print(f"Smoke test: window shown, viewer layers {layers}", flush=True)
            app.exit(0 if layers else 3)

        QTimer.singleShot(2000, finish)
    code = int(app.exec())
    log.info("HCS Studio closed (exit code %d)", code)
    return code
