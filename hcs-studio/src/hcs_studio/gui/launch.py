"""Start the Qt application, with an application log and an error handler that reports
unexpected errors instead of closing the window."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from hcs_studio import __version__
from hcs_studio.gui.main_window import MainWindow
from hcs_studio.gui.opengl import (
    SOFTWARE_GL_ENV,
    prepare_opengl,
    software_marker,
    use_software_opengl,
)
from hcs_studio.io.sidecar import config_dir

log = logging.getLogger("hcs_studio")
SMOKE_TIMEOUT_S = 120.0  # `gui --smoke-test` exits with code 4 when it has not finished by then


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


def is_opengl_error(exc: BaseException) -> bool:
    """The viewer could not get a working OpenGL context (PyOpenGL ``GLError``, vispy)."""
    for error in (exc, exc.__cause__, exc.__context__):
        if error is None:
            continue
        module = type(error).__module__ or ""
        if module.startswith(("OpenGL", "vispy")) or "OpenGL" in str(error):
            return True
    return False


def _restart_with_software_opengl() -> bool:
    """Start a new copy of the app with software OpenGL (installed app only)."""
    if not getattr(sys, "frozen", False):
        return False
    import subprocess

    env = {**os.environ, SOFTWARE_GL_ENV: "1"}
    subprocess.Popen([sys.executable, *sys.argv[1:]], env=env)
    return True


def run_gui(smoke_test: bool = False) -> int:
    """Run the application. ``smoke_test``: show the window and the test image, then quit
    (exit code 0 when everything loaded; 4 when stuck; 5 when OpenGL failed)."""
    software = use_software_opengl()
    opengl = prepare_opengl()  # normally done already by app.main, before Qt was imported
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("HCS Studio")
    setup_logging()
    log.info(
        "HCS Studio %s starting (Python %s%s)",
        __version__,
        sys.version.split()[0],
        f", {opengl}" if opengl else "",
    )
    if not smoke_test:
        return _run(app, smoke_test, software)
    import threading

    def stuck() -> None:  # e.g. a modal error dialog: fail instead of waiting forever
        print(f"Smoke test: no result after {SMOKE_TIMEOUT_S} s", flush=True)
        os._exit(4)

    watchdog = threading.Timer(SMOKE_TIMEOUT_S, stuck)
    watchdog.daemon = True
    watchdog.start()
    try:
        return _run(app, smoke_test, software)
    finally:
        watchdog.cancel()


def _run(app: QCoreApplication, smoke_test: bool, software: bool) -> int:
    if not smoke_test:
        from hcs_studio.gui.security_dialogs import require_sign_in

        if not require_sign_in():
            return 1
    try:
        window = MainWindow()
    except Exception as exc:
        if not is_opengl_error(exc) or software:
            raise
        log.exception("The image viewer could not start with the graphics driver")
        software_marker().parent.mkdir(parents=True, exist_ok=True)
        software_marker().write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        if smoke_test:
            print(f"Smoke test: OpenGL failed ({type(exc).__name__}); next start uses "
                  f"software OpenGL", flush=True)  # fmt: skip
            return 5
        restarted = _restart_with_software_opengl()
        QMessageBox.warning(
            None,
            "HCS Studio: graphics",
            "The graphics driver could not start the image viewer (common in remote desktop "
            "sessions and virtual machines). HCS Studio now uses software rendering"
            + (": it is starting again." if restarted else "; please start it again.")
            + " Updating the graphics driver gives faster rendering; then delete "
            f"{software_marker()} to use it.",
        )
        return 0 if restarted else 2
    install_error_handler(window)
    if not smoke_test:
        from hcs_studio.gui.security_dialogs import connect_server

        connect_server(window)
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
