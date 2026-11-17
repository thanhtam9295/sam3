"""Start the Qt application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from hcs_studio.gui.main_window import MainWindow


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("HCS Studio")
    window = MainWindow()
    window.show()
    window.show_test_image(switch_tab=False)
    return int(app.exec())
