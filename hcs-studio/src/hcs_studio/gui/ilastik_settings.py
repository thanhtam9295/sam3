"""Tools → ilastik settings: where ilastik and Fiji are, threads, RAM, parallel processes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.segmentation.ilastik_backend.settings import (
    TESTED,
    Detection,
    IlastikSettings,
    detect,
)

INTRO = (
    "HCS Studio applies projects trained in <b>ilastik</b> (<code>.ilp</code>) to whole plates. "
    f"Install the latest stable ilastik (tested: {TESTED}) and choose its program below "
    "(<code>ilastik.exe</code>, <code>run_ilastik.sh</code>, or the Python of an ilastik "
    "conda environment). <b>Fiji</b> with the <i>ilastik</i> update site is optional; it is "
    "used only by the Fiji runner."
)
COLOURS = {"ok": "#1b7f3b", "warning": "#a15c00", "error": "#b00020"}


class IlastikSettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ilastik settings")
        self.resize(660, 420)
        settings = IlastikSettings.load()
        self.detection: Detection | None = None
        intro = QLabel(INTRO)
        intro.setWordWrap(True)

        self.ilastik = QLineEdit(settings.ilastik)
        self.ilastik.setPlaceholderText(r"e.g. C:\Program Files\ilastik-1.4.2\ilastik.exe")
        self.fiji = QLineEdit(settings.fiji)
        self.fiji.setPlaceholderText(r"optional: e.g. C:\Fiji.app")
        self.threads = QSpinBox()
        self.threads.setRange(-1, 1024)
        self.threads.setSpecialValueText("no limit")
        self.threads.setValue(settings.threads)
        self.ram = QSpinBox()
        self.ram.setRange(0, 4_000_000)
        self.ram.setSingleStep(1024)
        self.ram.setSuffix(" MB")
        self.ram.setSpecialValueText("no limit")
        self.ram.setValue(settings.ram_mb)
        self.processes = QSpinBox()
        self.processes.setRange(1, 16)
        self.processes.setValue(settings.max_processes)
        self.processes.setToolTip(
            "ilastik is already multithreaded and memory-hungry: 1 is usually fastest"
        )
        self.runner = QComboBox()
        self.runner.addItems(["headless", "fiji"])
        self.runner.setCurrentText(settings.runner)
        self.version = QLabel("")
        self.version.setWordWrap(True)
        self.fiji_status = QLabel("")
        self.fiji_status.setWordWrap(True)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("ilastik program", self._row(self.ilastik, "ilastik program", folder=False))
        form.addRow("", self.version)
        form.addRow("Threads", self.threads)
        form.addRow("RAM limit", self.ram)
        form.addRow("ilastik processes at once", self.processes)
        form.addRow("Default runner", self.runner)
        form.addRow("Fiji (optional)", self._row(self.fiji, "Fiji folder", folder=True))
        form.addRow("", self.fiji_status)

        self.detect_button = QPushButton("Detect / check installation")
        self.detect_button.clicked.connect(self.run_detection)
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._show_detection)
        self.task.failed.connect(
            lambda message: self.status.setText(f"Detection failed: {message}")
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

    def _row(self, edit: QLineEdit, title: str, folder: bool) -> QHBoxLayout:
        button = QPushButton("…")
        button.setFixedWidth(32)

        def browse() -> None:
            chosen = (
                QFileDialog.getExistingDirectory(self, title)
                if folder
                else QFileDialog.getOpenFileName(self, title)[0]
            )
            if chosen:
                edit.setText(chosen)

        button.clicked.connect(browse)
        layout = QHBoxLayout()
        layout.addWidget(edit)
        layout.addWidget(button)
        return layout

    # ------------------------------------------------------------------ state

    def settings(self) -> IlastikSettings:
        """The settings as entered (raises ValueError when invalid)."""
        settings = replace(
            IlastikSettings(),
            ilastik=self.ilastik.text().strip(),
            fiji=self.fiji.text().strip(),
            threads=self.threads.value(),
            ram_mb=self.ram.value(),
            max_processes=self.processes.value(),
            runner=self.runner.currentText(),  # type: ignore[arg-type]
        )
        settings.validate()
        return settings

    def run_detection(self) -> None:
        try:
            settings = self.settings()  # captured here: the task must not read widgets
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Checking… (ilastik takes a few seconds to start)")
        self.detect_button.setEnabled(False)
        self.task.request(lambda: detect(settings))

    def _show_detection(self, found: Detection) -> None:
        self.detection = found
        self.detect_button.setEnabled(True)
        if found.ilastik and not self.ilastik.text().strip():
            self.ilastik.setText(found.ilastik)
        if found.fiji and not self.fiji.text().strip():
            self.fiji.setText(found.fiji)
        if found.ilastik:
            check = found.version_check
            self.version.setText(
                f'<span style="color:{COLOURS[check.level]}">{check.message}</span>'
            )
        else:
            self.version.setText(f'<span style="color:{COLOURS["error"]}">ilastik not found</span>')
        if not found.fiji:
            self.fiji_status.setText("Fiji not found (optional)")
        elif not found.fiji_launcher:
            self.fiji_status.setText("No Fiji launcher in this folder")
        elif not found.ilastik4ij:
            self.fiji_status.setText(
                "Fiji found, but not the ilastik4ij plugin: Help → Update → Manage update "
                "sites → tick ilastik → restart Fiji"
            )
        else:
            self.fiji_status.setText(f"Fiji with {Path(found.ilastik4ij).name}")
        runners = found.runners()
        self.status.setText(
            "Runners available: " + ", ".join(runners) if runners else "No runner available yet"
        )

    def save(self) -> None:
        try:
            settings = self.settings()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        from hcs_studio.services import context
        from hcs_studio.services.operations import save_ilastik_settings

        save_ilastik_settings(context(), settings)
        self.accept()
