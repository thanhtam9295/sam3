"""Jobs dock: a queue of pipeline runs, each in its own process, with live progress."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.model.plate import parse_well_name
from hcs_studio.pipeline.jobs import Job, JobProcess
from hcs_studio.pipeline.queue import JobState, LocalJobQueue, recover_interrupted


def job_for_folder(folder: Path, wells: list[str] | None = None) -> Job:
    """A job that resumes an earlier run folder (optionally only some wells)."""
    from hcs_studio.pipeline.schema import Pipeline

    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    pipeline = Pipeline.load(folder / "pipeline.yaml")
    plate = manifest["plate"]
    source = plate.get("source") or {"path": plate["path"]}
    chosen = wells if wells is not None else manifest.get("wells", [])
    return Job(
        plate_source=source,
        pipeline=pipeline.to_dict(),
        wells=[parse_well_name(w) for w in chosen] or None,
        resume=str(folder),
        checksums=manifest.get("input_checksums") == "sha256",
        label=f"{plate['name']} (resume {folder.name})",
    )


class JobsPanel(QWidget):
    """``job_finished(JobState)`` is emitted when a job ends (any status)."""

    job_finished = Signal(object)
    event_received = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: The queue does the scheduling; this panel shows it and forwards the buttons.
        self.queue = LocalJobQueue()
        try:  # jobs left running by a previous session are marked interrupted
            recover_interrupted()
        except Exception as exc:  # the registry is a record; the dock works without it
            import logging

            logging.getLogger(__name__).warning("Job history not updated: %s", exc)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.poll)

        self.jobs_table = QTableWidget(0, 4)
        self.jobs_table.setHorizontalHeaderLabels(["Job", "Status", "Progress", "Run folder"])
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        self.jobs_table.verticalHeader().setVisible(False)
        self.jobs_table.currentCellChanged.connect(lambda *_: self._show_wells())
        self.progress = QProgressBar()
        self.eta = QLabel("No job running.")
        self.wells_table = QTableWidget(0, 4)
        self.wells_table.setHorizontalHeaderLabels(["Well", "Status", "Fields", "Error"])
        self.wells_table.horizontalHeader().setStretchLastSection(True)
        self.wells_table.verticalHeader().setVisible(False)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.retry_button = QPushButton("Retry failed wells")
        self.retry_button.clicked.connect(self.retry_failed)
        self.resume_button = QPushButton("Resume a run folder…")
        self.resume_button.clicked.connect(self._choose_resume)
        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self._open_folder)

        buttons = QHBoxLayout()
        for button in (
            self.pause_button,
            self.cancel_button,
            self.retry_button,
            self.resume_button,
            self.open_button,
        ):
            buttons.addWidget(button)
        top = QHBoxLayout()
        top.addWidget(self.progress, 2)
        top.addWidget(self.eta, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        tables = QHBoxLayout()
        tables.addWidget(self.jobs_table, 3)
        tables.addWidget(self.wells_table, 2)
        layout.addLayout(top)
        layout.addLayout(tables, 1)
        layout.addLayout(buttons)
        self._update_buttons()

    # ----------------------------------------------------------------- queue

    @property
    def states(self) -> list[JobState]:
        return self.queue.states

    @property
    def _current_state(self) -> JobState | None:
        return next((self.queue.status(i) for i in self.queue.running), None)

    @property
    def current(self) -> JobProcess | None:
        return next(iter(self.queue.running.values()), None)

    def add_jobs(self, jobs: list[Job]) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.operations import submit_job

        ctx = context()
        for job in jobs:
            submit_job(ctx, self.queue, job)
        self._after_queue_change()

    def _after_queue_change(self) -> None:
        self._refresh_jobs()  # rows first, then select the running job
        state = self._current_state
        if state is not None:
            self.timer.start()
            self.jobs_table.setCurrentCell(self.states.index(state), 0)

    def poll(self) -> None:
        running_before = set(self.queue.running)
        for state, event in self.queue.poll():
            self.event_received.emit(event)
            if event.get("type") == "finished" and state.status != "running":
                self._refresh_progress(state)
                self.job_finished.emit(state)
            else:
                self._refresh_progress(state)
        if set(self.queue.running) != running_before:
            self._after_queue_change()
        if not self.queue.running:
            self.timer.stop()
            self._update_buttons()

    # --------------------------------------------------------------- control

    def toggle_pause(self) -> None:
        state = self._current_state
        if state is None:
            return
        self.queue.toggle_pause(state.id)
        self._refresh_jobs()

    def cancel(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.operations import cancel_job

        state = self._current_state
        if state is not None:
            cancel_job(context(), self.queue, state.id)
            self._refresh_progress(state)

    def selected_state(self) -> JobState | None:
        row = self.jobs_table.currentRow()
        return self.states[row] if 0 <= row < len(self.states) else None

    def retry_failed(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.operations import retry_job

        state = self.selected_state()
        if state is None or not state.folder or not state.failed:
            self.eta.setText("The selected job has no failed wells.")
            return
        retry_job(context(), self.queue, state.id)
        self._after_queue_change()

    def resume_folder(self, folder: Path) -> None:
        try:
            self.add_jobs([job_for_folder(folder)])
        except (OSError, ValueError, KeyError) as exc:
            self.eta.setText(f"Cannot resume {folder}: {exc}")

    def resume_selected(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.operations import resume_job

        state = self.selected_state()
        if state is not None and state.folder:
            resume_job(context(), self.queue, state.id)
            self._after_queue_change()

    def _choose_resume(self) -> None:
        state = self.selected_state()
        if state is not None and state.status in ("interrupted", "cancelled") and state.folder:
            self.resume_selected()
            return
        folder = QFileDialog.getExistingDirectory(self, "Run folder to resume")
        if folder:
            self.resume_folder(Path(folder))

    def _open_folder(self) -> None:
        state = self.selected_state()
        if state is not None and state.folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(state.folder))

    def shutdown(self) -> None:
        """On application exit: stop the running job (it can be resumed later)."""
        self.timer.stop()
        self.queue.shutdown()

    # --------------------------------------------------------------- display

    def _refresh_jobs(self) -> None:
        self.jobs_table.setRowCount(len(self.states))
        for row, state in enumerate(self.states):
            percent = f"{100 * state.done / state.total:.0f}%" if state.total else ""
            for column, text in enumerate(
                (state.job.label, state.status, percent, state.folder or "")
            ):
                self.jobs_table.setItem(row, column, QTableWidgetItem(text))
        self._update_buttons()

    def _refresh_progress(self, state: JobState) -> None:
        self.progress.setMaximum(max(1, state.total))
        self.progress.setValue(state.done)
        eta = ""
        if state.eta_s is not None and state.status == "running":
            minutes, seconds = divmod(int(state.eta_s), 60)
            eta = (
                f" — about {minutes} min {seconds:02d} s left"
                if minutes
                else f" — about {seconds} s left"
            )
        self.eta.setText(f"{state.status}: {state.done}/{state.total} fields{eta}. {state.message}")
        self._refresh_jobs()
        self._show_wells()

    def _show_wells(self) -> None:
        state = self.selected_state()
        wells = state.wells if state else {}
        self.wells_table.setRowCount(len(wells))
        for row, (name, info) in enumerate(sorted(wells.items())):
            values = (
                name,
                info.get("status", ""),
                str(info.get("fields", "")),
                info.get("error", "") or "",
            )
            for column, text in enumerate(values):
                self.wells_table.setItem(row, column, QTableWidgetItem(text))
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.current is not None
        self.pause_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        if running and self.current is not None:
            paused = self.current.pause.is_set()
            self.pause_button.setText("Continue" if paused else "Pause")
        state = self.selected_state()
        self.retry_button.setEnabled(bool(state and state.failed and not running))
        self.open_button.setEnabled(bool(state and state.folder))
