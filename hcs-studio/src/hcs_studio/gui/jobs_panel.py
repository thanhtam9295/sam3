"""Jobs dock: a queue of pipeline runs, each in its own process, with live progress."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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


@dataclass
class JobState:
    job: Job
    #: queued, running, paused, completed, completed_with_failures, cancelled, error, interrupted
    status: str = "queued"
    folder: str | None = None
    done: int = 0
    total: int = 0
    eta_s: float | None = None
    wells: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    message: str = ""


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
        self.states: list[JobState] = []
        self.current: JobProcess | None = None
        self._current_state: JobState | None = None
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

    def add_jobs(self, jobs: list[Job]) -> None:
        for job in jobs:
            self.states.append(JobState(job=job))
        self._refresh_jobs()
        self._start_next()

    def _start_next(self) -> None:
        if self.current is not None:
            return
        state = next((s for s in self.states if s.status == "queued"), None)
        if state is None:
            self._update_buttons()
            return
        self.current = JobProcess(state.job)
        self._current_state = state
        state.status = "running"
        self.current.start()
        self.timer.start()
        self._refresh_jobs()
        self.jobs_table.setCurrentCell(self.states.index(state), 0)

    def poll(self) -> None:
        process, state = self.current, self._current_state
        if process is None or state is None:
            return
        for event in process.poll():
            self._handle(state, event)
            self.event_received.emit(event)
        if state.status in ("running", "paused") and not process.alive():
            for event in process.poll():  # anything sent just before exiting
                self._handle(state, event)
            if state.status in ("running", "paused"):
                state.status = "interrupted"
                state.message = "The job stopped unexpectedly; resume it from its run folder."
                self._finish(state)

    def _handle(self, state: JobState, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "run_started":
            state.folder = event["folder"]
            state.total = event["total"]
            for well in event["wells"]:
                state.wells[well] = {"status": "queued", "fields": 0, "error": ""}
            for well in event["skipped"]:
                state.wells[well] = {"status": "done (earlier)", "fields": 0, "error": ""}
        elif kind == "well_started":
            state.wells.setdefault(event["well"], {"fields": 0, "error": ""})["status"] = "running"
        elif kind == "field_done":
            state.done, state.total, state.eta_s = event["done"], event["total"], event["eta_s"]
            state.wells.setdefault(event["well"], {"status": "running", "error": ""})["fields"] = (
                state.wells[event["well"]].get("fields", 0) + 1
            )
        elif kind == "well_finished":
            entry = state.wells.setdefault(event["well"], {"fields": 0})
            entry["status"], entry["error"] = event["status"], event.get("error") or ""
        elif kind == "status":
            state.message = event["text"]
        elif kind == "finished":
            state.status = event["status"]
            state.failed = dict(event.get("failed") or {})
            state.folder = event.get("folder", state.folder)
            if event.get("error"):
                state.message = event["error"]
            self._finish(state)
            return
        self._refresh_progress(state)

    def _finish(self, state: JobState) -> None:
        self.timer.stop()
        if self.current is not None:
            self.current.process.join(timeout=5)
        self.current, self._current_state = None, None
        self._refresh_jobs()
        self._refresh_progress(state)
        self.job_finished.emit(state)
        self._start_next()

    # --------------------------------------------------------------- control

    def toggle_pause(self) -> None:
        if self.current is None or self._current_state is None:
            return
        if self.current.pause.is_set():
            self.current.pause.clear()
            self._current_state.status = "running"
        else:
            self.current.pause.set()
            self._current_state.status = "paused"
        self._refresh_jobs()

    def cancel(self) -> None:
        if self.current is not None:
            self.current.pause.clear()
            self.current.cancel.set()
            if self._current_state is not None:
                self._current_state.message = "Cancelling: finishing the wells in progress …"
                self._refresh_progress(self._current_state)

    def selected_state(self) -> JobState | None:
        row = self.jobs_table.currentRow()
        return self.states[row] if 0 <= row < len(self.states) else None

    def retry_failed(self) -> None:
        state = self.selected_state()
        if state is None or not state.folder or not state.failed:
            self.eta.setText("The selected job has no failed wells.")
            return
        retry = replace(
            state.job,
            wells=[parse_well_name(w) for w in state.failed],
            resume=state.folder,
            label=f"{state.job.label} (retry {len(state.failed)} wells)",
        )
        self.add_jobs([retry])

    def resume_folder(self, folder: Path) -> None:
        try:
            self.add_jobs([job_for_folder(folder)])
        except (OSError, ValueError, KeyError) as exc:
            self.eta.setText(f"Cannot resume {folder}: {exc}")

    def resume_selected(self) -> None:
        state = self.selected_state()
        if state is not None and state.folder:
            self.add_jobs(
                [replace(state.job, resume=state.folder, label=f"{state.job.label} (resumed)")]
            )

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
        if self.current is not None:
            self.current.cancel.set()
            self.current.process.join(timeout=10)
            if self.current.alive():
                self.current.kill()

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
            self.pause_button.setText("Continue" if self.current.pause.is_set() else "Pause")
        state = self.selected_state()
        self.retry_button.setEnabled(bool(state and state.failed and not running))
        self.open_button.setEnabled(bool(state and state.folder))
