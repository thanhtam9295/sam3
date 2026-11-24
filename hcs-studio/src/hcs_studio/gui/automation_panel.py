"""The Automation tab (Phase 19): automatic runs and the lab-event log.

Top: every acquisition handled automatically (status, rule, run folder, detail); open its
results or start it again (with a reason). Bottom: the event-log viewer — every lab event
received or sent (accepted, duplicate, rejected with the schema errors, sent, send failed),
filterable, with the full message of the selected row. The automation itself runs headless
(``hcs-studio auto serve``, on the server or a lab PC); this tab only looks and retries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask

RUN_COLUMNS = ("status", "barcode", "instrument", "time point", "rule", "updated", "detail")
EVENT_COLUMNS = ("time", "dir", "channel", "type", "status", "barcode", "detail")


def _ask_reason(parent: QWidget, key: str) -> str | None:
    text, ok = QInputDialog.getText(parent, "Start again", f"Why should {key} run again?")
    return text.strip() if ok and text.strip() else None


class AutomationPanel(QWidget):
    log = Signal(str)
    open_run = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: asks for the reason of a retry (tests replace it)
        self.reason: Callable[[str], str | None] = lambda key: _ask_reason(self, key)
        self.runs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._loaded)
        self.task.failed.connect(lambda m: self.log.emit(f"Automation: {m}"))

        self.summary = QLabel("")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.open_button = QPushButton("Open results")
        self.open_button.clicked.connect(self._open)
        self.retry_button = QPushButton("Start again…")
        self.retry_button.clicked.connect(self.retry)
        top = QHBoxLayout()
        for widget in (self.summary, self.refresh_button, self.open_button, self.retry_button):
            top.addWidget(widget)
        top.insertStretch(1, 1)
        self.run_table = self._table(RUN_COLUMNS)

        self.status_filter = QComboBox()
        self.status_filter.addItems(
            ["all", "accepted", "duplicate", "rejected", "ignored", "sent", "send_failed"]
        )
        self.status_filter.currentTextChanged.connect(lambda _: self._fill_events())
        self.direction_filter = QComboBox()
        self.direction_filter.addItems(["in and out", "in", "out"])
        self.direction_filter.currentTextChanged.connect(lambda _: self._fill_events())
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Lab events:"))
        filters.addWidget(self.status_filter)
        filters.addWidget(self.direction_filter)
        filters.addStretch(1)
        self.event_table = self._table(EVENT_COLUMNS)
        self.event_table.itemSelectionChanged.connect(self._show_event)
        self.message = QPlainTextEdit()
        self.message.setReadOnly(True)

        events = QWidget()
        lower = QVBoxLayout(events)
        lower.setContentsMargins(0, 0, 0, 0)
        lower.addLayout(filters)
        split_events = QSplitter()
        split_events.addWidget(self.event_table)
        split_events.addWidget(self.message)
        split_events.setSizes([700, 300])
        lower.addWidget(split_events)
        runs = QWidget()
        upper = QVBoxLayout(runs)
        upper.setContentsMargins(0, 0, 0, 0)
        upper.addLayout(top)
        upper.addWidget(self.run_table)
        from PySide6.QtCore import Qt

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(runs)
        split.addWidget(events)
        layout = QVBoxLayout(self)
        layout.addWidget(split)

    @staticmethod
    def _table(columns: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def refresh(self) -> None:
        from hcs_studio.automation.status import auto_runs, lab_events

        self.task.request(lambda: (auto_runs(300), lab_events(1000)))

    def _loaded(self, result: tuple[list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        self.runs, self.events = result
        counts: dict[str, int] = {}
        for run in self.runs:
            counts[run["status"]] = counts.get(run["status"], 0) + 1
        self.summary.setText(
            ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "No automatic runs yet"
        )
        self.run_table.setRowCount(len(self.runs))
        for i, run in enumerate(self.runs):
            values = (
                run["status"],
                run["barcode"],
                run["instrument"],
                "" if run["time_point"] is None else str(run["time_point"]),
                run["route"],
                f"{run['updated']:%Y-%m-%d %H:%M}",
                run["detail"],
            )
            for j, value in enumerate(values):
                self.run_table.setItem(i, j, QTableWidgetItem(str(value)))
        self._fill_events()

    def visible_events(self) -> list[dict[str, Any]]:
        status = self.status_filter.currentText()
        direction = self.direction_filter.currentText()
        return [
            e
            for e in self.events
            if (status == "all" or e["status"] == status)
            and (direction == "in and out" or e["direction"] == direction)
        ]

    def _fill_events(self) -> None:
        rows = self.visible_events()
        self.event_table.setRowCount(len(rows))
        for i, event in enumerate(rows):
            values = (
                f"{event['time']:%Y-%m-%d %H:%M:%S}",
                event["direction"],
                event["channel"],
                event["type"],
                event["status"],
                event["barcode"],
                event["detail"],
            )
            for j, value in enumerate(values):
                self.event_table.setItem(i, j, QTableWidgetItem(str(value)))
        self.message.clear()

    def _show_event(self) -> None:
        rows = self.event_table.selectionModel().selectedRows()
        if not rows:
            return
        event = self.visible_events()[rows[0].row()]
        text = json.dumps(event["message"], indent=2, ensure_ascii=False, default=str)
        if event["detail"]:
            text = f"{event['status']}: {event['detail']}\n\n{text}"
        self.message.setPlainText(text)

    def selected_run(self) -> dict[str, Any] | None:
        rows = self.run_table.selectionModel().selectedRows()
        return self.runs[rows[0].row()] if rows else None

    def _open(self) -> None:
        run = self.selected_run()
        if run and run["run_folder"]:
            self.open_run.emit(Path(run["run_folder"]))

    def retry(self) -> None:
        run = self.selected_run()
        if run is None:
            return
        reason = self.reason(run["key"])
        if not reason:
            return
        from hcs_studio.services import context
        from hcs_studio.services.automation import retry_auto_run

        try:
            retry_auto_run(context(reason), run["key"])
        except Exception as exc:
            self.log.emit(f"Could not start {run['barcode']} again: {exc}")
            return
        self.log.emit(f"{run['barcode']} will run again at the automation's next round")
        self.refresh()
