"""The Models tab (Phase 25): the model registry and the spot-check queue.

Top: every registered model version with its status, validation and test scores and training
set. Change the status of the selected version with a reason (promotion to production asks for
confirmation — Phase 26 adds roles); compare it with the production version (or its base model)
on held-out ground truth in the background and open the saved report.

Bottom: fields queued for a human spot check (``model_qa`` run step or ``hcs-studio models
spot-check queue``) with their preview and outlines; accept or flag each one (flagging needs a
reason). The accept rate per model is shown above the queue.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask

MODEL_COLUMNS = (
    "model",
    "kind",
    "status",
    "validation",
    "test AP@0.5",
    "training set",
    "created",
    "author",
)
CHECK_COLUMNS = ("id", "barcode", "well", "field", "model")


def _score(metrics: dict[str, Any]) -> str:
    validation = metrics.get("validation", {})
    value = validation.get("overall", {}).get("ap50", validation.get("balanced_accuracy"))
    return f"{value:.3f}" if isinstance(value, float) else "–"


class ModelsPanel(QWidget):
    log = Signal(str)

    def __init__(
        self, parent: QWidget | None = None, confirm: Callable[[str], bool] | None = None
    ) -> None:
        super().__init__(parent)
        self.models: list[Any] = []
        self.checks = pd.DataFrame()
        self.confirm = confirm or self._ask
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._pending: Callable[[Any], None] | None = None

        self.model_table = QTableWidget(0, len(MODEL_COLUMNS))
        self.model_table.setHorizontalHeaderLabels(list(MODEL_COLUMNS))
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.status_combo = QComboBox()
        self.status_combo.addItems(["validated", "production", "retired", "candidate"])
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Reason (required)")
        self.status_button = QPushButton("Set status")
        self.status_button.clicked.connect(self.set_status)
        self.compare_button = QPushButton("Compare on held-out ground truth")
        self.compare_button.clicked.connect(self.compare)
        self.report_button = QPushButton("Open report")
        self.report_button.clicked.connect(self.open_report)

        self.rates = QLabel("")
        self.check_table = QTableWidget(0, len(CHECK_COLUMNS))
        self.check_table.setHorizontalHeaderLabels(list(CHECK_COLUMNS))
        self.check_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.check_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.itemSelectionChanged.connect(self._show_field)
        self.preview = QLabel("Select a spot check to see the field")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(256, 256)
        self.check_reason = QLineEdit()
        self.check_reason.setPlaceholderText("What is wrong (needed to flag)")
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(lambda: self.review("accepted"))
        self.flag_button = QPushButton("Flag")
        self.flag_button.clicked.connect(lambda: self.review("flagged"))
        self.status = QLabel("")
        self.status.setWordWrap(True)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        buttons = QHBoxLayout()
        for widget in (
            self.refresh_button,
            QLabel("Status:"),
            self.status_combo,
            self.reason,
            self.status_button,
            self.compare_button,
            self.report_button,
        ):
            buttons.addWidget(widget)
        top_layout.addWidget(self.model_table)
        top_layout.addLayout(buttons)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.addWidget(QLabel("Spot checks (pending)"))
        bottom_layout.addWidget(self.rates)
        row = QHBoxLayout()
        row.addWidget(self.check_table, 1)
        row.addWidget(self.preview, 1)
        bottom_layout.addLayout(row)
        review = QHBoxLayout()
        for widget in (self.check_reason, self.accept_button, self.flag_button):
            review.addWidget(widget)
        bottom_layout.addLayout(review)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.addWidget(self.status)

    # --- background work ---------------------------------------------------------------

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], message: str) -> None:
        self.status.setText(message)
        self._pending = done
        self.task.request(work)

    def _on_done(self, result: Any) -> None:
        done, self._pending = self._pending, None
        if done is not None:
            done(result)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.status.setText(f"Error: {message}")

    # --- models ---------------------------------------------------------------------------

    def refresh(self) -> None:
        def load() -> tuple[list[Any], pd.DataFrame, pd.DataFrame]:
            from hcs_studio.training.qa import spot_check_summary, spot_checks
            from hcs_studio.training.registry import list_models

            return list_models(), spot_checks("pending"), spot_check_summary()

        self._run(load, self._loaded, "Loading the model registry…")

    def _loaded(self, result: tuple[list[Any], pd.DataFrame, pd.DataFrame]) -> None:
        self.models, self.checks, summary = result
        self.model_table.setRowCount(len(self.models))
        for r, m in enumerate(self.models):
            test = m.metrics.get("test", {}).get("ap50")
            values = (
                f"{m.name}@{m.version}",
                m.kind,
                m.status,
                _score(m.metrics),
                f"{test:.3f}" if isinstance(test, float) else "–",
                m.training_set or "–",
                m.created[:10],
                m.author,
            )
            for c, value in enumerate(values):
                self.model_table.setItem(r, c, QTableWidgetItem(str(value)))
        self.check_table.setRowCount(len(self.checks))
        for r, record in enumerate(self.checks.to_dict("records")):
            for c, column in enumerate(CHECK_COLUMNS):
                self.check_table.setItem(r, c, QTableWidgetItem(str(record[column])))
        self.rates.setText(
            "; ".join(
                f"{s['model']}: {s['accepted']} accepted, {s['flagged']} flagged"
                + (f" ({s['accept_rate']:.0%})" if pd.notna(s["accept_rate"]) else "")
                for s in summary.to_dict("records")
            )
            or "No spot checks yet"
        )
        self.status.setText(
            f"{len(self.models)} model version(s), {len(self.checks)} pending check(s)"
        )

    def selected_model(self) -> Any | None:
        rows = self.model_table.selectionModel().selectedRows()
        return self.models[rows[0].row()] if rows else None

    def _ask(self, question: str) -> bool:
        answer = QMessageBox.question(self, "Promote to production", question)
        return answer == QMessageBox.StandardButton.Yes

    def set_status(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.training import set_model_status

        model = self.selected_model()
        status, reason = self.status_combo.currentText(), self.reason.text().strip()
        if model is None:
            self.status.setText("Select a model version first.")
            return
        if not reason:
            self.status.setText("Give a reason for the status change.")
            return
        approved = False
        if status == "production":
            approved = self.confirm(
                f"Make {model.name}@{model.version} the production version? Pipelines that "
                "pin it will use it, and the current production version becomes validated."
            )
            if not approved:
                self.status.setText("Promotion cancelled.")
                return
        ctx = context(reason)

        def done(record: Any) -> None:
            self.reason.clear()
            self.log.emit(f"Model {record.name}@{record.version}: {record.status} ({reason})")
            self.refresh()

        self._run(
            lambda: set_model_status(ctx, model.name, model.version, status, approved=approved),
            done,
            f"Setting {model.name}@{model.version} to {status}…",
        )

    def compare(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.training import compare_models

        model = self.selected_model()
        if model is None:
            self.status.setText("Select a model version first.")
            return
        ctx = context(self.reason.text().strip())

        def done(result: Any) -> None:
            _, report = result
            self.log.emit(f"Comparison of {model.name}@{model.version}: {report}")
            self.refresh()
            self.status.setText(f"Report saved: {report}")

        self._run(
            lambda: compare_models(ctx, model.ref),
            done,
            f"Comparing {model.name}@{model.version} on held-out fields…",
        )

    def open_report(self) -> None:
        model = self.selected_model()
        report = (model.metrics.get("test") or {}).get("report") if model is not None else None
        if not report or not Path(report).exists():
            self.status.setText("No comparison report yet: compare the model first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    # --- spot checks ----------------------------------------------------------------------

    def selected_check(self) -> dict[str, Any] | None:
        rows = self.check_table.selectionModel().selectedRows()
        if not rows:
            return None
        return {str(k): v for k, v in self.checks.to_dict("records")[rows[0].row()].items()}

    def _show_field(self) -> None:
        from hcs_studio.paths import resolve
        from hcs_studio.training.qa import field_files

        check = self.selected_check()
        if check is None:
            return
        try:
            files = field_files(
                Path(resolve(check["run"])), check["well"], int(check["field"]), check["labels"]
            )
        except (KeyError, ValueError):
            files = {}
        base = files.get("preview") or files.get("raw")
        if base is None:
            self.preview.setText("No preview saved for this field (run without images?)")
            return
        pixmap = QPixmap(str(base))
        if "outline" in files:
            painter = QPainter(pixmap)
            painter.drawPixmap(0, 0, QPixmap(str(files["outline"])).scaled(pixmap.size()))
            painter.end()
        self.preview.setPixmap(
            pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )

    def review(self, status: str) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.training import review_spot_check

        check = self.selected_check()
        reason = self.check_reason.text().strip()
        if check is None:
            self.status.setText("Select a spot check first.")
            return
        if status == "flagged" and not reason:
            self.status.setText("Say what is wrong to flag the field.")
            return
        ctx = context(reason)
        check_id = int(check["id"])

        def done(_: Any) -> None:
            self.check_reason.clear()
            self.log.emit(f"Spot check {check_id}: {status}")
            self.refresh()

        self._run(
            lambda: review_spot_check(ctx, check_id, status),
            done,
            f"Recording spot check {check_id}…",
        )
