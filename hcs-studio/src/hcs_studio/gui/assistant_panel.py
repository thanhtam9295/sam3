"""The Assistant tab (Phase 28): ask questions about results in plain language.

Options per question: a stronger model, compound codes instead of names, include free-text
notes, allow image thumbnails (each image is confirmed before it is sent). "Data to be sent"
shows the exact request first. Under each answer, "Show work" lists every tool call with its
SQL; numbers the grounding check could not find in any tool result are flagged. Charts open
locally; well highlights and comparisons go to the other tabs (``ui_action``); pipeline
proposals and methods drafts are shown for the user to apply or edit — the assistant never
changes anything itself.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask


def _confirm_image(parent: QWidget, request: dict[str, Any]) -> bool:
    answer = QMessageBox.question(
        parent,
        "Send an image?",
        f"The assistant asks for a preview of run {request['run_id']}, well {request['well']}, "
        f"field {request['field']}.\n\nA downsampled 8-bit image (at most 256 px) will be sent "
        "to the Claude API. Send it?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


class AssistantPanel(QWidget):
    log = Signal(str)
    #: highlight_wells / open_comparison actions for the main window
    ui_action = Signal(dict)

    def __init__(self, assistant: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._assistant = assistant
        #: asks the user before an image is sent (tests replace it)
        self.confirm: Callable[[dict[str, Any]], bool] = lambda r: _confirm_image(self, r)
        self.answers: list[Any] = []
        self.charts: list[QDialog] = []
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_answer)
        self.task.failed.connect(self._on_failed)
        self.preview_task = BackgroundTask(self)
        self.preview_task.finished.connect(self._on_preview)
        self.preview_task.failed.connect(self._on_failed)
        self.status_task = BackgroundTask(self)
        self.status_task.finished.connect(self._on_status)
        self.status_label = QLabel("")
        self.status_task.failed.connect(self.status_label.setText)

        self.status_label.setWordWrap(True)
        self.question = QPlainTextEdit()
        self.question.setPlaceholderText(
            "e.g. Which combinations had a ZIP score above 10 in both cell lines?"
        )
        self.question.setMaximumHeight(90)
        self.strong = QCheckBox("Stronger model")
        self.ids_only = QCheckBox("Send compound IDs only")
        self.notes = QCheckBox("Include free-text notes")
        self.images = QCheckBox("Allow image thumbnails")
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask)
        self.preview_button = QPushButton("Data to be sent…")
        self.preview_button.clicked.connect(self.preview)
        self.new_button = QPushButton("New conversation")
        self.new_button.clicked.connect(self.new_conversation)
        options = QHBoxLayout()
        for widget in (self.strong, self.ids_only, self.notes, self.images):
            options.addWidget(widget)
        options.addStretch(1)
        buttons = QHBoxLayout()
        for button in (self.ask_button, self.preview_button, self.new_button):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self.answer_view = QTextBrowser()
        self.answer_view.setOpenExternalLinks(False)
        self.work = QTreeWidget()
        self.work.setHeaderLabels(["Show work: tool", "result"])
        self.work.setColumnWidth(0, 220)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.answer_view)
        split.addWidget(self.work)
        split.setSizes([400, 200])

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.question)
        layout.addLayout(options)
        layout.addLayout(buttons)
        layout.addWidget(split, 1)

    # --- the assistant ---------------------------------------------------------------------

    @property
    def assistant(self) -> Any:
        if self._assistant is None:
            from hcs_studio.assistant.session import assistant_for_app

            window = self.window()
            self._assistant = assistant_for_app(getattr(window, "server", None))
        return self._assistant

    @assistant.setter
    def assistant(self, value: Any) -> None:
        self._assistant = value

    def refresh_status(self) -> None:
        assistant = self.assistant
        self.status_task.request(assistant.status)

    def _on_status(self, status: dict[str, Any]) -> None:
        if not status.get("enabled"):
            self.status_label.setText(
                "The lab assistant is off. An admin can enable it in Tools → Assistant settings."
            )
            self.ask_button.setEnabled(False)
            return
        self.ask_button.setEnabled(True)
        usage = status.get("usage", {})
        budget = status.get("budget") or 0
        used = f"{usage.get('weighted', 0):,} of {budget:,}" if budget else "no limit"
        self.status_label.setText(
            f"Model {status.get('model')} (stronger: {status.get('strong_model')}). "
            f"Tokens this month: {used}."
        )
        self.images.setEnabled(bool(status.get("allow_images")))
        if status.get("compound_ids_only"):
            self.ids_only.setChecked(True)

    def current_question(self) -> Any:
        from hcs_studio.assistant import Privacy, Question

        return Question(
            self.question.toPlainText().strip(),
            strong=self.strong.isChecked(),
            privacy=Privacy(
                compound_ids_only=self.ids_only.isChecked(),
                include_notes=self.notes.isChecked(),
            ),
            images=self.images.isChecked() and self.images.isEnabled(),
        )

    def ask(self) -> None:
        question = self.current_question()
        if not question.text:
            return
        assistant = self.assistant
        self.ask_button.setEnabled(False)
        self.answer_view.append(f"<p><b>You:</b> {html.escape(question.text)}</p>")
        self.log.emit("Asking the lab assistant…")
        self.task.request(lambda: assistant.ask(question))

    def preview(self) -> None:
        question = self.current_question()
        if not question.text:
            return
        assistant = self.assistant
        self.preview_task.request(lambda: assistant.preview(question))

    def _on_preview(self, body: dict[str, Any]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Data that will be sent to the Claude API")
        text = QPlainTextEdit(json.dumps(body, indent=2, ensure_ascii=False))
        text.setReadOnly(True)
        note = QLabel(
            "This is the first request. Tool results (rows of the curated views) are added as "
            "the assistant asks for them; every request is listed in the audit trail."
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(dialog)
        layout.addWidget(note)
        layout.addWidget(text)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.resize(800, 600)
        dialog.setModal(False)
        dialog.show()
        self.preview_dialog = dialog

    def new_conversation(self) -> None:
        assistant = self.assistant
        assistant.new_conversation()
        self.answer_view.clear()
        self.work.clear()

    def _on_failed(self, message: str) -> None:
        self.ask_button.setEnabled(True)
        self.answer_view.append(f"<p style='color:#b00'>{html.escape(message)}</p>")
        self.log.emit(f"Assistant: {message}")

    def _on_answer(self, answer: Any) -> None:
        if answer.status == "needs_confirmation" and answer.pending:
            approved = self.confirm(answer.pending)
            assistant = self.assistant
            self.task.request(lambda: assistant.resume(approved))
            return
        self.ask_button.setEnabled(True)
        self.answers.append(answer)
        self.show_answer(answer)
        self.refresh_status()

    def show_answer(self, answer: Any) -> None:
        text = answer.display_text or answer.text
        parts = [f"<p><b>Assistant</b> <small>({html.escape(answer.model)})</small>:</p>"]
        parts.append("<p>" + html.escape(text).replace("\n", "<br>") + "</p>")
        if answer.status != "answered":
            parts.append(f"<p><i>{html.escape(answer.message or answer.status)}</i></p>")
        if answer.unsupported_numbers:
            numbers = ", ".join(answer.unsupported_numbers)
            parts.append(
                "<p style='color:#a60'>Check these numbers: they were not found in any tool "
                f"result ({html.escape(numbers)}).</p>"
            )
        if not answer.steps and answer.status == "answered":
            parts.append("<p style='color:#a60'>No tool was used for this answer.</p>")
        self.answer_view.append("".join(parts))
        self.work.clear()
        for i, step in enumerate(answer.steps, start=1):
            item = QTreeWidgetItem([f"{i}. {step.tool}", step.summary])
            if step.is_error:
                item.setForeground(1, Qt.GlobalColor.red)
            for sql in step.sql:
                QTreeWidgetItem(item, ["SQL", sql]).setToolTip(1, sql)
            inputs = json.dumps(step.input, ensure_ascii=False)
            QTreeWidgetItem(item, ["input", inputs]).setToolTip(1, inputs)
            self.work.addTopLevelItem(item)
        for action in answer.ui:
            self.handle_action(action)

    def handle_action(self, action: dict[str, Any]) -> None:
        kind = action.get("action")
        if kind == "chart":
            self.charts.append(chart_dialog(self, action, self.assistant.reveal))
        elif kind == "pipeline_proposal":
            lines = [f"Proposed for run {action['run_id']}: {action.get('summary', '')}", ""]
            for change in action.get("changes", []):
                lines.append(
                    f"Step {change['step']} ({change['module']}): {change['parameter']} "
                    f"{change['old']!r} → {change['new']!r} — {change.get('reason', '')}"
                )
            lines += ["", "Nothing was changed. Apply it in the pipeline editor if you agree."]
            self._text_dialog("Proposed pipeline change", "\n".join(lines))
        elif kind == "methods_draft":
            draft = action.get("draft", {})
            text = "\n\n".join(p.get("text", "") for p in draft.get("paragraphs", []))
            missing = draft.get("missing") or []
            if missing:
                text += "\n\nMissing from the records: " + "; ".join(missing)
            self._text_dialog("Methods draft (edit before use)", self.assistant.reveal(text))
        elif kind in ("highlight_wells", "open_comparison"):
            self.ui_action.emit(action)

    def _text_dialog(self, title: str, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        edit = QPlainTextEdit(text)
        layout = QVBoxLayout(dialog)
        layout.addWidget(edit)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.resize(700, 400)
        dialog.show()
        self.charts.append(dialog)


def chart_dialog(parent: QWidget, spec: dict[str, Any], reveal: Callable[[str], str]) -> QDialog:
    """Draw a ``make_chart`` spec with pyqtgraph (the data never left the computer)."""
    import numpy as np
    import pyqtgraph as pg

    dialog = QDialog(parent)
    dialog.setWindowTitle(reveal(str(spec.get("title", "Chart"))))
    plot = pg.PlotWidget()
    plot.addLegend()
    columns = list(spec["columns"])
    rows = spec["rows"]
    x_at, y_at = columns.index(spec["x"]), columns.index(spec["y"])
    group_at = columns.index(spec["group"]) if spec.get("group") else None
    groups: dict[str, list[tuple[Any, Any]]] = {}
    for row in rows:
        key = reveal(str(row[group_at])) if group_at is not None else ""
        groups.setdefault(key, []).append((row[x_at], row[y_at]))
    categorical = any(isinstance(r[x_at], str) for r in rows)
    labels = sorted({str(r[x_at]) for r in rows}) if categorical else []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, points) in enumerate(groups.items()):
        xs = np.array(
            [labels.index(str(p[0])) if categorical else float(p[0] or 0) for p in points],
            dtype=float,
        )
        ys = np.array([float(p[1]) if p[1] is not None else np.nan for p in points], dtype=float)
        color = colors[i % len(colors)]
        if spec.get("log_x") and not categorical:
            xs = np.log10(np.where(xs > 0, xs, np.nan))
        if spec["kind"] == "bar":
            width = 0.8 / max(1, len(groups))
            plot.addItem(
                pg.BarGraphItem(
                    x=xs + i * width, height=np.nan_to_num(ys), width=width, brush=color, name=name
                )
            )
        elif spec["kind"] == "line":
            order = np.argsort(xs)
            plot.plot(xs[order], ys[order], pen=color, symbol="o", name=name or None)
        else:
            plot.plot(xs, ys, pen=None, symbol="o", symbolBrush=color, name=name or None)
    if categorical:
        plot.getAxis("bottom").setTicks([[(i, reveal(t)) for i, t in enumerate(labels)]])
    plot.setLabel("bottom", ("log10 " if spec.get("log_x") else "") + str(spec["x"]))
    plot.setLabel("left", str(spec["y"]))
    sql = QLabel(f"From: {html.escape(str(spec.get('sql', '')))}")
    sql.setWordWrap(True)
    layout = QVBoxLayout(dialog)
    layout.addWidget(plot)
    layout.addWidget(sql)
    dialog.resize(700, 500)
    dialog.show()
    dialog.plot = plot  # type: ignore[attr-defined]
    return dialog


class AssistantSettingsDialog(QDialog):
    """Admin settings of the assistant (through services: audited, admins only)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from hcs_studio.config import ASSISTANT_MODELS, load_settings

        self.setWindowTitle("Lab assistant settings")
        current = load_settings().assistant
        self.enabled = QCheckBox("Enable the lab assistant")
        self.enabled.setChecked(current.enabled)
        self.model = QComboBox()
        self.strong_model = QComboBox()
        for combo, value in (
            (self.model, current.model),
            (self.strong_model, current.strong_model),
        ):
            combo.setEditable(True)
            combo.addItems(list(ASSISTANT_MODELS))
            combo.setCurrentText(value)
        self.effort = QComboBox()
        self.effort.addItems(["low", "medium", "high", "xhigh", "max"])
        self.effort.setCurrentText(current.effort)
        self.budget = QSpinBox()
        self.budget.setRange(0, 1_000_000_000)
        self.budget.setSingleStep(100_000)
        self.budget.setValue(current.monthly_token_budget)
        self.budget.setSpecialValueText("no limit")
        self.ids_only = QCheckBox("Send compound IDs only by default")
        self.ids_only.setChecked(current.compound_ids_only)
        self.allow_images = QCheckBox("Users may enable image thumbnails (confirmed per image)")
        self.allow_images.setChecked(current.allow_images)
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(0.5, 120)
        self.timeout.setValue(current.query_timeout_s)
        self.max_rows = QSpinBox()
        self.max_rows.setRange(1, 5000)
        self.max_rows.setValue(current.max_rows)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("leave empty to keep the stored key")
        note = QLabel(
            "Before enabling: check your institution's data policy and the Anthropic API "
            "data-retention terms (docs/assistant.md). The key is stored in the OS keychain, "
            "never in settings files."
        )
        note.setWordWrap(True)
        form = QFormLayout()
        form.addRow(self.enabled)
        form.addRow("Model", self.model)
        form.addRow("Effort", self.effort)
        form.addRow("Stronger model", self.strong_model)
        form.addRow("Monthly tokens per user", self.budget)
        form.addRow(self.ids_only)
        form.addRow(self.allow_images)
        form.addRow("Query time limit (s)", self.timeout)
        form.addRow("Rows per query", self.max_rows)
        form.addRow("Claude API key", self.key)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def changes(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled.isChecked(),
            "model": self.model.currentText().strip(),
            "effort": self.effort.currentText(),
            "strong_model": self.strong_model.currentText().strip(),
            "monthly_token_budget": self.budget.value(),
            "compound_ids_only": self.ids_only.isChecked(),
            "allow_images": self.allow_images.isChecked(),
            "query_timeout_s": self.timeout.value(),
            "max_rows": self.max_rows.value(),
        }

    def save(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.assistant import configure_assistant, set_api_key

        ctx = context()
        try:
            configure_assistant(ctx, **self.changes())
            if self.key.text():
                set_api_key(ctx, self.key.text())
        except Exception as exc:
            QMessageBox.warning(self, "Assistant settings", str(exc))
            return
        self.key.clear()
        self.accept()
