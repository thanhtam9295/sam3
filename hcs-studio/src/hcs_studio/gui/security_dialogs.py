"""Audit-trail viewer and sign-in dialogs (Phase 26).

``AuditDialog`` (Tools → Audit trail…): filter the audit records by user, action, object and
dates; verify the hash chain; export the shown records as CSV or PDF.
``SignInDialog``: asks for a username and password when the settings require one (and for
re-authentication when signing a run).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask

AUDIT_COLUMNS = ("id", "UTC", "user", "role", "action", "object", "reason", "status")


class AuditDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audit trail")
        self.resize(1100, 600)
        self.records: list[dict[str, Any]] = []
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._loaded)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.task.failed.connect(lambda message: self.status.setText(f"Error: {message}"))
        self.user = QLineEdit()
        self.action = QLineEdit()
        self.action.setPlaceholderText("e.g. run or qc.review")
        self.object = QLineEdit()
        self.object.setPlaceholderText("plate, run folder …")
        self.since = QLineEdit()
        self.since.setPlaceholderText("YYYY-MM-DD")
        self.until = QLineEdit()
        self.until.setPlaceholderText("YYYY-MM-DD")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)
        self.verify_button = QPushButton("Verify chain")
        self.verify_button.clicked.connect(self.verify)
        self.csv_button = QPushButton("Export CSV…")
        self.csv_button.clicked.connect(lambda: self.export(".csv"))
        self.pdf_button = QPushButton("Export PDF…")
        self.pdf_button.clicked.connect(lambda: self.export(".pdf"))
        self.table = QTableWidget(0, len(AUDIT_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(AUDIT_COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        filters = QHBoxLayout()
        for label, widget in (
            ("User", self.user),
            ("Action", self.action),
            ("Object", self.object),
            ("From", self.since),
            ("To", self.until),
        ):
            filters.addWidget(QLabel(label))
            filters.addWidget(widget)
        filters.addWidget(self.search_button)
        buttons = QHBoxLayout()
        for button in (self.verify_button, self.csv_button, self.pdf_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(filters)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

    def _filters(self) -> dict[str, Any]:
        def when(text: str, end: bool = False) -> datetime | None:
            text = text.strip()
            if not text:
                return None
            value = datetime.fromisoformat(text)
            return (
                value.replace(hour=23, minute=59, second=59) if end and len(text) == 10 else value
            )

        return {
            "user": self.user.text().strip() or None,
            "action": self.action.text().strip() or None,
            "object_contains": self.object.text().strip() or None,
            "since": when(self.since.text()),
            "until": when(self.until.text(), True),
        }

    def search(self) -> None:
        from hcs_studio.security.audit import query

        try:
            filters = self._filters()
        except ValueError:
            self.status.setText("Dates are YYYY-MM-DD")
            return
        self.status.setText("Searching…")
        self.task.request(lambda: ("records", query(**filters)))

    def verify(self) -> None:
        from hcs_studio.security.audit import verify_chain

        self.status.setText("Verifying the hash chain…")
        self.task.request(lambda: ("verify", verify_chain()))

    def _loaded(self, result: tuple[str, Any]) -> None:
        from hcs_studio.security.audit import _utc_text

        kind, value = result
        if kind == "verify":
            self.status.setText(value.summary())
            return
        self.records = value
        self.table.setRowCount(len(value))
        for r, record in enumerate(value):
            cells = (
                record["id"],
                _utc_text(record["utc"])[:19],
                record["actor_id"],
                record["actor_role"],
                record["type"],
                f"{record['object_type']} {record['object_id']}",
                record["reason"],
                record["status"],
            )
            for c, cell in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(str(cell)))
        self.status.setText(f"{len(value)} record(s)")

    def export(self, suffix: str, path: Path | None = None) -> None:
        from hcs_studio.security.audit import export_csv, export_pdf

        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(self, "Export audit records", f"audit{suffix}")
            if not chosen:
                return
            path = Path(chosen)
        records = self.records
        (export_pdf if suffix == ".pdf" else export_csv)(records, path)
        self.status.setText(f"{len(records)} record(s) → {path}")


class SignInDialog(QDialog):
    """Username and password; ``accepted`` once :func:`authenticate` succeeds."""

    def __init__(
        self,
        parent: QWidget | None = None,
        username: str = "",
        title: str = "Sign in to HCS Studio",
        message: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.username = QLineEdit(username)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #b2182b")
        form = QFormLayout()
        form.addRow("User", self.username)
        form.addRow("Password", self.password)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.check)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        if message:
            note = QLabel(message)
            note.setWordWrap(True)
            layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.info: Any = None

    def check(self) -> None:
        from hcs_studio.security.accounts import AuthError, authenticate

        try:
            self.info = authenticate(self.username.text().strip(), self.password.text())
        except AuthError as exc:
            self.error.setText(str(exc))
            self.password.clear()
            return
        self.accept()


def require_sign_in(parent: QWidget | None = None) -> bool:
    """At start: when the settings require a password, sign in (False = the user gave up)."""
    from hcs_studio.config import load_settings
    from hcs_studio.security.accounts import list_accounts, os_username, sign_in

    if not load_settings().security.password_required or not list_accounts():
        return True
    dialog = SignInDialog(parent, os_username())
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    sign_in(dialog.username.text().strip(), dialog.password.text())
    return True


class SignRunDialog(QDialog):
    """Sign a run as reviewed or approved: state and signatures so far, the meaning of the
    signature, an optional comment and — when the settings require it — the password."""

    signed = Signal(object)

    def __init__(self, folder: Path, parent: QWidget | None = None) -> None:
        from hcs_studio.config import load_settings
        from hcs_studio.security import signoff

        super().__init__(parent)
        self.folder = Path(folder)
        self.setWindowTitle(f"Sign {self.folder.name}")
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._done)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.task.failed.connect(lambda message: self.status.setText(f"Not signed: {message}"))
        state = signoff.status(self.folder)
        self.reauth = load_settings().security.reauth_required
        lines = [
            f"State: <b>{state.state}</b>"
            + (" (locked)" if state.locked else "")
            + (f", version {state.version}" if state.version > 1 else "")
        ]
        lines += [
            f"{s['meaning']}: {s['display_name']} ({s['role']}), {s['utc']}"
            for s in state.signatures
        ]
        self.summary = QLabel("<br>".join(lines))
        self.meaning = QComboBox()
        self.meaning.addItems(["reviewed", "approved"])
        self.meaning.setCurrentText("approved" if state.state == "reviewed" else "reviewed")
        self.statement = QLabel("")
        self.statement.setWordWrap(True)
        self.meaning.currentTextChanged.connect(
            lambda m: self.statement.setText(signoff.STATEMENTS[m])
        )
        self.statement.setText(signoff.STATEMENTS[self.meaning.currentText()])
        self.comment = QLineEdit()
        self.comment.setPlaceholderText("Comment (optional)")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText(
            "required" if self.reauth else "optional: confirms your identity"
        )
        self.sign_button = QPushButton("Sign")
        self.sign_button.clicked.connect(self.sign)
        self.sign_button.setEnabled(not state.locked and state.state in ("completed", "reviewed"))
        form = QFormLayout()
        form.addRow("Meaning", self.meaning)
        form.addRow("", self.statement)
        form.addRow("Comment", self.comment)
        form.addRow("Password", self.password)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(form)
        layout.addWidget(self.sign_button)
        layout.addWidget(self.status)

    def sign(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.security import sign_run

        password = self.password.text() or None
        if self.reauth and not password:
            self.status.setText("Enter your password to sign.")
            return
        ctx, meaning, folder = (
            context(self.comment.text().strip()),
            self.meaning.currentText(),
            self.folder,
        )
        self.status.setText("Signing…")
        self.task.request(lambda: sign_run(ctx, folder, meaning, password))

    def _done(self, result: Any) -> None:
        self.password.clear()
        self.status.setText(
            f"Signed: the run is now {result.state}" + (" and locked." if result.locked else ".")
        )
        self.sign_button.setEnabled(False)
        self.signed.emit(result)
