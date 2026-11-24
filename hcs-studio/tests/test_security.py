"""Phase 26a: accounts, roles and permissions, the hash-chained audit trail."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hcs_studio.security import accounts, audit
from hcs_studio.security.roles import allowed, required_role
from hcs_studio.services import ReasonRequired, context, service


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


@service("test.change", object_type="thing")
def change(ctx, name: str, value: int):  # type: ignore[no-untyped-def]
    ctx.note(object_id=name, before={"value": value - 1}, after={"value": value})
    return value


@service("test.exclude", object_type="thing", reason_required=True)
def exclude(ctx, name: str):  # type: ignore[no-untyped-def]
    ctx.note(object_id=name)
    return name


def test_permission_matrix() -> None:
    assert (
        required_role("run.sign") == "reviewer" and required_role("run.sign.approved") == "reviewer"
    )
    assert required_role("run.analyze") == "analyst" and required_role("settings.app") == "admin"
    assert (
        required_role("models.status") == "reviewer" and required_role("models.train") == "analyst"
    )
    assert allowed("admin", "run.sign") and allowed("reviewer", "run.sign")
    assert not allowed("analyst", "run.sign") and not allowed("viewer", "run.analyze")
    assert not allowed("nobody", "run.analyze")


def test_accounts_passwords_and_lockout(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    me = accounts.os_username()
    assert accounts.role_of(me) == "admin"  # no accounts yet: a single-user installation
    accounts.create_account("anna", "reviewer", "Anna Smith", password="correct horse")
    assert accounts.role_of(me) == "viewer" and accounts.role_of("anna") == "reviewer"
    with pytest.raises(ValueError, match="exists"):
        accounts.create_account("anna", "analyst")
    with pytest.raises(ValueError, match="8 characters"):
        accounts.create_account("bob", "analyst", password="short")
    info = accounts.authenticate("anna", "correct horse")
    assert info.display_name == "Anna Smith" and info.has_password
    stored = accounts.list_accounts()[0]
    assert stored.last_login is not None
    for _ in range(5):
        with pytest.raises(accounts.AuthError, match="wrong password"):
            accounts.authenticate("anna", "wrong")
    with pytest.raises(accounts.AuthError, match="locked"):
        accounts.authenticate("anna", "correct horse")  # locked even with the right password
    accounts.update_account("anna", password="new password 1")  # an admin resets it
    assert accounts.authenticate("anna", "new password 1").username == "anna"
    accounts.update_account("anna", active=False)
    assert accounts.role_of("anna") == "viewer"
    with pytest.raises(accounts.AuthError, match="disabled"):
        accounts.authenticate("anna", "new password 1")
    from hcs_studio.db import session_scope
    from hcs_studio.db.security import Account

    with session_scope() as session:
        assert session.query(Account).one().password_hash.startswith("$argon2id$")


def test_sessions_time_out(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime, timedelta

    accounts.create_account("carl", "analyst", password="long enough")
    session = accounts.sign_in("carl", "long enough")
    assert accounts.current_session() is session
    session.last_active = datetime.now(UTC) - timedelta(minutes=31)
    assert accounts.current_session() is None
    accounts.sign_out()


def test_services_enforce_roles_and_record_denials(home: Path) -> None:
    ctx = context("fine")
    assert ctx.actor.role == "admin"
    assert change(ctx, "a", 1) == 1
    accounts.create_account(accounts.os_username(), "viewer")
    viewer = context("try")
    assert viewer.actor.role == "viewer"
    with pytest.raises(PermissionError, match="viewer"):
        change(viewer, "a", 2)
    records = audit.query(action="test.change")
    assert [r["status"] for r in records] == ["ok", "denied"]
    assert records[1]["actor_role"] == "viewer" and records[1]["error"].startswith(
        "PermissionError"
    )
    with pytest.raises(ReasonRequired):
        exclude(context(""), "x")


def test_chain_verifies_and_detects_tampering(home: Path) -> None:
    from sqlalchemy import text

    from hcs_studio.db import session_scope

    ctx = context("setting up")
    for i in range(6):
        change(ctx, f"t{i}", i)
    result = audit.verify_chain()
    assert result.ok and result.records == 6, result.breaks
    rows = audit.query()
    assert rows[0]["prev_hash"] == audit.GENESIS and rows[1]["prev_hash"] == rows[0]["hash"]
    assert all(len(r["hash"]) == 64 for r in rows)
    # someone edits a reason directly in the database
    with session_scope() as session:
        session.execute(text("UPDATE event SET reason = 'nothing to see' WHERE id = 3"))
    broken = audit.verify_chain()
    assert not broken.ok and broken.breaks[0] == {
        "id": 3,
        "problem": "content changed after it was written",
    }
    assert "BROKEN" in broken.summary()
    with session_scope() as session:  # put it back: intact again
        session.execute(text("UPDATE event SET reason = 'setting up' WHERE id = 3"))
    assert audit.verify_chain().ok
    with session_scope() as session:  # delete a record
        session.execute(text("DELETE FROM event WHERE id = 4"))
    problems = [b["problem"] for b in audit.verify_chain().breaks]
    assert any("missing" in p for p in problems) and any("does not follow" in p for p in problems)


def test_checkpoints_catch_a_rewritten_chain(home: Path) -> None:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    ctx = context("work")
    for i in range(3):
        change(ctx, f"c{i}", i)
    head = audit.checkpoint("test")
    assert head["id"] == 3
    files = audit.checkpoint_files()
    assert (
        files[0].exists()
        and json.loads(files[0].read_text().splitlines()[-1])["hash"] == head["hash"]
    )
    # a careful forger edits record 2 and recomputes every hash after it
    with session_scope() as session:
        rows = session.scalars(select(EventRecord).order_by(EventRecord.id)).all()
        rows[1].reason = "forged"
        prev = rows[0].hash
        for row in rows[1:]:
            row.prev_hash = prev
            row.hash = audit.record_hash(prev, audit.row_dict(row))
            prev = row.hash
    result = audit.verify_chain()
    assert not result.ok
    assert result.breaks == [
        {"id": 3, "problem": "differs from the checkpoint kept outside the database"}
    ]


def test_query_and_export(home: Path, tmp_path: Path) -> None:
    ctx = context("first")
    change(ctx, "PLATE01|run", 1)
    exclude(context("bad well"), "PLATE02|run")
    assert len(audit.query(object_contains="PLATE01")) == 1
    assert [r["type"] for r in audit.query(action="test")] == ["test.change", "test.exclude"]
    assert audit.query(user=accounts.os_username())
    rows = audit.query()
    csv_path = audit.export_csv(rows, tmp_path / "audit.csv")
    text = csv_path.read_text()
    assert "bad well" in text and text.splitlines()[0].startswith("id,utc,actor_id,actor_role")
    pdf = audit.export_pdf(rows, tmp_path / "audit.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "REVOKE UPDATE, DELETE" in audit.postgres_append_only_sql("hcs_app")


def test_old_unchained_records_are_chained(home: Path) -> None:
    from sqlalchemy import text

    from hcs_studio.db import session_scope

    change(context(), "x", 1)
    with session_scope() as session:  # simulate records written before Phase 26
        session.execute(text("UPDATE event SET hash = '', prev_hash = ''"))
        session.execute(text("DELETE FROM audit_head"))
    assert not audit.verify_chain().ok
    change(context(), "y", 2)  # the next event chains everything before it
    result = audit.verify_chain()
    assert result.ok and result.records == 2


def test_accounts_and_audit_commands(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hcs_studio.app import main

    me = accounts.os_username()
    assert main(["accounts", "add", me, "--role", "admin", "--reason", "first admin"]) == 0
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "reviewer pass")
    assert (
        main(
            [
                "accounts",
                "add",
                "rita",
                "--role",
                "reviewer",
                "--name",
                "Rita",
                "--password",
                "--reason",
                "new reviewer",
            ]
        )
        == 0
    )
    assert main(["accounts", "list"]) == 0 and main(["accounts", "matrix"]) == 0
    assert main(["accounts", "set", "rita", "--role", "analyst", "--reason", "moved"]) == 0
    assert accounts.get_account("rita").role == "analyst"  # type: ignore[union-attr]
    assert main(["audit", "list", "--action", "accounts"]) == 0
    assert main(["audit", "export", "--output", str(tmp_path / "a.csv")]) == 0
    assert main(["audit", "export", "--output", str(tmp_path / "a.pdf"), "--user", me]) == 0
    assert main(["audit", "checkpoint"]) == 0
    assert main(["audit", "verify"]) == 0
    from sqlalchemy import text

    from hcs_studio.db import session_scope

    with session_scope() as session:
        session.execute(text("UPDATE event SET object_id = 'x' WHERE id = 1"))
    assert main(["audit", "verify"]) == 1


@pytest.mark.gui
def test_audit_and_sign_in_dialogs(
    home: Path, tmp_path: Path, qtbot: Any, inline_tasks: None
) -> None:
    pytest.importorskip("PySide6")
    from hcs_studio.gui.security_dialogs import AuditDialog, SignInDialog

    change(context("one"), "PLATE9|a", 1)
    exclude(context("two"), "PLATE9|b")
    dialog = AuditDialog()
    qtbot.addWidget(dialog)
    dialog.search()
    assert dialog.table.rowCount() == 2 and "2 record" in dialog.status.text()
    dialog.action.setText("test.exclude")
    dialog.search()
    assert dialog.table.rowCount() == 1 and dialog.table.item(0, 6).text() == "two"  # type: ignore[union-attr]
    dialog.since.setText("not a date")
    dialog.search()
    assert "YYYY-MM-DD" in dialog.status.text()
    dialog.since.clear()
    dialog.verify()
    assert "intact" in dialog.status.text()
    dialog.export(".csv", tmp_path / "shown.csv")
    assert "two" in (tmp_path / "shown.csv").read_text()
    accounts.create_account("dora", "analyst", password="dora password")
    sign = SignInDialog(username="dora")
    qtbot.addWidget(sign)
    sign.password.setText("wrong")
    sign.check()
    assert "wrong password" in sign.error.text() and sign.info is None
    sign.password.setText("dora password")
    sign.check()
    assert sign.info is not None and sign.info.username == "dora"
