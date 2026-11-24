"""The tamper-evident audit trail (Phase 26).

Every service event is an append-only row of the ``event`` table with who (user and role), what
(action, object type and id, before/after), why (reason), when (UTC) and from where (client,
app version). Each row also stores

- ``hash`` = SHA-256 of ``prev_hash`` followed by the row's content as canonical JSON, and
- ``prev_hash`` = the hash of the row before it,

so the rows form a chain: changing, deleting or inserting a row breaks it from that point on.
``verify_chain`` recomputes every hash and reports each break. Rows written before the chain
existed are chained the first time an event is written (``backfill``).

The chain head (last id and hash) is also appended to ``<settings>/audit/chain_head.jsonl`` and
to any extra folders in the settings (``security.audit_head_copies``, e.g. a network share)
every ``security.audit_head_every`` events and on demand (``hcs-studio audit checkpoint``):
rewriting the whole chain then still disagrees with the copies kept elsewhere.

In PostgreSQL (server mode, Phase 27) the application role gets INSERT and SELECT on ``event``
but no UPDATE or DELETE (``postgres_append_only_sql``).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTENT = (
    "id",
    "type",
    "utc",
    "actor_id",
    "actor_name",
    "actor_kind",
    "actor_role",
    "reason",
    "object_type",
    "object_id",
    "before",
    "after",
    "client",
    "app_version",
    "status",
    "error",
)
GENESIS = "0" * 64


def _utc_text(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return str(value.isoformat(timespec="microseconds"))
    return str(value)


def canonical(row: dict[str, Any]) -> bytes:
    """The content of an event row as canonical JSON (what the hash covers)."""
    content = {key: row.get(key) for key in CONTENT}
    content["utc"] = _utc_text(content["utc"])
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def record_hash(prev_hash: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode("ascii") + canonical(row)).hexdigest()


def row_dict(record: Any) -> dict[str, Any]:
    return {key: getattr(record, key) for key in (*CONTENT, "prev_hash", "hash")}


def _head(session: Any) -> Any:
    from hcs_studio.db.security import AuditHead

    head = session.get(AuditHead, 1)
    if head is None:
        head = AuditHead(id=1, last_event_id=0, last_hash=GENESIS, count=0)
        session.add(head)
        session.flush()
    return head


def chain_new_rows(session: Any) -> int:
    """Chain every row after the head (new events, or rows written before Phase 26), in id
    order. Call inside the transaction that inserted them; returns the rows chained."""
    from sqlalchemy import select, update

    from hcs_studio.db.models import EventRecord
    from hcs_studio.db.security import AuditHead

    head = _head(session)
    # take the write lock first (UPDATE), so concurrent writers queue here
    session.execute(update(AuditHead).where(AuditHead.id == 1).values(count=AuditHead.count))
    session.refresh(head)
    rows = session.scalars(
        select(EventRecord).where(EventRecord.id > head.last_event_id).order_by(EventRecord.id)
    ).all()
    prev = head.last_hash or GENESIS
    for row in rows:
        row.prev_hash = prev
        row.hash = record_hash(prev, row_dict(row))
        prev = row.hash
    if rows:
        head.last_event_id, head.last_hash = rows[-1].id, prev
        head.count = head.count + len(rows)
    return len(rows)


@dataclass
class Verification:
    ok: bool
    records: int
    breaks: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: int = 0
    unchained: int = 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"Audit trail intact: {self.records} records, {self.checkpoints} checkpoint(s) "
                "agree"
            )
        first = self.breaks[0]
        return (
            f"Audit trail BROKEN: {len(self.breaks)} problem(s); first at record "
            f"{first['id']}: {first['problem']}"
        )


def checkpoint_files() -> list[Path]:
    from hcs_studio.config import load_settings
    from hcs_studio.io.sidecar import config_dir

    folders = [
        config_dir() / "audit",
        *(Path(p) for p in load_settings().security.audit_head_copies),
    ]
    return [f / "chain_head.jsonl" for f in folders]


def read_checkpoints() -> list[dict[str, Any]]:
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for path in checkpoint_files():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[(int(entry["id"]), str(entry["hash"]))] = entry
    return list(seen.values())


def verify_chain(session: Any | None = None) -> Verification:
    """Recompute every hash and compare with the stored ones and the checkpoints."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    def run(s: Any) -> Verification:
        rows = s.scalars(select(EventRecord).order_by(EventRecord.id)).all()
        breaks = []
        prev = GENESIS
        by_id: dict[int, str] = {}
        unchained = 0
        expected_id = None
        for row in rows:
            data = row_dict(row)
            if not row.hash:
                unchained += 1
                continue
            if expected_id is not None and row.id != expected_id:
                breaks.append(
                    {"id": row.id, "problem": f"records {expected_id}–{row.id - 1} missing"}
                )
            if row.prev_hash != prev:
                breaks.append(
                    {
                        "id": row.id,
                        "problem": "does not follow the previous record "
                        "(a record was deleted, inserted or changed before it)",
                    }
                )
            computed = record_hash(row.prev_hash or GENESIS, data)
            if computed != row.hash:
                breaks.append({"id": row.id, "problem": "content changed after it was written"})
            prev = row.hash
            by_id[row.id] = row.hash
            expected_id = row.id + 1
        checkpoints = read_checkpoints()
        for entry in checkpoints:
            stored = by_id.get(int(entry["id"]))
            if stored is None:
                breaks.append({"id": int(entry["id"]), "problem": "a checkpointed record is gone"})
            elif stored != entry["hash"]:
                breaks.append(
                    {
                        "id": int(entry["id"]),
                        "problem": "differs from the checkpoint kept outside the database",
                    }
                )
        if unchained:
            breaks.append({"id": 0, "problem": f"{unchained} record(s) never chained"})
        return Verification(not breaks, len(rows), breaks, len(checkpoints), unchained)

    if session is not None:
        return run(session)
    with session_scope() as s:
        return run(s)


def checkpoint(note: str = "") -> dict[str, Any]:
    """Append the chain head to every checkpoint file."""
    from hcs_studio.db import session_scope

    with session_scope() as session:
        chain_new_rows(session)
        head = _head(session)
        entry = {
            "id": head.last_event_id,
            "hash": head.last_hash,
            "count": head.count,
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": note,
        }
    if not entry["id"]:
        return entry
    for path in checkpoint_files():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:  # an unreachable copy must not stop work; verify reports fewer
            continue
    return entry


def maybe_checkpoint(count: int) -> None:
    from hcs_studio.config import load_settings

    every = load_settings().security.audit_head_every
    if every and count % every == 0:
        checkpoint("periodic")


# --- viewing and exporting -----------------------------------------------------------------


def query(
    *,
    user: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    action: str | None = None,
    object_contains: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Audit records, oldest first. ``action``: an event type or prefix (``run`` → run.*)."""
    from sqlalchemy import or_, select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    q = select(EventRecord).order_by(EventRecord.id)
    if user:
        q = q.where(or_(EventRecord.actor_id == user, EventRecord.actor_name == user))
    if since:
        q = q.where(EventRecord.utc >= since)
    if until:
        q = q.where(EventRecord.utc <= until)
    if action:
        q = q.where(or_(EventRecord.type == action, EventRecord.type.like(f"{action}.%")))
    if object_contains:
        q = q.where(EventRecord.object_id.contains(object_contains))
    if limit:
        q = q.limit(limit)
    with session_scope() as session:
        return [row_dict(r) for r in session.scalars(q)]


COLUMNS = (
    "id",
    "utc",
    "actor_id",
    "actor_role",
    "type",
    "object_type",
    "object_id",
    "reason",
    "status",
    "before",
    "after",
    "client",
    "app_version",
    "hash",
)


def export_csv(records: Iterable[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for r in records:
            writer.writerow(
                [
                    _utc_text(r[c])
                    if c == "utc"
                    else json.dumps(r[c], default=str)
                    if c in ("before", "after")
                    else r[c]
                    for c in COLUMNS
                ]
            )
    return path


def export_pdf(records: list[dict[str, Any]], path: Path, title: str = "Audit trail") -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small", fontSize=7, leading=8.5)
    verification = verify_chain()
    rows: list[list[Any]] = [["#", "UTC", "user (role)", "action", "object", "reason", "status"]]
    for r in records:
        rows.append(
            [
                str(r["id"]),
                _utc_text(r["utc"])[:19],
                f"{r['actor_id']} ({r['actor_role'] or '–'})",
                r["type"],
                Paragraph(_escape(f"{r['object_type']} {r['object_id']}"), small),
                Paragraph(_escape(r["reason"] or ""), small),
                r["status"],
            ]
        )
    table = Table(rows, repeatRows=1, colWidths=[28, 92, 100, 110, 260, 160, 40])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ]
        )
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        title=title,
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    doc.build(
        [
            Paragraph(title, styles["Title"]),
            Paragraph(
                f"{len(records)} records, exported {created}. {verification.summary()}",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            table,
        ]
    )
    return path


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def postgres_append_only_sql(app_role: str) -> str:
    """SQL for server mode: the application may add audit records, never change them."""
    return (
        f"REVOKE UPDATE, DELETE, TRUNCATE ON event FROM {app_role};\n"
        f"GRANT SELECT, INSERT ON event TO {app_role};\n"
        "-- the chain head row is updated by the application:\n"
        f"GRANT SELECT, INSERT, UPDATE ON audit_head TO {app_role};\n"
    )


def run_audit(run_key: str) -> list[dict[str, Any]]:
    """The audit records of one run (its rooted folder is the object id), for its report."""
    return [
        r
        for r in query(object_contains=run_key)
        if r["object_id"] == run_key or r["object_id"].startswith(run_key)
    ]
