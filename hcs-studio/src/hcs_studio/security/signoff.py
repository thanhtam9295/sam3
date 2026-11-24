"""Run states, electronic signatures, locking and run versions (Phase 26).

States: **draft** (running or unfinished) → **completed** → **reviewed** → **approved**. A
reviewer signs a completed run as *reviewed* and a reviewed run as *approved*; each signature
records the printed name, role, meaning (a fixed statement), time (UTC), an optional comment,
the SHA-256 of the run manifest at signing and whether the password was asked again
(``security.reauth_for_signoff``; on in regulated mode). In regulated mode the approver must also
differ from the reviewer.

An approved run is **locked**: services that change a run refuse it (``RunLocked``), and its
files are made read-only. Re-analysis creates a **new version** of the run — a copy
``<run>_v2`` (large unchanging files hard-linked) whose manifest names the original
(``parent_run``) — and works on that. An admin can unlock a run with a reason (recorded).

Signatures are kept in the registry (``signature``) and in the run folder
(``signatures.json``), so the run report shows them; unapproved reports carry a DRAFT mark.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATES = ("draft", "completed", "reviewed", "approved")
STATEMENTS = {
    "reviewed": "Reviewed: I have checked these results — the analysis, the QC decisions and the "
    "exclusions are appropriate.",
    "approved": "Approved: I release these results for use. The run is locked; any change "
    "creates a new version.",
}
#: Files of a run that never change after the run (hard-linked into new versions).
IMMUTABLE_DIRS = ("masks", "previews", "enhanced", "images", "corrected", "thumbs")
IMMUTABLE_FILES = ("objects.parquet", "coloc.csv", "fields.csv")


class RunLocked(PermissionError):
    """The run is approved and locked."""


@dataclass
class RunStatus:
    run: str
    state: str
    version: int = 1
    parent_run: str = ""
    locked: bool = False
    signatures: list[dict[str, Any]] = field(default_factory=list)


def run_key(folder: Path) -> str:
    from hcs_studio.output.manifest import rooted_text

    return rooted_text(Path(folder).absolute())


def _manifest(folder: Path) -> dict[str, Any]:
    path = Path(folder) / "run_manifest.json"
    if not path.exists():
        raise ValueError(f"{folder} is not a run folder (no run_manifest.json)")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def manifest_sha256(folder: Path) -> str:
    return hashlib.sha256((Path(folder) / "run_manifest.json").read_bytes()).hexdigest()


def _default_state(manifest: dict[str, Any]) -> str:
    """A run that finished (with or without failed wells) is completed; running, cancelled,
    failed or QC-stopped runs are drafts."""
    return (
        "completed"
        if manifest.get("status") in ("completed", "completed_with_failures")
        else "draft"
    )


def status(folder: Path) -> RunStatus:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.security import RunState, Signature

    folder = Path(folder)
    key = run_key(folder)
    manifest = _manifest(folder)
    with session_scope() as session:
        row = session.scalar(select(RunState).where(RunState.run == key))
        signatures = [
            _signature(s)
            for s in session.scalars(
                select(Signature).where(Signature.run == key).order_by(Signature.id)
            )
        ]
        if row is None:
            return RunStatus(
                key,
                _default_state(manifest),
                int(manifest.get("run_version", 1)),
                str(manifest.get("parent_run", "")),
                False,
                signatures,
            )
        return RunStatus(key, row.state, row.version, row.parent_run, row.locked, signatures)


def _signature(row: Any) -> dict[str, Any]:
    return {
        "meaning": row.meaning,
        "statement": row.statement,
        "username": row.username,
        "display_name": row.display_name,
        "role": row.role,
        "reason": row.reason,
        "manifest_sha256": row.manifest_sha256,
        "reauthenticated": row.reauthenticated,
        "utc": row.utc.isoformat(timespec="seconds") if row.utc else "",
    }


def check_unlocked(folder: Path) -> None:
    """Raise :class:`RunLocked` for an approved (locked) run."""
    if status(folder).locked:
        raise RunLocked(
            f"{Path(folder).name} is approved and locked: re-analyse it as a new "
            "version (--new-version), or ask an admin to unlock it"
        )


def _set_state(folder: Path, state: str, locked: bool | None = None) -> RunStatus:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.security import RunState

    key = run_key(folder)
    manifest = _manifest(folder)
    with session_scope() as session:
        row = session.scalar(select(RunState).where(RunState.run == key))
        if row is None:
            row = RunState(
                run=key,
                version=int(manifest.get("run_version", 1)),
                parent_run=str(manifest.get("parent_run", "")),
            )
            session.add(row)
        row.state = state
        if locked is not None:
            row.locked = locked
        row.updated = datetime.now(UTC)
    return status(folder)


def _write_files(folder: Path, signatures: list[dict[str, Any]], state: str) -> None:
    (Path(folder) / "signatures.json").write_text(
        json.dumps({"state": state, "signatures": signatures}, indent=1), encoding="utf-8"
    )
    try:
        from hcs_studio.output.report import write_report

        write_report(Path(folder))
    except Exception:  # the report is a convenience; the signature is in the registry
        pass


def sign(
    folder: Path,
    meaning: str,
    *,
    username: str,
    display_name: str = "",
    role: str = "",
    comment: str = "",
    reauthenticated: bool = False,
) -> RunStatus:
    """Record a signature; ``approved`` locks the run."""
    from sqlalchemy import select

    from hcs_studio.config import load_settings
    from hcs_studio.db import session_scope
    from hcs_studio.db.security import Signature

    if meaning not in STATEMENTS:
        raise ValueError(f"A signature means reviewed or approved, not {meaning!r}")
    folder = Path(folder)
    current = status(folder)
    settings = load_settings().security
    if current.locked:
        raise RunLocked(f"{folder.name} is already approved")
    if meaning == "reviewed" and current.state != "completed":
        raise ValueError(f"Only a completed run can be signed as reviewed (it is {current.state})")
    if meaning == "approved":
        if current.state != "reviewed":
            raise ValueError(
                f"Sign the run as reviewed before approving it (it is {current.state})"
            )
        reviewers = {s["username"] for s in current.signatures if s["meaning"] == "reviewed"}
        if settings.regulated_mode and username in reviewers:
            raise PermissionError("In regulated mode the approver must differ from the reviewer")
    if settings.reauth_required and not reauthenticated:
        raise PermissionError("Signing needs your password again (re-authentication)")
    with session_scope() as session:
        session.add(
            Signature(
                run=current.run,
                meaning=meaning,
                statement=STATEMENTS[meaning],
                username=username,
                display_name=display_name or username,
                role=role,
                reason=comment,
                manifest_sha256=manifest_sha256(folder),
                reauthenticated=reauthenticated,
            )
        )
    new = _set_state(folder, meaning, locked=meaning == "approved")
    _write_files(folder, new.signatures, new.state)
    if meaning == "approved":
        _set_read_only(folder, True)
    with session_scope() as session:  # the latest signature list, after the write
        rows = session.scalars(select(Signature).where(Signature.run == current.run))
        new.signatures = [_signature(r) for r in rows]
    return new


def unlock(folder: Path) -> RunStatus:
    """Admin action: unlock an approved run (back to reviewed; files writable again)."""
    folder = Path(folder)
    if not status(folder).locked:
        raise ValueError(f"{folder.name} is not locked")
    _set_read_only(folder, False)
    new = _set_state(folder, "reviewed", locked=False)
    _write_files(folder, new.signatures, new.state)
    return new


def _set_read_only(folder: Path, read_only: bool) -> None:
    for path in Path(folder).rglob("*"):
        if path.is_file() and not path.is_symlink():
            mode = path.stat().st_mode
            try:
                os.chmod(path, (mode & ~0o222) if read_only else (mode | stat.S_IWUSR))
            except OSError:
                continue


def _immutable(rel: Path) -> bool:
    return rel.name in IMMUTABLE_FILES or any(part in IMMUTABLE_DIRS for part in rel.parts[:-1])


def new_version(folder: Path, reason: str = "") -> Path:
    """A linked new version of a run to change instead of the locked original."""
    folder = Path(folder).absolute()
    original = status(folder)
    base = folder.name.split("_v")[0] if "_v" in folder.name else folder.name
    version = max(original.version, 1) + 1
    target = folder.parent / f"{base}_v{version}"
    while target.exists():
        version += 1
        target = folder.parent / f"{base}_v{version}"
    target.mkdir()
    try:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            out = target / rel
            if path.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif rel.name != "signatures.json":
                out.parent.mkdir(parents=True, exist_ok=True)
                if _immutable(rel):
                    try:
                        os.link(path, out)
                        continue
                    except OSError:
                        pass
                shutil.copy2(path, out)
                os.chmod(out, out.stat().st_mode | stat.S_IWUSR)
        manifest_path = target / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_version"] = version
        manifest["parent_run"] = original.run
        manifest.setdefault("history", []).append(
            {
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "action": "new_version",
                "from": original.run,
                "reason": reason,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except BaseException:
        _set_read_only(target, False)
        shutil.rmtree(target, ignore_errors=True)
        raise
    _set_state(target, "completed", locked=False)
    return target


def versions_of(folder: Path) -> list[dict[str, Any]]:
    """The chain of versions a run belongs to (from the registry), oldest first."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.security import RunState

    key = run_key(folder)
    with session_scope() as session:
        rows = {r.run: r for r in session.scalars(select(RunState))}
    chain = [key]
    while rows.get(chain[0]) is not None and rows[chain[0]].parent_run:
        chain.insert(0, rows[chain[0]].parent_run)
    children = True
    while children:
        nxt = [r.run for r in rows.values() if r.parent_run == chain[-1]]
        children = bool(nxt)
        if nxt:
            chain.append(nxt[0])
    return [
        {
            "run": run,
            "version": rows[run].version if run in rows else 1,
            "state": rows[run].state if run in rows else "",
            "locked": bool(rows[run].locked) if run in rows else False,
        }
        for run in chain
    ]
