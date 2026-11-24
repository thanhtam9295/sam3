"""Backups of the registry and the audit chain head (Phase 26)."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup_registry(output: Path) -> list[Path]:
    """A consistent copy of the local registry (SQLite online backup, safe while the app runs)
    and of the audit chain-head files, in ``output/registry-<UTC time>/``. Server mode uses the
    database server's own backup (pg_dump)."""
    from hcs_studio.db.engine import database_url
    from hcs_studio.security.audit import checkpoint, checkpoint_files

    url = database_url()
    if not url.startswith("sqlite:///"):
        raise ValueError(
            "Server-mode registries are backed up with the database server's tools "
            "(e.g. pg_dump); this command copies the local SQLite registry"
        )
    source = Path(url.removeprefix("sqlite:///"))
    checkpoint("backup")
    target = Path(output) / f"registry-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    target.mkdir(parents=True)
    copy = target / source.name
    with sqlite3.connect(source) as src, sqlite3.connect(copy) as dst:
        src.backup(dst)
    written = [copy]
    for head in checkpoint_files():
        if head.exists():
            out = target / f"{head.parent.name}_{head.name}"
            shutil.copy2(head, out)
            written.append(out)
    return written
