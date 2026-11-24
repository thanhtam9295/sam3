"""Read-only SQL on the curated snapshot (Phase 28.2 ``run_query``).

Writes are blocked by the connection, not by the prompt: the snapshot is opened with
``mode=ro`` and ``PRAGMA query_only``, and an authorizer allows only reading the curated
views (no ATTACH, no PRAGMA, no writes, no access to other tables). Each query has a row
limit and a time limit.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcs_studio.assistant.views import VIEW_NAMES


class QueryRejected(ValueError):
    """The statement is not a read of the curated views."""


class QueryTimeout(TimeoutError):
    pass


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_s: float

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "truncated": self.truncated,
        }


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_ALLOWED_ACTIONS.add(_RECURSIVE)
#: functions a read query may call (no load_extension, no randomblob floods)
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class ReadOnlyDatabase:
    """A read-only connection to a snapshot. Not thread-safe: one per conversation."""

    def __init__(self, path: Path, max_rows: int = 200, timeout_s: float = 5.0) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.db.execute("PRAGMA query_only = ON")
        self.db.set_authorizer(self._authorize)
        self._deadline = 0.0

    def _authorize(
        self, action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None
    ) -> int:
        if action not in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and db is not None and arg1 not in VIEW_NAMES:
            # a stored table that is not a curated view (sqlite_master, _snapshot, …);
            # common table expressions are read with db None
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _progress(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0

    def query(self, sql: str, max_rows: int | None = None) -> QueryResult:
        text = sql.strip().rstrip(";").strip()
        if not text:
            raise QueryRejected("Empty query")
        first = text.split(None, 1)[0].lower()
        if first not in ("select", "with"):
            raise QueryRejected("Only SELECT queries on the curated views are allowed")
        limit = min(max_rows or self.max_rows, self.max_rows)
        self._deadline = time.monotonic() + self.timeout_s
        self.db.set_progress_handler(self._progress, 10_000)
        started = time.monotonic()
        try:
            cursor = self.db.execute(text)
            rows = cursor.fetchmany(limit + 1)
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "interrupted" in message:
                raise QueryTimeout(f"The query took longer than {self.timeout_s:g} s") from exc
            if "not authorized" in message or "readonly" in message or "read-only" in message:
                raise QueryRejected(f"Not allowed: {message}") from exc
            raise QueryRejected(f"SQL error: {message}") from exc
        except sqlite3.Warning as exc:  # e.g. several statements at once
            raise QueryRejected(f"Not allowed: {exc}") from exc
        finally:
            self.db.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > limit
        out = [[_value(v) for v in r] for r in rows[:limit]]
        return QueryResult(text, columns, out, truncated, time.monotonic() - started)

    def close(self) -> None:
        self.db.close()


def _value(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, bytes):
        return "[binary]"
    return value
