"""The registry database (PROMPT_PLATFORM.md §2.1–2.2).

SQLAlchemy 2 models with Alembic migrations: SQLite in local mode (a file in the settings
folder, never on a network share), PostgreSQL in server mode, same code. Large per-object
data stays in Parquet files in run folders; the database holds the catalog and per-well
results.
"""

from hcs_studio.db.engine import get_engine, session_scope

__all__ = ["get_engine", "session_scope"]
