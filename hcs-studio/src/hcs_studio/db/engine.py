"""Engines, sessions and migrations.

Local mode: SQLite at ``<settings>/registry.sqlite`` (WAL, foreign keys on). SQLite must
never sit on a network share used by several computers (file locking there is unreliable
and corrupts databases): :func:`check_local_disk` refuses it, and shared use means server
mode (PostgreSQL). Every engine is migrated to the newest schema when first opened.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from hcs_studio.io.sidecar import config_dir

MIGRATIONS = Path(__file__).with_name("migrations")
NETWORK_FILESYSTEMS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb",
    "smb2",
    "smb3",
    "smbfs",
    "afs",
    "fuse.sshfs",
    "sshfs",
    "davfs",
    "fuse.davfs2",
    "glusterfs",
    "fuse.glusterfs",
    "ceph",
    "lustre",
    "9p",
    "webdav",
}

_ENGINES: dict[str, Engine] = {}
_LOCK = threading.Lock()


class NetworkDatabaseError(RuntimeError):
    pass


def default_url() -> str:
    return f"sqlite:///{config_dir() / 'registry.sqlite'}"


def database_url() -> str:
    """The URL from settings (``HCS_DATABASE_URL`` overrides; tests and servers use it)."""
    if os.environ.get("HCS_DATABASE_URL"):
        return os.environ["HCS_DATABASE_URL"]
    from hcs_studio.config import load_settings

    settings = load_settings().database
    if settings.mode == "server":
        if not settings.url:
            raise ValueError("Server mode needs a database URL (Settings → Database)")
        url = make_url(settings.url)
        if url.password is None:
            from hcs_studio.secrets import get_secret

            password = get_secret("database/password")
            if password:
                url = url.set(password=password)
        return url.render_as_string(hide_password=False)
    return settings.url or default_url()


# ---------------------------------------------------------------- network check


def _linux_mounts(mounts_text: str) -> list[tuple[str, str]]:
    out = []
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((parts[1].replace("\\040", " "), parts[2]))
    return out


def _mac_mounts(mount_output: str) -> list[tuple[str, str]]:
    out = []
    for line in mount_output.splitlines():  # "//u@srv/share on /Volumes/share (smbfs, …)"
        if " on " in line and "(" in line:
            point = line.split(" on ", 1)[1].rsplit(" (", 1)[0]
            fstype = line.rsplit("(", 1)[1].split(",")[0].strip(") ")
            out.append((point, fstype))
    return out


def filesystem_type(path: Path, mounts: list[tuple[str, str]] | None = None) -> str:
    """The filesystem type of the mount holding ``path`` (Linux and macOS)."""
    if mounts is None:
        if sys.platform == "darwin":
            output = subprocess.run(["mount"], capture_output=True, text=True).stdout
            mounts = _mac_mounts(output)
        else:
            try:
                mounts = _linux_mounts(Path("/proc/mounts").read_text(encoding="utf-8"))
            except OSError:
                return ""
    target = str(Path(path).absolute())
    best = ("", "")
    for point, fstype in mounts:
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) > len(best[0]):
            best = (point, fstype)
    return best[1]


def is_network_path(path: Path, mounts: list[tuple[str, str]] | None = None) -> bool:
    text = str(path)
    if sys.platform.startswith("win"):
        if text.startswith(("\\\\", "//")):
            return True
        drive = os.path.splitdrive(os.path.abspath(text))[0]
        if drive:
            import ctypes

            DRIVE_REMOTE = 4
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
            return bool(kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE)
        return False
    if text.startswith("//"):
        return True
    return filesystem_type(path, mounts).lower() in NETWORK_FILESYSTEMS


def check_local_disk(db_file: Path, mounts: list[tuple[str, str]] | None = None) -> None:
    if is_network_path(db_file.parent if not db_file.exists() else db_file, mounts):
        raise NetworkDatabaseError(
            f"The registry database {db_file} is on a network share. SQLite databases on "
            "network shares get corrupted when several computers use them. Keep the local "
            "database on this computer's own disk (the default), or use server mode "
            "(PostgreSQL) for a database shared by several computers."
        )


# ------------------------------------------------------------------- engines


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def make_engine(url: str, migrate: bool = True) -> Engine:
    parsed = make_url(url)
    kwargs: dict[str, Any] = {"future": True}
    if parsed.get_backend_name() == "sqlite":
        database = parsed.database or ""
        if database and database != ":memory:":
            path = Path(database)
            path.parent.mkdir(parents=True, exist_ok=True)
            check_local_disk(path)
        kwargs["connect_args"] = {"timeout": 30}
    engine = create_engine(url, **kwargs)
    if parsed.get_backend_name() == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    if migrate:
        upgrade(engine)
    return engine


def get_engine(url: str | None = None) -> Engine:
    """The shared engine for ``url`` (default: from settings), migrated to the newest schema."""
    url = url or database_url()
    with _LOCK:
        engine = _ENGINES.get(url)
        if engine is None:
            engine = make_engine(url)
            _ENGINES[url] = engine
        return engine


def dispose_engines() -> None:
    with _LOCK:
        for engine in _ENGINES.values():
            engine.dispose()
        _ENGINES.clear()


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """A session that commits on success and rolls back on error."""
    factory = sessionmaker(bind=get_engine(url), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------- migrations


def alembic_config(engine: Engine | None = None) -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    if engine is not None:
        cfg.attributes["engine"] = engine
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    from alembic import command

    with engine.begin() as connection:
        cfg = alembic_config()
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, revision)


def current_revision(engine: Engine) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def head_revision() -> str | None:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config()).get_current_head()
