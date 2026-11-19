"""Registry database: engines, migrations, the network-share guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from hcs_studio.db import engine as db
from hcs_studio.db.models import EventRecord


def test_migrations_build_exactly_the_model_schema(tmp_path: Path) -> None:
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    from hcs_studio.db.schema import metadata

    eng = db.make_engine(f"sqlite:///{tmp_path / 'r.sqlite'}")
    assert db.current_revision(eng) == db.head_revision()
    with eng.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), metadata)
        assert diff == [], diff  # a model change without a migration fails here
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    db.upgrade(eng)  # idempotent
    eng.dispose()


def test_sessions_commit_and_roll_back(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 's.sqlite'}"
    with db.session_scope(url) as s:
        s.add(EventRecord(type="test.ok", actor_id="me", actor_kind="os"))
    with pytest.raises(RuntimeError), db.session_scope(url) as s:
        s.add(EventRecord(type="test.rolled_back", actor_id="me", actor_kind="os"))
        raise RuntimeError("boom")
    with db.session_scope(url) as s:
        assert [e.type for e in s.query(EventRecord)] == ["test.ok"]
    db.dispose_engines()


MOUNTS = """\
/dev/vda / ext4 rw 0 0
server:/export /mnt/lab nfs4 rw 0 0
//nas/share /mnt/nas cifs rw 0 0
/dev/vdb /mnt/lab/local ext4 rw 0 0
"""


def test_sqlite_on_a_network_share_is_refused() -> None:
    mounts = db._linux_mounts(MOUNTS)
    assert db.filesystem_type(Path("/mnt/lab/x/r.sqlite"), mounts) == "nfs4"
    assert db.filesystem_type(Path("/mnt/lab/local/r.sqlite"), mounts) == "ext4"  # nested
    assert db.is_network_path(Path("/mnt/nas/r.sqlite"), mounts)
    assert not db.is_network_path(Path("/home/me/r.sqlite"), mounts)
    assert not db.is_network_path(Path("/mnt/laboratory/r.sqlite"), mounts)  # not a prefix
    with pytest.raises(db.NetworkDatabaseError, match="server mode"):
        db.check_local_disk(Path("/mnt/lab/registry.sqlite"), mounts)
    mac = db._mac_mounts("//me@nas/lab on /Volumes/lab (smbfs, nodev, nosuid)\n")
    assert mac == [("/Volumes/lab", "smbfs")]


def test_database_url_from_settings_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    assert db.database_url() == f"sqlite:///{tmp_path / 'registry.sqlite'}"
    from hcs_studio.config import AppSettings, DatabaseConfig, save_settings

    save_settings(AppSettings(database=DatabaseConfig(mode="server")))
    with pytest.raises(ValueError, match="database URL"):
        db.database_url()
    monkeypatch.setenv("HCS_DATABASE_URL", "sqlite:///:memory:")
    assert db.database_url() == "sqlite:///:memory:"
