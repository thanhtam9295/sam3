"""Tables for accounts, the audit-chain head and run sign-off (Phase 26)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base, utcnow


class Account(Base):
    """A local account: role, optional argon2 password hash, lockout state."""

    __tablename__ = "account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(256), unique=True)
    display_name: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[str] = mapped_column(String(16), default="analyst")
    password_hash: Mapped[str] = mapped_column(String(512), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(default=None)
    created: Mapped[datetime] = mapped_column(default=utcnow)
    last_login: Mapped[datetime | None] = mapped_column(default=None)


class AuditHead(Base):
    """One row: the last chained event (id and hash). Updating it first serialises writers,
    so two processes never chain to the same predecessor."""

    __tablename__ = "audit_head"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_event_id: Mapped[int] = mapped_column(Integer, default=0)
    last_hash: Mapped[str] = mapped_column(String(64), default="")
    count: Mapped[int] = mapped_column(Integer, default=0)


class RunState(Base):
    """Review state of a run: draft → completed → reviewed → approved (locked)."""

    __tablename__ = "run_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run: Mapped[str] = mapped_column(String(2048), unique=True)  # rooted run folder
    state: Mapped[str] = mapped_column(String(16), default="completed", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_run: Mapped[str] = mapped_column(String(2048), default="")
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    updated: Mapped[datetime] = mapped_column(default=utcnow)


class Signature(Base):
    """An electronic signature of a run (reviewed or approved)."""

    __tablename__ = "signature"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run: Mapped[str] = mapped_column(String(2048), index=True)
    meaning: Mapped[str] = mapped_column(String(16))  # reviewed | approved
    statement: Mapped[str] = mapped_column(Text, default="")
    username: Mapped[str] = mapped_column(String(256))
    display_name: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[str] = mapped_column(String(16), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    manifest_sha256: Mapped[str] = mapped_column(String(64), default="")
    reauthenticated: Mapped[bool] = mapped_column(Boolean, default=False)
    utc: Mapped[datetime] = mapped_column(default=utcnow)
