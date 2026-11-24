"""Foundation tables: data roots, service events, jobs (migration 0001).

The experiment catalog (Phase 17) lives in :mod:`hcs_studio.db.catalog`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base, utcnow


class DataRoot(Base):
    """A named data location with one path per operating system (mirrors settings.json)."""

    __tablename__ = "data_root"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    windows: Mapped[str] = mapped_column(String(1024), default="")
    linux: Mapped[str] = mapped_column(String(1024), default="")
    macos: Mapped[str] = mapped_column(String(1024), default="")
    description: Mapped[str] = mapped_column(Text, default="")


class EventRecord(Base):
    """A state change made through the service layer (who, what, why, before/after).

    Phase 26 adds the hash chain; rows are append-only (never updated or deleted).
    """

    __tablename__ = "event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    utc: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    type: Mapped[str] = mapped_column(String(128), index=True)
    actor_id: Mapped[str] = mapped_column(String(256))
    actor_name: Mapped[str] = mapped_column(String(256), default="")
    actor_kind: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, default="")
    object_type: Mapped[str] = mapped_column(String(64), default="")
    object_id: Mapped[str] = mapped_column(String(1024), default="")
    before: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    after: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    client: Mapped[str] = mapped_column(String(256), default="")
    app_version: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | failed | denied
    error: Mapped[str] = mapped_column(Text, default="")
    #: Phase 26: the actor's role and the hash chain (security/audit.py).
    actor_role: Mapped[str] = mapped_column(String(16), default="")
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    hash: Mapped[str] = mapped_column(String(64), default="", index=True)

    __table_args__ = (Index("ix_event_object", "object_type", "object_id"),)


class JobRecord(Base):
    """A queued or finished job (the local queue persists its jobs here)."""

    __tablename__ = "job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), default="pipeline")
    label: Mapped[str] = mapped_column(String(512), default="")
    owner: Mapped[str] = mapped_column(String(256), default="")
    priority: Mapped[str] = mapped_column(String(16), default="interactive")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    folder: Mapped[str] = mapped_column(String(2048), default="")
    resources: Mapped[dict[str, Any]] = mapped_column(default=dict)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    message: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    started: Mapped[datetime | None] = mapped_column(default=None)
    finished: Mapped[datetime | None] = mapped_column(default=None)
