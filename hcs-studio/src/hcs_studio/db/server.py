"""Tables of the server job queue (Phase 27): jobs, their progress events, the workers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base, utcnow


class QueueJob(Base):
    """A job of the server queue. Workers claim queued jobs (``SKIP LOCKED`` on PostgreSQL),
    keep ``heartbeat`` fresh while they run them, and a job whose worker stopped beating is
    queued again (resuming from its run folder)."""

    __tablename__ = "queue_job"
    __table_args__ = (Index("ix_queue_job_claim", "status", "priority_rank", "created"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), default="pipeline")
    label: Mapped[str] = mapped_column(String(512), default="")
    owner: Mapped[str] = mapped_column(String(256), default="", index=True)
    owner_role: Mapped[str] = mapped_column(String(16), default="")
    priority: Mapped[str] = mapped_column(String(16), default="interactive")
    priority_rank: Mapped[int] = mapped_column(Integer, default=0)
    #: queued | running | completed | completed_with_failures | cancelled | error | interrupted
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    gpu: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    worker: Mapped[str] = mapped_column(String(128), default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    folder: Mapped[str] = mapped_column(String(2048), default="")  # rooted run folder
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    events: Mapped[int] = mapped_column(Integer, default=0)
    failed_wells: Mapped[dict[str, Any]] = mapped_column(default=dict)
    message: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    started: Mapped[datetime | None] = mapped_column(default=None)
    heartbeat: Mapped[datetime | None] = mapped_column(default=None)
    finished: Mapped[datetime | None] = mapped_column(default=None)


class QueueEvent(Base):
    """A progress event of a job (``seq`` from 1), for clients and the WebSocket."""

    __tablename__ = "queue_event"
    __table_args__ = (Index("ix_queue_event_job_seq", "job_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36))
    seq: Mapped[int] = mapped_column(Integer)
    utc: Mapped[datetime] = mapped_column(default=utcnow)
    event: Mapped[dict[str, Any]] = mapped_column(default=dict)


class QueueWorker(Base):
    """A worker process: one per GPU, plus CPU workers."""

    __tablename__ = "queue_worker"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    host: Mapped[str] = mapped_column(String(256), default="")
    pid: Mapped[int] = mapped_column(Integer, default=0)
    gpu: Mapped[int | None] = mapped_column(Integer, default=None)  # GPU index, None = CPU
    status: Mapped[str] = mapped_column(String(16), default="idle")  # idle | busy | stopped
    job_id: Mapped[str] = mapped_column(String(36), default="")
    started: Mapped[datetime] = mapped_column(default=utcnow)
    heartbeat: Mapped[datetime] = mapped_column(default=utcnow)
    message: Mapped[str] = mapped_column(Text, default="")
