"""Tables of automatic runs (Phase 19): the lab-event log and one row per acquisition."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base, utcnow


class LabEventRecord(Base):
    """Every lab event received or sent, valid or not (the event-log viewer reads it)."""

    __tablename__ = "lab_event"
    __table_args__ = (Index("ix_lab_event_time", "logged_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    direction: Mapped[str] = mapped_column(String(8))  # in | out
    type: Mapped[str] = mapped_column(String(64), default="")
    channel: Mapped[str] = mapped_column(String(32), default="")  # http | inbox | watch | outbox
    #: accepted | duplicate | rejected | ignored | sent | send_failed
    status: Mapped[str] = mapped_column(String(32), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    acquisition_key: Mapped[str] = mapped_column(String(512), default="", index=True)
    barcode: Mapped[str] = mapped_column(String(256), default="", index=True)
    message: Mapped[dict[str, Any]] = mapped_column(default=dict)
    logged_at: Mapped[datetime] = mapped_column(default=utcnow)


class AutoRun(Base):
    """One acquisition handled automatically; the key makes each acquisition run once."""

    __tablename__ = "auto_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(512), unique=True)
    barcode: Mapped[str] = mapped_column(String(256), default="", index=True)
    acquisition_id: Mapped[str] = mapped_column(String(256), default="")
    instrument: Mapped[str] = mapped_column(String(256), default="")
    protocol: Mapped[str] = mapped_column(String(256), default="")
    time_point: Mapped[int | None] = mapped_column(Integer, default=None)
    raw: Mapped[str] = mapped_column(String(2048), default="")  # rooted or absolute
    event_id: Mapped[str] = mapped_column(String(64), default="")
    #: pending | waiting_data | running | completed | qc_failed | failed | unrouted
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    route: Mapped[str] = mapped_column(String(256), default="")
    job_id: Mapped[str] = mapped_column(String(36), default="")
    run_folder: Mapped[str] = mapped_column(String(2048), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    updated: Mapped[datetime] = mapped_column(default=utcnow)
