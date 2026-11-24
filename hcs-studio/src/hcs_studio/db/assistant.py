"""Token usage of the lab assistant (Phase 28.4): one row per user, month and model, for the
per-user monthly budget and the usage shown in settings."""

from __future__ import annotations

from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base


class AssistantUsage(Base):
    __tablename__ = "assistant_usage"
    __table_args__ = (UniqueConstraint("user", "month", "model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String(256), index=True)
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM (UTC)
    model: Mapped[str] = mapped_column(String(64))
    requests: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_creation_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
