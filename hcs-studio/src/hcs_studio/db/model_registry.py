"""Tables of the model registry and of spot checks (Phase 25)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from hcs_studio.db.base import Base, utcnow


class ModelVersion(Base):
    """One registered model version. The file is copied into the model store and never
    changes; status moves candidate → validated → production → retired."""

    __tablename__ = "model_version"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    #: cellpose | steps | classifier | qc_classifier | ilastik | micro_sam
    kind: Mapped[str] = mapped_column(String(32), index=True)
    base_model: Mapped[str] = mapped_column(String(256), default="")
    training_set: Mapped[str] = mapped_column(String(256), default="")  # name@version
    training_set_sha256: Mapped[str] = mapped_column(String(64), default="")
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    #: Object-feature quantiles of the training data, for drift alerts.
    feature_summary: Mapped[dict[str, Any]] = mapped_column(default=dict)
    file: Mapped[str] = mapped_column(String(2048))  # rooted path in the model store
    sha256: Mapped[str] = mapped_column(String(64))
    author: Mapped[str] = mapped_column(String(256), default="")
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    status_changed: Mapped[datetime] = mapped_column(default=utcnow)


class SpotCheck(Base):
    """A field queued for a human spot-check of a run's segmentation (accept or flag)."""

    __tablename__ = "spot_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run: Mapped[str] = mapped_column(String(2048), index=True)  # rooted run folder
    barcode: Mapped[str] = mapped_column(String(256), default="")
    well: Mapped[str] = mapped_column(String(8))
    field: Mapped[int] = mapped_column(Integer)
    labels: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(256), default="", index=True)  # name@version
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    reviewer: Mapped[str] = mapped_column(String(256), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    decided: Mapped[datetime | None] = mapped_column(default=None)
