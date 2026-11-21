"""The experiment catalog (Phase 17a, migration 0002).

Project → Experiment → Plate → Acquisition (one imaging session) → Run (one analysis) →
results: per-well readouts in long form, dose–response fits, combination scores, plate QC.
Plus reference tables: compounds, library plates, cell lines, event annotations, saved
searches. Paths are stored rooted (``ROOT:relative``, see ``hcs_studio.paths``). Per-object
data stays in the runs' Parquet files.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hcs_studio.db.base import Base, utcnow

plate_cell_line = Table(
    "plate_cell_line",
    Base.metadata,
    Column("plate_id", ForeignKey("plate.id", ondelete="CASCADE"), primary_key=True),
    Column("cell_line_id", ForeignKey("cell_line.id", ondelete="CASCADE"), primary_key=True),
)

run_acquisition = Table(
    "run_acquisition",
    Base.metadata,
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), primary_key=True),
    Column("acquisition_id", ForeignKey("acquisition.id", ondelete="CASCADE"), primary_key=True),
)


class Project(Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    owner: Mapped[str] = mapped_column(String(256), default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    experiments: Mapped[list[Experiment]] = relationship(back_populates="project")


class Experiment(Base):
    __tablename__ = "experiment"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("project.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(256))
    assay: Mapped[str] = mapped_column(String(256), default="", index=True)
    purpose: Mapped[str] = mapped_column(Text, default="")
    owner: Mapped[str] = mapped_column(String(256), default="")
    start_date: Mapped[date | None] = mapped_column(Date, default=None)
    end_date: Mapped[date | None] = mapped_column(Date, default=None)
    notes: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    project: Mapped[Project | None] = relationship(back_populates="experiments")
    plates: Mapped[list[Plate]] = relationship(back_populates="experiment")


class CellLine(Base):
    __tablename__ = "cell_line"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    rrid: Mapped[str] = mapped_column(String(64), default="")  # Cellosaurus, e.g. CVCL_0031
    cancer_type: Mapped[str] = mapped_column(String(256), default="")
    passage: Mapped[str] = mapped_column(String(64), default="")
    authentication_date: Mapped[date | None] = mapped_column(Date, default=None)
    mycoplasma_date: Mapped[date | None] = mapped_column(Date, default=None)
    notes: Mapped[str] = mapped_column(Text, default="")


class Compound(Base):
    __tablename__ = "compound"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    external_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    synonyms: Mapped[list[Any]] = mapped_column(default=list)
    moa: Mapped[str] = mapped_column(String(256), default="")
    target: Mapped[str] = mapped_column(String(256), default="")
    smiles: Mapped[str] = mapped_column(Text, default="")
    supplier: Mapped[str] = mapped_column(String(256), default="")
    lot: Mapped[str] = mapped_column(String(128), default="")
    stock_concentration_m: Mapped[float | None] = mapped_column(Float, default=None)
    solvent: Mapped[str] = mapped_column(String(64), default="")
    notes: Mapped[str] = mapped_column(Text, default="")


class LibraryPlate(Base):
    __tablename__ = "library_plate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barcode: Mapped[str] = mapped_column(String(128), unique=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    wells: Mapped[list[LibraryWell]] = relationship(
        back_populates="library_plate", cascade="all, delete-orphan"
    )


class LibraryWell(Base):
    __tablename__ = "library_well"
    __table_args__ = (UniqueConstraint("library_plate_id", "well"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    library_plate_id: Mapped[int] = mapped_column(
        ForeignKey("library_plate.id", ondelete="CASCADE")
    )
    well: Mapped[str] = mapped_column(String(8))
    compound_id: Mapped[int | None] = mapped_column(ForeignKey("compound.id", ondelete="SET NULL"))
    concentration_m: Mapped[float | None] = mapped_column(Float, default=None)
    volume_nl: Mapped[float | None] = mapped_column(Float, default=None)
    library_plate: Mapped[LibraryPlate] = relationship(back_populates="wells")


class Plate(Base):
    __tablename__ = "plate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: barcode:<barcode>, else name:<name>|<rooted raw path> (stable across scans)
    key: Mapped[str] = mapped_column(String(2048), unique=True)
    barcode: Mapped[str] = mapped_column(String(128), default="", index=True)
    name: Mapped[str] = mapped_column(String(512), default="")
    rows: Mapped[int] = mapped_column(Integer, default=0)
    cols: Mapped[int] = mapped_column(Integer, default=0)
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiment.id", ondelete="SET NULL")
    )
    layout_version: Mapped[int | None] = mapped_column(Integer, default=None)
    layout_sha256: Mapped[str] = mapped_column(String(64), default="")
    assay: Mapped[str] = mapped_column(String(256), default="")
    seeded_at: Mapped[datetime | None] = mapped_column(default=None)
    treated_at: Mapped[datetime | None] = mapped_column(default=None)
    notes: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    updated: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    experiment: Mapped[Experiment | None] = relationship(back_populates="plates")
    cell_lines: Mapped[list[CellLine]] = relationship(secondary=plate_cell_line)
    acquisitions: Mapped[list[Acquisition]] = relationship(
        back_populates="plate", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="plate")
    wells: Mapped[list[PlateWell]] = relationship(
        back_populates="plate", cascade="all, delete-orphan"
    )

    @property
    def plate_type(self) -> int:
        return self.rows * self.cols


class PlateWell(Base):
    """The annotation of a well (from the plate's layout), for queries by condition."""

    __tablename__ = "plate_well"
    __table_args__ = (UniqueConstraint("plate_id", "well"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plate_id: Mapped[int] = mapped_column(ForeignKey("plate.id", ondelete="CASCADE"))
    well: Mapped[str] = mapped_column(String(8))
    cell_line: Mapped[str] = mapped_column(String(256), default="", index=True)
    role: Mapped[str] = mapped_column(String(32), default="")
    treatment: Mapped[str] = mapped_column(String(1024), default="")
    compounds: Mapped[list[Any]] = mapped_column(default=list)  # [[name, concentration_base]]
    plate: Mapped[Plate] = relationship(back_populates="wells")


class Acquisition(Base):
    """One imaging session of a plate (time courses and re-imaging add more)."""

    __tablename__ = "acquisition"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: <rooted raw path>|t<time point>
    key: Mapped[str] = mapped_column(String(2048), unique=True)
    plate_id: Mapped[int] = mapped_column(ForeignKey("plate.id", ondelete="CASCADE"))
    instrument: Mapped[str] = mapped_column(String(256), default="")
    protocol: Mapped[str] = mapped_column(String(256), default="")
    started: Mapped[datetime | None] = mapped_column(default=None)
    finished: Mapped[datetime | None] = mapped_column(default=None)
    time_point_index: Mapped[int | None] = mapped_column(Integer, default=None)
    raw_path: Mapped[str] = mapped_column(String(2048), default="")  # rooted
    status: Mapped[str] = mapped_column(String(32), default="imaged")
    created: Mapped[datetime] = mapped_column(default=utcnow)
    plate: Mapped[Plate] = relationship(back_populates="acquisitions")


class Run(Base):
    """One analysis (a run folder) of one or more acquisitions."""

    __tablename__ = "run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    folder: Mapped[str] = mapped_column(String(2048), unique=True)  # rooted
    name: Mapped[str] = mapped_column(String(512), default="")
    plate_id: Mapped[int | None] = mapped_column(ForeignKey("plate.id", ondelete="SET NULL"))
    pipeline_name: Mapped[str] = mapped_column(String(256), default="")
    pipeline_hash: Mapped[str] = mapped_column(String(64), default="")
    software_version: Mapped[str] = mapped_column(String(64), default="")
    git_commit: Mapped[str] = mapped_column(String(64), default="")
    models: Mapped[list[Any]] = mapped_column(default=list)
    status: Mapped[str] = mapped_column(String(32), default="", index=True)
    started: Mapped[datetime | None] = mapped_column(default=None, index=True)
    finished: Mapped[datetime | None] = mapped_column(default=None)
    actor: Mapped[str] = mapped_column(String(256), default="")
    fingerprint: Mapped[str] = mapped_column(String(64), default="")  # what was indexed
    indexed_at: Mapped[datetime] = mapped_column(default=utcnow)
    #: Image QC of the run (Phase 18): passed, reasons, counts, rules; empty without QC.
    image_qc: Mapped[dict[str, Any]] = mapped_column(default=dict)
    plate: Mapped[Plate | None] = relationship(back_populates="runs")
    acquisitions: Mapped[list[Acquisition]] = relationship(secondary=run_acquisition)


class WellResult(Base):
    """A per-well readout, in long form."""

    __tablename__ = "well_result"
    __table_args__ = (
        Index("ix_well_result_run_readout", "run_id", "readout_id"),
        Index("ix_well_result_readout", "readout_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"))
    well: Mapped[str] = mapped_column(String(8))
    readout_id: Mapped[str] = mapped_column(String(256))
    value: Mapped[float | None] = mapped_column(Float, default=None)
    n_objects: Mapped[int | None] = mapped_column(Integer, default=None)
    qc_flag: Mapped[str] = mapped_column(String(64), default="")


class DoseResponseFit(Base):
    __tablename__ = "dose_response_fit"
    __table_args__ = (Index("ix_fit_compound_line", "compound", "cell_line"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), index=True)
    readout_id: Mapped[str] = mapped_column(String(256))
    cell_line: Mapped[str] = mapped_column(String(256), default="")
    compound: Mapped[str] = mapped_column(String(256), default="")
    unit: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(32), default="")
    ec50_m: Mapped[float | None] = mapped_column(Float, default=None)
    ec50_ci_low_m: Mapped[float | None] = mapped_column(Float, default=None)
    ec50_ci_high_m: Mapped[float | None] = mapped_column(Float, default=None)
    log_ec50: Mapped[float | None] = mapped_column(Float, default=None)
    abs_ic50_m: Mapped[float | None] = mapped_column(Float, default=None)
    hill: Mapped[float | None] = mapped_column(Float, default=None)
    top: Mapped[float | None] = mapped_column(Float, default=None)
    bottom: Mapped[float | None] = mapped_column(Float, default=None)
    emax: Mapped[float | None] = mapped_column(Float, default=None)
    auc: Mapped[float | None] = mapped_column(Float, default=None)
    r2: Mapped[float | None] = mapped_column(Float, default=None)
    n_points: Mapped[int | None] = mapped_column(Integer, default=None)
    flags: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[dict[str, Any]] = mapped_column(default=dict)


class CombinationScore(Base):
    __tablename__ = "combination_score"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), index=True)
    readout_id: Mapped[str] = mapped_column(String(256))
    cell_line: Mapped[str] = mapped_column(String(256), default="")
    drug_a: Mapped[str] = mapped_column(String(256), default="")
    drug_b: Mapped[str] = mapped_column(String(256), default="")
    model: Mapped[str] = mapped_column(String(16))
    mean: Mapped[float | None] = mapped_column(Float, default=None)
    max_3x3: Mapped[float | None] = mapped_column(Float, default=None)
    p: Mapped[float | None] = mapped_column(Float, default=None)
    label: Mapped[str] = mapped_column(String(64), default="")


class PlateQCRecord(Base):
    __tablename__ = "plate_qc"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), index=True)
    readout_id: Mapped[str] = mapped_column(String(256))
    z_prime: Mapped[float | None] = mapped_column(Float, default=None)
    ssmd: Mapped[float | None] = mapped_column(Float, default=None)
    signal_window: Mapped[float | None] = mapped_column(Float, default=None)
    cv_vehicle_pct: Mapped[float | None] = mapped_column(Float, default=None)
    cv_positive_pct: Mapped[float | None] = mapped_column(Float, default=None)
    vehicle_mean: Mapped[float | None] = mapped_column(Float, default=None)
    positive_mean: Mapped[float | None] = mapped_column(Float, default=None)
    n_vehicle: Mapped[int | None] = mapped_column(Integer, default=None)
    n_positive: Mapped[int | None] = mapped_column(Integer, default=None)
    flag: Mapped[str] = mapped_column(String(256), default="")


class EventAnnotation(Base):
    """Something that may change assay behaviour (shown on control charts, Phase 17c)."""

    __tablename__ = "event_annotation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))  # reagent_lot, cell_vial, service, …
    on: Mapped[date] = mapped_column(Date)
    instrument: Mapped[str] = mapped_column(String(256), default="")
    assay: Mapped[str] = mapped_column(String(256), default="")
    cell_line: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    actor: Mapped[str] = mapped_column(String(256), default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)


class SavedSearch(Base):
    __tablename__ = "saved_search"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    owner: Mapped[str] = mapped_column(String(256), default="")
    query: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created: Mapped[datetime] = mapped_column(default=utcnow)


class UniformityMapRecord(Base):
    """A spatial correction map from a uniformity plate (Phase 17b), for later plates of
    the same instrument, plate type and assay."""

    __tablename__ = "uniformity_map"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    instrument: Mapped[str] = mapped_column(String(256), default="")
    assay: Mapped[str] = mapped_column(String(256), default="")
    readout_id: Mapped[str] = mapped_column(String(256))
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    degree: Mapped[int] = mapped_column(Integer, default=2)
    factors: Mapped[list[Any]] = mapped_column(default=list)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("run.id", ondelete="SET NULL"), default=None
    )
    actor: Mapped[str] = mapped_column(String(256), default="")
    created: Mapped[datetime] = mapped_column(default=utcnow)
