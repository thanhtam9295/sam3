"""Service functions for the experiment catalog (Phase 17a): scans and catalog edits."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from hcs_studio.db import session_scope
from hcs_studio.db.catalog import (
    CellLine,
    Compound,
    EventAnnotation,
    Experiment,
    Plate,
    Project,
    SavedSearch,
)
from hcs_studio.db.queries import PlateQuery
from hcs_studio.db.scanner import ScanReport
from hcs_studio.services import ServiceContext, service

COMPOUND_FIELDS = (
    "external_id",
    "synonyms",
    "moa",
    "target",
    "smiles",
    "supplier",
    "lot",
    "stock_concentration_m",
    "solvent",
    "notes",
)
CELL_LINE_FIELDS = (
    "rrid",
    "cancer_type",
    "passage",
    "authentication_date",
    "mycoplasma_date",
    "notes",
)
EVENT_KINDS = ("reagent_lot", "cell_vial", "instrument_service", "instrument_change", "other")


@service("registry.scan", object_type="registry")
def scan_folders(
    ctx: ServiceContext, paths: Iterable[Path], *, raw: bool = False, force: bool = False
) -> ScanReport:
    """Index run folders (and, with ``raw``, raw plate folders) below ``paths``."""
    from hcs_studio.db.scanner import scan

    paths = [Path(p) for p in paths]
    with session_scope() as session:
        report = scan(session, paths, raw=raw, force=force)
    ctx.note(object_id=";".join(str(p) for p in paths), after=report.summary())
    return report


def _fields(obj: Any, names: Iterable[str]) -> dict[str, Any]:
    return {n: getattr(obj, n) for n in names}


@service("registry.compound", object_type="compound")
def save_compound(ctx: ServiceContext, name: str, **values: Any) -> None:
    unknown = set(values) - set(COMPOUND_FIELDS)
    if unknown:
        raise ValueError(f"Unknown compound field(s): {', '.join(sorted(unknown))}")
    with session_scope() as session:
        compound = session.scalars(select(Compound).where(Compound.name == name)).first()
        before = _fields(compound, COMPOUND_FIELDS) if compound else None
        if compound is None:
            compound = Compound(name=name)
            session.add(compound)
        for key, value in values.items():
            setattr(compound, key, value)
        session.flush()
        ctx.note(object_id=name, before=before, after=_fields(compound, COMPOUND_FIELDS))


@service("registry.cell_line", object_type="cell_line")
def save_cell_line(ctx: ServiceContext, name: str, **values: Any) -> None:
    unknown = set(values) - set(CELL_LINE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown cell-line field(s): {', '.join(sorted(unknown))}")
    with session_scope() as session:
        line = session.scalars(select(CellLine).where(CellLine.name == name)).first()
        before = _fields(line, CELL_LINE_FIELDS) if line else None
        if line is None:
            line = CellLine(name=name)
            session.add(line)
        for key, value in values.items():
            setattr(line, key, value)
        session.flush()
        ctx.note(object_id=name, before=before, after=_fields(line, CELL_LINE_FIELDS))


@service("registry.experiment", object_type="experiment")
def save_experiment(
    ctx: ServiceContext,
    name: str,
    *,
    project: str = "",
    assay: str = "",
    purpose: str = "",
    owner: str = "",
    plates: Iterable[str] = (),
) -> int:
    """Create or update an experiment and assign plates (by barcode or name) to it."""
    with session_scope() as session:
        project_row = None
        if project:
            project_row = session.scalars(select(Project).where(Project.name == project)).first()
            if project_row is None:
                project_row = Project(name=project, owner=owner or ctx.actor.id)
                session.add(project_row)
                session.flush()
        stmt = select(Experiment).where(Experiment.name == name)
        stmt = stmt.where(
            Experiment.project_id == project_row.id
            if project_row
            else Experiment.project_id.is_(None)
        )
        experiment = session.scalars(stmt).first()
        if experiment is None:
            experiment = Experiment(name=name, project_id=project_row.id if project_row else None)
            session.add(experiment)
        experiment.assay = assay or experiment.assay
        experiment.purpose = purpose or experiment.purpose
        experiment.owner = owner or experiment.owner or ctx.actor.id
        session.flush()
        assigned = []
        for key in plates:
            plate = session.scalars(
                select(Plate).where((Plate.barcode == key) | (Plate.name == key))
            ).first()
            if plate is None:
                raise KeyError(f"No plate {key!r} in the registry (scan its run first)")
            plate.experiment_id = experiment.id
            assigned.append(key)
        ctx.note(
            object_id=f"{project}/{name}" if project else name,
            after={"assay": experiment.assay, "plates": assigned},
        )
        return experiment.id


@service("registry.event_annotation", object_type="event_annotation")
def add_event_annotation(
    ctx: ServiceContext,
    kind: str,
    on: date,
    *,
    instrument: str = "",
    assay: str = "",
    cell_line: str = "",
    description: str = "",
) -> int:
    """Record something that may shift an assay (new reagent lot, cell vial, service …)."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"Unknown event kind {kind!r} ({', '.join(EVENT_KINDS)})")
    with session_scope() as session:
        row = EventAnnotation(
            kind=kind,
            on=on,
            instrument=instrument,
            assay=assay,
            cell_line=cell_line,
            description=description,
            actor=ctx.actor.id,
        )
        session.add(row)
        session.flush()
        ctx.note(object_id=str(row.id), after={"kind": kind, "on": on.isoformat()})
        return row.id


@service("registry.saved_search", object_type="saved_search")
def save_search(ctx: ServiceContext, name: str, query: PlateQuery) -> None:
    with session_scope() as session:
        row = session.scalars(select(SavedSearch).where(SavedSearch.name == name)).first()
        before = dict(row.query) if row else None
        if row is None:
            row = SavedSearch(name=name, owner=ctx.actor.id)
            session.add(row)
        row.query = query.to_dict()
        ctx.note(object_id=name, before=before, after=query.to_dict())


@service("registry.saved_search_delete", object_type="saved_search")
def delete_search(ctx: ServiceContext, name: str) -> None:
    with session_scope() as session:
        row = session.scalars(select(SavedSearch).where(SavedSearch.name == name)).first()
        if row is None:
            raise KeyError(f"No saved search {name!r}")
        ctx.note(object_id=name, before=dict(row.query))
        session.delete(row)


@service("registry.uniformity_map", object_type="uniformity_map")
def save_uniformity_map(
    ctx: ServiceContext,
    name: str,
    run: str,
    readout: str,
    *,
    degree: int = 2,
    edge_rings: int = 1,
    instrument: str = "",
    assay: str = "",
    replace: bool = False,
) -> dict[str, Any]:
    """Fit a correction map on a uniformity plate (a run of a vehicle-only plate) and store it.
    Replacing an existing map needs ``replace`` (and is recorded with the old statistics)."""
    from hcs_studio.db.catalog import UniformityMapRecord
    from hcs_studio.db.normalization import find_run, plate_values
    from hcs_studio.normalization.uniformity import build_map

    with session_scope() as session:
        source = find_run(session, run)
        data = plate_values(session, source, readout)
        roles = set(data.roles().values())
        if roles - {"negative_control", "untreated", "sample"} or any(
            w.compounds for w in data.wells
        ):
            raise ValueError(
                f"Run {source.name} is not a uniformity plate: its wells hold "
                f"{', '.join(sorted(roles))} or compounds; use a vehicle-only plate"
            )
        mapped = build_map(data.grid, degree, edge_rings)
        record = session.scalars(
            select(UniformityMapRecord).where(UniformityMapRecord.name == name)
        ).first()
        before = dict(record.stats) if record else None
        if record is not None and not replace:
            raise ValueError(f"A uniformity map {name!r} exists; replace it explicitly")
        if record is None:
            record = UniformityMapRecord(name=name)
            session.add(record)
        record.instrument = instrument or data.instrument
        record.assay = assay or (source.plate.assay if source.plate else "")
        record.readout_id = readout
        record.rows, record.cols = data.rows, data.cols
        record.degree = degree
        record.factors = mapped.factors.tolist()
        record.stats = {**mapped.stats, "edge_rings": edge_rings}
        record.source_run_id = source.id
        record.actor = ctx.actor.id
        ctx.note(object_id=name, before=before, after={"run": source.name, **mapped.stats})
        return dict(mapped.stats)
