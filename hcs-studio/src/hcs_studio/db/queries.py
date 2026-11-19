"""Read-only queries of the experiment catalog (Phase 17a): plate search, plate pages,
potency across runs, readouts over time, compound and cell-line pages, exports.

Every function takes a session and returns plain data (DataFrames or dicts), so the GUI, the
command line and (Phase 28) the assistant share them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from hcs_studio.db.catalog import (
    Acquisition,
    CellLine,
    CombinationScore,
    Compound,
    DoseResponseFit,
    Experiment,
    Plate,
    PlateQCRecord,
    PlateWell,
    Run,
    SavedSearch,
    WellResult,
    plate_cell_line,
)


@dataclass
class PlateQuery:
    """Search criteria (all optional; text matches barcode, plate name, experiment, assay)."""

    text: str = ""
    barcode: str = ""
    experiment: str = ""
    assay: str = ""
    cell_line: str = ""
    compound: str = ""
    instrument: str = ""
    run_status: str = ""
    date_from: date | None = None
    date_to: date | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("date_from", "date_to"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlateQuery:
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("date_from", "date_to"):
            if values.get(key):
                values[key] = date.fromisoformat(str(values[key]))
        return cls(**values)


def _started_key(run: Run) -> float:
    return run.started.timestamp() if run.started else 0.0


def _like(text: str) -> str:
    return f"%{text.strip()}%"


def search_plates(session: Session, query: PlateQuery | None = None) -> pd.DataFrame:
    """Plates matching ``query``, newest run first, with their latest run."""
    q = query or PlateQuery()
    stmt = select(Plate).outerjoin(Experiment, Plate.experiment_id == Experiment.id)
    if q.text:
        t = _like(q.text)
        stmt = stmt.where(
            or_(
                Plate.barcode.ilike(t),
                Plate.name.ilike(t),
                Plate.assay.ilike(t),
                Experiment.name.ilike(t),
                Experiment.assay.ilike(t),
            )
        )
    if q.barcode:
        stmt = stmt.where(Plate.barcode.ilike(_like(q.barcode)))
    if q.experiment:
        stmt = stmt.where(Experiment.name.ilike(_like(q.experiment)))
    if q.assay:
        stmt = stmt.where(
            or_(Plate.assay.ilike(_like(q.assay)), Experiment.assay.ilike(_like(q.assay)))
        )
    if q.cell_line:
        stmt = stmt.where(
            Plate.id.in_(
                select(plate_cell_line.c.plate_id)
                .join(CellLine, CellLine.id == plate_cell_line.c.cell_line_id)
                .where(CellLine.name.ilike(_like(q.cell_line)))
            )
        )
    if q.compound:
        # compounds are stored per well as JSON; a LIKE on the text form is portable
        stmt = stmt.where(
            Plate.id.in_(
                select(PlateWell.plate_id).where(
                    cast(PlateWell.compounds, String).ilike(_like(q.compound))
                )
            )
        )
    if q.instrument:
        stmt = stmt.where(
            Plate.id.in_(
                select(Acquisition.plate_id).where(
                    Acquisition.instrument.ilike(_like(q.instrument))
                )
            )
        )
    if q.run_status or q.date_from or q.date_to:
        runs = select(Run.plate_id)
        if q.run_status:
            runs = runs.where(Run.status == q.run_status)
        if q.date_from:
            runs = runs.where(Run.started >= datetime.combine(q.date_from, datetime.min.time()))
        if q.date_to:
            runs = runs.where(Run.started <= datetime.combine(q.date_to, datetime.max.time()))
        stmt = stmt.where(Plate.id.in_(runs))
    rows = []
    for plate in session.scalars(stmt.distinct()).all():
        latest = max(plate.runs, key=_started_key) if plate.runs else None
        rows.append(
            {
                "plate_id": plate.id,
                "barcode": plate.barcode,
                "name": plate.name,
                "plate_type": plate.plate_type,
                "experiment": plate.experiment.name if plate.experiment else "",
                "assay": plate.assay or (plate.experiment.assay if plate.experiment else ""),
                "cell_lines": ", ".join(c.name for c in plate.cell_lines),
                "acquisitions": len(plate.acquisitions),
                "runs": len(plate.runs),
                "latest_run": latest.name if latest else "",
                "latest_status": latest.status if latest else "",
                "latest_started": latest.started if latest else None,
            }
        )
    columns = [
        "plate_id",
        "barcode",
        "name",
        "plate_type",
        "experiment",
        "assay",
        "cell_lines",
        "acquisitions",
        "runs",
        "latest_run",
        "latest_status",
        "latest_started",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("latest_started", ascending=False, na_position="last").reset_index(
        drop=True
    )


def plate_page(session: Session, plate_id: int) -> dict[str, Any]:
    plate = session.get(Plate, plate_id)
    if plate is None:
        raise KeyError(f"No plate {plate_id}")
    runs = sorted(plate.runs, key=_started_key, reverse=True)
    qc = session.scalars(
        select(PlateQCRecord).where(PlateQCRecord.run_id.in_([r.id for r in runs]))
    ).all()
    roles = session.execute(
        select(PlateWell.role, func.count())
        .where(PlateWell.plate_id == plate.id)
        .group_by(PlateWell.role)
    ).all()
    return {
        "plate": {
            "id": plate.id,
            "barcode": plate.barcode,
            "name": plate.name,
            "plate_type": plate.plate_type,
            "assay": plate.assay,
            "experiment": plate.experiment.name if plate.experiment else "",
            "cell_lines": [c.name for c in plate.cell_lines],
            "layout_version": plate.layout_version,
            "layout_sha256": plate.layout_sha256,
        },
        "layout_roles": {str(role or "unannotated"): int(n) for role, n in roles},
        "acquisitions": [
            {
                "id": a.id,
                "instrument": a.instrument,
                "protocol": a.protocol,
                "started": a.started,
                "time_point_index": a.time_point_index,
                "raw_path": a.raw_path,
                "status": a.status,
            }
            for a in plate.acquisitions
        ],
        "runs": [
            {
                "id": r.id,
                "name": r.name,
                "folder": r.folder,
                "status": r.status,
                "pipeline": r.pipeline_name,
                "started": r.started,
                "software_version": r.software_version,
            }
            for r in runs
        ],
        "qc": [
            {
                "run_id": q.run_id,
                "readout": q.readout_id,
                "z_prime": q.z_prime,
                "ssmd": q.ssmd,
                "cv_vehicle_pct": q.cv_vehicle_pct,
                "flag": q.flag,
            }
            for q in qc
        ],
    }


# ----------------------------------------------------------------- potency


def _fits(
    session: Session, compound: str = "", cell_line: str = "", readout: str = ""
) -> pd.DataFrame:
    stmt = (
        select(
            DoseResponseFit,
            Run.name.label("run"),
            Run.folder.label("run_folder"),
            Run.started.label("run_started"),
            Plate.barcode.label("barcode"),
        )
        .join(Run, Run.id == DoseResponseFit.run_id)
        .outerjoin(Plate, Plate.id == Run.plate_id)
    )
    if compound:
        stmt = stmt.where(DoseResponseFit.compound == compound)
    if cell_line:
        stmt = stmt.where(DoseResponseFit.cell_line == cell_line)
    if readout:
        stmt = stmt.where(DoseResponseFit.readout_id == readout)
    rows = []
    for fit, run, folder, started, barcode in session.execute(stmt).all():
        rows.append(
            {
                "compound": fit.compound,
                "cell_line": fit.cell_line,
                "readout": fit.readout_id,
                "ec50_m": fit.ec50_m,
                "ec50_ci_low_m": fit.ec50_ci_low_m,
                "ec50_ci_high_m": fit.ec50_ci_high_m,
                "log_ec50": fit.log_ec50,
                "hill": fit.hill,
                "r2": fit.r2,
                "flags": fit.flags,
                "run": run,
                "run_folder": folder,
                "run_started": started,
                "barcode": barcode or "",
            }
        )
    columns = [
        "compound",
        "cell_line",
        "readout",
        "ec50_m",
        "ec50_ci_low_m",
        "ec50_ci_high_m",
        "log_ec50",
        "hill",
        "r2",
        "flags",
        "run",
        "run_folder",
        "run_started",
        "barcode",
    ]
    return pd.DataFrame(rows, columns=columns)


def ic50s(session: Session, compound: str, cell_line: str, readout: str = "") -> pd.DataFrame:
    """Every fitted EC50/IC50 of ``compound`` in ``cell_line`` across runs, oldest first."""
    frame = _fits(session, compound, cell_line, readout)
    return frame.sort_values("run_started", na_position="first").reset_index(drop=True)


def potency_table(
    session: Session, compound: str = "", cell_line: str = "", readout: str = ""
) -> pd.DataFrame:
    """compound × cell line × readout: n runs, geometric mean and SD of log10 EC50, the latest
    value and its run."""
    fits = _fits(session, compound, cell_line, readout)
    rows = []
    for (cpd, line, rd), group in fits.groupby(["compound", "cell_line", "readout"], sort=True):
        logs = group["log_ec50"].dropna().to_numpy(float)
        latest = group.sort_values("run_started", na_position="first").iloc[-1]
        rows.append(
            {
                "compound": cpd,
                "cell_line": line,
                "readout": rd,
                "n": len(logs),
                "mean_log_ec50": float(np.mean(logs)) if len(logs) else math.nan,
                "sd_log_ec50": float(np.std(logs, ddof=1)) if len(logs) > 1 else math.nan,
                "geomean_ec50_m": float(10 ** np.mean(logs)) if len(logs) else math.nan,
                "latest_ec50_m": latest["ec50_m"],
                "latest_run": latest["run"],
                "latest_started": latest["run_started"],
            }
        )
    columns = [
        "compound",
        "cell_line",
        "readout",
        "n",
        "mean_log_ec50",
        "sd_log_ec50",
        "geomean_ec50_m",
        "latest_ec50_m",
        "latest_run",
        "latest_started",
    ]
    return pd.DataFrame(rows, columns=columns)


# -------------------------------------------------------- readout over time


def readout_over_time(
    session: Session,
    readout: str,
    *,
    compound: str = "",
    cell_line: str = "",
    role: str = "",
) -> pd.DataFrame:
    """Per run: mean, SD and n of a per-well readout over the wells matching the filters
    (compound present at any dose, cell line, role), oldest run first."""
    stmt = (
        select(
            WellResult.value,
            Run.id.label("run_id"),
            Run.name.label("run"),
            Run.started.label("started"),
            Plate.barcode.label("barcode"),
            PlateWell.compounds,
        )
        .join(Run, Run.id == WellResult.run_id)
        .join(Plate, Plate.id == Run.plate_id)
        .join(PlateWell, (PlateWell.plate_id == Plate.id) & (PlateWell.well == WellResult.well))
        .where(WellResult.readout_id == readout)
        .where(WellResult.qc_flag != "excluded")
    )
    if cell_line:
        stmt = stmt.where(PlateWell.cell_line == cell_line)
    if role:
        stmt = stmt.where(PlateWell.role == role)
    frame = pd.DataFrame(
        session.execute(stmt).all(),
        columns=["value", "run_id", "run", "started", "barcode", "compounds"],
    )
    if compound and not frame.empty:
        keep = frame["compounds"].map(lambda items: any(c[0] == compound for c in items or []))
        frame = frame[keep]
    rows = []
    for (_run_id, run, started, barcode), group in frame.groupby(
        ["run_id", "run", "started", "barcode"], dropna=False, sort=False
    ):
        values = group["value"].dropna().to_numpy(float)
        rows.append(
            {
                "run": run,
                "barcode": barcode,
                "started": started,
                "n": len(values),
                "mean": float(values.mean()) if len(values) else math.nan,
                "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
            }
        )
    out = pd.DataFrame(rows, columns=["run", "barcode", "started", "n", "mean", "sd"])
    return out.sort_values("started", na_position="first").reset_index(drop=True)


def readouts_in_registry(session: Session) -> list[str]:
    return [r for (r,) in session.execute(select(WellResult.readout_id).distinct()).all()]


# ---------------------------------------------------------- reference pages


def compound_page(session: Session, name: str) -> dict[str, Any]:
    compound = session.scalars(select(Compound).where(Compound.name == name)).first()
    if compound is None:
        raise KeyError(f"No compound {name!r}")
    plates = session.scalars(
        select(Plate).where(
            Plate.id.in_(
                select(PlateWell.plate_id).where(
                    cast(PlateWell.compounds, String).ilike(_like(name))
                )
            )
        )
    ).all()
    combos = session.scalars(
        select(CombinationScore).where(
            or_(CombinationScore.drug_a == name, CombinationScore.drug_b == name)
        )
    ).all()
    return {
        "compound": {
            "name": compound.name,
            "external_id": compound.external_id,
            "synonyms": compound.synonyms,
            "moa": compound.moa,
            "target": compound.target,
            "smiles": compound.smiles,
            "supplier": compound.supplier,
            "lot": compound.lot,
            "stock_concentration_m": compound.stock_concentration_m,
            "solvent": compound.solvent,
        },
        "plates": [p.barcode or p.name for p in plates],
        "fits": _fits(session, compound=name),
        "combinations": pd.DataFrame(
            [
                {
                    "cell_line": c.cell_line,
                    "drug_a": c.drug_a,
                    "drug_b": c.drug_b,
                    "model": c.model,
                    "mean": c.mean,
                    "label": c.label,
                    "run_id": c.run_id,
                }
                for c in combos
            ]
        ),
    }


def cell_line_page(session: Session, name: str) -> dict[str, Any]:
    line = session.scalars(select(CellLine).where(CellLine.name == name)).first()
    if line is None:
        raise KeyError(f"No cell line {name!r}")
    plates = session.scalars(
        select(Plate).join(plate_cell_line).where(plate_cell_line.c.cell_line_id == line.id)
    ).all()
    return {
        "cell_line": {
            "name": line.name,
            "rrid": line.rrid,
            "cancer_type": line.cancer_type,
            "passage": line.passage,
            "authentication_date": line.authentication_date,
            "mycoplasma_date": line.mycoplasma_date,
        },
        "plates": [p.barcode or p.name for p in plates],
        "fits": _fits(session, cell_line=name),
    }


def names(session: Session, kind: str) -> list[str]:
    model = {"compound": Compound, "cell_line": CellLine, "experiment": Experiment}[kind]
    return sorted(n for (n,) in session.execute(select(model.name)).all())  # type: ignore[attr-defined]


def saved_searches(session: Session) -> dict[str, PlateQuery]:
    return {
        s.name: PlateQuery.from_dict(s.query)
        for s in session.scalars(select(SavedSearch).order_by(SavedSearch.name)).all()
    }


def export_table(frame: pd.DataFrame, path: Path) -> Path:
    """CSV or XLSX by extension (times without zone for Excel)."""
    path = Path(path)
    out = frame.copy()
    for column in out.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(out[column])
            or out[column].map(lambda v: isinstance(v, datetime)).any()
        ):
            out[column] = out[column].map(
                lambda v: (
                    v.replace(tzinfo=None).isoformat(sep=" ") if isinstance(v, datetime) else v
                )
            )
    if path.suffix.lower() == ".xlsx":
        out.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        out.to_csv(path, index=False)
    else:
        raise ValueError("Export as .csv or .xlsx")
    return path
