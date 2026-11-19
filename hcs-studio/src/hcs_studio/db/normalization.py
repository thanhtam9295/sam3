"""Registry side of Phase 17b: read a run's plate of values and annotations, normalize it,
compare replicate runs, and collect day and plate effects across runs.

Read-only; storing a uniformity map goes through ``services.registry.save_uniformity_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from hcs_studio.db.catalog import (
    Acquisition,
    DoseResponseFit,
    Experiment,
    Plate,
    PlateWell,
    Run,
    UniformityMapRecord,
    WellResult,
)
from hcs_studio.normalization.batch import BatchEffects, batch_effects
from hcs_studio.normalization.concordance import Concordance, replicate_concordance
from hcs_studio.normalization.plate import NormalizedPlate, WellInfo, normalize_plate, to_grid
from hcs_studio.normalization.uniformity import UniformityMap


@dataclass
class PlateValues:
    run: str
    run_id: int
    barcode: str
    rows: int
    cols: int
    readout: str
    values: dict[str, float]
    wells: list[WellInfo]
    instrument: str = ""

    @property
    def grid(self) -> np.ndarray:
        return to_grid(self.values, self.rows, self.cols)

    def roles(self) -> dict[str, str]:
        return {w.well: w.role for w in self.wells}


def find_run(session: Session, key: str) -> Run:
    """A run by id, folder name or plate barcode (its newest run)."""
    if key.isdigit():
        run = session.get(Run, int(key))
        if run is not None:
            return run
    run = session.scalars(select(Run).where(Run.name == key)).first()
    if run is not None:
        return run
    runs = session.scalars(
        select(Run).join(Plate, Plate.id == Run.plate_id).where(Plate.barcode == key)
    ).all()
    if runs:
        return max(runs, key=lambda r: r.started.timestamp() if r.started else 0.0)
    raise KeyError(f"No run {key!r} in the registry (a run id, run folder name or barcode)")


def plate_values(session: Session, run: Run, readout: str) -> PlateValues:
    plate = run.plate
    if plate is None or not plate.rows:
        raise ValueError(f"Run {run.name} has no plate geometry in the registry")
    values = {
        w: v
        for w, v in session.execute(
            select(WellResult.well, WellResult.value).where(
                WellResult.run_id == run.id, WellResult.readout_id == readout
            )
        ).all()
        if v is not None
    }
    if not values:
        raise ValueError(f"Run {run.name} has no values of {readout!r}")
    annotations = {
        pw.well: pw
        for pw in session.scalars(select(PlateWell).where(PlateWell.plate_id == plate.id))
    }
    wells = []
    for well in values:
        pw = annotations.get(well)
        compounds = tuple((str(c[0]), float(c[1])) for c in (pw.compounds if pw else []) or [])
        wells.append(WellInfo(well, (pw.role if pw and pw.role else "sample"), compounds))
    instrument = next((a.instrument for a in run.acquisitions if a.instrument), "")
    return PlateValues(
        run.name,
        run.id,
        plate.barcode or plate.name,
        plate.rows,
        plate.cols,
        readout,
        values,
        wells,
        instrument,
    )


def load_map(session: Session, name: str) -> tuple[UniformityMapRecord, UniformityMap]:
    record = session.scalars(
        select(UniformityMapRecord).where(UniformityMapRecord.name == name)
    ).first()
    if record is None:
        raise KeyError(f"No uniformity map {name!r}")
    stats = dict(record.stats)
    rings = int(stats.get("edge_rings", 1))
    return record, UniformityMap(np.array(record.factors, float), record.degree, stats, rings)


def normalize_run(
    session: Session, run_key: str, readout: str, method: str, map_name: str = ""
) -> tuple[PlateValues, NormalizedPlate]:
    run = find_run(session, run_key)
    data = plate_values(session, run, readout)
    factors = None
    if method == "uniformity":
        if not map_name:
            raise ValueError("Choose a uniformity map")
        _, mapped = load_map(session, map_name)
        factors = mapped.factors
    result = normalize_plate(data.values, data.wells, data.rows, data.cols, method, factors=factors)
    return data, result


def uniformity_maps(session: Session) -> pd.DataFrame:
    rows = [
        {
            "name": m.name,
            "instrument": m.instrument,
            "assay": m.assay,
            "readout": m.readout_id,
            "plate": f"{m.rows}×{m.cols}",
            "degree": m.degree,
            "edge_effect_pct": m.stats.get("before_edge_effect_pct"),
            "cv_before_pct": m.stats.get("before_cv_pct"),
            "cv_after_pct": m.stats.get("after_cv_pct"),
            "created": m.created,
            "actor": m.actor,
        }
        for m in session.scalars(select(UniformityMapRecord).order_by(UniformityMapRecord.name))
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ replicates


def run_fits(session: Session, run: Run, readout: str) -> pd.DataFrame:
    rows = session.execute(
        select(
            DoseResponseFit.compound,
            DoseResponseFit.cell_line,
            DoseResponseFit.readout_id,
            DoseResponseFit.ec50_m,
        ).where(DoseResponseFit.run_id == run.id, DoseResponseFit.readout_id == readout)
    ).all()
    return pd.DataFrame(rows, columns=["compound", "cell_line", "readout", "ec50_m"])


def concordance(
    session: Session,
    run_a: str,
    run_b: str,
    readout: str,
    *,
    r_min: float = 0.8,
    fold_max: float = 3.0,
) -> tuple[PlateValues, PlateValues, Concordance]:
    """Compare two replicate runs on their sample wells (conditions: the wells' treatments)."""
    a, b = find_run(session, run_a), find_run(session, run_b)
    va, vb = plate_values(session, a, readout), plate_values(session, b, readout)
    treatments = {
        pw.well: pw.treatment or pw.well
        for pw in session.scalars(select(PlateWell).where(PlateWell.plate_id == a.plate_id))
        if pw.role == "sample"
    }
    if not treatments:  # no layout: compare all wells, each its own condition
        treatments = {w: w for w in va.values}
    result = replicate_concordance(
        va.values,
        vb.values,
        treatments,
        fits_a=run_fits(session, a, readout),
        fits_b=run_fits(session, b, readout),
        r_min=r_min,
        fold_max=fold_max,
    )
    return va, vb, result


# ---------------------------------------------------------------- day effects


def batch_frame(
    session: Session, readout: str, *, runs: list[str] | None = None, experiment: str = ""
) -> pd.DataFrame:
    """Wells of several runs: plate, day (treatment date of the plate, else run start), role,
    value."""
    stmt = select(Run).join(Plate, Plate.id == Run.plate_id)
    if experiment:
        stmt = stmt.join(Experiment, Experiment.id == Plate.experiment_id).where(
            Experiment.name == experiment
        )
    chosen = [find_run(session, k) for k in runs] if runs else list(session.scalars(stmt).all())
    if not chosen:
        raise ValueError("No runs selected")
    frames = []
    for run in chosen:
        data = plate_values(session, run, readout)
        plate = run.plate
        day_source = (plate.treated_at if plate else None) or run.started
        day = day_source.date().isoformat() if day_source else "unknown"
        roles = data.roles()
        frames.append(
            pd.DataFrame(
                {
                    "plate": data.barcode or run.name,
                    "run": run.name,
                    "day": day,
                    "well": list(data.values),
                    "role": [roles.get(w, "sample") for w in data.values],
                    "value": list(data.values.values()),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def day_effects(
    session: Session, readout: str, *, runs: list[str] | None = None, experiment: str = ""
) -> BatchEffects:
    return batch_effects(batch_frame(session, readout, runs=runs, experiment=experiment))


def instruments(session: Session) -> list[str]:
    return sorted({i for (i,) in session.execute(select(Acquisition.instrument).distinct()) if i})


def describe(value: Any) -> str:
    return "" if value is None else str(value)
