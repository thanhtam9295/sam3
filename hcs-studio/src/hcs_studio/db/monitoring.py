"""Registry side of Phase 17c: per-run control metrics, reference-compound potency over time,
control charts with event annotations, MSR history, and instrument bridging.

Control metrics are computed from the stored per-well results and the wells' roles, so every
scanned run can be charted (pharmacology need not have run). Read-only.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from hcs_studio.db.catalog import (
    DoseResponseFit,
    EventAnnotation,
    Experiment,
    Plate,
    PlateWell,
    Run,
)
from hcs_studio.db.normalization import plate_values
from hcs_studio.monitoring.bridging import Bridging, bridge
from hcs_studio.monitoring.charts import (
    DEFAULT_RULES,
    ControlChart,
    Rule,
    control_chart,
    events_for,
)
from hcs_studio.monitoring.msr import msr_from_runs, rolling_msr

METRICS = {
    "z_prime": "Z′",
    "vehicle_mean": "vehicle mean",
    "vehicle_cv_pct": "vehicle CV %",
    "positive_mean": "positive-control mean",
    "positive_cv_pct": "positive-control CV %",
    "log_ec50": "log10 EC50 of the reference compound",
}


def _when(run: Run) -> Any:
    plate = run.plate
    return (plate.treated_at if plate is not None else None) or run.started


def _instrument(run: Run) -> str:
    return next((a.instrument for a in run.acquisitions if a.instrument), "")


def _assay(run: Run) -> str:
    plate = run.plate
    if plate is None:
        return ""
    if plate.assay:
        return plate.assay
    return plate.experiment.assay if plate.experiment is not None else ""


def select_runs(
    session: Session, *, assay: str = "", instrument: str = "", experiment: str = ""
) -> list[Run]:
    stmt = select(Run).join(Plate, Plate.id == Run.plate_id)
    if experiment:
        stmt = stmt.join(Experiment, Experiment.id == Plate.experiment_id).where(
            Experiment.name == experiment
        )
    runs = [
        r
        for r in session.scalars(stmt).all()
        if (not assay or _assay(r) == assay) and (not instrument or _instrument(r) == instrument)
    ]
    return sorted(runs, key=lambda r: _timestamp(_when(r)))


def _timestamp(value: Any) -> float:
    return value.timestamp() if value is not None else 0.0


def control_metrics(values: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    def cv(x: np.ndarray) -> float:
        return float(100 * x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else math.nan

    out = {
        "vehicle_mean": float(values.mean()) if len(values) else math.nan,
        "vehicle_cv_pct": cv(values),
        "positive_mean": float(positive.mean()) if len(positive) else math.nan,
        "positive_cv_pct": cv(positive),
        "n_vehicle": len(values),
        "n_positive": len(positive),
        "z_prime": math.nan,
    }
    if len(values) > 1 and len(positive) > 1:
        window = abs(values.mean() - positive.mean())
        if window > 0:
            out["z_prime"] = float(1 - 3 * (values.std(ddof=1) + positive.std(ddof=1)) / window)
    return out


def run_metrics(
    session: Session,
    readout: str,
    *,
    assay: str = "",
    instrument: str = "",
    cell_line: str = "",
    experiment: str = "",
) -> pd.DataFrame:
    """One row per run: when, run, barcode, instrument, assay and the control metrics of
    ``readout`` (vehicle and positive-control mean, CV, Z′), for ``cell_line`` if given."""
    rows = []
    for run in select_runs(session, assay=assay, instrument=instrument, experiment=experiment):
        try:
            data = plate_values(session, run, readout)
        except ValueError:
            continue
        lines = {
            pw.well: pw.cell_line
            for pw in session.scalars(select(PlateWell).where(PlateWell.plate_id == run.plate_id))
        }
        wells = [w for w in data.wells if not cell_line or lines.get(w.well) == cell_line]
        vehicle = np.array([data.values[w.well] for w in wells if w.role == "negative_control"])
        positive = np.array([data.values[w.well] for w in wells if w.role == "positive_control"])
        if not len(vehicle):
            continue
        rows.append(
            {
                "when": _when(run),
                "run": run.name,
                "barcode": data.barcode,
                "instrument": _instrument(run),
                "assay": _assay(run),
                **control_metrics(vehicle, positive),
            }
        )
    columns = [
        "when",
        "run",
        "barcode",
        "instrument",
        "assay",
        *[m for m in METRICS if m != "log_ec50"],
        "n_vehicle",
        "n_positive",
    ]
    return pd.DataFrame(rows, columns=columns)


def reference_potency(
    session: Session, compound: str, *, cell_line: str = "", readout: str = "", instrument: str = ""
) -> pd.DataFrame:
    """EC50 of a reference compound in every run (oldest first): when, run, barcode, ec50_m,
    log_ec50."""
    stmt = (
        select(DoseResponseFit, Run)
        .join(Run, Run.id == DoseResponseFit.run_id)
        .where(DoseResponseFit.compound == compound)
    )
    if cell_line:
        stmt = stmt.where(DoseResponseFit.cell_line == cell_line)
    if readout:
        stmt = stmt.where(DoseResponseFit.readout_id == readout)
    rows = [
        {
            "when": _when(run),
            "run": run.name,
            "barcode": run.plate.barcode if run.plate else "",
            "instrument": _instrument(run),
            "cell_line": fit.cell_line,
            "readout": fit.readout_id,
            "ec50_m": fit.ec50_m,
            "log_ec50": math.log10(fit.ec50_m) if fit.ec50_m and fit.ec50_m > 0 else math.nan,
        }
        for fit, run in session.execute(stmt).all()
        if not instrument or _instrument(run) == instrument
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "when",
            "run",
            "barcode",
            "instrument",
            "cell_line",
            "readout",
            "ec50_m",
            "log_ec50",
        ],
    )
    if frame["cell_line"].nunique() > 1 or frame["readout"].nunique() > 1:
        raise ValueError(
            f"{compound} was fitted in several cell lines or readouts; choose one of each"
        )
    frame["_t"] = frame["when"].map(_timestamp)
    return frame.sort_values("_t").drop(columns="_t").reset_index(drop=True)


def event_annotations(session: Session) -> pd.DataFrame:
    rows = [
        {
            "on": e.on,
            "kind": e.kind,
            "assay": e.assay,
            "instrument": e.instrument,
            "cell_line": e.cell_line,
            "description": e.description,
        }
        for e in session.scalars(select(EventAnnotation).order_by(EventAnnotation.on))
    ]
    return pd.DataFrame(
        rows, columns=["on", "kind", "assay", "instrument", "cell_line", "description"]
    )


def chart(
    session: Session,
    metric: str,
    *,
    readout: str = "",
    compound: str = "",
    cell_line: str = "",
    assay: str = "",
    instrument: str = "",
    experiment: str = "",
    rules: Sequence[Rule] = DEFAULT_RULES,
    baseline_n: int = 20,
) -> ControlChart:
    """A control chart of one metric over runs, with the event annotations that concern it."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r} ({', '.join(METRICS)})")
    if metric == "log_ec50":
        if not compound:
            raise ValueError("Choose the reference compound")
        series = reference_potency(
            session, compound, cell_line=cell_line, readout=readout, instrument=instrument
        )
        name = f"log10 EC50 of {compound}" + (f" in {cell_line}" if cell_line else "")
    else:
        if not readout:
            raise ValueError("Choose the readout")
        series = run_metrics(
            session,
            readout,
            assay=assay,
            instrument=instrument,
            cell_line=cell_line,
            experiment=experiment,
        )
        name = f"{METRICS[metric]} of {readout}"
    points = series.rename(columns={metric: "value"})
    points = points[np.isfinite(points["value"].astype(float))]
    events = events_for(
        event_annotations(session), assay=assay, instrument=instrument, cell_line=cell_line
    )
    return control_chart(points, name=name, rules=rules, baseline_n=baseline_n, events=events)


def msr_history(
    session: Session, compound: str, *, cell_line: str = "", readout: str = "", window: int = 6
) -> dict[str, Any]:
    series = reference_potency(session, compound, cell_line=cell_line, readout=readout)
    potencies = series["ec50_m"].dropna()
    potencies = potencies[potencies > 0]
    rolling = (
        rolling_msr(potencies.tolist(), window)
        if len(potencies) >= window
        else pd.Series(dtype=float)
    )
    return {
        "compound": compound,
        "n_runs": len(potencies),
        "msr": msr_from_runs(potencies.tolist()) if len(potencies) >= 2 else math.nan,
        "rolling": series.loc[potencies.index].assign(
            rolling_msr=rolling.to_numpy() if len(rolling) else math.nan
        ),
        "window": window,
    }


def bridging(
    session: Session, instrument_a: str, instrument_b: str, readout: str, msr: float = 3.0
) -> Bridging:
    """Plates imaged on both instruments (same barcode): per-plate agreement of ``readout`` and
    potency ratios A/B of every compound fitted on both."""
    by_plate: dict[str, dict[str, Run]] = {}
    for run in session.scalars(select(Run)).all():
        inst = _instrument(run)
        if inst in (instrument_a, instrument_b) and run.plate is not None and run.plate.barcode:
            current = by_plate.setdefault(run.plate.barcode, {}).get(inst)
            if current is None or _timestamp(run.started) > _timestamp(current.started):
                by_plate[run.plate.barcode][inst] = run
    pairs: list[tuple[str, dict[str, float], dict[str, float]]] = []
    fits_a: list[pd.DataFrame] = []
    fits_b: list[pd.DataFrame] = []
    for barcode, runs in sorted(by_plate.items()):
        if instrument_a not in runs or instrument_b not in runs:
            continue
        a, b = runs[instrument_a], runs[instrument_b]
        with contextlib.suppress(ValueError):
            pairs.append(
                (
                    barcode,
                    plate_values(session, a, readout).values,
                    plate_values(session, b, readout).values,
                )
            )
        for run, out in ((a, fits_a), (b, fits_b)):
            frame = _fits(session, run, readout)
            frame["compound"] = barcode + ": " + frame["compound"]  # pair fits per plate
            out.append(frame)
    fa = pd.concat(fits_a, ignore_index=True) if fits_a else pd.DataFrame()
    fb = pd.concat(fits_b, ignore_index=True) if fits_b else pd.DataFrame()
    return bridge(instrument_a, instrument_b, readout, pairs, fa, fb, msr)


def _fits(session: Session, run: Run, readout: str) -> pd.DataFrame:
    rows = session.execute(
        select(
            DoseResponseFit.compound,
            DoseResponseFit.cell_line,
            DoseResponseFit.readout_id,
            DoseResponseFit.ec50_m,
            DoseResponseFit.ec50_ci_low_m,
            DoseResponseFit.ec50_ci_high_m,
        ).where(DoseResponseFit.run_id == run.id, DoseResponseFit.readout_id == readout)
    ).all()
    return pd.DataFrame(
        rows,
        columns=["compound", "cell_line", "readout", "ec50_m", "ec50_ci_low_m", "ec50_ci_high_m"],
    )
