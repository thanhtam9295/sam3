"""Per-well readouts + plate annotation → normalized responses, plate QC and outliers.

Normalization is done per plate and cell line, relative to that cell line's vehicle
(negative-control) wells on the same plate (falling back to the plate's vehicle wells, with a
note, when a cell line has none):

- ``percent_of_control``: 100 · x / median(vehicle); effect = 100 − value.
- ``percent_inhibition``: 100 · (median(vehicle) − x) / (median(vehicle) − median(positive));
  0 % = vehicle, 100 % = positive control (works for falling and rising readouts).
- ``gr``: growth-rate inhibition (Hafner et al. 2016) from time-zero values x0: wells marked
  time point 0 h (same cell line, vehicle) on the plate or on a separate time-zero plate, or a
  doubling time and assay duration (x0 = x_vehicle / 2^(duration / doubling time)).

``effect`` is the response on the % inhibition scale used by the combination models.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.plate import well_name

NORMALIZATIONS = ("percent_of_control", "percent_inhibition", "gr")
SKIP_ROLES = ("empty", "excluded", "blank")
LABELS = {
    "percent_of_control": "% of control",
    "percent_inhibition": "% inhibition",
    "gr": "GR value",
}


# ----------------------------------------------------------------- well table


def treatment_key(treatments: Iterable[tuple[str, float, str]]) -> str:
    """Order-free text key of the active treatments, e.g. 'DrugA 1e-06 + DrugB 3e-07'."""
    return " + ".join(f"{c} {v:.6g}" for c, v, _ in sorted(treatments))


def drug_treatments(annotation: Any) -> tuple[tuple[str, float, str], ...]:
    """The drugs of a well for pharmacology: active treatments, without vehicles (treatments
    in % v/v) and without anything in control wells (their vehicle is not a dose)."""
    if annotation.role in ("negative_control", "positive_control", "untreated"):
        return ()
    return tuple(
        (t.compound, float(t.concentration), t.unit)
        for t in sorted(annotation.active_treatments, key=lambda t: t.compound)
        if t.kind != "percent"
    )


def wells_table(
    summary: pd.DataFrame, layout: PlateLayout | None, plate: str, run: str = ""
) -> pd.DataFrame:
    """One row per well of a run: annotation, active treatments and every readout.
    ``summary`` is ``per_well_summary`` indexed by well name."""
    rows = []
    wells = set(summary.index.astype(str))
    if layout is not None:
        wells |= {well_name(*k) for k in layout.wells}
    for name in sorted(wells):
        from hcs_studio.model.plate import parse_well_name

        key = parse_well_name(name)
        annotation = layout.get(key) if layout is not None else None
        active = drug_treatments(annotation) if annotation is not None else ()
        rows.append(
            {
                "plate": plate,
                "run": run,
                "well": name,
                "row": key[0],
                "col": key[1],
                "cell_line": annotation.cell_line if annotation else "",
                "role": annotation.role if annotation else "empty",
                "treatments": active,
                "n_active": len(active),
                "treatment": treatment_key(active),
                "replicate_group": annotation.replicate_group if annotation else "",
                "time_point_h": annotation.time_point_h if annotation else None,
                "vehicle_only": bool(
                    annotation is not None and not active and annotation.treatments
                ),
            }
        )
    frame = pd.DataFrame(rows)
    readouts = summary.select_dtypes("number")
    # per-well summaries of runs carry the annotation columns too (conc_1, time_point_h …):
    # those are not readouts, and the well table already has the annotation
    from hcs_studio.io.layout_io import is_annotation_column

    drop = [
        c
        for c in readouts
        if c in ("row", "col")
        or c in frame.columns
        or (layout is not None and is_annotation_column(str(c), layout))
    ]
    readouts = readouts.drop(columns=drop)
    return frame.merge(readouts, left_on="well", right_index=True, how="left")


def load_wells(folders: Iterable[Path]) -> pd.DataFrame:
    """Well tables of finished runs (per-well summary + the run's plate layout)."""
    from hcs_studio.output.results import RunResults

    frames = []
    names: set[str] = set()
    for folder in folders:
        results = RunResults(Path(folder))
        plate = results.plate_name
        if plate in names:  # the same plate name in two runs: keep them apart
            plate = f"{plate} ({Path(folder).name})"
        names.add(plate)
        frames.append(
            wells_table(results.summary, results.layout_document, plate, Path(folder).name)
        )
    if not frames:
        raise ValueError("No runs given")
    return pd.concat(frames, ignore_index=True)


def readout_choices(wells: pd.DataFrame) -> list[str]:
    """Numeric readout columns of a well table."""
    fixed = {"row", "col", "n_active", "time_point_h", "vehicle_only"}
    return [
        c
        for c in wells.columns
        if c not in fixed
        and pd.api.types.is_numeric_dtype(wells[c])
        and not pd.api.types.is_bool_dtype(wells[c])
        and wells[c].notna().any()
    ]


# --------------------------------------------------------------- normalization


@dataclass
class Normalization:
    method: str = "percent_inhibition"
    #: GR: separate time-zero wells (a table like ``load_wells``), matched by cell line
    time_zero: pd.DataFrame | None = None
    doubling_time_h: float | None = None
    assay_duration_h: float | None = None


@dataclass
class Normalized:
    wells: pd.DataFrame  # + value, normalized, effect, excluded, exclusion_reason
    controls: pd.DataFrame  # per plate and cell line: vehicle / positive / x0 levels
    notes: list[str] = field(default_factory=list)
    readout: str = ""
    method: str = ""


def _is_vehicle(frame: pd.DataFrame) -> pd.Series:
    return (frame["role"] == "negative_control") | (
        (frame["role"] == "untreated") & (frame["n_active"] == 0)
    )


def normalize(wells: pd.DataFrame, readout: str, options: Normalization) -> Normalized:
    if options.method not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {options.method!r} (use {', '.join(NORMALIZATIONS)})"
        )
    if readout not in wells:
        raise ValueError(f"Readout {readout!r} is not in the per-well tables")
    frame = wells[~wells["role"].isin(SKIP_ROLES)].copy()
    frame["value"] = pd.to_numeric(frame[readout], errors="coerce")
    frame = frame[frame["value"].notna()]
    # time-zero wells (GR) are references, not samples or controls of the assay
    is_t0 = pd.to_numeric(frame["time_point_h"], errors="coerce") == 0
    t0_wells, frame = frame[is_t0], frame[~is_t0].copy()
    notes: list[str] = []
    control_rows = []
    normalized = pd.Series(np.nan, index=frame.index)
    gi = pd.Series(np.nan, index=frame.index)
    vehicle = _is_vehicle(frame)
    positive = frame["role"] == "positive_control"
    for plate, plate_frame in frame.groupby("plate"):
        plate_vehicle = plate_frame[vehicle.loc[plate_frame.index]]
        plate_positive = plate_frame[positive.loc[plate_frame.index]]
        for line_key, group in plate_frame.groupby("cell_line"):
            line = str(line_key)
            veh = plate_vehicle[plate_vehicle["cell_line"] == line]["value"]
            if veh.empty:
                veh = plate_vehicle["value"]
                if veh.empty:
                    raise ValueError(f"Plate {plate}: no vehicle (negative-control) wells")
                notes.append(
                    f"{plate}, {line or 'no cell line'}: no vehicle wells of this cell line; "
                    "all vehicle wells of the plate used"
                )
            pos = plate_positive[plate_positive["cell_line"] == line]["value"]
            if pos.empty and not plate_positive.empty:
                pos = plate_positive["value"]
                if options.method == "percent_inhibition":
                    notes.append(
                        f"{plate}, {line or 'no cell line'}: no positive controls of this "
                        "cell line; all positive controls of the plate used"
                    )
            v_med = float(veh.median())
            p_med = float(pos.median()) if not pos.empty else math.nan
            x = group["value"].to_numpy(float)
            x0 = math.nan
            if options.method == "percent_of_control":
                if not v_med > 0:
                    raise ValueError(
                        f"Plate {plate}, {line or 'no cell line'}: the vehicle median of "
                        f"{readout} is {v_med:g}; % of control needs a positive vehicle level "
                        "(use % inhibition for readouts that are ~0 in vehicle wells)"
                    )
                values = 100 * x / v_med
            elif options.method == "percent_inhibition":
                if pos.empty:
                    raise ValueError(
                        f"Plate {plate}: % inhibition needs positive-control wells (mark them "
                        "in the plate layout) — or use % of control"
                    )
                if v_med == p_med:
                    raise ValueError(f"Plate {plate}, {line}: vehicle and positive controls equal")
                values = 100 * (v_med - x) / (v_med - p_med)
            else:
                x0 = _time_zero(t0_wells, plate, line, v_med, readout, options, notes)
                from hcs_studio.pharmacology.curves import gi_values, gr_values

                values = gr_values(x, x0, v_med)
                gi.loc[group.index] = gi_values(x, x0, v_med)
            normalized.loc[group.index] = values
            control_rows.append(
                {
                    "plate": plate,
                    "cell_line": line,
                    "vehicle_median": v_med,
                    "vehicle_n": int(veh.size),
                    "positive_median": p_med,
                    "positive_n": int(pos.size),
                    "time_zero": x0,
                }
            )
    frame["normalized"] = normalized
    if options.method == "gr":
        frame["gi"] = gi  # NCI growth inhibition (for GI50)
    if options.method == "percent_of_control":
        frame["effect"] = 100 - frame["normalized"]
    elif options.method == "percent_inhibition":
        frame["effect"] = frame["normalized"]
    else:
        frame["effect"] = 100 * (1 - frame["normalized"]) / 2  # GR 1 → 0 %, GR −1 → 100 %
    frame["excluded"] = False
    frame["exclusion_reason"] = ""
    return Normalized(frame, pd.DataFrame(control_rows), notes, readout, options.method)


def _time_zero(
    t0_wells: pd.DataFrame,
    plate: Any,
    line: Any,
    vehicle_median: float,
    readout: str,
    options: Normalization,
    notes: list[str],
) -> float:
    t0 = t0_wells[(t0_wells["plate"] == plate) & (t0_wells["cell_line"] == line)]["value"]
    if t0.empty and options.time_zero is not None and readout in options.time_zero:
        other = options.time_zero
        mine = other[(other["cell_line"] == line) & (other["n_active"] == 0)]
        t0 = pd.to_numeric(mine[readout], errors="coerce").dropna()
    if not t0.empty:
        return float(t0.median())
    if options.doubling_time_h and options.assay_duration_h:
        notes.append(
            f"{plate}, {line}: time-zero value from the doubling time "
            f"({options.doubling_time_h:g} h over {options.assay_duration_h:g} h)"
        )
        return vehicle_median / 2 ** (options.assay_duration_h / options.doubling_time_h)
    raise ValueError(
        f"GR for {plate}, {line or 'no cell line'}: give time-zero wells (time point 0 h), a "
        "time-zero plate, or a doubling time and assay duration"
    )


# --------------------------------------------------------------------- outliers


def exclude_outliers(
    data: Normalized,
    mad_threshold: float | None = 3.0,
    manual: Iterable[tuple[str, str]] = (),
    manual_include: Iterable[tuple[str, str]] = (),
) -> Normalized:
    """Mark wells beyond median ± k·MAD of their replicate group (≥ 3 wells; groups of
    cell line + treatments across plates; the MAD is at least that of the cell line's
    vehicle wells), plus wells excluded by hand ((plate, well)).
    Excluded wells stay in the table (``excluded`` = True) but are left out of fits."""
    frame = data.wells
    frame["excluded"] = False
    frame["exclusion_reason"] = ""
    if mad_threshold:
        # with 3–4 replicates the MAD can be tiny by chance: the spread is never taken as
        # smaller than that of the vehicle wells (pooled over plates) of the cell line
        vehicle = frame[_is_vehicle(frame)]
        floors: dict[Any, float] = {
            line: 1.4826 * float(np.nanmedian(np.abs(v - np.nanmedian(v))))
            for line, v in (
                (k, g["normalized"].to_numpy(float)) for k, g in vehicle.groupby("cell_line")
            )
        }
        samples = frame[frame["n_active"] > 0]
        for (line, _), group in samples.groupby(["cell_line", "treatment"]):
            values = group["normalized"].to_numpy(float)
            if np.isfinite(values).sum() < 3:
                continue
            median = float(np.nanmedian(values))
            mad = 1.4826 * float(np.nanmedian(np.abs(values - median)))
            mad = max(mad, floors.get(line, 0.0))
            if mad <= 0:
                continue
            far = np.abs(values - median) > mad_threshold * mad
            index = group.index[far]
            frame.loc[index, "excluded"] = True
            frame.loc[index, "exclusion_reason"] = (
                f"outside median ± {mad_threshold:g} MAD of its replicates"
            )
    for plate, well in manual:
        hit = (frame["plate"] == plate) & (frame["well"] == well)
        frame.loc[hit, "excluded"] = True
        frame.loc[hit, "exclusion_reason"] = "excluded by hand"
    for plate, well in manual_include:
        hit = (frame["plate"] == plate) & (frame["well"] == well)
        frame.loc[hit, "excluded"] = False
        frame.loc[hit, "exclusion_reason"] = ""
    return data


# ---------------------------------------------------------------------- plate QC


def plate_qc(data: Normalized, z_threshold: float = 0.5) -> pd.DataFrame:
    """Z′, SSMD, signal window and control CVs per plate (raw readout values).

    Z′ = 1 − 3(σp + σn)/|μp − μn|; SSMD = (μp − μn)/√(σp² + σn²);
    signal window = (|μp − μn| − 3(σp + σn))/σn; CV = 100 σ/μ."""
    rows = []
    frame = data.wells
    for plate, group in frame.groupby("plate"):
        negative = group[_is_vehicle(group)]["value"].to_numpy(float)
        positive = group[group["role"] == "positive_control"]["value"].to_numpy(float)
        row: dict[str, Any] = {
            "plate": plate,
            "readout": data.readout,
            "n_vehicle": negative.size,
            "n_positive": positive.size,
            "vehicle_mean": float(negative.mean()) if negative.size else math.nan,
            "positive_mean": float(positive.mean()) if positive.size else math.nan,
            "cv_vehicle_pct": _cv(negative),
            "cv_positive_pct": _cv(positive),
            "z_prime": math.nan,
            "ssmd": math.nan,
            "signal_window": math.nan,
        }
        if negative.size >= 2 and positive.size >= 2:
            mn, mp = negative.mean(), positive.mean()
            sn, sp = negative.std(ddof=1), positive.std(ddof=1)
            gap = abs(mp - mn)
            row["z_prime"] = float(1 - 3 * (sp + sn) / gap) if gap else math.nan
            row["ssmd"] = float((mp - mn) / math.sqrt(sp**2 + sn**2)) if sp + sn else math.nan
            row["signal_window"] = float((gap - 3 * (sp + sn)) / sn) if sn else math.nan
        z = row["z_prime"]
        row["flag"] = (
            f"Z′ below {z_threshold:g}"
            if math.isfinite(z) and z < z_threshold
            else (
                "Z′ not computed (needs ≥ 2 vehicle and ≥ 2 positive controls)"
                if not math.isfinite(z)
                else ""
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cv(values: Any) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values.mean() == 0:
        return math.nan
    return float(100 * values.std(ddof=1) / abs(values.mean()))


def plate_trends(data: Normalized, plate: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized values of all wells as a (rows, cols) array, and row / column medians of
    treated and vehicle wells — to spot edge, gradient or dispense effects."""
    group = data.wells[data.wells["plate"] == plate]
    rows = int(group["row"].max()) + 1 if not group.empty else 0
    cols = int(group["col"].max()) + 1 if not group.empty else 0
    rows, cols = _plate_shape(rows, cols)
    grid = np.full((rows, cols), np.nan)
    for r, c, v in group[["row", "col", "normalized"]].itertuples(index=False):
        grid[int(r), int(c)] = v
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_medians = np.nanmedian(grid, axis=1)
            col_medians = np.nanmedian(grid, axis=0)
    return grid, row_medians, col_medians


def _plate_shape(rows: int, cols: int) -> tuple[int, int]:
    for r, c in ((8, 12), (16, 24), (32, 48)):
        if rows <= r and cols <= c:
            return r, c
    return rows, cols
