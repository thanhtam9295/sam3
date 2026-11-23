"""Dose–response over time from a time course (Phase 22b).

Works on a time-course table (one row per well and time point: ``well``, ``time_h``, optionally
``hours_since_treatment``, the readout) and a well table (``conditions``: role, cell line and the
single compound with its concentration in base units, from the plate layout).

- **IC50(t)** (``ic50_over_time``): at each time point after treatment, the readout as % of the
  median of the vehicle wells *at that time point* (same cell line; else all vehicle wells), and a
  4PL fit per compound × cell line (`pharmacology.curves.fit_curve`, as everywhere else).
- **GR(t)** from treatment (``gr_over_time``): Hafner et al. 2016,
  GR(c, t) = 2^(log2(x(c, t)/x₀) / log2(x_ctrl(t)/x₀,ctrl)) − 1. A live-cell plate images every
  well before treatment, so x₀ is each well's **own** value at the last time point at or before
  treatment (no separate time-zero plate), and the control term is the median over the vehicle
  wells of log2(x_ctrl(t)/x₀,ctrl). Without a time point before treatment the first time point
  is used, with a note. GR50, GRmax, GR_AOC per compound and time (`gr_metrics`).
- **GR over intervals** (``gr_intervals``): the same formula with the start of each interval as
  time zero. For an interval [t₁, t₂], with the growth rate k = log2(x(t₂)/x(t₁)) / (t₂ − t₁),
  GR = 2^(k(c)/k_ctrl) − 1 — the drug's effect on the growth rate *during* the interval (an
  effect that starts late or wears off shows here, while GR from treatment averages it out).
  Intervals are consecutive time points, or with ``interval_h`` the first later time point at
  least that far; each is reported at its midpoint. The control growth rate is the median over
  the vehicle wells; an interval in which the controls grow less than ``min_control_doublings``
  gives no GR (GR is undefined when the controls do not grow).

The endpoint formula is the one of Hafner et al. 2016 (checked against the GRmetrics and
gr_metrics reference code); the interval form follows from it algebraically. The paper's own
time-dependent definition could not be read from here (see docs/validation.md): the choice of
intervals is this module's convention.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

VEHICLE_ROLES = ("negative_control",)


def conditions(layout: Any) -> pd.DataFrame:
    """One row per annotated well: role, cell line, compound and concentration (single agents;
    combinations are marked and left out of the fits)."""
    from hcs_studio.model.plate import well_name
    from hcs_studio.pharmacology.normalize import drug_treatments

    rows = []
    if layout is None:
        return pd.DataFrame(columns=["well", "role", "cell_line", "compound", "concentration"])
    for key, annotation in layout.wells.items():
        drugs = drug_treatments(annotation)
        role = annotation.role
        if role == "untreated" and not drugs:
            role = "negative_control"
        rows.append(
            {
                "well": well_name(*key),
                "role": role,
                "cell_line": annotation.cell_line or "",
                "compound": drugs[0][0] if len(drugs) == 1 else ("combination" if drugs else ""),
                "concentration": drugs[0][1] if len(drugs) == 1 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def time_column(course: pd.DataFrame) -> str:
    return "hours_since_treatment" if "hours_since_treatment" in course else "time_h"


def _vehicle(info: pd.DataFrame, line: str) -> list[str]:
    vehicle = info[info["role"].isin(VEHICLE_ROLES)]
    mine = vehicle[vehicle["cell_line"] == line]
    return list((mine if not mine.empty else vehicle)["well"])


def _series(course: pd.DataFrame, readout: str, time: str) -> pd.DataFrame:
    """Wells × times matrix of the readout."""
    return course.pivot_table(index="well", columns=time, values=readout, aggfunc="mean")


FLAT = (
    "flat response (no concentration differs from the others by 3 × the replicate SD): not fitted"
)


def responds(conc: Any, values: Any) -> bool:
    """Whether the concentrations differ: the range of the concentration means exceeds
    3 × the pooled SD of replicates (the spread of the means when there are no replicates is
    compared with the median absolute step between neighbouring concentrations). Flat series
    are not fitted: a 4PL on a flat line has no answer, and searching for one is slow."""
    frame = pd.DataFrame({"c": np.asarray(conc, float), "v": np.asarray(values, float)}).dropna()
    if frame["c"].nunique() < 3:
        return False
    groups = frame.groupby("c")["v"]
    means = groups.mean()
    spread = float(means.max() - means.min())
    deviations = frame["v"] - groups.transform("mean")
    dof = len(frame) - len(means)
    if dof > 0:
        sd = math.sqrt(float((deviations**2).sum()) / dof)
    else:
        sd = float(np.median(np.abs(np.diff(means.to_numpy())))) / math.sqrt(2)
    return spread > 3 * sd


def ic50_over_time(
    course: pd.DataFrame, info: pd.DataFrame, readout: str, notes: list[str]
) -> pd.DataFrame:
    from hcs_studio.pharmacology.curves import fit_curve

    time = time_column(course)
    matrix = _series(course, readout, time)
    rows = []
    samples = info[(info["compound"] != "") & (info["compound"] != "combination")]
    for (compound, line), group in samples.groupby(["compound", "cell_line"]):
        vehicle = [w for w in _vehicle(info, str(line)) if w in matrix.index]
        if not vehicle:
            notes.append(f"IC50(t): no vehicle wells for {compound} ({line or 'no cell line'})")
            continue
        wells = [w for w in group["well"] if w in matrix.index]
        conc = group.set_index("well").loc[wells, "concentration"].to_numpy(float)
        for t in matrix.columns:
            if time == "hours_since_treatment" and float(t) <= 0:
                continue
            control = float(matrix.loc[vehicle, t].median())
            if not control > 0:
                continue
            values = 100 * matrix.loc[wells, t].to_numpy(float) / control
            if not responds(conc, values):
                rows.append({"compound": compound, "cell_line": line, "time_h": float(t),
                             "ic50": math.nan, "vehicle_median": control,
                             "flags": FLAT})  # fmt: skip
                continue
            fit = fit_curve(conc, values, half=50.0)
            rows.append({
                "compound": compound, "cell_line": line, "time_h": float(t),
                "ic50": fit.abs_ic50, "ec50": fit.ec50 if fit.converged else math.nan,
                "hill": fit.hill, "top": fit.top, "bottom": fit.bottom, "r2": fit.r2,
                "vehicle_median": control, "flags": "; ".join(fit.flags),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _baseline(matrix: pd.DataFrame, time: str, notes: list[str]) -> pd.Series:
    """Each well's value at the last time point at or before treatment."""
    times = np.asarray(matrix.columns, dtype=float)
    if time == "hours_since_treatment" and np.any(times <= 1e-6):
        column = matrix.columns[int(np.nonzero(times <= 1e-6)[0][-1])]
    else:
        column = matrix.columns[0]
        notes.append(
            "GR: no time point at or before treatment; the first time point is time zero"
            if time == "hours_since_treatment"
            else "GR: treatment time unknown; the first time point is time zero"
        )
    return matrix[column]


def gr_over_time(
    course: pd.DataFrame,
    info: pd.DataFrame,
    readout: str,
    notes: list[str],
    min_control_doublings: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GR values per well and time from treatment, and GR metrics per compound and time."""
    time = time_column(course)
    matrix = _series(course, readout, time)
    x0 = _baseline(matrix, time, notes)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fold = pd.DataFrame(
            np.log2(matrix.div(x0, axis=0).to_numpy(float)),
            index=matrix.index,
            columns=matrix.columns,
        )
    start = float(str(x0.name)) if x0.name is not None else float(matrix.columns[0])
    values: list[dict[str, Any]] = []
    for line, line_info in info.groupby("cell_line"):
        vehicle = [w for w in _vehicle(info, str(line)) if w in matrix.index]
        if not vehicle:
            continue
        control = log_fold.loc[vehicle].median()
        for _, row in line_info.iterrows():
            if row["well"] not in matrix.index:
                continue
            for t in matrix.columns:
                if float(t) <= start:
                    continue
                c = float(control[t])
                gr = (
                    2 ** (float(log_fold.loc[row["well"], t]) / c) - 1
                    if c >= min_control_doublings
                    else math.nan
                )
                values.append({"well": row["well"], "cell_line": line, "compound": row["compound"],
                               "concentration": row["concentration"], "role": row["role"],
                               "time_h": float(t), "gr": gr, "control_doublings": c})  # fmt: skip
    frame = pd.DataFrame(values)
    if not frame.empty and frame["gr"].isna().any():
        notes.append(
            f"GR: undefined where the controls grew less than {min_control_doublings:g} doublings"
        )
    return frame, _gr_metrics(frame, "time_h")


def _intervals(times: np.ndarray, interval_h: float | None) -> list[tuple[int, int]]:
    pairs = []
    for i, t in enumerate(times):
        if interval_h is None:
            if i + 1 < len(times):
                pairs.append((i, i + 1))
            continue
        later = np.nonzero(times >= t + interval_h - 1e-6)[0]
        if later.size:
            pairs.append((i, int(later[0])))
    if interval_h is not None:  # non-overlapping intervals from the first time point
        chosen, last = [], None
        for i, j in pairs:
            if last is None or i >= last:
                chosen.append((i, j))
                last = j
        pairs = chosen
    return pairs


def gr_intervals(
    course: pd.DataFrame,
    info: pd.DataFrame,
    readout: str,
    notes: list[str],
    interval_h: float | None = None,
    min_control_doublings: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GR over intervals per well, and GR metrics per compound and interval midpoint."""
    time = time_column(course)
    matrix = _series(course, readout, time)
    times = np.asarray(matrix.columns, dtype=float)
    values = []
    for i, j in _intervals(times, interval_h):
        t1, t2 = times[i], times[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.log2(matrix.iloc[:, j] / matrix.iloc[:, i]) / (t2 - t1)
        for line, line_info in info.groupby("cell_line"):
            vehicle = [w for w in _vehicle(info, str(line)) if w in matrix.index]
            if not vehicle:
                continue
            k_ctrl = float(rate.loc[vehicle].median())
            enough = k_ctrl * (t2 - t1) >= min_control_doublings
            for _, row in line_info.iterrows():
                if row["well"] not in matrix.index:
                    continue
                k = float(rate.loc[row["well"]])
                values.append({
                    "well": row["well"], "cell_line": line, "compound": row["compound"],
                    "concentration": row["concentration"], "role": row["role"],
                    "start_h": t1, "end_h": t2, "time_h": (t1 + t2) / 2,
                    "rate_per_h": k * math.log(2), "control_rate_per_h": k_ctrl * math.log(2),
                    "gr": 2 ** (k / k_ctrl) - 1 if enough and math.isfinite(k) else math.nan,
                })  # fmt: skip
    frame = pd.DataFrame(values)
    if not frame.empty and frame["gr"].isna().any():
        notes.append("GR over intervals: undefined where the controls did not grow "
                     f"{min_control_doublings:g} doublings in the interval")  # fmt: skip
    return frame, _gr_metrics(frame, "time_h")


def _gr_metrics(values: pd.DataFrame, time: str) -> pd.DataFrame:
    from hcs_studio.pharmacology.curves import gr_metrics

    if values.empty:
        return pd.DataFrame()
    rows = []
    samples = values[(values["compound"] != "") & (values["compound"] != "combination")
                     & values["concentration"].notna()]  # fmt: skip
    for keys, group in samples.groupby(["compound", "cell_line", time]):
        compound, line, t = keys[0], keys[1], float(str(keys[2]))
        group = group[group["gr"].notna()]
        if group["concentration"].nunique() < 3:
            continue
        conc, gr = group["concentration"].to_numpy(float), group["gr"].to_numpy(float)
        if not responds(conc, gr):
            rows.append({"compound": compound, "cell_line": line, time: t,
                         "GR50": math.nan, "flags": FLAT})  # fmt: skip
            continue
        metrics = gr_metrics(conc, gr)
        rows.append({"compound": compound, "cell_line": line, time: t, **metrics})
    return pd.DataFrame(rows)
