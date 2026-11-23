"""Quality control of a time-lapse plate (Phase 22c).

Checks of the time course that single time points cannot reveal. Each check returns a status
(``pass``, ``warn``, ``fail`` or ``not assessed``) with a message; wells get flags.

- **time points**: gaps in the acquisition (an interval longer than ``gap_factor`` × the
  median interval: the instrument stopped or skipped) and wells missing time points (no value
  where other wells have one — a failed read, a well not imaged in one acquisition).
- **focus drift**: the per-field focus score (``focus_score``, written by the ``qc`` step)
  relative to the same well's median over time. A time point below ``focus_fraction`` × that is
  flagged in the well; a plate whose median relative focus in the last third of the time points
  is below ``drift_fraction`` × the first third drifts (autofocus failing, plate warping).
- **edge evaporation**: medium evaporates fastest from the outer wells, concentrating drugs and
  salts, so their cells grow differently as time goes on. For every condition (role, cell line,
  compound, concentration) imaged both on the edge and inside, the log2 fold change since the
  first time point of the edge wells is compared with the interior wells' median; the median of
  these differences over conditions, at each time point, is the edge effect. Beyond
  ``edge_log2`` (default log2 1.25) at the end it is reported, and edge wells deviating that much
  from their condition are flagged. Without a condition on both the edge and inside, the check is
  not assessed (often deliberate: edges filled with medium).
- **control growth**: the vehicle wells' doubling times (growth fits) — none growing fails; a
  vehicle well whose doubling time differs from the vehicle median by more than ``outlier_ratio``
  is flagged; and the vehicle median is compared with earlier runs of the same cell line in the
  registry (median of each run's vehicle doubling time): outside median ± ``history_k`` robust
  SD (at least ``min_history_change`` relative) warns. Fewer than ``min_history`` earlier runs:
  history not assessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "not assessed"


@dataclass
class KineticQcOptions:
    gap_factor: float = 1.5
    focus_fraction: float = 0.5
    drift_fraction: float = 0.8
    edge_log2: float = math.log2(1.25)
    outlier_ratio: float = 1.5
    history_k: float = 3.0
    min_history_change: float = 0.15
    min_history: int = 3


@dataclass
class KineticQc:
    checks: list[dict[str, Any]] = field(default_factory=list)
    well_flags: dict[str, list[str]] = field(default_factory=dict)
    #: Per time point: focus relative to each well's median (plate median), edge effect (log2).
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)

    def add(self, check: str, status: str, message: str, **values: Any) -> None:
        self.checks.append({"check": check, "status": status, "message": message, **values})

    def flag(self, well: str, text: str) -> None:
        flags = self.well_flags.setdefault(str(well), [])
        if text not in flags:
            flags.append(text)

    @property
    def status(self) -> str:
        statuses = {c["status"] for c in self.checks}
        return FAIL if FAIL in statuses else WARN if WARN in statuses else PASS

    def wells_table(self) -> pd.DataFrame:
        rows = [{"well": w, "flags": "; ".join(f)} for w, f in sorted(self.well_flags.items())]
        return pd.DataFrame(rows, columns=["well", "flags"])


def _hours(value: float) -> str:
    return f"{value:g} h"


def check_time_points(
    qc: KineticQc, course: pd.DataFrame, readout: str, time: str, options: KineticQcOptions
) -> None:
    times = np.sort(course[time].dropna().unique().astype(float))
    if times.size < 2:
        qc.add("time points", NA, "fewer than two time points")
        return
    steps = np.diff(times)
    median = float(np.median(steps))
    gaps = [
        (float(times[i]), float(times[i + 1]))
        for i in np.nonzero(steps > options.gap_factor * median)[0].tolist()
    ]
    matrix = course.pivot_table(index="well", columns=time, values=readout, aggfunc="mean")
    matrix = matrix.reindex(columns=times)
    missing = {
        str(w): [float(t) for t in matrix.columns[row.isna().to_numpy()]]
        for w, row in matrix.iterrows()
    }
    missing = {w: ts for w, ts in missing.items() if ts}
    for well, ts in missing.items():
        qc.flag(well, "missing time point(s) " + ", ".join(_hours(t) for t in ts))
    parts = []
    if gaps:
        spans = ", ".join(f"{_hours(a)}–{_hours(b)}" for a, b in gaps)
        parts.append(f"gap(s) in the acquisition: {spans} (median interval {_hours(median)})")
    if missing:
        parts.append(f"{len(missing)} well(s) missing time points")
    qc.add("time points", WARN if parts else PASS,
           "; ".join(parts) or f"{times.size} time points every {_hours(median)}",
           gaps=gaps, wells_missing=len(missing), median_interval_h=median)  # fmt: skip


def check_focus(
    qc: KineticQc, course: pd.DataFrame, time: str, options: KineticQcOptions
) -> pd.Series | None:
    if "focus_score" not in course or course["focus_score"].notna().sum() == 0:
        qc.add("focus drift", NA, "no focus score in the time course (add the qc step)")
        return None
    matrix = course.pivot_table(index="well", columns=time, values="focus_score", aggfunc="mean")
    relative = matrix.div(matrix.median(axis=1), axis=0)
    for well, row in relative.iterrows():
        low = [float(str(t)) for t, v in row.items() if v < options.focus_fraction]
        if low:
            qc.flag(str(well), "out of focus at " + ", ".join(_hours(t) for t in low))
    plate = relative.median(axis=0)
    third = max(len(plate) // 3, 1)
    early, late = float(plate.iloc[:third].median()), float(plate.iloc[-third:].median())
    drifting = late < options.drift_fraction * early
    out_of_focus = int((relative < options.focus_fraction).to_numpy().sum())
    status = WARN if drifting or out_of_focus else PASS
    message = (f"relative focus falls from {early:.2f} to {late:.2f} over the time course"
               if drifting else f"no drift (relative focus {early:.2f} → {late:.2f})")  # fmt: skip
    if out_of_focus:
        message += f"; {out_of_focus} well × time point(s) out of focus"
    qc.add("focus drift", status, message, early=early, late=late, out_of_focus=out_of_focus)
    return plate


def _is_edge(well: str, rows: int, cols: int) -> bool:
    from hcs_studio.model.plate import parse_well_name

    row, col = parse_well_name(well)
    return row in (0, rows - 1) or col in (0, cols - 1)


def check_edges(
    qc: KineticQc,
    course: pd.DataFrame,
    readout: str,
    time: str,
    info: pd.DataFrame,
    rows: int,
    cols: int,
    options: KineticQcOptions,
) -> pd.Series | None:
    matrix = course.pivot_table(index="well", columns=time, values=readout, aggfunc="mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = pd.DataFrame(np.log2(matrix.to_numpy(float) / matrix.iloc[:, [0]].to_numpy(float)),
                            index=matrix.index, columns=matrix.columns)  # fmt: skip
    keys = pd.Series("", index=matrix.index, dtype=object)
    if not info.empty:
        labels = info.set_index("well")
        for well in matrix.index:
            if well in labels.index:
                r = labels.loc[well]
                keys[well] = f"{r['role']}|{r['cell_line']}|{r['compound']}|{r['concentration']}"
    keys = keys[~keys.str.startswith(("empty", "blank", "excluded"))]
    edge = pd.Series({w: _is_edge(str(w), rows, cols) for w in keys.index})
    differences, deviating = [], {}
    for _key, wells in keys.groupby(keys):
        on_edge = [w for w in wells.index if edge[w]]
        inside = [w for w in wells.index if not edge[w]]
        if not on_edge or not inside:
            continue
        interior = fold.loc[inside].median()
        differences.append(fold.loc[on_edge].median() - interior)
        for well in on_edge:
            deviating[str(well)] = float((fold.loc[well] - interior).iloc[-1])
    if not differences:
        qc.add("edge evaporation", NA, "no condition is imaged both on the edge and inside")
        return None
    effect = pd.concat(differences, axis=1).median(axis=1)
    final = float(effect.iloc[-1])
    for well, value in deviating.items():
        if abs(value) > options.edge_log2:
            qc.flag(well, f"edge well {'below' if value < 0 else 'above'} its condition by "
                          f"{100 * (2**value - 1):+.0f} % at the end")  # fmt: skip
    status = WARN if abs(final) > options.edge_log2 else PASS
    qc.add("edge evaporation", status,
           f"edge wells end at {100 * (2**final - 1):+.0f} % of the interior wells of the same "
           f"condition (fold change since the first time point; {len(differences)} condition(s))",
           final_log2=final, conditions=len(differences))  # fmt: skip
    return effect


def check_controls(
    qc: KineticQc,
    fits: pd.DataFrame,
    info: pd.DataFrame,
    options: KineticQcOptions,
    history: dict[str, list[float]] | None = None,
) -> None:
    if info.empty or not (info["role"] == "negative_control").any():
        qc.add("control growth", NA, "no vehicle wells in the layout")
        return
    vehicle = fits.merge(info[["well", "role", "cell_line"]], on="well")
    vehicle = vehicle[vehicle["role"] == "negative_control"]
    if vehicle.empty:
        qc.add("control growth", NA, "the vehicle wells have no data in the time course")
        return
    for line, group in vehicle.groupby("cell_line"):
        name = str(line) or "no cell line"
        doubling = pd.to_numeric(group["doubling_h"], errors="coerce")
        growing = doubling.dropna()
        if growing.empty:
            qc.add("control growth", FAIL, f"{name}: no vehicle well grows", cell_line=line)
            continue
        median = float(growing.median())
        outliers = 0
        for well, value in zip(group["well"], doubling, strict=True):
            if not math.isfinite(value):
                qc.flag(str(well), "vehicle well not growing")
                outliers += 1
            elif max(value / median, median / value) > options.outlier_ratio:
                qc.flag(str(well), f"vehicle doubling {value:.3g} h vs median {median:.3g} h")
                outliers += 1
        message = f"{name}: vehicle doubling time {median:.3g} h ({len(growing)} wells)"
        status = WARN if outliers else PASS
        if outliers:
            message += f"; {outliers} vehicle well(s) off"
        past = (history or {}).get(str(line), [])
        record: dict[str, Any] = {
            "cell_line": line,
            "doubling_h": median,
            "history_runs": len(past),
        }
        if len(past) >= options.min_history:
            centre = float(np.median(past))
            spread = 1.4826 * float(np.median(np.abs(np.asarray(past) - centre)))
            allowed = max(options.history_k * spread, options.min_history_change * centre)
            record.update(history_median_h=centre, history_allowed_h=allowed)
            if abs(median - centre) > allowed:
                status = WARN
                message += (f"; outside its history ({centre:.3g} ± {allowed:.2g} h over "
                            f"{len(past)} earlier runs)")  # fmt: skip
            else:
                message += f"; within its history ({centre:.3g} h over {len(past)} runs)"
        else:
            message += f"; history: {len(past)} earlier run(s), {options.min_history} needed"
        qc.add("control growth", status, message, **record)


def control_history(
    cell_lines: list[str], exclude_folders: set[str] | None = None
) -> dict[str, list[float]]:
    """Median vehicle doubling time of each earlier run per cell line, from the registry
    (``kn_doubling_h`` of vehicle wells; ``exclude_folders``: rooted run folders left out).
    {} without a registry."""
    try:
        from sqlalchemy import select

        from hcs_studio.db.catalog import PlateWell, Run, WellResult
        from hcs_studio.db.engine import session_scope

        out: dict[str, list[float]] = {}
        with session_scope() as session:
            rows = session.execute(
                select(Run.folder, PlateWell.cell_line, WellResult.value)
                .join(WellResult, WellResult.run_id == Run.id)
                .join(PlateWell, (PlateWell.plate_id == Run.plate_id)
                      & (PlateWell.well == WellResult.well))
                .where(WellResult.readout_id == "kn_doubling_h",
                       PlateWell.role == "negative_control",
                       PlateWell.cell_line.in_(cell_lines))
            ).all()  # fmt: skip
        frame = pd.DataFrame(rows, columns=["folder", "cell_line", "value"]).dropna()
        if exclude_folders:
            frame = frame[~frame["folder"].isin(exclude_folders)]
        for (line, _folder), group in frame.groupby(["cell_line", "folder"]):
            out.setdefault(str(line), []).append(float(group["value"].median()))
        return out
    except Exception:  # no registry, an older schema: history is simply not assessed
        return {}


def kinetic_qc(
    course: pd.DataFrame,
    fits: pd.DataFrame,
    info: pd.DataFrame,
    readout: str,
    time: str,
    plate_shape: tuple[int, int],
    options: KineticQcOptions | None = None,
    history: dict[str, list[float]] | None = None,
) -> KineticQc:
    options = options or KineticQcOptions()
    qc = KineticQc()
    check_time_points(qc, course, readout, time, options)
    focus = check_focus(qc, course, time, options)
    edge = check_edges(qc, course, readout, time, info, *plate_shape, options)
    check_controls(qc, fits, info, options, history)
    timeline = {}
    if focus is not None:
        timeline["relative_focus"] = focus
    if edge is not None:
        timeline["edge_log2"] = edge
    if timeline:
        qc.timeline = pd.DataFrame(timeline).rename_axis("time_h").reset_index()
    return qc
