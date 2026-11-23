"""Kinetic analysis of a time-lapse run (Phase 22b): the ``kinetics`` table step.

After every well of a time-lapse run, the step builds the time course (``timecourse``) and:

- fits a growth curve to every well (``growth.fit_growth``: exponential, lag-exponential or
  logistic, chosen by AICc) — doubling time, growth rate, lag, capacity — plus the model-free
  AUC, fold changes and times to a fold change or level (``growth.curve_metrics``);
- with compounds in the layout: IC50 at each time point, GR from treatment per time point and
  GR over intervals, with GR50/GRmax per compound and time (``response``).

Per-well values go to ``per_well_summary.csv`` with the prefix ``kn_``; everything else to
``kinetics/`` in the run folder (CSV tables, ``kinetics.json``, figures). Separate acquisitions
of one plate are analysed with ``analyse_runs`` (CLI ``hcs-studio kinetics analyse``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from hcs_studio.analysis.table_steps import TableModule, WellContext
from hcs_studio.kinetics.growth import MODELS, curve_metrics, fit_growth
from hcs_studio.processing.base import MODULES, ModuleError, ModuleParams

PREFIX = "kn_"


class KineticsParams(ModuleParams):
    readout: str = Field(
        default="auto",
        description="Time-course column to analyse; auto: the object count "
        "(<labels>_objects_measured), else confluence_pct.",
    )
    models: list[str] = Field(default=list(MODELS), description="Growth models to compare.")
    fit_window_h: tuple[float, float] | None = Field(
        default=None, description="Fit only this time range (h; since treatment when known)."
    )
    fold: float = Field(default=2.0, gt=1, description="Report the time to this fold change.")
    level: float | None = Field(default=None, description="Report the time to reach this value.")
    dose_response: bool = Field(default=True, description="IC50(t) and GR when compounds exist.")
    interval_h: float | None = Field(
        default=None, gt=0, description="GR over intervals of at least this length (None: "
        "consecutive time points).",
    )  # fmt: skip
    min_control_doublings: float = Field(
        default=0.25, ge=0, description="GR from treatment is undefined where the controls "
        "grew less (doublings).",
    )  # fmt: skip
    min_interval_doublings: float = Field(
        default=0.1, ge=0, description="GR over an interval is undefined where the controls "
        "grew less in it (doublings).",
    )  # fmt: skip
    treated_at: str | None = Field(
        default=None, description="Treatment time (ISO 8601); default: layout, then registry."
    )


@dataclass
class KineticsResult:
    readout: str
    time_column: str
    fits: pd.DataFrame
    ic50: pd.DataFrame = field(default_factory=pd.DataFrame)
    gr_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    gr_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    gr_interval_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    gr_interval_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    #: Wells × times matrix of the readout and the wells' conditions (for figures).
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    conditions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def per_well(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self.fits.to_dict("records"):
            values = {f"{PREFIX}{k}": v for k, v in row.items() if k not in ("well", "n_points")}
            out[str(row["well"])] = values
        if not self.gr_values.empty:
            last = self.gr_values.sort_values("time_h").groupby("well").tail(1)
            for row in last.to_dict("records"):
                out.setdefault(str(row["well"]), {})[f"{PREFIX}gr_final"] = row["gr"]
        return out


def choose_readout(course: pd.DataFrame, readout: str) -> str:
    if readout != "auto":
        if readout not in course:
            raise ModuleError(f"The time course has no column {readout!r}")
        return readout
    counts = [c for c in course.columns if str(c).endswith("_objects_measured")]
    if counts:
        return str(counts[0])
    if "confluence_pct" in course:
        return "confluence_pct"
    raise ModuleError("No object counts or confluence in the time course: set the readout")


def analyse(course: pd.DataFrame, layout: Any, params: KineticsParams) -> KineticsResult:
    """Growth fits and dose–response over time of a time-course table (see module docs)."""
    from hcs_studio.kinetics.response import (
        conditions,
        gr_intervals,
        gr_over_time,
        ic50_over_time,
        time_column,
    )

    if course is None or course.empty:
        raise ModuleError("The run has no time course (a time-lapse plate is needed)")
    if "time_h" not in course:
        raise ModuleError(
            "The time course has no times: set the time interval in the Metadata panel"
        )
    readout = choose_readout(course, params.readout)
    time = time_column(course)
    notes: list[str] = []
    if time == "time_h":
        notes.append("Treatment time unknown: times are from the first time point")
    info = conditions(layout)
    rows = []
    for well, group in course.groupby("well"):
        group = group.sort_values(time)
        t = group[time].to_numpy(float)
        y = pd.to_numeric(group[readout], errors="coerce").to_numpy(float)
        fit = fit_growth(t, y, params.models, params.fit_window_h)
        metrics = curve_metrics(t, y, params.fold, params.level)
        row = {"well": str(well), **fit.as_dict(), **metrics}
        row["growth_rate_per_h"] = row.pop("rate_per_h")
        row.pop("aicc")
        rows.append(row)
    fits = pd.DataFrame(rows)
    matrix = course.pivot_table(index="well", columns=time, values=readout, aggfunc="mean")
    result = KineticsResult(readout, time, fits, notes=notes, matrix=matrix, conditions=info)
    if not info.empty:
        _auc_vs_vehicle(result, info)
    has_doses = not info.empty and bool((info["concentration"] > 0).any())
    if params.dose_response and has_doses:
        result.ic50 = ic50_over_time(course, info, readout, notes)
        result.gr_values, result.gr_metrics = gr_over_time(
            course, info, readout, notes, params.min_control_doublings
        )
        interval = course
        if time == "hours_since_treatment":  # intervals after treatment only
            interval = course[course[time] >= -1e-6]
        result.gr_interval_values, result.gr_interval_metrics = gr_intervals(
            interval, info, readout, notes, params.interval_h, params.min_interval_doublings
        )
    return result


def _auc_vs_vehicle(result: KineticsResult, info: pd.DataFrame) -> None:
    """AUC as % of the median AUC of the vehicle wells (same cell line, else all)."""
    fits = result.fits.merge(info[["well", "role", "cell_line"]], on="well", how="left")
    vehicle = fits[fits["role"] == "negative_control"]
    values = []
    for row in fits.to_dict("records"):
        mine = vehicle[vehicle["cell_line"] == row["cell_line"]]
        reference = float((mine if not mine.empty else vehicle)["auc"].median())
        values.append(100 * row["auc"] / reference if reference > 0 else math.nan)
    result.fits["auc_vs_vehicle_pct"] = values


# -------------------------------------------------------------------- files


def write_kinetics(result: KineticsResult, out: Path, info: dict[str, Any]) -> list[str]:
    """Tables, ``kinetics.json`` and figures; returns the file names written."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    tables = {
        "growth_fits.csv": result.fits,
        "ic50_over_time.csv": result.ic50,
        "gr_values.csv": result.gr_values,
        "gr_metrics_over_time.csv": result.gr_metrics,
        "gr_interval_values.csv": result.gr_interval_values,
        "gr_interval_metrics.csv": result.gr_interval_metrics,
    }
    for name, table in tables.items():
        if table is not None and not table.empty:
            table.to_csv(out / name, index=False)
            files.append(name)
        else:
            (out / name).unlink(missing_ok=True)
    summary = {
        "readout": result.readout,
        "time": result.time_column,
        "notes": result.notes,
        **info,
        "files": files,
    }
    (out / "kinetics.json").write_text(json.dumps(summary, indent=1, default=_json), "utf-8")
    files += _figures(result, out)
    return files


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _figures(result: KineticsResult, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    if _growth_figure(result, out / "growth_curves.png"):
        written.append("growth_curves.png")
    if not result.ic50.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for (compound, line), group in result.ic50.groupby(["compound", "cell_line"]):
            label = f"{compound}" + (f" ({line})" if line else "")
            axes[0].plot(group["time_h"], group["ic50"], "o-", label=label)
        gr = result.gr_metrics
        for (compound, line), group in (
            gr.groupby(["compound", "cell_line"]) if not gr.empty else []
        ):
            label = f"{compound}" + (f" ({line})" if line else "")
            axes[1].plot(group["time_h"], group["GR50"], "o-", label=label)
        for axis, title in zip(axes, ("IC50 (% of control)", "GR50 (from treatment)"), strict=True):
            axis.set_yscale("log")
            axis.set_xlabel(_time_label(result.time_column))
            axis.set_ylabel("concentration (M)")
            axis.set_title(title)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "potency_over_time.png", dpi=120)
        plt.close(fig)
        written.append("potency_over_time.png")
    return written


def _growth_figure(result: KineticsResult, path: Path) -> bool:
    """Readout over time (log scale): vehicle wells in black, each compound in its own panel
    with concentrations from light to dark; wells without a layout in grey."""
    import matplotlib.pyplot as plt

    matrix = result.matrix
    if matrix.empty:
        path.unlink(missing_ok=True)
        return False
    info = result.conditions.set_index("well") if not result.conditions.empty else None
    compounds: list[str] = []
    if info is not None:
        compounds = sorted(c for c in info["compound"].unique() if c and c != "combination")
    panels = compounds[:6] or [""]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), squeeze=False)
    times = np.asarray(matrix.columns, dtype=float)
    for axis, compound in zip(axes[0], panels, strict=True):
        if info is None:
            for well in matrix.index:
                axis.plot(times, matrix.loc[well], color="0.6", lw=0.8)
        else:
            for well in info.index[info["role"] == "negative_control"]:
                if well in matrix.index:
                    axis.plot(times, matrix.loc[well], color="black", lw=1.0)
            mine = info[info["compound"] == compound].sort_values("concentration")
            doses = sorted(mine["concentration"].unique())
            cmap = plt.get_cmap("viridis_r")
            for well, row in mine.iterrows():
                well = str(well)
                if well in matrix.index:
                    shade = doses.index(row["concentration"]) / max(len(doses) - 1, 1)
                    axis.plot(times, matrix.loc[well], color=cmap(0.15 + 0.8 * shade), lw=0.9)
        positive = matrix.to_numpy(float)
        if np.nanmin(positive) > 0:
            axis.set_yscale("log")
        axis.set_title(compound or "all wells", fontsize=9)
        axis.set_xlabel(_time_label(result.time_column))
        axis.set_ylabel(result.readout, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _time_label(column: str) -> str:
    return "hours since treatment" if column == "hours_since_treatment" else "hours"


# ----------------------------------------------------------------- table step


@MODULES.register("kinetics")
class Kinetics(TableModule):
    """Growth curves, doubling times, IC50 and GR over time of a time-lapse run."""

    name = "kinetics"
    title = "Kinetics (growth, IC50(t), GR)"
    Params = KineticsParams

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        return pd.DataFrame(index=objects.index)

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        from hcs_studio.kinetics.timecourse import _layout, build_time_course, time_info

        params: KineticsParams = self.params
        course = build_time_course(folder, treated_at=params.treated_at)
        if course is None:
            raise ModuleError("kinetics needs a time-lapse run (more than one time point)")
        result = analyse(course, _layout(folder), params)
        for well, values in result.per_well().items():
            per_well.setdefault(well, {}).update(values)
        write_kinetics(result, Path(folder) / "kinetics", time_info(folder, params.treated_at))


def analyse_runs(
    folders: list[Path], params: KineticsParams, output: Path
) -> tuple[KineticsResult, list[str]]:
    """Kinetics of one plate imaged as separate acquisitions (``timecourse.assemble``)."""
    from hcs_studio.kinetics.timecourse import _layout, assemble, time_info

    course = assemble(folders, treated_at=params.treated_at)
    result = analyse(course, _layout(folders[0]), params)
    info = time_info(folders[0], params.treated_at)
    info["runs"] = [Path(f).name for f in folders]
    return result, write_kinetics(result, output, info)
