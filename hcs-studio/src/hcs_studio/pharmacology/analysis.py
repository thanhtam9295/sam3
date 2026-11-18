"""The pharmacology workflow: wells → normalization → QC → curves → combinations.

``analyse`` works on a well table in memory (no files); ``run_pharmacology`` loads finished
runs, analyses them and writes ``pharmacology/`` into the first run folder (or ``output``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations as pairs
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.pharmacology.combination import (
    MODELS,
    CombinationResult,
    find_matrices,
    score_matrix,
    scores_table,
)
from hcs_studio.pharmacology.curves import (
    CurveFit,
    activity_area,
    auc_normalized,
    compare_ec50,
    concentration_means,
    fit_curve,
    gr_metrics,
)
from hcs_studio.pharmacology.normalize import (
    Normalization,
    Normalized,
    exclude_outliers,
    load_wells,
    normalize,
    plate_qc,
)

CURVE_MODELS = ("4pl", "5pl", "3pl_bottom", "3pl_top")


@dataclass
class PharmOptions:
    readouts: list[str]
    normalization: str = "percent_inhibition"
    models: tuple[str, ...] = MODELS
    curve_model: str = "4pl"
    robust: bool = False
    bootstrap: int = 0
    outlier_mad: float | None = None
    z_threshold: float = 0.5
    doubling_time_h: float | None = None
    assay_duration_h: float | None = None
    time_zero_runs: list[str] = field(default_factory=list)
    excluded_wells: list[tuple[str, str]] = field(default_factory=list)
    included_wells: list[tuple[str, str]] = field(default_factory=list)
    figures: bool = True
    surfaces: bool = False
    pdf: bool = True

    def check(self) -> None:
        if not self.readouts:
            raise ValueError("Choose at least one readout")
        if self.curve_model not in CURVE_MODELS:
            raise ValueError(
                f"Unknown curve model {self.curve_model!r} ({', '.join(CURVE_MODELS)})"
            )
        unknown = [m for m in self.models if m not in MODELS]
        if unknown:
            raise ValueError(f"Unknown synergy model(s) {unknown} (use {', '.join(MODELS)})")


@dataclass
class CurveResult:
    readout: str
    cell_line: str
    compound: str
    unit: str
    fit: CurveFit
    points: pd.DataFrame  # conc (base), normalized, excluded, plate, well
    auc: float
    activity_area: float
    emax_observed: float
    emax_fitted: float
    gr: dict[str, float] = field(default_factory=dict)
    gi50: float = math.nan

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.readout, self.cell_line, self.compound)


@dataclass
class ReadoutResult:
    readout: str
    data: Normalized
    qc: pd.DataFrame
    curves: list[CurveResult]
    combinations: list[CombinationResult]
    comparisons: pd.DataFrame
    notes: list[str] = field(default_factory=list)


@dataclass
class PharmResult:
    options: PharmOptions
    readouts: dict[str, ReadoutResult]
    runs: list[str] = field(default_factory=list)
    folder: Path | None = None
    files: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


# --------------------------------------------------------------------- curves


def _curve_kwargs(options: PharmOptions) -> dict[str, Any]:
    gr = options.normalization == "gr"
    top = 1.0 if gr else 100.0
    kwargs: dict[str, Any] = {
        "robust": options.robust,
        "bootstrap": options.bootstrap,
        "half": 0.5 if gr else 50.0,
    }
    if options.curve_model == "5pl":
        kwargs["model"] = "5pl"
    elif options.curve_model == "3pl_bottom":
        # % of control and GR fall to the bottom; % inhibition starts at it
        kwargs["fix_bottom"] = -1.0 if gr else 0.0
    elif options.curve_model == "3pl_top":
        kwargs["fix_top"] = top
    return kwargs


def fit_single_agents(data: Normalized, options: PharmOptions) -> list[CurveResult]:
    frame = data.wells[data.wells["n_active"] == 1]
    method = data.method
    out = []
    for (line, compound), group in frame.groupby(
        [frame["cell_line"], frame["treatments"].map(lambda ts: ts[0][0])]
    ):
        conc = group["treatments"].map(lambda ts: ts[0][1]).to_numpy(float)
        points = pd.DataFrame(
            {
                "conc": conc,
                "normalized": group["normalized"].to_numpy(float),
                "effect": group["effect"].to_numpy(float),
                "excluded": group["excluded"].to_numpy(bool),
                "plate": group["plate"].to_numpy(str),
                "well": group["well"].to_numpy(str),
            }
        )
        used = points[~points["excluded"]]
        fit = fit_curve(used["conc"], used["normalized"], **_curve_kwargs(options))
        levels, means, _, _ = concentration_means(used["conc"], used["normalized"])
        _, effect_means, _, _ = concentration_means(used["conc"], used["effect"])
        if method == "percent_inhibition":
            reference, decreasing, full = 0.0, False, 100.0
        elif method == "percent_of_control":
            reference, decreasing, full = 100.0, True, 100.0
        else:
            reference, decreasing, full = 1.0, True, 1.0
        emax_fitted = math.nan
        if math.isfinite(fit.top) and levels.size:
            top_value = float(fit.predict(levels[-1]))
            if method == "percent_inhibition":
                emax_fitted = top_value
            elif method == "percent_of_control":
                emax_fitted = 100 - top_value
            else:
                emax_fitted = 100 * (1 - top_value) / 2
        result = CurveResult(
            readout=data.readout,
            cell_line=str(line),
            compound=str(compound),
            unit=str(group["treatments"].iloc[0][0][2]),
            fit=fit,
            points=points,
            auc=auc_normalized(levels, means),
            activity_area=activity_area(fit, reference, decreasing, full),
            emax_observed=float(np.nanmax(effect_means)) if effect_means.size else math.nan,
            emax_fitted=emax_fitted,
        )
        if method == "gr":
            result.gr = gr_metrics(used["conc"], used["normalized"])
            gi = group.loc[~group["excluded"], "gi"].to_numpy(float)
            result.gi50 = fit_curve(used["conc"], gi, half=50.0).abs_ic50
        out.append(result)
    return out


def _pow10(value: float) -> float:
    return float(10**value) if math.isfinite(value) else math.nan


def curves_table(curves: Sequence[CurveResult], molar: bool = True) -> pd.DataFrame:
    rows = []
    for c in curves:
        f = c.fit
        row: dict[str, Any] = {
            "readout": c.readout,
            "cell_line": c.cell_line,
            "compound": c.compound,
            "unit": c.unit,
            "model": f.model,
            "ec50": f.ec50 if math.isfinite(f.log_ec50) else math.nan,
            "ec50_ci_low": _pow10(f.log_ec50_ci[0]),
            "ec50_ci_high": _pow10(f.log_ec50_ci[1]),
            "log_ec50": f.log_ec50,
            "abs_ic50": f.abs_ic50,
            "abs_ic50_ci_low": _pow10(f.log_abs_ic50_ci[0]),
            "abs_ic50_ci_high": _pow10(f.log_abs_ic50_ci[1]),
            "hill": f.hill,
            "hill_ci_low": f.hill_ci[0],
            "hill_ci_high": f.hill_ci[1],
            "top": f.top,
            "bottom": f.bottom,
            "asymmetry": f.asymmetry if f.model == "5pl" else math.nan,
            "emax_observed": c.emax_observed,
            "emax_fitted": c.emax_fitted,
            "auc": c.auc,
            "activity_area": c.activity_area,
            "r2": f.r2,
            "rmse": f.rmse,
            "n_points": f.n_points,
            "n_concentrations": f.n_concentrations,
            "n_replicates": f.n_replicates,
            "n_excluded": int(c.points["excluded"].sum()),
            "ci_method": f.ci_method,
            "converged": f.converged,
            "flags": "; ".join(f.flags),
        }
        if c.gr:
            row.update(c.gr)
            row["GI50"] = c.gi50
        rows.append(row)
    return pd.DataFrame(rows)


def ic50_comparisons(curves: Sequence[CurveResult]) -> pd.DataFrame:
    """Log-EC50 shift between cell lines for each compound and readout (F test)."""
    rows = []
    by_compound: dict[tuple[str, str], list[CurveResult]] = {}
    for c in curves:
        by_compound.setdefault((c.readout, c.compound), []).append(c)
    for (readout, compound), group in by_compound.items():
        for a, b in pairs(sorted(group, key=lambda c: c.cell_line), 2):
            pa = a.points[~a.points["excluded"]]
            pb = b.points[~b.points["excluded"]]
            test = compare_ec50(pa["conc"], pa["normalized"], pb["conc"], pb["normalized"])
            rows.append(
                {
                    "readout": readout,
                    "compound": compound,
                    "cell_line_a": a.cell_line,
                    "cell_line_b": b.cell_line,
                    "delta_log_ec50": test["delta_log_ec50"],
                    "fold_change": 10 ** test["delta_log_ec50"]
                    if math.isfinite(test["delta_log_ec50"])
                    else math.nan,
                    "f": test["f"],
                    "p": test["p"],
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- workflow


def analyse(
    wells: pd.DataFrame, options: PharmOptions, time_zero: pd.DataFrame | None = None
) -> PharmResult:
    options.check()
    results: dict[str, ReadoutResult] = {}
    for readout in options.readouts:
        data = normalize(
            wells,
            readout,
            Normalization(
                options.normalization,
                time_zero=time_zero,
                doubling_time_h=options.doubling_time_h,
                assay_duration_h=options.assay_duration_h,
            ),
        )
        exclude_outliers(data, options.outlier_mad, options.excluded_wells, options.included_wells)
        qc = plate_qc(data, options.z_threshold)
        curves = fit_single_agents(data, options)
        combos: list[CombinationResult] = []
        notes = list(data.notes)
        matrices = find_matrices(data)
        if matrices and options.models:
            if options.normalization == "gr":
                notes.append(
                    "Combination scores use (1 − GR)/2 × 100 as the effect; Bliss and ZIP "
                    "assume effects between 0 and 100 %"
                )
            combos = [score_matrix(m, options.models) for m in matrices]
        results[readout] = ReadoutResult(
            readout, data, qc, curves, combos, ic50_comparisons(curves), notes
        )
    return PharmResult(options, results)


def run_pharmacology(
    folders: Sequence[Path],
    options: PharmOptions,
    output: Path | None = None,
    write: bool = True,
) -> PharmResult:
    """Analyse finished runs and write ``pharmacology/`` (in the first run folder, or in
    ``output``)."""
    import time

    started = time.perf_counter()
    folders = [Path(f) for f in folders]
    wells = load_wells(folders)
    time_zero = (
        load_wells([Path(f) for f in options.time_zero_runs]) if options.time_zero_runs else None
    )
    result = analyse(wells, options, time_zero)
    result.runs = [str(f) for f in folders]
    if write:
        from hcs_studio.pharmacology.report import write_outputs

        target = Path(output) if output is not None else folders[0] / "pharmacology"
        write_outputs(result, target)
    result.seconds = time.perf_counter() - started
    return result


def options_dict(options: PharmOptions) -> dict[str, Any]:
    data = asdict(options)
    data["models"] = list(options.models)
    data["excluded_wells"] = [list(w) for w in options.excluded_wells]
    data["included_wells"] = [list(w) for w in options.included_wells]
    return data


def combination_tables(result: ReadoutResult) -> pd.DataFrame:
    table = scores_table(result.combinations)
    if not table.empty:
        table.insert(0, "readout", result.readout)
    return table
