"""Per-plate normalization: control-based, robust Z-score and B-score.

- **Robust Z**: (x − median of the sample wells) / (1.4826 · MAD of the sample wells).
- **B-score** (Brideau et al. 2003, J Biomol Screen 8:634): Tukey's two-way median polish of
  the sample wells removes row and column effects; the residuals are divided by
  1.4826 · MAD of all residuals. Controls and empty wells are left out of the polish.
- **Guard rail**: B-score (and any row/column correction) assumes compounds are placed at
  random with respect to rows and columns. On dose–response layouts (a compound diluted along
  a row or column) and combination matrices the "row/column effect" *is* the signal, so the
  correction is blocked with an explanation; use a uniformity-plate correction instead.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from hcs_studio.model.plate import parse_well_name, well_name

MAD_SCALE = 1.4826  # MAD → SD for normally distributed data
Method = Literal["percent_of_control", "percent_inhibition", "robust_z", "b_score", "uniformity"]
METHODS: tuple[str, ...] = (
    "percent_of_control",
    "percent_inhibition",
    "robust_z",
    "b_score",
    "uniformity",
)
POSITIONAL = ("b_score",)  # methods that remove row/column effects
SAMPLE = "sample"


class PositionalCorrectionBlocked(ValueError):
    """A row/column correction was asked for on a layout where it would remove real signal."""


@dataclass(frozen=True)
class WellInfo:
    """What normalization needs to know about a well: role and drugs (name, concentration)."""

    well: str
    role: str = SAMPLE
    compounds: tuple[tuple[str, float], ...] = ()

    @property
    def position(self) -> tuple[int, int]:
        return parse_well_name(self.well)


# ---------------------------------------------------------------------- grids


def to_grid(values: Mapping[str, float], rows: int, cols: int) -> NDArray[np.float64]:
    grid = np.full((rows, cols), np.nan)
    for well, value in values.items():
        r, c = parse_well_name(well)
        if r < rows and c < cols and value is not None and math.isfinite(float(value)):
            grid[r, c] = float(value)
    return grid


def from_grid(grid: NDArray[np.float64], wells: Iterable[str]) -> dict[str, float]:
    out = {}
    for well in wells:
        r, c = parse_well_name(well)
        out[well] = float(grid[r, c])
    return out


def mad(values: NDArray[np.float64]) -> float:
    """Scaled median absolute deviation (an SD estimate), ignoring NaN."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return math.nan
    return float(MAD_SCALE * np.median(np.abs(finite - np.median(finite))))


# --------------------------------------------------------------------- methods


def robust_z(values: Mapping[str, float], reference: Iterable[str]) -> dict[str, float]:
    """Robust Z of every well against the ``reference`` wells (usually the samples)."""
    ref = np.array([values[w] for w in reference if w in values], dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 3:
        raise ValueError("Robust Z needs at least 3 reference wells with values")
    centre, spread = float(np.median(ref)), mad(ref)
    if not spread > 0:
        raise ValueError("The reference wells have no spread (MAD = 0); robust Z is undefined")
    return {w: (float(v) - centre) / spread for w, v in values.items()}


@dataclass
class MedianPolish:
    residuals: NDArray[np.float64]
    overall: float
    row_effects: NDArray[np.float64]
    col_effects: NDArray[np.float64]
    iterations: int


def median_polish(grid: NDArray[np.float64], max_iter: int = 20, tol: float = 1e-9) -> MedianPolish:
    """Tukey's median polish of a plate (NaN = not part of the polish)."""
    residuals = np.array(grid, dtype=float)
    rows, cols = residuals.shape
    row_effects, col_effects = np.zeros(rows), np.zeros(cols)
    overall = 0.0
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        with warnings.catch_warnings():  # empty rows (no samples) give NaN, handled below
            warnings.simplefilter("ignore", RuntimeWarning)
            row_med = np.nanmedian(residuals, axis=1)
        row_med = np.where(np.isfinite(row_med), row_med, 0.0)
        residuals -= row_med[:, None]
        row_effects += row_med
        shift = float(np.median(col_effects))
        col_effects -= shift
        overall += shift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_med = np.nanmedian(residuals, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        residuals -= col_med[None, :]
        col_effects += col_med
        shift = float(np.median(row_effects))
        row_effects -= shift
        overall += shift
        if max(np.abs(row_med).max(initial=0), np.abs(col_med).max(initial=0)) < tol:
            break
    return MedianPolish(residuals, overall, row_effects, col_effects, iterations)


def b_score(
    values: Mapping[str, float], samples: Iterable[str], rows: int, cols: int
) -> tuple[dict[str, float], MedianPolish]:
    """B-scores of the sample wells (others NaN) and the polish (row/column effects)."""
    samples = list(samples)
    grid = to_grid({w: values[w] for w in samples if w in values}, rows, cols)
    if np.isfinite(grid).sum() < 3:
        raise ValueError("B-score needs at least 3 sample wells with values")
    polish = median_polish(grid)
    spread = mad(polish.residuals)
    if not spread > 0:
        raise ValueError("The residuals have no spread (MAD = 0); B-score is undefined")
    scores = polish.residuals / spread
    return {w: float(scores[parse_well_name(w)]) for w in values}, polish


# ------------------------------------------------------------------ guard rail


@dataclass(frozen=True)
class Guard:
    allowed: bool
    reason: str = ""
    series: tuple[str, ...] = field(default_factory=tuple)


def positional_guard(wells: Sequence[WellInfo], min_points: int = 3) -> Guard:
    """Is a row/column correction safe on this layout?

    Blocked when a compound is diluted along a row or a column (≥ ``min_points`` distinct
    concentrations in one row or column, monotonic in position: |Spearman ρ| ≥ 0.9), or when
    the plate holds combination wells (two or more drugs in ≥ 4 wells: matrices put each
    drug's concentration along rows or columns).
    """
    combos = [w for w in wells if len([c for c in w.compounds if c[1] > 0]) >= 2]
    if len(combos) >= 4:
        return Guard(
            False,
            f"{len(combos)} wells hold drug combinations (combination matrices place each "
            "drug's concentration along rows or columns). A row/column correction would remove "
            "the dose effect; use a uniformity-plate correction instead.",
            ("combinations",),
        )
    series: list[str] = []
    by_compound: dict[str, list[tuple[int, int, float]]] = {}
    for w in wells:
        if w.role != SAMPLE:
            continue
        r, c = w.position
        for name, conc in w.compounds:
            if conc > 0:
                by_compound.setdefault(name, []).append((r, c, float(conc)))
    for name, points in sorted(by_compound.items()):
        if len({conc for _, _, conc in points}) < min_points:
            continue  # a single-concentration compound (library screens) is never a series
        frame = pd.DataFrame(points, columns=["row", "col", "conc"])
        for axis, along in (("row", "col"), ("col", "row")):
            for line, group in frame.groupby(axis):
                if group["conc"].nunique() < min_points:
                    continue
                rho = stats.spearmanr(group[along], np.log10(group["conc"])).statistic
                if abs(float(rho)) >= 0.9:
                    index = int(str(line))
                    row_name = "".join(ch for ch in well_name(index, 0) if ch.isalpha())
                    where = f"row {row_name}" if axis == "row" else f"column {index + 1}"
                    series.append(f"{name} ({where})")
    if series:
        shown = ", ".join(series[:4]) + (f" and {len(series) - 4} more" if len(series) > 4 else "")
        return Guard(
            False,
            f"Dose–response series run along rows or columns: {shown}. B-score and other "
            "row/column corrections assume compounds are placed at random and would remove the "
            "dose effect; use a uniformity-plate correction instead.",
            tuple(series),
        )
    return Guard(True)


# ------------------------------------------------------------------ one plate


@dataclass
class NormalizedPlate:
    method: str
    frame: pd.DataFrame  # well, role, raw, normalized
    guard: Guard
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def normalize_plate(
    values: Mapping[str, float],
    wells: Sequence[WellInfo],
    rows: int,
    cols: int,
    method: str,
    *,
    factors: NDArray[np.float64] | None = None,
) -> NormalizedPlate:
    """Normalize one plate's readout. ``factors``: a uniformity map (``method="uniformity"``)."""
    if method not in METHODS:
        raise ValueError(f"Unknown normalization {method!r} ({', '.join(METHODS)})")
    info = {w.well: w for w in wells}
    roles = {w: (info[w].role if w in info else SAMPLE) for w in values}
    samples = [w for w, role in roles.items() if role == SAMPLE]
    guard = positional_guard(wells)
    notes: list[str] = []
    details: dict[str, Any] = {}
    if method in POSITIONAL and not guard.allowed:
        raise PositionalCorrectionBlocked(guard.reason)
    if method in ("percent_of_control", "percent_inhibition"):
        neg = np.array([values[w] for w, r in roles.items() if r == "negative_control"], float)
        neg = neg[np.isfinite(neg)]
        if not neg.size:
            raise ValueError("No negative-control (vehicle) wells with values on this plate")
        neg_median = float(np.median(neg))
        if method == "percent_of_control":
            if not neg_median > 0:
                raise ValueError("The vehicle median is not positive; use % inhibition")
            out = {w: 100 * float(v) / neg_median for w, v in values.items()}
        else:
            pos = np.array([values[w] for w, r in roles.items() if r == "positive_control"], float)
            pos = pos[np.isfinite(pos)]
            if not pos.size:
                raise ValueError("% inhibition needs positive-control wells")
            span = neg_median - float(np.median(pos))
            if span == 0:
                raise ValueError("Vehicle and positive controls have the same median")
            out = {w: 100 * (neg_median - float(v)) / span for w, v in values.items()}
        details["vehicle_median"] = neg_median
    elif method == "robust_z":
        out = robust_z(values, samples)
    elif method == "b_score":
        out, polish = b_score(values, samples, rows, cols)
        details["row_effects"] = polish.row_effects.tolist()
        details["col_effects"] = polish.col_effects.tolist()
        notes.append("B-scores are computed for sample wells only (controls are left out).")
    else:
        if factors is None:
            raise ValueError("Uniformity correction needs a uniformity map")
        if factors.shape != (rows, cols):
            raise ValueError(
                f"The uniformity map is for {factors.shape[0]}×{factors.shape[1]} plates; this "
                f"plate is {rows}×{cols}"
            )
        out = {w: float(v) / float(factors[parse_well_name(w)]) for w, v in values.items()}
    frame = pd.DataFrame(
        {
            "well": list(values),
            "role": [roles[w] for w in values],
            "raw": [values[w] for w in values],
            "normalized": [out.get(w, math.nan) for w in values],
        }
    )
    positions = [parse_well_name(w) for w in frame["well"]]
    order = sorted(range(len(frame)), key=lambda i: positions[i])
    frame = frame.iloc[order].reset_index(drop=True)
    return NormalizedPlate(method, frame, guard, notes, details)


def wells_from_layout(layout: Any) -> list[WellInfo]:
    """``WellInfo`` of every annotated well of a ``PlateLayout`` (drugs as for pharmacology)."""
    from hcs_studio.pharmacology.normalize import drug_treatments

    return [
        WellInfo(
            well_name(r, c),
            a.role,
            tuple((name, conc) for name, conc, _ in drug_treatments(a)),
        )
        for (r, c), a in sorted(layout.wells.items())
    ]


def all_wells(rows: int, cols: int) -> list[str]:
    return [well_name(r, c) for r in range(rows) for c in range(cols)]
