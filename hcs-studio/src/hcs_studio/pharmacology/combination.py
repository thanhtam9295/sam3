"""Combination matrices (drug A × drug B × cell line) and synergy scores.

Responses are on the % inhibition scale (``effect``: 0 = vehicle, 100 = full effect), as in
SynergyFinder. Rows are drug A (ascending, the 0 row first), columns drug B.

Reference models (expected response E of the combination; synergy = observed − E):

- Bliss independence: E = yA + yB − yA·yB/100, with the observed single-agent responses
  (the 0 row and column of the matrix).
- HSA (highest single agent): E = max(yA, yB), observed single agents.
- Loewe additivity: E solves a/ICA(E) + b/ICB(E) = 1, with ICX the inverse of drug X's
  4PL fit to its single-agent responses. Where no solution exists (an effect beyond what
  either drug reaches), the point is flagged and HSA is used, as SynergyFinder does.
  The combination index (Chou–Talalay, generalised to 4PL fits) is
  CI = a/ICA(y) + b/ICB(y) at the observed response y (< 1 synergy, > 1 antagonism).
- ZIP (Yadav et al. 2015): each row is fitted as a 4PL in drug B with its lower asymptote
  fixed at drug A's fitted single-agent response and its top at 100 (columns likewise);
  the combination response is the mean of the row and column fits, and δ = that − the
  ZIP reference yA + yB − yA·yB/100 from the fitted single-agent curves.

Summary scores use the combination cells only (both drugs > 0): the mean, the mean of the
most synergistic 3 × 3 window, and the maximum cell. Replicate matrices (the k-th well of
every cell) give a one-sample t test of the mean against 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcs_studio.pharmacology.curves import CurveFit, fit_curve, logistic
from hcs_studio.pharmacology.normalize import Normalized

Array = NDArray[np.float64]
MODELS = ("bliss", "hsa", "loewe", "zip")
#: SynergyFinder's guide: < −10 likely antagonistic, −10 to 10 likely additive, > 10
#: likely synergistic.
THRESHOLD = 10.0


@dataclass
class Matrix:
    cell_line: str
    drug_a: str
    drug_b: str
    unit_a: str
    unit_b: str
    conc_a: Array  # base units, ascending, 0 first
    conc_b: Array
    response: Array  # (len(conc_a), len(conc_b)) mean % inhibition; NaN = no well
    sd: Array
    n: Array
    wells: dict[tuple[int, int], list[tuple[str, str]]] = field(default_factory=dict)
    replicates: list[Array] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.cell_line or 'no cell line'} · {self.drug_a} × {self.drug_b}"

    @property
    def slug(self) -> str:
        import re

        text = f"{self.cell_line or 'none'}_{self.drug_a}_x_{self.drug_b}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)

    @property
    def missing(self) -> int:
        return int(np.isnan(self.response).sum())


def matrix_from_values(
    conc_a: Any,
    conc_b: Any,
    response: Any,
    drug_a: str = "A",
    drug_b: str = "B",
    cell_line: str = "",
    unit_a: str = "µM",
    unit_b: str = "µM",
    replicates: Sequence[Any] = (),
) -> Matrix:
    """A matrix from given concentrations (0 first) and mean responses (% inhibition)."""
    response = np.asarray(response, dtype=np.float64)
    reps = [np.asarray(r, dtype=np.float64) for r in replicates]
    sd = np.std(np.stack(reps), axis=0, ddof=1) if len(reps) > 1 else np.full_like(response, np.nan)
    return Matrix(
        cell_line,
        drug_a,
        drug_b,
        unit_a,
        unit_b,
        np.asarray(conc_a, dtype=np.float64),
        np.asarray(conc_b, dtype=np.float64),
        response,
        sd,
        np.full_like(response, max(1, len(reps))),
        {},
        reps,
    )


# ------------------------------------------------------------- block detection


def find_matrices(data: Normalized) -> list[Matrix]:
    """Combination blocks of the annotation: every (cell line, drug A, drug B) with wells
    containing both drugs, plus the single-agent wells at the same concentrations and the
    cell line's vehicle wells (the 0 × 0 cell). Excluded wells are left out."""
    frame = data.wells[~data.wells["excluded"]]
    out: list[Matrix] = []
    for line_key, line_frame in frame.groupby("cell_line"):
        line = str(line_key)
        combos = line_frame[line_frame["n_active"] == 2]
        pairs = sorted({tuple(t[0] for t in ts) for ts in combos["treatments"]})
        for drug_a, drug_b in pairs:
            mine = combos[
                combos["treatments"].map(lambda ts, p=(drug_a, drug_b): (ts[0][0], ts[1][0]) == p)
            ]
            a_levels = sorted({_canon(ts[0][1]) for ts in mine["treatments"]})
            b_levels = sorted({_canon(ts[1][1]) for ts in mine["treatments"]})
            unit_a = mine["treatments"].iloc[0][0][2]
            unit_b = mine["treatments"].iloc[0][1][2]
            conc_a = np.array([0.0, *a_levels])
            conc_b = np.array([0.0, *b_levels])
            cells: dict[tuple[int, int], list[tuple[float, str, str]]] = {}

            def add(
                i: int,
                j: int,
                rows: pd.DataFrame,
                cells: dict[tuple[int, int], list[tuple[float, str, str]]] = cells,
            ) -> None:
                for value, plate, well in rows[["effect", "plate", "well"]].itertuples(index=False):
                    cells.setdefault((i, j), []).append((float(value), str(plate), str(well)))

            for i, a in enumerate(conc_a):
                for j, b in enumerate(conc_b):
                    if i and j:
                        hit = mine[
                            mine["treatments"].map(
                                lambda ts, a=a, b=b: _canon(ts[0][1]) == a and _canon(ts[1][1]) == b
                            )
                        ]
                    elif i:
                        hit = _single(line_frame, drug_a, a)
                    elif j:
                        hit = _single(line_frame, drug_b, b)
                    else:
                        hit = line_frame[
                            (line_frame["n_active"] == 0)
                            & line_frame["role"].isin(("negative_control", "untreated"))
                        ]
                    add(i, j, hit)
            shape = (conc_a.size, conc_b.size)
            response = np.full(shape, np.nan)
            sd = np.full(shape, np.nan)
            n = np.zeros(shape)
            wells: dict[tuple[int, int], list[tuple[str, str]]] = {}
            n_reps = max((len(v) for (i, j), v in cells.items() if i and j), default=1)
            replicates = [np.full(shape, np.nan) for _ in range(n_reps)]
            for (i, j), values in cells.items():
                v = np.array([x[0] for x in values])
                response[i, j] = v.mean()
                sd[i, j] = v.std(ddof=1) if v.size > 1 else np.nan
                n[i, j] = v.size
                wells[(i, j)] = [(p, w) for _, p, w in values]
                ordered = [x[0] for x in sorted(values, key=lambda x: (x[1], x[2]))]
                for k in range(n_reps):
                    # single agents and vehicle are shared: every replicate uses their mean
                    replicates[k][i, j] = (
                        ordered[k]
                        if (i and j and k < len(ordered))
                        else (v.mean() if not (i and j) else np.nan)
                    )
            out.append(
                Matrix(
                    line,
                    drug_a,
                    drug_b,
                    unit_a,
                    unit_b,
                    conc_a,
                    conc_b,
                    response,
                    sd,
                    n,
                    wells,
                    replicates,
                )
            )
    return out


def _canon(value: float) -> float:
    return float(f"{value:.12g}")


def _single(frame: pd.DataFrame, drug: str, conc: float) -> pd.DataFrame:
    return frame[
        (frame["n_active"] == 1)
        & frame["treatments"].map(
            lambda ts: len(ts) == 1 and ts[0][0] == drug and _canon(ts[0][1]) == conc
        )
    ]


# ------------------------------------------------------------------- models


def single_agent_fit(conc: Array, response: Array) -> CurveFit:
    """4PL of one drug's single-agent responses (the matrix edge, 0 included as a point far
    below the lowest concentration, like a dose of 0 in a log-logistic fit)."""
    c = np.asarray(conc, dtype=np.float64).copy()
    positive = c[c > 0]
    if positive.size:
        c[c <= 0] = float(np.min(positive)) * 1e-6
    keep = np.isfinite(response)
    return fit_curve(c[keep], np.asarray(response)[keep], half=50.0)


def _value(fit: CurveFit, conc: float, observed: float) -> float:
    if conc <= 0:
        return 0.0 if not math.isfinite(observed) else observed
    if not math.isfinite(fit.top):
        return observed
    return float(fit.predict(math.log10(conc)))


def bliss_expected(m: Matrix) -> Array:
    ya, yb = m.response[:, :1], m.response[:1, :]
    return np.asarray(ya + yb - ya * yb / 100.0, dtype=np.float64)


def hsa_expected(m: Matrix) -> Array:
    return np.asarray(np.maximum(m.response[:, :1], m.response[:1, :]), dtype=np.float64)


def _ic(fit: CurveFit, effect: float) -> float:
    """Concentration of a single drug that gives ``effect`` (inf when never reached)."""
    log_c = fit.inverse(effect)
    return 10**log_c if math.isfinite(log_c) else math.inf


def loewe_expected(m: Matrix, fit_a: CurveFit, fit_b: CurveFit) -> tuple[Array, Array]:
    """Expected Loewe responses and a mask of points where Loewe is undefined (HSA used)."""
    from scipy.optimize import brentq

    expected = hsa_expected(m).copy()
    undefined = np.zeros(m.response.shape, dtype=bool)
    ok = all(math.isfinite(f.top) for f in (fit_a, fit_b))
    for i, a in enumerate(m.conc_a):
        for j, b in enumerate(m.conc_b):
            if not (i and j):
                continue
            if not ok:
                undefined[i, j] = True
                continue

            def excess(e: float, a: float = a, b: float = b) -> float:
                return a / _ic(fit_a, e) + b / _ic(fit_b, e) - 1.0

            lo = max(min(fit_a.bottom, fit_a.top), min(fit_b.bottom, fit_b.top)) + 1e-9
            hi = max(max(fit_a.bottom, fit_a.top), max(fit_b.bottom, fit_b.top)) - 1e-9
            # the sum falls from +∞ (tiny effects need tiny doses... ) — scan for a sign change
            grid = np.linspace(min(lo, hi), max(lo, hi), 400)
            values = np.array([excess(float(e)) for e in grid])
            sign = np.flatnonzero(
                np.isfinite(values[:-1])
                & np.isfinite(values[1:])
                & (np.sign(values[:-1]) != np.sign(values[1:]))
            )
            if sign.size == 0:
                undefined[i, j] = True
                continue
            k = int(sign[0])
            expected[i, j] = brentq(excess, float(grid[k]), float(grid[k + 1]))
    return expected, undefined


def combination_index(m: Matrix, fit_a: CurveFit, fit_b: CurveFit) -> Array:
    """CI = a/ICA(y) + b/ICB(y) at the observed response y (NaN where undefined)."""
    ci = np.full(m.response.shape, np.nan)
    for i, a in enumerate(m.conc_a):
        for j, b in enumerate(m.conc_b):
            y = m.response[i, j]
            if i and j and math.isfinite(y):
                value = a / _ic(fit_a, y) + b / _ic(fit_b, y)
                ci[i, j] = (
                    value if (_ic(fit_a, y) < math.inf and _ic(fit_b, y) < math.inf) else math.nan
                )
    return ci


def _fit_fixed(logx: Array, y: Array, bottom: float) -> Array | None:
    """4PL with bottom and top (100) fixed; returns fitted values at ``logx`` (ZIP)."""
    from scipy.optimize import least_squares

    keep = np.isfinite(y)
    if keep.sum() < 2 or bottom >= 100:
        return None
    x, v = logx[keep], y[keep]

    def residuals(th: Array) -> Array:
        return np.asarray(logistic(x, bottom, 100.0, th[0], th[1]) - v, dtype=np.float64)

    best = None
    xmin, xmax = float(np.min(x)), float(np.max(x))
    for start in ((float(np.median(x)), 1.0), (float(np.median(x)), 3.0), (xmin, 1.0)):
        try:
            result = least_squares(
                residuals, np.array(start), bounds=([xmin - 6, 1e-3], [xmax + 6, 20.0])
            )
        except ValueError:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        return None
    return np.asarray(logistic(logx, bottom, 100.0, best.x[0], best.x[1]), dtype=np.float64)


def zip_delta(m: Matrix, fit_a: CurveFit, fit_b: CurveFit) -> tuple[Array, Array, Array]:
    """ZIP δ (observed-fitted − reference), the fitted combination surface and the ZIP
    reference, after SynergyFinder."""
    ya = np.array([_value(fit_a, a, m.response[i, 0]) for i, a in enumerate(m.conc_a)])
    yb = np.array([_value(fit_b, b, m.response[0, j]) for j, b in enumerate(m.conc_b)])
    ya[0], yb[0] = 0.0, 0.0
    shape = m.response.shape
    rows = np.full(shape, np.nan)
    cols = np.full(shape, np.nan)
    log_b = np.log10(np.where(m.conc_b > 0, m.conc_b, np.nan))
    log_a = np.log10(np.where(m.conc_a > 0, m.conc_a, np.nan))
    for i in range(1, shape[0]):
        fitted = _fit_fixed(log_b[1:], m.response[i, 1:], ya[i])
        rows[i, 1:] = fitted if fitted is not None else m.response[i, 1:]
    for j in range(1, shape[1]):
        fitted = _fit_fixed(log_a[1:], m.response[1:, j], yb[j])
        cols[1:, j] = fitted if fitted is not None else m.response[1:, j]
    surface = (rows + cols) / 2
    reference = ya[:, None] + yb[None, :] - ya[:, None] * yb[None, :] / 100.0
    delta = surface - reference
    delta[0, :] = 0.0
    delta[:, 0] = 0.0
    return delta, surface, reference


# ----------------------------------------------------------------- summaries


@dataclass
class ModelResult:
    model: str
    expected: Array
    synergy: Array  # observed − expected (ZIP: δ), % points; 0 row/column = 0
    mean: float
    window_mean: float  # most synergistic 3 × 3 window
    window: tuple[int, int] | None  # top-left (i, j) of that window
    max_value: float
    max_cell: tuple[int, int] | None
    replicate_means: list[float]
    p_value: float
    ci95: tuple[float, float]
    label: str
    notes: list[str] = field(default_factory=list)


@dataclass
class CombinationResult:
    matrix: Matrix
    fit_a: CurveFit
    fit_b: CurveFit
    models: dict[str, ModelResult]
    combination_index: Array


def _synergy(
    m: Matrix, model: str, fit_a: CurveFit, fit_b: CurveFit
) -> tuple[Array, Array, list[str]]:
    notes: list[str] = []
    if model == "bliss":
        expected = bliss_expected(m)
        synergy = m.response - expected
    elif model == "hsa":
        expected = hsa_expected(m)
        synergy = m.response - expected
    elif model == "loewe":
        expected, undefined = loewe_expected(m, fit_a, fit_b)
        synergy = m.response - expected
        if undefined[1:, 1:].any():
            notes.append(
                f"Loewe undefined at {int(undefined[1:, 1:].sum())} point(s) (an effect beyond "
                "what a single drug reaches): HSA used there"
            )
    elif model == "zip":
        synergy, _, expected = zip_delta(m, fit_a, fit_b)
    else:
        raise ValueError(f"Unknown synergy model {model!r} (use {', '.join(MODELS)})")
    synergy = np.asarray(synergy, dtype=np.float64).copy()
    synergy[0, :] = 0.0
    synergy[:, 0] = 0.0
    return np.asarray(expected, dtype=np.float64), synergy, notes


def summarize(
    synergy: Array,
) -> tuple[float, float, tuple[int, int] | None, float, tuple[int, int] | None]:
    inner = synergy[1:, 1:]
    if not np.isfinite(inner).any():
        return math.nan, math.nan, None, math.nan, None
    mean = float(np.nanmean(inner))
    best, where = -math.inf, None
    rows, cols = inner.shape
    for i in range(max(1, rows - 2)):
        for j in range(max(1, cols - 2)):
            window = inner[i : i + 3, j : j + 3]
            if np.isfinite(window).any():
                value = float(np.nanmean(window))
                if value > best:
                    best, where = value, (i + 1, j + 1)
    k = int(np.nanargmax(inner))
    cell = (k // cols + 1, k % cols + 1)
    return mean, best, where, float(inner.flat[k]), cell


def interpret(score: float) -> str:
    if not math.isfinite(score):
        return "not computed"
    if score < -THRESHOLD:
        return "likely antagonistic"
    if score > THRESHOLD:
        return "likely synergistic"
    return "likely additive"


def score_matrix(m: Matrix, models: Sequence[str] = MODELS) -> CombinationResult:
    from scipy import stats

    fit_a = single_agent_fit(m.conc_a, m.response[:, 0])
    fit_b = single_agent_fit(m.conc_b, m.response[0, :])
    results: dict[str, ModelResult] = {}
    for model in models:
        expected, synergy, notes = _synergy(m, model, fit_a, fit_b)
        mean, window_mean, window, max_value, max_cell = summarize(synergy)
        replicate_means = []
        if len(m.replicates) > 1:
            for rep in m.replicates:
                rep_matrix = Matrix(
                    m.cell_line,
                    m.drug_a,
                    m.drug_b,
                    m.unit_a,
                    m.unit_b,
                    m.conc_a,
                    m.conc_b,
                    rep,
                    m.sd,
                    m.n,
                )
                ra = (
                    single_agent_fit(rep_matrix.conc_a, rep[:, 0])
                    if model in ("loewe", "zip")
                    else fit_a
                )
                rb = (
                    single_agent_fit(rep_matrix.conc_b, rep[0, :])
                    if model in ("loewe", "zip")
                    else fit_b
                )
                _, rep_syn, _ = _synergy(rep_matrix, model, ra, rb)
                replicate_means.append(summarize(rep_syn)[0])
        finite = [v for v in replicate_means if math.isfinite(v)]
        p, ci = math.nan, (math.nan, math.nan)
        if len(finite) >= 2:
            import warnings

            with warnings.catch_warnings():  # identical replicates: p is NaN, not an error
                warnings.simplefilter("ignore", RuntimeWarning)
                test = stats.ttest_1samp(finite, 0.0)
            p = float(test.pvalue)
            half = float(
                stats.t.ppf(0.975, len(finite) - 1)
                * np.std(finite, ddof=1)
                / math.sqrt(len(finite))
            )
            ci = (float(np.mean(finite)) - half, float(np.mean(finite)) + half)
        else:
            notes.append("no replicate matrices: no test against 0")
        results[model] = ModelResult(
            model,
            expected,
            synergy,
            mean,
            window_mean,
            window,
            max_value,
            max_cell,
            replicate_means,
            p,
            ci,
            interpret(mean),
            notes,
        )
    return CombinationResult(m, fit_a, fit_b, results, combination_index(m, fit_a, fit_b))


def scores_table(results: Sequence[CombinationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.matrix
        row: dict[str, Any] = {
            "cell_line": m.cell_line,
            "drug_a": m.drug_a,
            "drug_b": m.drug_b,
            "size": f"{m.conc_a.size}×{m.conc_b.size}",
            "missing_cells": m.missing,
            "replicates": len(m.replicates),
            "max_response": float(np.nanmax(m.response))
            if np.isfinite(m.response).any()
            else math.nan,
        }
        for name, res in r.models.items():
            row[f"{name}_mean"] = res.mean
            row[f"{name}_max_3x3"] = res.window_mean
            row[f"{name}_max_cell"] = res.max_value
            if res.max_cell is not None:
                i, j = res.max_cell
                row[f"{name}_max_at"] = (
                    f"{m.drug_a} {m.conc_a[i]:.4g} + {m.drug_b} {m.conc_b[j]:.4g}"
                )
            row[f"{name}_p"] = res.p_value
            row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = res.ci95
            row[f"{name}_label"] = res.label
            if res.notes:
                row[f"{name}_notes"] = "; ".join(res.notes)
        finite_ci = r.combination_index[np.isfinite(r.combination_index)]
        row["ci_median"] = float(np.median(finite_ci)) if finite_ci.size else math.nan
        rows.append(row)
    return pd.DataFrame(rows)
