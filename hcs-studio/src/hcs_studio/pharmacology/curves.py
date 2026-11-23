"""Dose–response curves on log10(concentration), parametrised as in GraphPad Prism.

4PL ("log(inhibitor) vs. response — variable slope"):
    Y = Bottom + (Top − Bottom) / (1 + 10^((LogEC50 − X) · HillSlope))
5PL ("asymmetric sigmoidal, X is log(concentration)"):
    Y = Bottom + (Top − Bottom) / (1 + 10^((LogXb − X) · HillSlope))^S,
    LogEC50 = LogXb + log10(2^(1/S) − 1) / HillSlope
3PL: the 4PL with Bottom or Top fixed.

X is log10 of the concentration in mol/L (or the base unit of the compound). The curve
is reported with Top ≥ Bottom; a falling curve has a negative Hill slope (as in Prism).
Confidence intervals come from the covariance matrix (t distribution, n − p degrees of
freedom; EC50 intervals are asymmetric because they are computed on log10), or from a
bootstrap when asked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
# np.trapezoid is NumPy ≥ 2.0; np.trapz (same integral) is what NumPy 1.26 has
_trapezoid: Any = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def logistic(
    logx: Any, bottom: float, top: float, log_ec50: float, hill: float, s: float = 1.0
) -> Any:
    """The 4PL (``s = 1``) or 5PL curve; for 5PL ``log_ec50`` is Prism's LogXb."""
    z = np.clip((log_ec50 - np.asarray(logx, dtype=np.float64)) * hill, -300, 300)
    return bottom + (top - bottom) / (1.0 + 10.0**z) ** s


@dataclass
class CurveFit:
    model: str
    bottom: float
    top: float
    log_ec50: float  # midpoint: the relative EC50/IC50 (for 5PL: Prism's LogEC50)
    hill: float
    asymmetry: float = 1.0  # 5PL S
    log_xb: float | None = None  # 5PL inflection parameter
    log_ec50_ci: tuple[float, float] = (math.nan, math.nan)
    hill_ci: tuple[float, float] = (math.nan, math.nan)
    top_ci: tuple[float, float] = (math.nan, math.nan)
    bottom_ci: tuple[float, float] = (math.nan, math.nan)
    log_abs_ic50: float = math.nan  # where the curve crosses ``half``
    log_abs_ic50_ci: tuple[float, float] = (math.nan, math.nan)
    half: float = 50.0
    r2: float = math.nan
    rmse: float = math.nan
    ssr: float = math.nan
    n_points: int = 0
    n_concentrations: int = 0
    n_replicates: int = 0
    converged: bool = False  # also True when the search stopped at a flat fit (see _solve)
    flat_p: float = math.nan  # F test against a horizontal line
    log_range: tuple[float, float] = (math.nan, math.nan)
    ci_method: str = "covariance"
    flags: list[str] = field(default_factory=list)

    def predict(self, logx: Any) -> Any:
        center = self.log_xb if self.log_xb is not None else self.log_ec50
        return logistic(logx, self.bottom, self.top, center, self.hill, self.asymmetry)

    def inverse(self, y: float) -> float:
        """log10 concentration where the curve equals ``y`` (NaN if never reached)."""
        lo, hi = min(self.bottom, self.top), max(self.bottom, self.top)
        if not lo < y < hi or self.hill == 0:
            return math.nan
        center = self.log_xb if self.log_xb is not None else self.log_ec50
        ratio = ((self.top - self.bottom) / (y - self.bottom)) ** (1 / self.asymmetry) - 1
        if ratio <= 0:
            return math.nan
        return center - math.log10(ratio) / self.hill

    @property
    def ec50(self) -> float:
        return float(10**self.log_ec50)

    @property
    def abs_ic50(self) -> float:
        return float(10**self.log_abs_ic50) if math.isfinite(self.log_abs_ic50) else math.nan


# ----------------------------------------------------------------------- fitting

_NAMES = ("bottom", "top", "log_center", "hill", "s")
# Sigmoidal data converge in tens of evaluations per starting point. A flat response does not:
# with nothing to fit, the optimiser creeps along the Hill slope and LogEC50 (towards a step at
# the slope bound) for thousands of evaluations, seconds per curve.
_QUICK_NFEV = 60
_MAX_NFEV = 4000


def _unpack(
    theta: Array, free: list[str], fixed: dict[str, float]
) -> tuple[float, float, float, float, float]:
    values = dict(fixed)
    values.update(zip(free, (float(v) for v in theta), strict=True))
    return (
        values["bottom"],
        values["top"],
        values["log_center"],
        values["hill"],
        values.get("s", 1.0),
    )


def _solve(
    logx: Array,
    y: Array,
    free: list[str],
    fixed: dict[str, float],
    starts: list[dict[str, float]],
    bounds: dict[str, tuple[float, float]],
    robust: bool,
    flat: Callable[[Any, float], bool],
) -> tuple[Any, float, bool] | None:
    """Best fit over the starting points: (least_squares result, SSR, stopped as flat).

    Every start first runs with ``_QUICK_NFEV`` evaluations. Starts still running then are
    restarted from the beginning with ``_MAX_NFEV``, so every result is that of one
    uninterrupted run (``max_nfev`` only ends the search, it does not change its path):
    curves are fitted exactly as with one ``_MAX_NFEV`` run per start. The exception is a best
    fit that is already ``flat`` (not better than a horizontal line): further iterations
    only move parameters a flat response does not determine, so the search stops there."""
    from scipy.optimize import least_squares

    lower = np.array([bounds[n][0] for n in free])
    upper = np.array([bounds[n][1] for n in free])

    def residuals(theta: Array) -> Array:
        return np.asarray(logistic(logx, *_unpack(theta, free, fixed)) - y, dtype=np.float64)

    def run(start: dict[str, float], max_nfev: int) -> tuple[Any, float] | None:
        x0 = np.clip([start[n] for n in free], lower + 1e-9, upper - 1e-9)
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                loss="soft_l1" if robust else "linear",
                f_scale=max(float(np.std(y)) * 0.1, 1e-9) if robust else 1.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except (ValueError, FloatingPointError):
            return None
        return result, float(np.sum(result.fun**2))

    def best_of(runs: list[tuple[Any, float] | None]) -> tuple[Any, float] | None:
        best: tuple[Any, float] | None = None
        for item in runs:  # in start order, first of equal SSRs wins
            if item is not None and (best is None or item[1] < best[1]):
                best = item
        return best

    runs = [run(start, _QUICK_NFEV) for start in starts]
    unfinished = [k for k, item in enumerate(runs) if item is not None and item[0].status == 0]
    best = best_of(runs)
    if not unfinished or best is None:
        return None if best is None else (*best, False)
    if flat(*best):
        return (*best, True)
    for k in unfinished:
        runs[k] = run(starts[k], _MAX_NFEV)
    best = best_of(runs)
    return None if best is None else (*best, False)


def _flat_p(total: float, ssr: float, p: int, df: int) -> float:
    """p of the F test of a ``p``-parameter curve against a horizontal line."""
    from scipy import stats

    f_value = ((total - ssr) / (p - 1)) / (ssr / df) if ssr > 0 else math.inf
    return float(stats.f.sf(f_value, p - 1, df)) if math.isfinite(f_value) else 0.0


def _is_flat(flat_p: float, change: float, rmse: float) -> bool:
    """The flat-response rule: F test p > 0.05, or a change over the tested range of less
    than 3 × RMSE."""
    return flat_p > 0.05 or change < 3 * rmse


def fit_curve(
    conc: Any,
    response: Any,
    *,
    model: str = "4pl",
    fix_bottom: float | None = None,
    fix_top: float | None = None,
    robust: bool = False,
    half: float = 50.0,
    bootstrap: int = 0,
    seed: int = 0,
) -> CurveFit:
    """Fit concentrations (> 0, any base unit) and responses. ``model``: "4pl" or "5pl";
    a fixed bottom or top makes a 3PL. ``half``: response defining the absolute IC50
    (50 for % of control or % inhibition)."""
    from scipy import stats

    c = np.asarray(conc, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    keep = np.isfinite(c) & np.isfinite(r) & (c > 0)
    logx, y = np.log10(c[keep]), r[keep]
    levels, counts = np.unique(np.round(logx, 9), return_counts=True)
    fit = CurveFit(
        model=model if fix_bottom is None and fix_top is None else "3pl",
        bottom=math.nan,
        top=math.nan,
        log_ec50=math.nan,
        hill=math.nan,
        half=half,
        n_points=int(y.size),
        n_concentrations=int(levels.size),
        n_replicates=int(np.max(counts)) if counts.size else 0,
        log_range=(float(levels.min()), float(levels.max())) if levels.size else fit_nan2(),
    )
    n_free = 4 + (model == "5pl") - (fix_bottom is not None) - (fix_top is not None)
    if levels.size < 3 or y.size <= n_free:
        fit.flags.append("too few concentrations to fit")
        return fit

    span = max(float(np.ptp(y)), 1e-9)
    ylo, yhi = float(y.min()) - 5 * span, float(y.max()) + 5 * span
    xlo, xhi = float(levels.min()), float(levels.max())
    bounds = {
        "bottom": (ylo, yhi),
        "top": (ylo, yhi),
        "log_center": (xlo - 3, xhi + 3),
        "hill": (-10.0, 10.0),
        "s": (0.05, 20.0),
    }
    fixed: dict[str, float] = {}
    if fix_bottom is not None:
        fixed["bottom"] = float(fix_bottom)
    if fix_top is not None:
        fixed["top"] = float(fix_top)
    if model != "5pl":
        fixed["s"] = 1.0
    free = [n for n in _NAMES if n not in fixed]

    means = np.array([y[np.isclose(logx, v)].mean() for v in levels])
    low, high = float(means[0]), float(means[-1])
    starts = []
    for sign in (1.0, -1.0):
        for center in (float(np.median(levels)), xlo + (xhi - xlo) / 4, xhi - (xhi - xlo) / 4):
            # Top ≥ Bottom: a rising curve has a positive slope, a falling one negative
            top, bottom = max(low, high), min(low, high)
            starts.append(
                {"bottom": bottom, "top": top, "log_center": center, "hill": sign, "s": 1.0}
            )
    n, p = y.size, len(free)
    df = n - p
    total = float(np.sum((y - y.mean()) ** 2))

    def flat(result: Any, ssr: float) -> bool:
        if df <= 0 or total <= 0:
            return False
        params = _unpack(result.x, free, fixed)
        change = abs(float(logistic(xhi, *params)) - float(logistic(xlo, *params)))
        return _is_flat(_flat_p(total, ssr, p, df), change, math.sqrt(ssr / n))

    best = _solve(logx, y, free, fixed, starts, bounds, robust, flat)
    if best is None:
        fit.flags.append("curve not converged")
        return fit
    result, ssr, stopped_flat = best
    bottom, top, center, hill, s = _unpack(result.x, free, fixed)
    if top < bottom and "bottom" not in fixed and "top" not in fixed and s == 1.0:
        bottom, top, hill = top, bottom, -hill  # same curve, Prism's orientation
    fit.bottom, fit.top, fit.hill, fit.asymmetry = bottom, top, hill, s
    if model == "5pl":
        fit.log_xb = center
        fit.log_ec50 = center + math.log10(2 ** (1 / s) - 1) / hill if hill else math.nan
    else:
        fit.log_ec50 = center
    # a search stopped at a flat fit counts as converged: the answer (flat) is settled
    fit.converged = bool(result.status > 0) or stopped_flat
    if not fit.converged:
        fit.flags.append("curve not converged")
    fit.ssr = ssr
    fit.rmse = math.sqrt(ssr / n)
    fit.r2 = 1 - ssr / total if total > 0 else math.nan
    fit.log_abs_ic50 = fit.inverse(half)

    # covariance-based intervals
    if df > 0:
        t = float(stats.t.ppf(0.975, df))
        try:
            jac = np.asarray(result.jac, dtype=np.float64)
            cov = np.linalg.inv(jac.T @ jac) * (ssr / df)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            cov, se = None, None
            fit.flags.append("parameters not identifiable (singular covariance)")
        if se is not None and cov is not None:
            index = {name: k for k, name in enumerate(free)}
            # if the orientation was swapped, Top/Bottom/Hill errors swap with it
            swapped = fit.top != _unpack(result.x, free, fixed)[1]

            def ci(name: str, value: float) -> tuple[float, float]:
                if name not in index:
                    return (value, value)
                k = index[name]
                return (value - t * se[k], value + t * se[k])

            fit.hill_ci = ci("hill", hill)
            fit.top_ci = ci("bottom" if swapped else "top", top)
            fit.bottom_ci = ci("top" if swapped else "bottom", bottom)
            if model == "5pl":
                grad = _gradient(
                    lambda th: _log_ec50_5pl(_unpack(th, free, fixed)), np.asarray(result.x)
                )
                sd = math.sqrt(max(float(grad @ cov @ grad), 0.0))
                fit.log_ec50_ci = (fit.log_ec50 - t * sd, fit.log_ec50 + t * sd)
            else:
                fit.log_ec50_ci = ci("log_center", fit.log_ec50)
            if math.isfinite(fit.log_abs_ic50):
                grad = _gradient(
                    lambda th: _abs_from(_unpack(th, free, fixed), half), np.asarray(result.x)
                )
                if np.all(np.isfinite(grad)):
                    sd = math.sqrt(max(float(grad @ cov @ grad), 0.0))
                    fit.log_abs_ic50_ci = (fit.log_abs_ic50 - t * sd, fit.log_abs_ic50 + t * sd)

    if bootstrap:
        _bootstrap(fit, logx, y, model, fix_bottom, fix_top, robust, half, bootstrap, seed)

    # flat response: F test of the curve against a horizontal line
    if df > 0 and total > 0:
        fit.flat_p = _flat_p(total, ssr, p, df)
        change = abs(float(fit.predict(xhi)) - float(fit.predict(xlo)))
        if _is_flat(fit.flat_p, change, fit.rmse):
            fit.flags.append("flat response (curve not better than a horizontal line)")
    if math.isfinite(fit.log_ec50) and not xlo <= fit.log_ec50 <= xhi:
        fit.flags.append("EC50 extrapolated beyond the tested range")
    if not math.isfinite(fit.log_abs_ic50):
        fit.flags.append(f"absolute IC50 not reached (curve never crosses {half:g})")
    elif not xlo <= fit.log_abs_ic50 <= xhi:
        fit.flags.append("absolute IC50 extrapolated beyond the tested range")
    if levels.size < 5:
        fit.flags.append("fewer than 5 concentrations")
    return fit


def fit_nan2() -> tuple[float, float]:
    return (math.nan, math.nan)


def _log_ec50_5pl(params: tuple[float, float, float, float, float]) -> float:
    _, _, center, hill, s = params
    return center + math.log10(2 ** (1 / s) - 1) / hill if hill else math.nan


def _abs_from(params: tuple[float, float, float, float, float], half: float) -> float:
    bottom, top, center, hill, s = params
    if not min(bottom, top) < half < max(bottom, top) or hill == 0:
        return math.nan
    ratio = ((top - bottom) / (half - bottom)) ** (1 / s) - 1
    return center - math.log10(ratio) / hill if ratio > 0 else math.nan


def _gradient(function: Any, theta: Array, step: float = 1e-6) -> Array:
    grad = np.zeros_like(theta, dtype=np.float64)
    for k in range(theta.size):
        h = step * max(1.0, abs(float(theta[k])))
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        grad[k] = (function(up) - function(down)) / (2 * h)
    return grad


def _bootstrap(
    fit: CurveFit,
    logx: Array,
    y: Array,
    model: str,
    fix_bottom: float | None,
    fix_top: float | None,
    robust: bool,
    half: float,
    n: int,
    seed: int,
) -> None:
    """Percentile intervals from refits of points resampled within each concentration."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.isclose(logx, v)) for v in np.unique(np.round(logx, 9))]
    ec50s, hills, abs50 = [], [], []
    for _ in range(n):
        pick = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        sample = fit_curve(
            10 ** logx[pick],
            y[pick],
            model=model,
            fix_bottom=fix_bottom,
            fix_top=fix_top,
            robust=robust,
            half=half,
        )
        if sample.converged:
            ec50s.append(sample.log_ec50)
            hills.append(sample.hill)
            abs50.append(sample.log_abs_ic50)

    def interval(values: list[float]) -> tuple[float, float]:
        finite = np.asarray([v for v in values if math.isfinite(v)])
        if finite.size < max(10, n // 4):
            return (math.nan, math.nan)
        return (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))

    fit.log_ec50_ci, fit.hill_ci, fit.log_abs_ic50_ci = (
        interval(ec50s),
        interval(hills),
        interval(abs50),
    )
    fit.ci_method = f"bootstrap ({n})"


# ------------------------------------------------------------------ summaries


def concentration_means(conc: Any, response: Any) -> tuple[Array, Array, Array, Array]:
    """log10 concentration levels (> 0), mean, SD and count of the responses at each."""
    c = np.asarray(conc, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    keep = np.isfinite(c) & np.isfinite(r) & (c > 0)
    logx = np.round(np.log10(c[keep]), 9)
    levels = np.unique(logx)
    groups = [r[keep][logx == v] for v in levels]
    return (
        levels,
        np.array([g.mean() for g in groups]),
        np.array([g.std(ddof=1) if g.size > 1 else math.nan for g in groups]),
        np.array([g.size for g in groups], dtype=np.float64),
    )


def auc_normalized(levels: Array, means: Array) -> float:
    """Trapezoidal area under the mean responses on log10 concentration, divided by the
    tested log range: the average response over the range (same unit as the response)."""
    if levels.size < 2:
        return math.nan
    return float(_trapezoid(means, levels) / (levels[-1] - levels[0]))


def activity_area(
    fit: CurveFit, reference: float, decreasing: bool, full: float = 100.0, points: int = 200
) -> float:
    """Area between the fitted curve and the no-effect level over the tested range, in
    effect fraction × log10 units (effect = reference − Y for falling readouts such as %
    of control, Y − reference for rising ones such as % inhibition, divided by ``full``,
    the effect counted as 1), negatives as 0."""
    lo, hi = fit.log_range
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo and math.isfinite(fit.top)):
        return math.nan
    xs = np.linspace(lo, hi, points)
    values = fit.predict(xs)
    effect = (reference - values) if decreasing else (values - reference)
    return float(_trapezoid(np.clip(effect, 0, None) / full, xs))


# --------------------------------------------------------------- comparisons


def compare_ec50(conc_a: Any, response_a: Any, conc_b: Any, response_b: Any) -> dict[str, float]:
    """Extra-sum-of-squares F test: do two 4PL curves need different EC50s? Compares
    separate fits with a global fit sharing LogEC50 (other parameters free)."""
    from scipy import stats
    from scipy.optimize import least_squares

    fa = fit_curve(conc_a, response_a)
    fb = fit_curve(conc_b, response_b)
    out = {"delta_log_ec50": fb.log_ec50 - fa.log_ec50, "f": math.nan, "p": math.nan}
    if not (fa.converged and fb.converged):
        return out
    xa = np.log10(np.asarray(conc_a, dtype=float))
    xb = np.log10(np.asarray(conc_b, dtype=float))
    ya, yb = np.asarray(response_a, dtype=float), np.asarray(response_b, dtype=float)

    def residuals(th: Array) -> Array:
        ba, ta, ha, bb, tb, hb, c = th
        return np.concatenate([logistic(xa, ba, ta, c, ha) - ya, logistic(xb, bb, tb, c, hb) - yb])

    start = np.array(
        [fa.bottom, fa.top, fa.hill, fb.bottom, fb.top, fb.hill, (fa.log_ec50 + fb.log_ec50) / 2]
    )
    shared = least_squares(residuals, start, max_nfev=4000)
    ssr_shared = float(np.sum(shared.fun**2))
    ssr_separate = fa.ssr + fb.ssr
    df_separate = ya.size + yb.size - 8
    df_shared = ya.size + yb.size - 7
    if df_separate <= 0 or ssr_separate <= 0:
        return out
    f_value = ((ssr_shared - ssr_separate) / (df_shared - df_separate)) / (
        ssr_separate / df_separate
    )
    out["f"] = float(f_value)
    out["p"] = float(stats.f.sf(max(f_value, 0.0), df_shared - df_separate, df_separate))
    return out


# --------------------------------------------------------------- GR metrics


def gr_values(x: Any, x0: Any, x_ctrl: Any) -> Array:
    """Growth-rate inhibition (Hafner et al. 2016): GR = 2^(log2(x/x0) / log2(xctrl/x0)) − 1."""
    x, x0, xc = (np.asarray(v, dtype=np.float64) for v in (x, x0, x_ctrl))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(2 ** (np.log2(x / x0) / np.log2(xc / x0)) - 1, dtype=np.float64)


def gi_values(x: Any, x0: Any, x_ctrl: Any) -> Array:
    """NCI growth inhibition: 100 · (x − x0) / (xctrl − x0) (GI50 where it is 50)."""
    x, x0, xc = (np.asarray(v, dtype=np.float64) for v in (x, x0, x_ctrl))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(100 * (x - x0) / (xc - x0), dtype=np.float64)


def gr_metrics(conc: Any, gr: Any) -> dict[str, float]:
    """GR50, GRmax, GR_AOC, GEC50, GRinf, h_GR from GR values (fit with Top fixed at 1)."""
    fit = fit_curve(conc, gr, fix_top=1.0, half=0.5)
    levels, means, _, _ = concentration_means(conc, gr)
    aoc = (
        float(_trapezoid(1 - means, levels) / (levels[-1] - levels[0]))
        if levels.size >= 2
        else math.nan
    )
    gr_max = float(fit.predict(levels[-1])) if fit.converged else math.nan
    return {
        "GR50": fit.abs_ic50,
        "GRmax": gr_max,
        "GR_AOC": aoc,
        "GEC50": fit.ec50 if fit.converged else math.nan,
        "GRinf": fit.bottom if fit.converged else math.nan,
        "h_GR": fit.hill if fit.converged else math.nan,
        "GR_r2": fit.r2,
    }
