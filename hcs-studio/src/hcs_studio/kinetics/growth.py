"""Growth curves of one well over time (Phase 22b).

``fit_growth(t_h, y)`` fits the readout of one well (cell count, confluence …) against time with
three models, on the **log** of the readout (counting noise and growth are multiplicative, so
every time point weighs the same):

- ``exponential``: ln y = ln N₀ + μ·t;
- ``lag_exponential``: ln y = ln N₀ until the lag λ, then ln N₀ + μ·(t − λ) (cells recovering
  from seeding or treatment);
- ``logistic``: y = K / (1 + (K/N₀ − 1)·e^(−μt)) (growth slowing towards a capacity K —
  confluence, contact inhibition).

The model with the lowest AICc (small-sample Akaike criterion; plain AIC when there are too
few time points for it, e.g. three acquisitions) is chosen; a more complex model must earn its
parameters, and a constant must do worse. μ is the specific growth rate (per hour, natural
log), the doubling time ln 2/μ (for the logistic model: the initial doubling time, while cells
are sparse). A well whose readout fits a constant as well is reported as model ``constant``,
not growing; one with μ < 0 as shrinking; neither has a doubling time. Values ≤ 0 cannot enter
a log fit and are left out (noted).

``curve_metrics`` adds model-free numbers: the AUC (trapezoid, readout·h), the mean fold change
(AUC of y/y₀ over the duration), the fold change at the end, and the times at which the readout
first reaches ``fold`` × its first value or a given ``level`` (linear interpolation between time
points; None when never reached).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODELS = ("exponential", "lag_exponential", "logistic")
_trapezoid: Any = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


@dataclass
class GrowthFit:
    model: str = "none"
    n0: float = math.nan
    rate_per_h: float = math.nan  # specific growth rate μ (ln units)
    doubling_h: float = math.nan
    lag_h: float = math.nan
    capacity: float = math.nan
    r2: float = math.nan  # on the log scale
    aicc: float = math.nan
    n_points: int = 0
    candidates: dict[str, float] = field(default_factory=dict)  # model → AICc
    flags: list[str] = field(default_factory=list)

    def predict(self, t_h: Any) -> NDArray[np.float64]:
        t = np.asarray(t_h, dtype=np.float64)
        if self.model == "exponential":
            return np.asarray(self.n0 * np.exp(self.rate_per_h * t), dtype=np.float64)
        if self.model == "lag_exponential":
            return np.asarray(
                self.n0 * np.exp(self.rate_per_h * np.clip(t - self.lag_h, 0, None)),
                dtype=np.float64,
            )
        if self.model == "logistic":
            return _logistic(t, self.n0, self.rate_per_h, self.capacity)
        if self.model == "constant":
            return np.full(t.shape, self.n0)
        return np.full(t.shape, math.nan)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["flags"] = "; ".join(self.flags)
        out.pop("candidates")
        return out


def _logistic(t: NDArray[Any], n0: float, rate: float, capacity: float) -> NDArray[np.float64]:
    with np.errstate(over="ignore"):
        value = capacity / (1 + (capacity / n0 - 1) * np.exp(-rate * t))
    return np.asarray(value, dtype=np.float64)


def _aicc(ssr: float, n: int, k: int, corrected: bool = True) -> float:
    """AICc (or plain AIC when ``corrected`` is False)."""
    ssr = max(ssr, 1e-12 * n)
    aic = n * math.log(ssr / n) + 2 * k
    if not corrected:
        return aic
    return aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf


def _fit_exponential(t: NDArray[Any], logy: NDArray[Any]) -> tuple[dict[str, float], float]:
    slope, intercept = np.polyfit(t, logy, 1)
    ssr = float(np.sum((logy - (intercept + slope * t)) ** 2))
    return {"n0": math.exp(intercept), "rate": float(slope)}, ssr


def _fit_lag(t: NDArray[Any], logy: NDArray[Any]) -> tuple[dict[str, float], float] | None:
    """Exact profile over the lag: for a given λ the model is linear in (ln N₀, μ); the best λ
    is searched on a fine grid between the first and the third-last time point."""
    if len(t) < 5:
        return None
    best: tuple[float, float, float, float] | None = None
    for lag in np.linspace(t[0], t[-3], 200):
        x = np.clip(t - lag, 0, None)
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, logy, rcond=None)
        if coef[1] <= 0:
            continue
        ssr = float(np.sum((logy - design @ coef) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, float(coef[0]), float(coef[1]), float(lag))
    if best is None:
        return None
    ssr, intercept, rate, lag = best
    return {"n0": math.exp(intercept), "rate": rate, "lag": lag}, ssr


def _fit_logistic(t: NDArray[Any], logy: NDArray[Any]) -> tuple[dict[str, float], float] | None:
    from scipy.optimize import least_squares

    if len(t) < 5:
        return None
    y = np.exp(logy)
    top = float(y.max())
    slope = max(float(np.polyfit(t, logy, 1)[0]), 1e-3)

    def residuals(p: NDArray[Any]) -> NDArray[Any]:
        n0, rate, capacity = math.exp(p[0]), p[1], math.exp(p[2])
        fitted = np.log(np.clip(_logistic(t, n0, rate, capacity), 1e-300, None))
        return np.asarray(fitted - logy, dtype=np.float64)

    best = None
    for capacity in (1.1 * top, 2 * top, 5 * top):
        start = np.array([logy[0], 2 * slope, math.log(capacity)])
        lower = [logy.min() - 5, 1e-6, math.log(top * 0.5)]
        upper = [logy.max() + 5, 10.0, math.log(top * 1e4)]
        try:
            result = least_squares(residuals, np.clip(start, lower, upper), bounds=(lower, upper))
        except (ValueError, FloatingPointError):
            continue
        ssr = float(np.sum(result.fun**2))
        if best is None or ssr < best[1]:
            best = (result.x, ssr)
    if best is None:
        return None
    p, ssr = best
    return {"n0": math.exp(p[0]), "rate": float(p[1]), "capacity": math.exp(p[2])}, ssr


def fit_growth(
    t_h: Sequence[float] | NDArray[Any],
    y: Sequence[float] | NDArray[Any],
    models: Sequence[str] = MODELS,
    window_h: tuple[float, float] | None = None,
) -> GrowthFit:
    """The best of ``models`` (AICc) for one well; ``window_h`` limits the fit to a time range
    (e.g. the exponential phase)."""
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown growth model(s) {unknown} (use {', '.join(MODELS)})")
    t = np.asarray(t_h, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(v)
    if window_h is not None:
        keep &= (t >= window_h[0]) & (t <= window_h[1])
    out = GrowthFit()
    nonpositive = int(np.sum(keep & (v <= 0)))
    if nonpositive:
        out.flags.append(f"{nonpositive} time point(s) ≤ 0 left out")
    keep &= v > 0
    order = np.argsort(t[keep])
    t, logy = t[keep][order], np.log(v[keep][order])
    out.n_points = int(t.size)
    if t.size < 3:
        out.flags.append("too few time points")
        return out
    t0 = t - t[0]  # times from the first point fitted; the parameters refer to it
    ssr_const = float(np.sum((logy - logy.mean()) ** 2))
    fits: dict[str, tuple[dict[str, float], float, int]] = {}
    for model in models:
        if model == "exponential":
            params, ssr = _fit_exponential(t0, logy)
            fits[model] = (params, ssr, 2)
        elif model == "lag_exponential":
            result = _fit_lag(t0, logy)
            if result is not None:
                fits[model] = (result[0], result[1], 3)
        else:
            result = _fit_logistic(t0, logy)
            if result is not None:
                fits[model] = (result[0], result[1], 3)
    # AICc needs n > k + 1 for every candidate; with fewer time points plain AIC is used
    corrected = all(t.size - k - 1 > 0 for _, _, k in fits.values())
    if not corrected:
        out.flags.append("few time points: AIC instead of AICc")
    scores = {m: _aicc(ssr, t.size, k, corrected) for m, (_, ssr, k) in fits.items()}
    out.candidates = scores
    constant = _aicc(ssr_const, t.size, 1, corrected)
    if not scores:
        out.flags.append("no model could be fitted")
        return out
    model = min(scores, key=lambda m: scores[m])
    params, ssr, _k = fits[model]
    out.model = model
    out.aicc = scores[model]
    out.n0 = params["n0"]
    out.rate_per_h = params["rate"]
    out.lag_h = params.get("lag", 0.0 if model != "logistic" else math.nan)
    out.capacity = params.get("capacity", math.nan)
    out.r2 = 1 - ssr / ssr_const if ssr_const > 0 else math.nan
    if constant <= scores[model]:  # a flat line is as good: no growth to report
        out.model, out.n0, out.rate_per_h = "constant", math.exp(float(logy.mean())), 0.0
        out.lag_h = out.capacity = math.nan
        out.aicc, out.r2 = constant, 0.0
        out.flags.append("not growing")
        return out
    if out.rate_per_h <= 0:
        out.flags += ["not growing", "shrinking"]
        return out
    out.doubling_h = math.log(2) / out.rate_per_h
    if t[-1] - t[0] < 0.5 * out.doubling_h:
        out.flags.append("observed for less than half a doubling time")
    return out


def _first_crossing(t: NDArray[Any], y: NDArray[Any], level: float) -> float | None:
    above = np.nonzero(y >= level)[0]
    if above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(t[0])
    t1, t2, y1, y2 = t[i - 1], t[i], y[i - 1], y[i]
    return float(t1 + (level - y1) * (t2 - t1) / (y2 - y1)) if y2 != y1 else float(t2)


def curve_metrics(
    t_h: Sequence[float] | NDArray[Any],
    y: Sequence[float] | NDArray[Any],
    fold: float = 2.0,
    level: float | None = None,
) -> dict[str, float | None]:
    """AUC, mean and final fold change, time to ``fold`` × the first value and to ``level``."""
    t = np.asarray(t_h, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(v)
    order = np.argsort(t[keep])
    t, v = t[keep][order], v[keep][order]
    out: dict[str, float | None] = {
        "auc": math.nan,
        "mean_fold": math.nan,
        "fold_change": math.nan,
        "time_to_fold_h": None,
        "time_to_level_h": None,
    }
    if t.size < 2:
        return out
    duration = float(t[-1] - t[0])
    out["auc"] = float(_trapezoid(v, t))
    if v[0] > 0:
        out["mean_fold"] = float(_trapezoid(v / v[0], t) / duration) if duration > 0 else math.nan
        out["fold_change"] = float(v[-1] / v[0])
        crossing = _first_crossing(t, v, fold * float(v[0]))
        out["time_to_fold_h"] = crossing - float(t[0]) if crossing is not None else None
    if level is not None:
        crossing = _first_crossing(t, v, level)
        out["time_to_level_h"] = crossing - float(t[0]) if crossing is not None else None
    return out
