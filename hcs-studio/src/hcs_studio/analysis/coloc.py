"""Colocalization statistics for two channels within a mask.

Definitions follow Manders et al. (1993), Costes et al. (2004) and Li et al. (2004), as
implemented in Fiji's Coloc 2:

- Pearson's r and Spearman's rho of the two intensities, inside the mask only.
- M1 = Σ A(where B > 0) / Σ A, and M2 likewise (after background subtraction).
- tM1 = Σ A(where A > T_A and B > T_B) / Σ A(where A > T_A), with Costes thresholds T.
- Costes thresholds: on the orthogonal regression line B = m·A + c, the highest T_A at
  which the pixels below both thresholds no longer correlate (r ≤ 0): the first crossing
  when T is lowered from the maximum (Coloc 2 bisects, which can stop at a lower crossing
  when r(T) is not monotonic; the difference is within the noise floor).
- Costes randomization: Pearson's r of A scrambled in blocks of about the PSF size;
  p = (1 + #(r_random ≥ r)) / (1 + n). Small p means the correlation is unlikely by chance.
  (Coloc 2 reports the complementary value, 1 − p, as "Costes P-Value".)
- Li's ICQ = fraction of pixels where (A − Ā)(B − B̄) > 0, minus 0.5.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Below = Literal["and", "or"]


def _f64(values: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64).ravel()


def pearson(a: NDArray[Any], b: NDArray[Any]) -> float:
    a, b = _f64(a), _f64(b)
    if a.size < 3:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    denominator = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / denominator) if denominator > 0 else float("nan")


def spearman(a: NDArray[Any], b: NDArray[Any]) -> float:
    from scipy.stats import rankdata

    a, b = _f64(a), _f64(b)
    if a.size < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def li_icq(a: NDArray[Any], b: NDArray[Any]) -> float:
    a, b = _f64(a), _f64(b)
    if a.size == 0:
        return float("nan")
    product = (a - a.mean()) * (b - b.mean())
    return float((product > 0).mean() - 0.5)


def orthogonal_regression(a: NDArray[Any], b: NDArray[Any]) -> tuple[float, float]:
    """Slope and intercept of b = m·a + c minimising perpendicular distances (as Coloc 2)."""
    a, b = _f64(a), _f64(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a = ((a - mean_a) ** 2).mean()
    var_b = ((b - mean_b) ** 2).mean()
    cov = ((a - mean_a) * (b - mean_b)).mean()
    if cov == 0:
        return 0.0, float(mean_b)
    slope = (var_b - var_a + np.sqrt((var_b - var_a) ** 2 + 4 * cov * cov)) / (2 * cov)
    return float(slope), float(mean_b - slope * mean_a)


@dataclass(frozen=True)
class CostesResult:
    threshold_a: float
    threshold_b: float
    slope: float
    intercept: float
    r_below: float
    #: Empty when the thresholds are valid; otherwise why they should not be trusted.
    warning: str = ""


def costes_threshold(
    a: NDArray[Any], b: NDArray[Any], below: Below = "and", iterations: int = 60
) -> CostesResult:
    """Costes automatic thresholds along the orthogonal regression line."""
    a, b = _f64(a), _f64(b)
    slope, intercept = orthogonal_regression(a, b)
    if not slope > 0:
        return CostesResult(
            float(np.max(a)),
            float(np.max(b)),
            slope,
            intercept,
            float("nan"),
            "no positive correlation: Costes thresholds are not defined",
        )

    def r_below(t: float) -> float:
        tb = slope * t + intercept
        mask = (a < t) & (b < tb) if below == "and" else (a < t) | (b < tb)
        return pearson(a[mask], b[mask]) if mask.sum() >= 3 else float("nan")

    low, high = float(np.min(a)), float(np.max(a))
    r_high = r_below(high)
    if np.isnan(r_high) or r_high <= 0:
        return CostesResult(
            high,
            slope * high + intercept,
            slope,
            intercept,
            r_high,
            "the data below the maximum is already uncorrelated",
        )
    # Costes: lower T from the maximum until r(below) <= 0. r(T) is not monotonic near the
    # noise floor, so scan downward to bracket the *first* crossing, then refine by bisection.
    steps = np.linspace(high, low, 257)
    for upper, lower in itertools.pairwise(steps):
        r = r_below(float(lower))
        if np.isnan(r) or r <= 0:
            high, low = float(upper), float(lower)
            break
    else:
        r_low = r_below(low)
        return CostesResult(
            low,
            slope * low + intercept,
            slope,
            intercept,
            r_low,
            "no uncorrelated background found: thresholds at the minimum",
        )
    for _ in range(iterations):
        middle = (low + high) / 2
        r = r_below(middle)
        if np.isnan(r) or r <= 0:
            low = middle
        else:
            high = middle
        if high - low <= 1e-6 * max(1.0, abs(high)):
            break
    threshold = low
    return CostesResult(
        threshold, slope * threshold + intercept, slope, intercept, r_below(threshold)
    )


def manders(
    a: NDArray[Any], b: NDArray[Any], threshold_a: float = 0.0, threshold_b: float = 0.0
) -> tuple[float, float]:
    """M1, M2: fraction of A where B is above its threshold, and vice versa."""
    a, b = _f64(a), _f64(b)
    total_a, total_b = a.sum(), b.sum()
    m1 = a[b > threshold_b].sum() / total_a if total_a > 0 else float("nan")
    m2 = b[a > threshold_a].sum() / total_b if total_b > 0 else float("nan")
    return float(m1), float(m2)


def thresholded_manders(
    a: NDArray[Any], b: NDArray[Any], threshold_a: float, threshold_b: float
) -> tuple[float, float]:
    """tM1, tM2 as Coloc 2: restricted to pixels above their own channel's threshold."""
    a, b = _f64(a), _f64(b)
    above_a, above_b = a > threshold_a, b > threshold_b
    both = above_a & above_b
    total_a, total_b = a[above_a].sum(), b[above_b].sum()
    tm1 = a[both].sum() / total_a if total_a > 0 else float("nan")
    tm2 = b[both].sum() / total_b if total_b > 0 else float("nan")
    return float(tm1), float(tm2)


def overlap_coefficient(a: NDArray[Any], b: NDArray[Any]) -> float:
    a, b = _f64(a), _f64(b)
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def block_scramble(
    image: NDArray[Any], block: tuple[int, ...], rng: np.random.Generator
) -> NDArray[Any]:
    """Shuffle non-overlapping blocks of an image (the partial border blocks stay in place)."""
    out = np.array(image, copy=True)
    counts = [n // max(1, s) for n, s in zip(image.shape, block, strict=True)]
    if np.prod(counts) < 2:
        return out
    starts = np.stack(
        np.meshgrid(*[np.arange(c) * s for c, s in zip(counts, block, strict=True)], indexing="ij"),
        -1,
    ).reshape(-1, image.ndim)
    order = rng.permutation(len(starts))
    source = [
        image[tuple(slice(p, p + s) for p, s in zip(start, block, strict=True))].copy()
        for start in starts
    ]
    for target, index in zip(starts, order, strict=True):
        out[tuple(slice(p, p + s) for p, s in zip(target, block, strict=True))] = source[index]
    return out


def costes_p_value(
    a: NDArray[Any],
    b: NDArray[Any],
    mask: NDArray[np.bool_],
    block: tuple[int, ...],
    n: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(p, mean random r, SD random r) for the observed Pearson's r inside ``mask``."""
    observed = pearson(a[mask], b[mask])
    rng = np.random.default_rng(seed)
    random = np.array([pearson(block_scramble(a, block, rng)[mask], b[mask]) for _ in range(n)])
    random = random[~np.isnan(random)]
    if np.isnan(observed) or random.size == 0:
        return float("nan"), float("nan"), float("nan")
    p = (1 + int((random >= observed).sum())) / (1 + random.size)
    return float(p), float(random.mean()), float(random.std())


def channel_shift(a: NDArray[Any], b: NDArray[Any]) -> tuple[float, ...]:
    """Estimated (Z,) Y, X shift of b relative to a in pixels, by phase correlation."""
    from skimage.registration import phase_cross_correlation

    # Plain cross-correlation: phase normalisation whitens the spectrum, which makes smooth
    # fluorescence images noise-dominated and returns no shift.
    shift, _, _ = phase_cross_correlation(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        upsample_factor=10,
        normalization=None,
    )
    return tuple(float(-s) for s in shift)


@dataclass
class ColocResult:
    n_pixels: int
    pearson: float
    spearman: float
    icq: float
    m1: float
    m2: float
    tm1: float
    tm2: float
    overlap: float
    overlap_above_thresholds: float
    threshold_a: float
    threshold_b: float
    threshold_method: str
    costes_slope: float
    costes_r_below: float
    p_value: float = float("nan")
    random_r_mean: float = float("nan")
    random_r_sd: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = "; ".join(self.warnings)
        return data


def colocalize(
    a: NDArray[Any],
    b: NDArray[Any],
    mask: NDArray[np.bool_] | None = None,
    *,
    thresholds: tuple[float, float] | None = None,
    below: Below = "and",
    randomizations: int = 0,
    block: tuple[int, ...] | None = None,
    seed: int = 0,
) -> ColocResult:
    """All statistics for two background-subtracted channels inside ``mask``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    values_a, values_b = a[mask], b[mask]
    warnings: list[str] = []
    if values_a.size < 10:
        warnings.append("fewer than 10 pixels")
    if thresholds is None:
        costes = costes_threshold(values_a, values_b, below)
        threshold_a, threshold_b, method = (
            costes.threshold_a,
            costes.threshold_b,
            f"costes ({below})",
        )
        slope, r_below = costes.slope, costes.r_below
        if costes.warning:
            warnings.append(f"Costes: {costes.warning}")
    else:
        threshold_a, threshold_b = thresholds
        method, slope, r_below = "manual", float("nan"), float("nan")
    above = (values_a > threshold_a) & (values_b > threshold_b)
    m1, m2 = manders(values_a, values_b)
    tm1, tm2 = thresholded_manders(values_a, values_b, threshold_a, threshold_b)
    result = ColocResult(
        n_pixels=int(values_a.size),
        pearson=pearson(values_a, values_b),
        spearman=spearman(values_a, values_b),
        icq=li_icq(values_a, values_b),
        m1=m1,
        m2=m2,
        tm1=tm1,
        tm2=tm2,
        overlap=overlap_coefficient(values_a, values_b),
        overlap_above_thresholds=overlap_coefficient(values_a[above], values_b[above]),
        threshold_a=float(threshold_a),
        threshold_b=float(threshold_b),
        threshold_method=method,
        costes_slope=slope,
        costes_r_below=r_below,
        warnings=warnings,
    )
    if randomizations > 0:
        if block is None:
            raise ValueError("The randomization test needs a block size")
        result.p_value, result.random_r_mean, result.random_r_sd = costes_p_value(
            a, b, mask, block, randomizations, seed
        )
    return result
