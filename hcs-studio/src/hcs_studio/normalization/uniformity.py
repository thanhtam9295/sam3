"""Spatial correction maps from uniformity plates (vehicle-only plates).

A uniformity plate has the same treatment in every well, so any pattern across the plate is
positional (edge evaporation, temperature gradients, dispenser or imager artifacts). A smooth
2D surface — a polynomial in the row and column position plus a step for each of the
outermost well rings (edge evaporation is a step, not a smooth trend), fitted robustly
(points beyond 3 MAD are dropped and the fit repeated) — describes it; the map is that
surface divided by its median (factor 1 = typical well). Later plates of the same
instrument, plate type and assay are corrected by dividing each well by its factor. Unlike
B-score this needs no assumption about where compounds sit, so it also suits dose–response
and combination plates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.normalization.plate import mad

MIN_FACTOR = 0.2  # refuse maps that would multiply a well by more than 5


@dataclass
class UniformityMap:
    factors: NDArray[np.float64]  # rows × cols, median 1
    degree: int
    stats: dict[str, float] = field(default_factory=dict)
    edge_rings: int = 1

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.factors.shape
        return int(rows), int(cols)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": self.factors.tolist(),
            "degree": self.degree,
            "edge_rings": self.edge_rings,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniformityMap:
        return cls(
            np.array(data["factors"], dtype=float),
            int(data["degree"]),
            dict(data["stats"]),
            int(data.get("edge_rings", 1)),
        )


def _design(
    rows: int, cols: int, degree: int, edge_rings: int = 0
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    r, c = np.mgrid[0:rows, 0:cols]
    y = (r - (rows - 1) / 2) / max((rows - 1) / 2, 1)
    x = (c - (cols - 1) / 2) / max((cols - 1) / 2, 1)
    powers = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    columns = [(x**i * y**j).ravel() for i, j in powers]
    ring = np.minimum(np.minimum(r, rows - 1 - r), np.minimum(c, cols - 1 - c))
    for k in range(edge_rings):  # a step for each outer ring (k = 0: the edge wells)
        columns.append((ring == k).astype(float).ravel())
        powers.append((-1, k))
    return np.column_stack(columns), powers


def fit_surface(
    grid: NDArray[np.float64], degree: int = 2, robust: bool = True, edge_rings: int = 1
) -> NDArray[np.float64]:
    """Robust least-squares polynomial surface through the finite wells of ``grid``."""
    if not 0 <= degree <= 6:
        raise ValueError("Surface degree must be 0–6")
    if not 0 <= edge_rings <= 3:
        raise ValueError("Edge rings must be 0–3")
    rows, cols = grid.shape
    design, powers = _design(rows, cols, degree, edge_rings)
    values = grid.ravel()
    keep = np.isfinite(values)
    if keep.sum() < 2 * len(powers):
        raise ValueError(
            f"A degree-{degree} surface needs at least {2 * len(powers)} wells with values"
        )
    coef = np.zeros(len(powers))
    for _ in range(6 if robust else 1):
        coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coef
        spread = mad(residual[keep])
        # never call a well an outlier for a deviation below 0.1 % of the typical value
        floor = 1e-3 * float(np.median(np.abs(values[keep])))
        if not robust or not spread > 0:
            break
        new_keep = np.isfinite(values) & (np.abs(residual) <= max(3 * spread, floor))
        if np.array_equal(new_keep, keep) or new_keep.sum() < max(2 * len(powers), keep.sum() // 2):
            break  # converged, or the outlier rule would drop too much of the plate
        keep = new_keep
    return np.asarray((design @ coef).reshape(rows, cols), dtype=float)


def edge_mask(rows: int, cols: int, width: int = 1) -> NDArray[np.bool_]:
    r, c = np.mgrid[0:rows, 0:cols]
    return np.asarray((r < width) | (r >= rows - width) | (c < width) | (c >= cols - width))


def plate_stats(grid: NDArray[np.float64]) -> dict[str, float]:
    """Edge effect (% edge vs interior median), CV and row/column trends (% of median per
    plate span) of a plate's values."""
    rows, cols = grid.shape
    finite = np.isfinite(grid)
    edge = edge_mask(rows, cols) & finite
    inner = ~edge_mask(rows, cols) & finite
    median = float(np.nanmedian(grid))
    values = grid[finite]
    out = {
        "median": median,
        "cv_pct": float(100 * np.std(values, ddof=1) / np.mean(values))
        if values.size > 1
        else math.nan,
        "edge_effect_pct": float(100 * (np.median(grid[edge]) / np.median(grid[inner]) - 1))
        if edge.any() and inner.any()
        else math.nan,
    }
    with np.errstate(all="ignore"):
        row_means = np.nanmedian(grid, axis=1)
        col_means = np.nanmedian(grid, axis=0)
    for name, series in (("row_trend_pct", row_means), ("col_trend_pct", col_means)):
        ok = np.isfinite(series)
        if ok.sum() >= 2:
            slope = np.polyfit(np.arange(len(series))[ok], series[ok], 1)[0]
            out[name] = float(100 * slope * (len(series) - 1) / median) if median else math.nan
        else:
            out[name] = math.nan
    return out


def build_map(grid: NDArray[np.float64], degree: int = 2, edge_rings: int = 1) -> UniformityMap:
    """A correction map from a uniformity plate (``grid``: rows × cols, NaN = not used)."""
    surface = fit_surface(grid, degree, edge_rings=edge_rings)
    reference = float(np.median(surface[np.isfinite(grid)]))
    if not reference > 0:
        raise ValueError("The uniformity plate's typical value is not positive")
    factors = surface / reference
    if np.nanmin(factors) < MIN_FACTOR:
        raise ValueError(
            "The fitted surface drops below 20 % of the median somewhere on the plate; check "
            "the uniformity plate (empty or failed wells) or lower the degree"
        )
    before = plate_stats(grid)
    after = plate_stats(grid / factors)
    stats = {f"before_{k}": v for k, v in before.items()} | {
        f"after_{k}": v for k, v in after.items()
    }
    stats["factor_min"] = float(np.min(factors))
    stats["factor_max"] = float(np.max(factors))
    return UniformityMap(factors, degree, stats, edge_rings)


def apply_map(grid: NDArray[np.float64], uniformity: UniformityMap) -> NDArray[np.float64]:
    if grid.shape != uniformity.shape:
        raise ValueError(
            f"The map is for {uniformity.shape[0]}×{uniformity.shape[1]} plates, the plate is "
            f"{grid.shape[0]}×{grid.shape[1]}"
        )
    return np.asarray(grid / uniformity.factors, dtype=float)
