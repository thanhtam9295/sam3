"""Agreement between replicate plates.

For two plates with the same layout: Pearson and Spearman correlation of the sample wells,
Bland–Altman bias and 95 % limits of agreement (mean difference ± 1.96 SD), the CV of each
condition across both plates, and the fold difference of each IC50 between the plates. A
pair is flagged discordant when the correlation is below ``r_min`` or any IC50 differs by
more than ``fold_max``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Concordance:
    n: int
    pearson: float
    spearman: float
    bias: float
    sd_difference: float
    loa_low: float
    loa_high: float
    conditions: pd.DataFrame  # condition, n, mean, sd, cv_pct
    ic50: pd.DataFrame  # compound, cell_line, readout, ec50_a, ec50_b, fold
    flags: list[str] = field(default_factory=list)

    @property
    def discordant(self) -> bool:
        return bool(self.flags)

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "bias": self.bias,
            "loa_low": self.loa_low,
            "loa_high": self.loa_high,
            "max_ic50_fold": float(self.ic50["fold"].max()) if not self.ic50.empty else math.nan,
            "discordant": self.discordant,
        }


def bland_altman(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Bias (mean of a − b), SD of the differences and the 95 % limits of agreement."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else math.nan
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def replicate_concordance(
    a: Mapping[str, float],
    b: Mapping[str, float],
    conditions: Mapping[str, str],
    *,
    fits_a: pd.DataFrame | None = None,
    fits_b: pd.DataFrame | None = None,
    r_min: float = 0.8,
    fold_max: float = 3.0,
) -> Concordance:
    """``a``/``b``: values by well of the two plates; ``conditions``: well → condition label
    (only these wells are compared, usually the samples); ``fits_*``: dose–response fits
    (compound, cell_line, readout, ec50_m) of each plate."""
    wells = [w for w in conditions if w in a and w in b and np.isfinite(a[w]) and np.isfinite(b[w])]
    if len(wells) < 3:
        raise ValueError("Fewer than 3 wells have values on both plates")
    x = np.array([a[w] for w in wells], float)
    y = np.array([b[w] for w in wells], float)
    pearson = float(stats.pearsonr(x, y).statistic) if x.std() and y.std() else math.nan
    spearman = float(stats.spearmanr(x, y).statistic) if x.std() and y.std() else math.nan
    bias, sd, low, high = bland_altman(x, y)
    long = pd.DataFrame(
        {
            "condition": [conditions[w] for w in wells] * 2,
            "value": np.concatenate([x, y]),
        }
    )
    grouped = long.groupby("condition")["value"]
    cond = pd.DataFrame(
        {"n": grouped.size(), "mean": grouped.mean(), "sd": grouped.std(ddof=1)}
    ).reset_index()
    cond["cv_pct"] = 100 * cond["sd"] / cond["mean"].abs()
    ic50 = pd.DataFrame(columns=["compound", "cell_line", "readout", "ec50_a", "ec50_b", "fold"])
    if fits_a is not None and fits_b is not None and not fits_a.empty and not fits_b.empty:
        keys = ["compound", "cell_line", "readout"]
        merged = fits_a[[*keys, "ec50_m"]].merge(
            fits_b[[*keys, "ec50_m"]], on=keys, suffixes=("_a", "_b")
        )
        merged = merged.dropna()
        merged = merged[(merged["ec50_m_a"] > 0) & (merged["ec50_m_b"] > 0)]
        ic50 = merged.rename(columns={"ec50_m_a": "ec50_a", "ec50_m_b": "ec50_b"})
        ic50["fold"] = np.maximum(ic50["ec50_a"], ic50["ec50_b"]) / np.minimum(
            ic50["ec50_a"], ic50["ec50_b"]
        )
        ic50 = ic50.reset_index(drop=True)
    flags = []
    if math.isfinite(pearson) and pearson < r_min:
        flags.append(f"Pearson r {pearson:.2f} < {r_min:g}")
    far = ic50[ic50["fold"] > fold_max] if not ic50.empty else ic50
    for row in far.itertuples():
        flags.append(f"{row.compound} in {row.cell_line}: IC50 differs {row.fold:.1f}-fold")
    return Concordance(len(wells), pearson, spearman, bias, sd, low, high, cond, ic50, flags)
