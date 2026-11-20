"""Minimum significant ratio (MSR) of potencies (Eastwood et al. 2006, J Biomol Screen 11:253).

The MSR is the smallest ratio between the potencies of two compounds (or of one compound in
two runs) that is statistically significant; 1 is perfect reproducibility, and an assay is
commonly considered reproducible below 3.

- **One compound across runs** (e.g. a reference compound in every run): with s the SD of
  log10 potency between runs, the difference of two runs has SD √2·s, and
  ``MSR = 10^(2·√2·s)``.
- **Test–retest** (a set of compounds in two independent runs): ``d_i = log10 P1_i −
  log10 P2_i``, mean d̄ and SD s_d; ``MSR = 10^(2·s_d)``, mean ratio ``MR = 10^d̄`` and
  ratio limits of agreement ``LsA = 10^(d̄ ± 2·s_d)``.
- **Rolling MSR**: the single-compound MSR over the last ``window`` runs, to monitor an assay.

The multiplier 2 approximates the 95 % normal quantile (1.96), as in the published method.

**Verification status**: these formulas follow Eastwood et al. 2006; they have not yet been
checked against the NCATS Assay Guidance Manual text (the NCBI Bookshelf could not be read
from the build environment). The window length and acceptance limits are settings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

MULTIPLIER = 2.0
ACCEPTABLE_MSR = 3.0


def _log10(potencies: Sequence[float]) -> np.ndarray:
    values = np.asarray(potencies, dtype=float)
    values = values[np.isfinite(values)]
    if (values <= 0).any():
        raise ValueError("Potencies must be positive (EC50/IC50 in concentration units)")
    return np.log10(values)


def msr_from_runs(potencies: Sequence[float]) -> float:
    """MSR of one compound from its potencies in independent runs: 10^(2·√2·s)."""
    logs = _log10(potencies)
    if len(logs) < 2:
        raise ValueError("The MSR needs the compound in at least 2 runs")
    s = float(np.std(logs, ddof=1))
    return float(10 ** (MULTIPLIER * math.sqrt(2) * s))


@dataclass(frozen=True)
class RetestStatistics:
    n: int
    mean_difference: float  # d̄ (log10)
    sd_difference: float  # s_d (log10)
    msr: float
    mr: float
    lsa_low: float
    lsa_high: float

    def acceptable(self, msr_max: float = ACCEPTABLE_MSR) -> bool:
        return self.msr < msr_max and 1 / msr_max <= self.lsa_low and self.lsa_high <= msr_max


def retest_statistics(run1: Sequence[float], run2: Sequence[float]) -> RetestStatistics:
    """Compounds measured in two independent runs (paired by position)."""
    a = np.asarray(run1, dtype=float)
    b = np.asarray(run2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Both runs need the same compounds, in the same order")
    keep = np.isfinite(a) & np.isfinite(b)
    if (a[keep] <= 0).any() or (b[keep] <= 0).any():
        raise ValueError("Potencies must be positive")
    d = np.log10(a[keep]) - np.log10(b[keep])
    if len(d) < 3:
        raise ValueError("Test–retest needs at least 3 compounds in both runs")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    return RetestStatistics(
        n=len(d),
        mean_difference=mean,
        sd_difference=sd,
        msr=float(10 ** (MULTIPLIER * sd)),
        mr=float(10**mean),
        lsa_low=float(10 ** (mean - MULTIPLIER * sd)),
        lsa_high=float(10 ** (mean + MULTIPLIER * sd)),
    )


def rolling_msr(potencies: Sequence[float], window: int = 6) -> pd.Series:
    """MSR over the last ``window`` runs at each run (NaN until ``window`` runs exist)."""
    if window < 2:
        raise ValueError("The rolling window needs at least 2 runs")
    logs = pd.Series(np.log10(np.asarray(potencies, dtype=float)))
    sd = logs.rolling(window, min_periods=window).std(ddof=1)
    return 10 ** (MULTIPLIER * math.sqrt(2) * sd)
