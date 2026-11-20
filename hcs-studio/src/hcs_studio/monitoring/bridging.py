"""Instrument bridging: the same plates or reference compounds measured on two instruments.

Per readout, the per-well values of plate pairs (the same barcode imaged on instrument A and
on instrument B) are compared with correlation and Bland–Altman statistics. Per compound and
cell line, the potency ratio A/B gets a 95 % confidence interval from the two fits' own
intervals (on the log scale, the standard errors add in quadrature) and is judged against an
MSR: within when the ratio lies inside [1/MSR, MSR]. Across compounds, the test–retest
statistics (MR and limits of agreement) summarize the systematic shift between instruments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from hcs_studio.monitoring.msr import ACCEPTABLE_MSR, RetestStatistics, retest_statistics
from hcs_studio.normalization.concordance import bland_altman

Z95 = 1.959964


def log_se_from_ci(low: float, high: float) -> float:
    """SE of log10 potency from a 95 % CI (NaN when the CI is missing)."""
    if not (low and high and low > 0 and high > 0 and math.isfinite(low) and math.isfinite(high)):
        return math.nan
    return (math.log10(high) - math.log10(low)) / (2 * Z95)


def potency_ratios(fits_a: pd.DataFrame, fits_b: pd.DataFrame, msr: float) -> pd.DataFrame:
    """Join fits (compound, cell_line, readout, ec50_m, ec50_ci_low_m, ec50_ci_high_m) of the
    two instruments; ratio A/B with its 95 % CI and whether it lies within the MSR."""
    keys = ["compound", "cell_line", "readout"]
    columns = [*keys, "ec50_m", "ec50_ci_low_m", "ec50_ci_high_m"]
    merged = fits_a[columns].merge(fits_b[columns], on=keys, suffixes=("_a", "_b"))
    merged = merged[(merged["ec50_m_a"] > 0) & (merged["ec50_m_b"] > 0)].copy()
    rows = []
    for r in merged.to_dict("records"):
        log_ratio = math.log10(r["ec50_m_a"]) - math.log10(r["ec50_m_b"])
        se = math.hypot(
            log_se_from_ci(r["ec50_ci_low_m_a"], r["ec50_ci_high_m_a"]),
            log_se_from_ci(r["ec50_ci_low_m_b"], r["ec50_ci_high_m_b"]),
        )
        ratio = 10**log_ratio
        rows.append(
            {
                "compound": r["compound"],
                "cell_line": r["cell_line"],
                "readout": r["readout"],
                "ec50_a_m": r["ec50_m_a"],
                "ec50_b_m": r["ec50_m_b"],
                "ratio": ratio,
                "ratio_ci_low": 10 ** (log_ratio - Z95 * se) if math.isfinite(se) else math.nan,
                "ratio_ci_high": 10 ** (log_ratio + Z95 * se) if math.isfinite(se) else math.nan,
                "within_msr": 1 / msr <= ratio <= msr,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "compound", "cell_line", "readout", "ec50_a_m", "ec50_b_m", "ratio",
            "ratio_ci_low", "ratio_ci_high", "within_msr",
        ],
    )  # fmt: skip


def readout_agreement(pairs: list[tuple[str, dict[str, float], dict[str, float]]]) -> pd.DataFrame:
    """Per plate pair (label, values on A, values on B): n, Pearson r, Bland–Altman bias and
    limits (in the readout's unit), and the bias as % of the mean."""
    rows = []
    for label, a, b in pairs:
        wells = [w for w in a if w in b and math.isfinite(a[w]) and math.isfinite(b[w])]
        if len(wells) < 3:
            continue
        x = np.array([a[w] for w in wells])
        y = np.array([b[w] for w in wells])
        bias, _sd, low, high = bland_altman(x, y)
        mean = float(np.mean(np.concatenate([x, y])))
        rows.append(
            {
                "plate": label,
                "n": len(wells),
                "pearson": float(stats.pearsonr(x, y).statistic)
                if x.std() and y.std()
                else math.nan,
                "bias": bias,
                "loa_low": low,
                "loa_high": high,
                "bias_pct": 100 * bias / mean if mean else math.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["plate", "n", "pearson", "bias", "loa_low", "loa_high", "bias_pct"]
    )


@dataclass
class Bridging:
    instrument_a: str
    instrument_b: str
    readout: str
    msr: float
    agreement: pd.DataFrame
    ratios: pd.DataFrame
    retest: RetestStatistics | None
    notes: list[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        """Every potency ratio within the MSR and, with ≥ 3 compounds, the limits of agreement
        within [1/MSR, MSR]."""
        if self.ratios.empty:
            return False
        if not bool(self.ratios["within_msr"].all()):
            return False
        if self.retest is not None:
            return 1 / self.msr <= self.retest.lsa_low and self.retest.lsa_high <= self.msr
        return True


def bridge(
    instrument_a: str,
    instrument_b: str,
    readout: str,
    pairs: list[tuple[str, dict[str, float], dict[str, float]]],
    fits_a: pd.DataFrame,
    fits_b: pd.DataFrame,
    msr: float = ACCEPTABLE_MSR,
) -> Bridging:
    if not msr > 1:
        raise ValueError("The MSR must be above 1")
    agreement = readout_agreement(pairs)
    ratios = (
        potency_ratios(fits_a, fits_b, msr)
        if not fits_a.empty and not fits_b.empty
        else pd.DataFrame(columns=["compound", "cell_line", "readout", "ratio", "within_msr"])
    )
    notes = []
    retest = None
    if len(ratios) >= 3:
        retest = retest_statistics(ratios["ec50_a_m"].tolist(), ratios["ec50_b_m"].tolist())
    elif not ratios.empty:
        notes.append("Fewer than 3 compounds on both instruments: no limits of agreement")
    if agreement.empty:
        notes.append("No plate was imaged on both instruments with this readout")
    return Bridging(instrument_a, instrument_b, readout, msr, agreement, ratios, retest, notes)
