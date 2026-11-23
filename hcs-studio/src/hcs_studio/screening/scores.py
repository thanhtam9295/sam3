"""Scores of screening wells (Phase 23a).

Per plate and readout (``plate_scores``):

- ``pct``: % activity against the on-plate controls — % inhibition (0 = vehicle median,
  100 = positive-control median) when the plate has positive controls, else % of the vehicle
  median (``activity="percent_of_control"``);
- ``z``: robust Z of the sample wells, (x − median of the sample wells) / (1.4826 · MAD) — most
  library compounds are inactive, so the sample wells are their own reference;
- ``b``: B-score (Brideau et al. 2003; median polish of the sample wells, residuals / (1.4826 ·
  MAD)) — row and column effects removed. Only for randomised libraries: the Phase 17 guard rail
  blocks it on dose–response or combination layouts (the score is then missing, with a note).

Plate quality: Z′ = 1 − 3(σ₊ + σ₋)/|μ₊ − μ₋| of the controls (Zhang et al. 1999); robust Z′
with medians and 1.4826 · MAD.

Replicates (``ssmd``): SSMD (strictly standardised mean difference, Zhang 2007) of a compound
tested on n replicate plates, from its differences to the plate's reference (the sample-well
median) d_j = x_j − m_j: the method-of-moments estimate d̄ / s, the UMVUE
√(2/(n−1)) · Γ((n−1)/2) / Γ((n−2)/2) · d̄ / s (unbiased for n ≥ 3), and the robust estimate
median(d) / (1.4826 · MAD(d)) (n ≥ 3). For a single replicate, SSMD* = (x − m) / (√2 · 1.4826 ·
MAD of the sample wells) (Zhang 2011).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826


def robust_sd(values: Any) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def z_prime(positive: Any, negative: Any, robust: bool = False) -> float:
    """Z′ of two control groups (robust: medians and MAD)."""
    p = np.asarray(positive, dtype=float)
    n = np.asarray(negative, dtype=float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if p.size < 2 or n.size < 2:
        return math.nan
    if robust:
        mp, mn, sp, sn = float(np.median(p)), float(np.median(n)), robust_sd(p), robust_sd(n)
    else:
        mp, mn = float(p.mean()), float(n.mean())
        sp, sn = float(p.std(ddof=1)), float(n.std(ddof=1))
    if mp == mn:
        return math.nan
    return 1 - 3 * (sp + sn) / abs(mp - mn)


def plate_scores(
    values: dict[str, float],
    wells: list[Any],
    rows: int,
    cols: int,
    activity: str = "auto",
    positional: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``well``, ``role``, ``raw``, ``pct``, ``z``, ``b`` for one plate and readout, and plate
    facts (Z′, which % activity, guard, notes)."""
    from hcs_studio.normalization.plate import (
        PositionalCorrectionBlocked,
        normalize_plate,
    )

    roles = {w.well: w.role for w in wells}
    has_positive = any(roles.get(w) == "positive_control" for w in values)
    method = activity
    if activity == "auto":
        method = "percent_inhibition" if has_positive else "percent_of_control"
    info: dict[str, Any] = {"activity": method, "notes": []}
    frame = normalize_plate(values, wells, rows, cols, "robust_z").frame.rename(
        columns={"normalized": "z"}
    )
    try:
        pct = normalize_plate(values, wells, rows, cols, method).frame.set_index("well")
        frame["pct"] = frame["well"].map(pct["normalized"])
    except ValueError as exc:
        frame["pct"] = math.nan
        info["notes"].append(f"% activity: {exc}")
    frame["b"] = math.nan
    info["b_score"] = False
    if positional:
        try:
            result = normalize_plate(values, wells, rows, cols, "b_score")
            frame["b"] = frame["well"].map(result.frame.set_index("well")["normalized"])
            info["b_score"] = True
            info["row_effects"] = result.details.get("row_effects")
            info["col_effects"] = result.details.get("col_effects")
        except PositionalCorrectionBlocked as exc:
            info["notes"].append(f"B-score blocked: {exc}")
        except ValueError as exc:
            info["notes"].append(f"B-score: {exc}")
    sample = frame["role"] == "sample"
    frame.loc[~sample, ["z", "b"]] = math.nan
    negative = frame.loc[frame["role"] == "negative_control", "raw"]
    positive = frame.loc[frame["role"] == "positive_control", "raw"]
    info["z_prime"] = z_prime(positive, negative)
    info["z_prime_robust"] = z_prime(positive, negative, robust=True)
    info["sample_median"] = float(frame.loc[sample, "raw"].median())
    info["sample_mad_sd"] = robust_sd(frame.loc[sample, "raw"])
    frame["ssmd_single"] = (frame["raw"] - info["sample_median"]) / (
        math.sqrt(2) * info["sample_mad_sd"]
    )
    frame.loc[~sample, "ssmd_single"] = math.nan
    return frame[["well", "role", "raw", "pct", "z", "b", "ssmd_single"]], info


def ssmd(differences: Any) -> dict[str, float]:
    """SSMD of one compound's replicate differences to their plates' reference."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    out = {"n": float(n), "ssmd_mm": math.nan, "ssmd_umvue": math.nan, "ssmd_robust": math.nan}
    if n < 2:
        return out
    sd = float(d.std(ddof=1))
    if sd > 0:
        out["ssmd_mm"] = float(d.mean()) / sd
        if n >= 3:
            factor = math.sqrt(2 / (n - 1)) * math.exp(
                math.lgamma((n - 1) / 2) - math.lgamma((n - 2) / 2)
            )
            out["ssmd_umvue"] = factor * float(d.mean()) / sd
    spread = robust_sd(d)
    if n >= 3 and spread > 0:
        out["ssmd_robust"] = float(np.median(d)) / spread
    return out
