"""Compound-related artifacts (Phase 18): autofluorescence and precipitation.

Both are judged per well against the plate's vehicle wells and along each compound's dose
series (single-agent wells of one compound and cell line):

- **autofluorescence** (per channel): the background outside cells (``c<n>_background``, well
  median over its good fields) is raised — robust Z against the vehicle wells above
  ``CompoundChecks.z`` — and rises with concentration along the series (Spearman ρ ≥
  ``min_rho`` over ≥ ``min_concentrations`` concentrations);
- **precipitation**: the same test on ``particle_fraction`` of the brightfield channel,
  limited to the upper half of the series' concentrations.

A raised well whose series does not rise with concentration is not flagged (one bad well is
the business of the field rules). Flags are reported per well and carried into pharmacology;
with ``CompoundChecks.exclude`` the wells fail (and are left out of fits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.plate import well_name
from hcs_studio.qc.decide import ROBUST
from hcs_studio.qc.rules import CompoundChecks

VEHICLE_ROLES = ("negative_control", "untreated")


def _series(layout: PlateLayout) -> pd.DataFrame:
    """Well, role, cell line and the single active compound + concentration of each well."""
    from hcs_studio.pharmacology.normalize import drug_treatments

    rows = []
    for key, annotation in layout.wells.items():
        active = drug_treatments(annotation)
        rows.append(
            {
                "well": well_name(*key),
                "role": annotation.role,
                "cell_line": annotation.cell_line,
                "compound": active[0][0] if len(active) == 1 else "",
                "conc": float(active[0][1]) if len(active) == 1 else np.nan,
                "vehicle": annotation.role in VEHICLE_ROLES,
            }
        )
    return pd.DataFrame(rows)


def _flag(
    values: pd.Series,
    series: pd.DataFrame,
    checks: CompoundChecks,
    *,
    upper_half: bool,
) -> tuple[set[str], dict[str, float]]:
    """Wells whose ``values`` are raised and rise with concentration (see module docs)."""
    frame = series.assign(value=series["well"].map(values)).dropna(subset=["value"])
    vehicle = frame[frame["vehicle"]]["value"]
    if len(vehicle) < 3:
        return set(), {}
    median = float(vehicle.median())
    spread = ROBUST * float((vehicle - median).abs().median())
    spread = max(spread, 0.01 * abs(median), 1e-9)
    frame = frame.assign(z=(frame["value"] - median) / spread)
    flagged: set[str] = set()
    for _, group in frame[frame["compound"] != ""].groupby(["compound", "cell_line"]):
        by_conc = group.groupby("conc")["z"].median().sort_index()
        if len(by_conc) < checks.min_concentrations:
            continue
        rho = spearmanr(by_conc.index, by_conc.to_numpy()).statistic
        if not np.isfinite(rho) or rho < checks.min_rho:
            continue
        raised = group[group["z"] > checks.z]
        if upper_half:
            cut = float(np.median(by_conc.index))
            raised = raised[raised["conc"] > cut]
        flagged |= set(raised["well"])
    return flagged, {str(w): float(z) for w, z in zip(frame["well"], frame["z"], strict=True)}


def compound_flags(
    fields: pd.DataFrame, layout: PlateLayout | None, checks: CompoundChecks
) -> pd.DataFrame:
    """One row per well with ``autofluorescence`` (channels, e.g. "c2"), ``precipitation``
    (bool) and the robust Z values used. Empty without a layout."""
    columns = ["well", "autofluorescence", "precipitation"]
    if layout is None or not layout.wells or fields.empty:
        return pd.DataFrame(columns=columns)
    good = fields[~fields["excluded"]] if "excluded" in fields else fields
    per_well = good.groupby("well").median(numeric_only=True)
    series = _series(layout)
    result = pd.DataFrame({"well": sorted(set(fields["well"]))})
    autofluorescent: dict[str, list[str]] = {}
    if checks.autofluorescence:
        for column in [c for c in per_well.columns if str(c).endswith("_background")]:
            channel = str(column).split("_")[0]
            flagged, z = _flag(per_well[column], series, checks, upper_half=False)
            result[f"{channel}_background_z"] = result["well"].map(z)
            for well in flagged:
                autofluorescent.setdefault(well, []).append(channel)
    result["autofluorescence"] = result["well"].map(
        lambda w: ",".join(sorted(autofluorescent.get(w, [])))
    )
    result["precipitation"] = False
    if checks.precipitation and "particle_fraction" in per_well:
        flagged, z = _flag(per_well["particle_fraction"], series, checks, upper_half=True)
        result["particle_z"] = result["well"].map(z)
        result["precipitation"] = result["well"].isin(flagged)
    return result
