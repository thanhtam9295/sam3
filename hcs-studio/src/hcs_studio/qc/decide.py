"""Field flags and well / plate decisions from QC metrics and rules (Phase 18).

Input: one row per field (columns ``well``, ``row``, ``col``, ``field`` and metrics). Output
(:class:`QcResult`): the field table with ``flag_<name>`` columns, ``flags``, ``severity``,
``excluded`` and ``reasons``; one row per well with ``failed`` and ``reasons``; a plate
summary with ``passed`` and ``reasons``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hcs_studio.model.plate import well_name
from hcs_studio.qc.rules import FieldRule, QcRules

if TYPE_CHECKING:
    from hcs_studio.model.annotation import PlateLayout

ROBUST = 1.4826


@dataclass
class QcResult:
    fields: pd.DataFrame
    wells: pd.DataFrame
    plate: dict[str, Any] = field(default_factory=dict)

    @property
    def excluded_fields(self) -> set[tuple[str, int]]:
        rows = self.fields[self.fields["excluded"]]
        return {(str(w), int(f)) for w, f in zip(rows["well"], rows["field"], strict=True)}

    @property
    def failed_wells(self) -> list[str]:
        return [str(w) for w in self.wells.loc[self.wells["failed"], "well"]]


def n_channels(fields: pd.DataFrame) -> int:
    numbers = [int(m.group(1)) for c in fields.columns if (m := re.match(r"c(\d+)_", str(c)))]
    return max(numbers, default=0)


def robust_z(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values.astype(float))]
    if finite.empty:
        return pd.Series(np.nan, index=values.index)
    median = float(finite.median())
    mad = float((finite - median).abs().median()) * ROBUST
    scale = mad if mad > 0 else max(abs(median) * 1e-3, 1e-9)
    return (values.astype(float) - median) / scale


def add_focus_offsets(fields: pd.DataFrame) -> pd.DataFrame:
    """``c<n>_best_z_offset``: |best plane − plate trend| in planes, the trend being a robust
    plane through the best-focus positions over the plate (row, column)."""
    out = fields.copy()
    for column in [c for c in fields.columns if str(c).endswith("_best_z")]:
        z = out[column].astype(float)
        target = column + "_offset"
        good = np.isfinite(z.to_numpy())
        if good.sum() < 4:
            out[target] = np.nan
            continue
        design = np.column_stack([np.ones(len(out)), out["row"], out["col"]]).astype(float)
        keep = good.copy()
        coefficients = np.zeros(3)
        for _ in range(5):
            coefficients, *_ = np.linalg.lstsq(design[keep], z.to_numpy()[keep], rcond=None)
            residual = np.abs(z.to_numpy() - design @ coefficients)
            new = good & (residual <= max(1.0, 3 * ROBUST * np.median(residual[keep])))
            if np.array_equal(new, keep) or new.sum() < 4:
                break
            keep = new
        out[target] = np.abs(z.to_numpy() - design @ coefficients)
    return out


def _evaluate(rule: FieldRule, values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(fired, severity ≥ 1 where fired) of one rule on one column."""
    v = values.astype(float)
    fired = pd.Series(False, index=v.index)
    severity = pd.Series(0.0, index=v.index)

    def mark(hit: pd.Series, score: pd.Series) -> None:
        nonlocal fired, severity
        hit = hit.fillna(False).astype(bool)
        fired = fired | hit
        score = score.clip(lower=1.0, upper=100.0).fillna(1.0)
        severity = severity.where(~hit, severity.combine(score, max))

    if rule.above is not None:
        limit = rule.above
        mark(v > limit, v / limit if limit > 0 else 1.0 + (v - limit))
    if rule.below is not None:
        limit = rule.below
        mark(v < limit, limit / v.clip(lower=limit / 100) if limit > 0 else 1.0 + (limit - v))
    if rule.z_above is not None or rule.z_below is not None:
        z = robust_z(v)
        if rule.z_above is not None:
            mark(z > rule.z_above, z / max(abs(rule.z_above), 1e-9))
        if rule.z_below is not None:
            mark(z < rule.z_below, -z / max(abs(rule.z_below), 1e-9))
    if rule.rel_above is not None or rule.rel_below is not None:
        median = float(v[np.isfinite(v)].median()) if np.isfinite(v).any() else np.nan
        rel = v / median if median and np.isfinite(median) else pd.Series(np.nan, index=v.index)
        if rule.rel_above is not None:
            mark(rel > rule.rel_above, rel / rule.rel_above)
        if rule.rel_below is not None:
            mark(rel < rule.rel_below, rule.rel_below / rel.clip(lower=1e-9))
    return fired, severity


def flag_fields(fields: pd.DataFrame, rules: QcRules) -> pd.DataFrame:
    out = add_focus_offsets(fields).reset_index(drop=True)
    channels = n_channels(out)
    content = rules.settings.content_channel
    severity = pd.Series(0.0, index=out.index)
    excluded = pd.Series(False, index=out.index)
    reasons: list[list[str]] = [[] for _ in range(len(out))]
    for flag in rules.flags:
        out[f"flag_{flag}"] = False
    for rule in rules.field:
        for column in rule.columns(channels, content):
            if column not in out:
                continue  # e.g. best-focus offsets of single-plane plates
            fired, score = _evaluate(rule, out[column])
            out[f"flag_{rule.flag}"] |= fired
            severity = severity.combine(score.where(fired, 0.0), max)
            if rule.exclude:
                excluded |= fired
            values = out[column].to_numpy(dtype=float)
            for i in np.flatnonzero(fired.to_numpy()).tolist():
                text = f"{rule.flag} ({column} = {values[i]:.4g})"
                if text not in reasons[i]:
                    reasons[i].append(text)
    flag_columns = [f"flag_{f}" for f in rules.flags]
    out["flags"] = [
        ";".join(f[5:] for f in flag_columns if row[f])
        for row in out[flag_columns].to_dict("records")
    ]
    out["severity"] = severity.astype(float)
    out["excluded"] = excluded.astype(bool)
    out["reasons"] = ["; ".join(r) for r in reasons]
    return out


def decide_wells(fields: pd.DataFrame, rules: QcRules) -> pd.DataFrame:
    decision = rules.well
    rows = []
    for well, group in fields.groupby("well", sort=True):
        good = group[~group["excluded"]]
        cells = float(good["object_count"].sum()) if "object_count" in good else np.nan
        reasons = []
        if len(good) < decision.min_good_fields:
            reasons.append(f"{len(good)} good field(s) < {decision.min_good_fields}")
        if len(group) and len(good) / len(group) < decision.min_good_field_fraction:
            reasons.append(
                f"{len(good)} of {len(group)} fields good "
                f"(< {decision.min_good_field_fraction:.0%})"
            )
        if decision.min_cells and np.isfinite(cells) and cells < decision.min_cells:
            reasons.append(f"{int(cells)} cells < {decision.min_cells}")
        flags = sorted({f for text in group["flags"] for f in str(text).split(";") if f})
        rows.append(
            {
                "well": well,
                "row": int(group["row"].iloc[0]),
                "col": int(group["col"].iloc[0]),
                "n_fields": len(group),
                "n_good_fields": len(good),
                "cells": cells,
                "flags": ";".join(flags),
                "failed": bool(reasons),
                "reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def decide_plate(
    wells: pd.DataFrame,
    rules: QcRules,
    roles: Mapping[str, str] | None = None,
    z_prime: float | None = None,
) -> dict[str, Any]:
    decision = rules.plate
    reasons = []
    n = len(wells)
    failed = int(wells["failed"].sum()) if n else 0
    fraction = failed / n if n else 0.0
    if n and fraction > decision.max_failed_well_fraction:
        reasons.append(
            f"{failed} of {n} wells failed "
            f"({fraction:.0%} > {decision.max_failed_well_fraction:.0%})"
        )
    controls: dict[str, Any] = {}
    if roles:
        for role in decision.controls:
            members = [w for w, r in roles.items() if r == role]
            present = wells[wells["well"].isin(members)]
            if present.empty:
                continue
            bad = int(present["failed"].sum())
            share = bad / len(present)
            controls[role] = {"wells": len(present), "failed": bad}
            if share > decision.max_failed_control_fraction:
                reasons.append(f"{bad} of {len(present)} {role} wells failed")
    if decision.min_z_prime is not None and z_prime is not None and z_prime < decision.min_z_prime:
        reasons.append(f"Z′ {z_prime:.2f} < {decision.min_z_prime}")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "wells": n,
        "failed_wells": failed,
        "failed_well_fraction": fraction,
        "excluded_fields": 0,
        "controls": controls,
        "z_prime": z_prime,
    }


def add_compound_flags(
    wells: pd.DataFrame, fields: pd.DataFrame, rules: QcRules, layout: PlateLayout | None
) -> pd.DataFrame:
    """Autofluorescence / precipitation per well (:mod:`hcs_studio.qc.compound`)."""
    from hcs_studio.qc.compound import compound_flags

    out = wells.copy()
    out["autofluorescence"] = ""
    out["precipitation"] = False
    found = compound_flags(fields, layout, rules.compound)
    if found.empty:
        return out
    found = found.set_index("well")
    for column in found.columns:
        out[column] = out["well"].map(found[column])
    out["autofluorescence"] = out["autofluorescence"].fillna("").astype(str)
    out["precipitation"] = out["precipitation"].fillna(False).astype(bool)
    auto = out["autofluorescence"].tolist()
    for n, i in enumerate(out.index):
        extra = []
        if auto[n]:
            extra.append(f"autofluorescence ({auto[n]})")
        if out.at[i, "precipitation"]:
            extra.append("precipitation")
        if not extra:
            continue
        names = [e.split(" ")[0] for e in extra]
        current = [f for f in str(out.at[i, "flags"]).split(";") if f]
        out.at[i, "flags"] = ";".join([*current, *names])
        if rules.compound.exclude:
            reasons = [r for r in str(out.at[i, "reasons"]).split("; ") if r]
            out.at[i, "reasons"] = "; ".join([*reasons, *extra])
            out.at[i, "failed"] = True
    return out


def apply_rules(
    fields: pd.DataFrame,
    rules: QcRules,
    *,
    roles: Mapping[str, str] | None = None,
    z_prime: float | None = None,
    layout: PlateLayout | None = None,
) -> QcResult:
    """Flag fields, then decide wells (with compound artifacts when a layout is given) and the
    plate. Control roles come from ``roles`` or the layout."""
    for column in ("well", "row", "col", "field"):
        if column not in fields:
            raise ValueError(f"The field table needs a {column!r} column")
    if roles is None and layout is not None:
        roles = {well_name(*k): a.role for k, a in layout.wells.items()}
    flagged = flag_fields(fields, rules)
    wells = add_compound_flags(decide_wells(flagged, rules), flagged, rules, layout)
    plate = decide_plate(wells, rules, roles, z_prime)
    plate["excluded_fields"] = int(flagged["excluded"].sum())
    plate["fields"] = len(flagged)
    plate["flag_counts"] = {f: int(flagged[f"flag_{f}"].sum()) for f in rules.flags}
    plate["compound_flags"] = {
        "autofluorescence": int((wells["autofluorescence"] != "").sum()),
        "precipitation": int(wells["precipitation"].sum()),
    }
    return QcResult(flagged, wells, plate)
