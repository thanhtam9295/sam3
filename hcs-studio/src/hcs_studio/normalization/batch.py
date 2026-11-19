"""Day and batch effects across plates.

``frame`` rows are wells of several plates: ``plate``, ``day`` (date the plate was treated or
imaged), ``role`` and ``value``. Normalizing to same-day controls divides every well by the
median of the vehicle wells of all plates of that day (× 100); normalizing per plate uses
each plate's own vehicle wells. The share of the vehicle wells' variance explained by day
and by plate (one-way R²) shows how much of the spread is a day or plate effect, before and
after normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTROL = "negative_control"


def r_squared(values: pd.Series, groups: pd.Series) -> float:
    """Between-group share of the variance (one-way ANOVA R²)."""
    data = pd.DataFrame({"v": values.astype(float), "g": groups}).dropna()
    if data["g"].nunique() < 2 or len(data) < 3:
        return math.nan
    total = float(((data["v"] - data["v"].mean()) ** 2).sum())
    if total == 0:
        return 0.0
    means = data.groupby("g")["v"].transform("mean")
    between = float(((means - data["v"].mean()) ** 2).sum())
    return between / total


def normalize_to_controls(
    frame: pd.DataFrame, by: str = "day", control: str = CONTROL
) -> pd.Series:
    """100 × value / median of the control wells in the same ``by`` group (day or plate)."""
    controls = frame[frame["role"] == control].groupby(by)["value"].median()
    missing = sorted(set(frame[by]) - set(controls.index[controls > 0]))
    if missing:
        raise ValueError(f"No positive vehicle median for {by} {', '.join(map(str, missing))}")
    return 100 * frame["value"] / frame[by].map(controls)


@dataclass
class BatchEffects:
    per_plate: pd.DataFrame  # plate, day, vehicle median raw / per day / per plate
    summary: pd.DataFrame  # normalization × (R² day, R² plate, vehicle CV %)
    notes: list[str] = field(default_factory=list)


def batch_effects(frame: pd.DataFrame, control: str = CONTROL) -> BatchEffects:
    """Day and plate effects of the vehicle wells: raw, normalized to same-day controls, and
    normalized per plate."""
    needed = {"plate", "day", "role", "value"}
    if not needed <= set(frame.columns):
        raise ValueError(f"Need columns {', '.join(sorted(needed))}")
    data = frame.copy()
    data["per_day"] = normalize_to_controls(data, "day", control)
    data["per_plate"] = normalize_to_controls(data, "plate", control)
    vehicle = data[data["role"] == control]
    rows = []
    for column, label in (
        ("value", "raw"),
        ("per_day", "same-day controls"),
        ("per_plate", "per-plate controls"),
    ):
        values = vehicle[column].astype(float)
        rows.append(
            {
                "normalization": label,
                "r2_day": r_squared(values, vehicle["day"]),
                "r2_plate": r_squared(values, vehicle["plate"]),
                "vehicle_cv_pct": float(100 * values.std(ddof=1) / values.mean())
                if len(values) > 1 and values.mean()
                else math.nan,
            }
        )
    per_plate = (
        vehicle.groupby(["plate", "day"])
        .agg(
            vehicle_median=("value", "median"),
            vehicle_median_per_day=("per_day", "median"),
            vehicle_median_per_plate=("per_plate", "median"),
            n_vehicle=("value", "size"),
        )
        .reset_index()
        .sort_values(["day", "plate"])
        .reset_index(drop=True)
    )
    notes = []
    if data["day"].nunique() < 2:
        notes.append("All plates are from one day: no day effect can be estimated")
    plate_share = float(rows[1]["r2_plate"])  # type: ignore[arg-type]
    if np.isfinite(plate_share) and plate_share > 0.3:
        notes.append(
            "After same-day normalization plates still differ (R² plate > 0.3): normalize per "
            "plate as well"
        )
    return BatchEffects(per_plate, pd.DataFrame(rows), notes)
