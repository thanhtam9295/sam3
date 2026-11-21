"""Object-level QC (Phase 18): objects that are likely segmentation errors.

The ``object_qc`` table step learns, from objects sampled across the plate, the median and
robust spread (1.4826 × MAD) of a few features — by default size (log area), shape (solidity)
and the mean intensity of the first channel measured (log) — and flags objects whose robust Z
lies beyond ``threshold`` in any of them: merged cells (large, low solidity), fragments (small),
out-of-focus or debris-like objects (intensity). Columns:

- ``qc_outlier``: flagged (empty for other object types);
- ``qc_excluded``: flagged and ``exclude`` is on; such objects are left out of the per-well
  feature statistics (means, medians, SDs) and of later table steps (gates, classifiers,
  clusters). Counts are not changed: a merged object still holds cells.
- ``qc_reason``: which features, e.g. "area_um2 high; solidity low".

Per well: ``qc_<type>_outlier_count`` and ``_fraction``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from hcs_studio.analysis.table_steps import TableModule, TableParams, WellContext
from hcs_studio.processing.base import MODULES, ModuleError

ROBUST = 1.4826
#: Features whose logarithm is used (sizes and intensities are right-skewed).
LOG_SUFFIXES = ("area_um2", "volume_um3", "_intensity")


class ObjectQcParams(TableParams):
    features: list[str] = Field(
        default_factory=list,
        description="Features to check; empty = area (or volume), solidity and the mean "
        "intensity of the first channel.",
    )
    threshold: float = Field(default=5.0, gt=0, description="Robust Z beyond which an object "
                             "is an outlier.")  # fmt: skip
    exclude: bool = Field(
        default=True, description="Leave outliers out of per-well feature statistics."
    )
    max_fit_objects: int = Field(default=200_000, ge=100)
    seed: int = 0


def default_features(columns: list[str]) -> list[str]:
    chosen: list[str] = []
    for name in ("area_um2", "volume_um3"):
        if name in columns:
            chosen.append(name)
            break
    if "solidity" in columns:
        chosen.append("solidity")
    suffix = "_mean_intensity"
    intensity = [c for c in columns if c.endswith(suffix) and len(c) > len(suffix)]
    direct = [c for c in intensity if "_" not in c[: -len(suffix)]]  # <channel>_mean_intensity
    if direct or intensity:
        chosen.append((direct or intensity)[0])
    return chosen


def _transform(values: pd.Series, feature: str) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if feature.endswith(LOG_SUFFIXES):
        return pd.Series(np.log10(v.where(v > 0)), index=v.index)
    return v


@MODULES.register("object_qc")
class ObjectQc(TableModule):
    """Flag likely segmentation errors (merged cells, fragments, out-of-focus objects) by
    robust outlier detection on size, shape and intensity."""

    name = "object_qc"
    title = "Object QC: outliers"
    Params = ObjectQcParams
    needs_fit: ClassVar[bool] = True

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        params: ObjectQcParams = self.params
        features = params.features or default_features([str(c) for c in sample.columns])
        if not features:
            raise ModuleError("no size, shape or intensity features to check")
        missing = [f for f in features if f not in sample.columns]
        if missing:
            raise ModuleError(f"feature(s) not in the object table: {', '.join(missing)}")
        state: dict[str, Any] = {"features": {}}
        for feature in features:
            values = _transform(sample[feature], feature).dropna()
            if len(values) < 10:
                continue
            median = float(values.median())
            spread = ROBUST * float((values - median).abs().median())
            if spread > 0:
                state["features"][feature] = {"median": median, "spread": spread}
        if not state["features"]:
            raise ModuleError("the features do not vary across objects")
        return state

    def describe_state(self, state: Any) -> dict[str, Any]:
        return dict(state)

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        params: ObjectQcParams = self.params
        rows = self.selected(objects)
        outlier = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        reason = pd.Series("", index=objects.index, dtype=object)
        flags = pd.Series(False, index=objects.index)
        for feature, ref in state["features"].items():
            if feature not in objects:
                continue
            z = (_transform(objects[feature], feature) - ref["median"]) / ref["spread"]
            high = (z > params.threshold).fillna(False)
            low = (z < -params.threshold).fillna(False)
            for mask, word in ((high, "high"), (low, "low")):
                hit = mask & rows
                reason[hit] = reason[hit].map(
                    lambda r, w=f"{feature} {word}": f"{r}; {w}" if r else w
                )
                flags |= hit
        outlier[rows] = flags[rows]
        excluded = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        excluded[rows] = flags[rows] & params.exclude
        return pd.DataFrame({"qc_outlier": outlier, "qc_excluded": excluded, "qc_reason": reason})

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kind = str(self.params.object_type)
        values = objects["qc_outlier"].dropna()
        n, hits = len(values), int(values.astype(bool).sum())
        reasons = objects.loc[objects["qc_outlier"].fillna(False).astype(bool), "qc_reason"]
        return {
            f"qc_{kind}_outlier_count": hits,
            f"qc_{kind}_outlier_fraction": hits / n if n else math.nan,
        }, {"reasons": reasons.value_counts().to_dict()}
