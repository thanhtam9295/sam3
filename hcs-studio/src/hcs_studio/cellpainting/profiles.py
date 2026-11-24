"""The profiling pipeline (Phase 24b): single cells → well profiles → treatment profiles.

Every stage calls pycytominer (``aggregate``, ``annotate``, ``normalize``, ``consensus`` and the
feature-selection operations), so the profiles are pycytominer's — HCS Studio decides what goes
in (single-cell QC, which wells are the reference, per plate or across plates) and records
every choice:

1. **single-cell QC**: cells flagged by object QC (``Metadata_QCExcluded``, Phase 18) and cells
   missing more than ``max_missing_fraction`` of the features are dropped (counted in the log);
2. **aggregate**: the ``aggregate`` (median by default) of the cells of each well
   (``Metadata_Plate``, ``Metadata_Well``), with the number of cells (``Metadata_Object_Count``);
   the other ``Metadata_`` columns, constant within a well, are carried over;
3. **annotate** (optional): a plate map joined on the well (``pycytominer.annotate``) — for data
   imported from CellProfiler; HCS Studio's own runs carry their layout already;
4. **normalize**: ``mad_robustize`` (x − median) / (1.4826 · MAD) by default, against the **vehicle
   wells** (``Metadata_Role == 'negative_control'``) of each plate separately (``per_plate``), or
   against all wells; ``spherize`` (ZCA-cor whitening against the vehicle wells of all plates
   together) to remove batch effects;
5. **feature selection**: pycytominer's operations in order — ``variance_threshold``,
   ``correlation_threshold``, ``drop_na_columns``, ``blocklist`` (CellProfiler's standard list of
   uninformative features), ``drop_outliers`` (features with |value| > ``outlier_cutoff``) — each
   on the features left by the ones before, exactly as ``pycytominer.feature_select`` does; the
   log lists what each removed;
6. **consensus**: the median (or ``modz``) profile of each treatment (``consensus_by``).

``run_profiling`` writes ``profiling/`` with every stage as Parquet, ``feature_selection.json``
and ``profiling.json`` (options, counts, pycytominer version).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

STRATA = ["Metadata_Plate", "Metadata_Well"]
OPERATIONS = (
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
)
VEHICLE = "Metadata_Role == 'negative_control'"


class ProfilingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregate: Literal["median", "mean"] = "median"
    normalize: Literal["mad_robustize", "robustize", "standardize", "spherize", "none"] = (
        "mad_robustize"
    )
    reference: Literal["vehicle", "all"] = Field(
        default="vehicle", description="Wells the normalization is relative to."
    )
    per_plate: bool = Field(default=True, description="Normalize each plate on its own.")
    spherize_method: Literal["PCA", "ZCA", "PCA-cor", "ZCA-cor"] = "ZCA-cor"
    operations: list[str] = Field(default_factory=lambda: list(OPERATIONS))
    na_cutoff: float = Field(default=0.05, ge=0, le=1)
    corr_threshold: float = Field(default=0.9, gt=0, le=1)
    freq_cut: float = 0.05
    unique_cut: float = 0.01
    outlier_cutoff: float = 500.0
    consensus: Literal["median", "modz"] = "median"
    consensus_by: list[str] = Field(
        default_factory=lambda: ["Metadata_Treatment", "Metadata_CellLine"]
    )
    max_missing_fraction: float = Field(default=0.1, ge=0, le=1)


@dataclass
class Profiles:
    single_cells: int
    aggregated: pd.DataFrame
    normalized: pd.DataFrame
    selected: pd.DataFrame
    consensus: pd.DataFrame
    features: list[str]
    selection: dict[str, list[str]] = field(default_factory=dict)
    log: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_features(self) -> list[str]:
        return [c for c in self.selected.columns if c in set(self.features)]


def _features(frame: pd.DataFrame) -> list[str]:
    from pycytominer.cyto_utils import infer_cp_features

    return list(infer_cp_features(frame))


def single_cell_qc(
    cells: pd.DataFrame, features: list[str], options: ProfilingOptions
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"cells": len(cells)}
    keep = pd.Series(True, index=cells.index)
    if "Metadata_QCExcluded" in cells:
        flagged = cells["Metadata_QCExcluded"].fillna(False).astype(bool)
        counts["object_qc_excluded"] = int(flagged.sum())
        keep &= ~flagged
    missing = cells[features].isna().mean(axis=1)
    too_many = missing > options.max_missing_fraction
    counts["too_many_missing"] = int((too_many & keep).sum())
    keep &= ~too_many
    counts["kept"] = int(keep.sum())
    return cells[keep], counts


def aggregate_wells(
    cells: pd.DataFrame, features: list[str], operation: str = "median"
) -> pd.DataFrame:
    """pycytominer ``aggregate`` per well, other well-level Metadata carried over."""
    from pycytominer import aggregate

    profiles = aggregate(
        cells,
        strata=STRATA,
        features=features,
        operation=operation,
        compute_object_count=True,
        object_feature="Metadata_ObjectNumber",
    )
    extra = [
        c
        for c in cells.columns
        if c.startswith("Metadata_")
        and c not in STRATA
        and c
        not in (
            "Metadata_ObjectNumber",
            "Metadata_Site",
            "Metadata_TimePoint",
            "Metadata_QCExcluded",
        )
        and cells.groupby(STRATA)[c].nunique(dropna=False).max() <= 1
    ]
    if extra:
        well_meta = cells.groupby(STRATA, as_index=False)[extra].first()
        profiles = profiles.merge(well_meta, on=STRATA, how="left")
    meta = [c for c in profiles.columns if c.startswith("Metadata_")]
    return pd.DataFrame(profiles[meta + [c for c in profiles.columns if c not in meta]])


def normalize_profiles(
    profiles: pd.DataFrame, features: list[str], options: ProfilingOptions
) -> pd.DataFrame:
    from pycytominer import normalize

    if options.normalize == "none":
        return profiles.copy()
    samples = VEHICLE if options.reference == "vehicle" else "all"
    if samples != "all" and (
        "Metadata_Role" not in profiles
        or not (profiles["Metadata_Role"] == "negative_control").any()
    ):
        raise ValueError(
            "No vehicle (negative-control) wells to normalize against; use reference: all"
        )
    meta = [c for c in profiles.columns if c not in features]
    kwargs: dict[str, Any] = {
        "features": features,
        "meta_features": meta,
        "samples": samples,
        "method": options.normalize,
    }
    if options.normalize == "spherize":
        kwargs["spherize_method"] = options.spherize_method
        return pd.DataFrame(normalize(profiles, **kwargs))  # batch effects: all plates together
    if not options.per_plate:
        return pd.DataFrame(normalize(profiles, **kwargs))
    parts = []
    for plate, group in profiles.groupby("Metadata_Plate", sort=False):
        if samples != "all" and not (group["Metadata_Role"] == "negative_control").any():
            raise ValueError(f"Plate {plate}: no vehicle wells to normalize against")
        parts.append(normalize(group.reset_index(drop=True), **kwargs))
    return pd.DataFrame(pd.concat(parts, ignore_index=True))


def select_features(
    profiles: pd.DataFrame, features: list[str], options: ProfilingOptions
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """pycytominer's feature-selection operations in order, each on the features left (as
    ``pycytominer.feature_select`` does), with what each removed."""
    from pycytominer.cyto_utils import drop_outlier_features, get_blocklist_features
    from pycytominer.operations import (
        correlation_threshold,
        frequency_threshold,
        get_na_columns,
        variance_threshold,
    )

    left = list(features)
    removed: dict[str, list[str]] = {}
    for op in options.operations:
        if op == "variance_threshold":
            exclude = variance_threshold(population_df=profiles, features=left, samples="all")
        elif op == "frequency_threshold":
            exclude = frequency_threshold(
                population_df=profiles,
                features=left,
                samples="all",
                freq_cut=options.freq_cut,
                unique_cut=options.unique_cut,
            )
        elif op == "correlation_threshold":
            exclude = correlation_threshold(
                population_df=profiles,
                features=left,
                samples="all",
                threshold=options.corr_threshold,
                method="pearson",
            )
        elif op == "drop_na_columns":
            exclude = get_na_columns(
                population_df=profiles, features=left, samples="all", cutoff=options.na_cutoff
            )
        elif op == "blocklist":
            exclude = get_blocklist_features(population_df=profiles)
        elif op == "drop_outliers":
            exclude = drop_outlier_features(
                population_df=profiles,
                features=left,
                samples="all",
                outlier_cutoff=options.outlier_cutoff,
            )
        else:
            raise ValueError(
                f"Unknown feature-selection operation {op!r} ({', '.join(OPERATIONS)})"
            )
        removed[op] = sorted(set(exclude) & set(left))
        left = [f for f in left if f not in set(exclude)]
    dropped = sorted(set(features) - set(left))
    return profiles.drop(columns=dropped), removed


def consensus_profiles(
    profiles: pd.DataFrame, features: list[str], options: ProfilingOptions
) -> pd.DataFrame:
    from pycytominer import consensus

    by = [c for c in options.consensus_by if c in profiles and profiles[c].notna().any()]
    if not by:  # imported data without treatments: one profile per well
        by = STRATA
    frame = profiles.copy()
    for column in by:
        frame[column] = frame[column].fillna("").astype(str)
    return pd.DataFrame(
        consensus(frame, replicate_columns=by, operation=options.consensus, features=features)
    )


def build_profiles(
    cells: pd.DataFrame,
    options: ProfilingOptions | None = None,
    platemap: pd.DataFrame | None = None,
    platemap_join: str = "Metadata_well_position",
) -> Profiles:
    options = options or ProfilingOptions()
    features = _features(cells)
    if not features:
        raise ValueError("No CellProfiler-style features (Cells_/Nuclei_/Cytoplasm_ columns)")
    kept, qc = single_cell_qc(cells, features, options)
    aggregated = aggregate_wells(kept, features, options.aggregate)
    if platemap is not None:
        from pycytominer import annotate

        aggregated = annotate(aggregated, platemap, join_on=[platemap_join, "Metadata_Well"])
    normalized = normalize_profiles(aggregated, features, options)
    selected, removed = select_features(normalized, features, options)
    kept_features = [f for f in features if f in selected.columns]
    consensus = consensus_profiles(selected, kept_features, options)
    used = [
        c for c in options.consensus_by if c in selected and selected[c].notna().any()
    ] or STRATA
    log = {
        "single_cell_qc": qc,
        "consensus_by": used,
        "wells": len(aggregated),
        "features": len(features),
        "selected_features": len(kept_features),
        "removed": {op: len(v) for op, v in removed.items()},
        "treatments": len(consensus),
    }
    return Profiles(len(cells), aggregated, normalized, selected, consensus, features, removed, log)


def load_single_cells(sources: list[Path]) -> pd.DataFrame:
    """Single-cell tables: HCS Studio run folders (``profiling/single_cell.parquet``, written
    when missing) or Parquet files (own or imported)."""
    from hcs_studio.cellpainting.single_cell import write_single_cells

    frames = []
    for source in sources:
        source = Path(source)
        if source.is_dir():
            path = source / "profiling" / "single_cell.parquet"
            if not path.exists():
                path, _ = write_single_cells(source)
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_parquet(source))
    if not frames:
        raise ValueError("No single-cell tables given")
    return pd.concat(frames, ignore_index=True)


def write_profiles(result: Profiles, out: Path, options: ProfilingOptions) -> list[str]:
    import importlib.metadata as metadata

    out.mkdir(parents=True, exist_ok=True)
    stages = {
        "aggregated.parquet": result.aggregated,
        "normalized.parquet": result.normalized,
        "feature_selected.parquet": result.selected,
        "consensus.parquet": result.consensus,
    }
    for name, frame in stages.items():
        frame.to_parquet(out / name, index=False)
    (out / "feature_selection.json").write_text(
        json.dumps(
            {
                "features_in": len(result.features),
                "features_kept": len(result.selected_features),
                "removed": result.selection,
                "kept": result.selected_features,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (out / "profiling.json").write_text(
        json.dumps(
            {
                "options": options.model_dump(mode="json"),
                "log": result.log,
                "pycytominer": metadata.version("pycytominer"),
            },
            indent=1,
            default=_json,
        ),
        encoding="utf-8",
    )
    return [*stages, "feature_selection.json", "profiling.json"]


def _json(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else str(value)


def run_profiling(
    sources: list[Path],
    out: Path,
    options: ProfilingOptions | None = None,
    platemap: Path | None = None,
) -> Profiles:
    options = options or ProfilingOptions()
    cells = load_single_cells(sources)
    plate_map = pd.read_csv(platemap, sep=None, engine="python") if platemap else None
    result = build_profiles(cells, options, plate_map)
    write_profiles(result, Path(out), options)
    return result
