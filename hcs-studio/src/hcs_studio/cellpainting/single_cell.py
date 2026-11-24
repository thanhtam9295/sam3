"""Single-cell tables in CellProfiler form (Phase 24a).

``well_single_cells(well_dir, …)`` turns one well's objects into one row per cell: the features
of each compartment (``nuclei``, ``cells``, ``cytoplasm`` by default) renamed by
``naming.rename_objects`` and joined on the shared object ID (cells grown from nuclei and the
cytoplasm made from them keep the nucleus ID), plus ``Metadata_`` columns:

``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Site`` (field), ``Metadata_TimePoint``,
``Metadata_ObjectNumber``, and from the plate layout ``Metadata_Role``, ``Metadata_Compound``,
``Metadata_Concentration`` (M), ``Metadata_CellLine``, ``Metadata_Treatment``,
``Metadata_ReplicateGroup``; ``Metadata_QCExcluded`` marks objects excluded by object QC
(Phase 18) — they are kept, flagged, so that the QC can be reviewed; the profiling pipeline
drops them before aggregation.

The first compartment that is present is the base: a cell needs it (a nucleus); compartments
missing for a cell leave its columns empty. Stitched duplicates are left out.

``write_single_cells(run, out)`` writes a run's table well by well into one Parquet file (the
union of all wells' columns; one well in memory at a time).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.cellpainting.naming import compartment, rename_objects

COMPARTMENTS = ("nuclei", "cells", "cytoplasm")
KEY = ["field", "timepoint", "object_id"]


def channel_prefixes_of(run: Path) -> list[str]:
    from hcs_studio.analysis.features import column_name

    manifest = json.loads((Path(run) / "run_manifest.json").read_text(encoding="utf-8"))
    channels = (manifest.get("plate", {}).get("metadata") or {}).get("channels") or []
    names = [column_name(str(c.get("name", ""))) for c in channels]
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _well_annotations(layout: Any, well: str) -> dict[str, Any]:
    from hcs_studio.model.plate import parse_well_name
    from hcs_studio.pharmacology.normalize import drug_treatments, treatment_key

    if layout is None:
        return {}
    annotation = layout.wells.get(parse_well_name(well))
    if annotation is None:
        return {"Metadata_Role": "empty"}
    drugs = drug_treatments(annotation)
    role = annotation.role
    if role == "untreated" and not drugs:
        role = "negative_control"
    return {
        "Metadata_Role": role,
        "Metadata_Compound": " + ".join(d[0] for d in drugs)
        if drugs
        else ("DMSO" if role == "negative_control" else ""),
        "Metadata_Concentration": drugs[0][1]
        if len(drugs) == 1
        else (0.0 if not drugs else np.nan),
        "Metadata_CellLine": annotation.cell_line or "",
        "Metadata_Treatment": treatment_key(drugs) or role,
        "Metadata_ReplicateGroup": annotation.replicate_group or "",
    }


def well_single_cells(
    well_dir: Path,
    plate: str,
    prefixes: list[str],
    channel_map: dict[str, str],
    *,
    compartments: Sequence[str] = COMPARTMENTS,
    layout: Any = None,
    radial_bins: int = 3,
) -> pd.DataFrame:
    """One row per cell of one well (see the module docstring)."""
    from hcs_studio.analysis.table_steps import read_well_objects

    objects = read_well_objects(Path(well_dir))
    if "in_overlap_duplicate" in objects:
        objects = objects[~objects["in_overlap_duplicate"].fillna(False).astype(bool)]
    if "timepoint" not in objects:
        objects["timepoint"] = 0
    present = [c for c in compartments if c in set(objects["object_type"])]
    if not present:
        return pd.DataFrame()
    merged: pd.DataFrame | None = None
    excluded = pd.Series(dtype=bool)
    for name in present:
        part = objects[objects["object_type"] == name]
        part = part.dropna(axis=1, how="all")
        # summaries of a compartment profiled on its own (nuclei carry their cells' and
        # cytoplasm's counts and means as children) would duplicate it
        others = tuple(f"{o}_" for o in present if o != name)
        features = [
            c
            for c in part.columns
            if c not in KEY
            and pd.api.types.is_numeric_dtype(part[c])
            and not str(c).startswith(("qc_", "trk_", "gate_", "class_", "cluster_", "spatial_"))
            and not (str(c).startswith(others) and not str(c).endswith("_correlation"))
        ]
        renamed = rename_objects(features, name, prefixes, channel_map, radial_bins)
        table = part[KEY + list(renamed)].rename(columns=renamed)
        if "qc_excluded" in part:
            flags = part.set_index(KEY)["qc_excluded"].fillna(False).astype(bool)
            excluded = flags if excluded.empty else excluded.combine(flags, max, fill_value=False)
        merged = table if merged is None else merged.merge(table, on=KEY, how="left")
    assert merged is not None
    well = Path(well_dir).name
    meta = pd.DataFrame(
        {
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Site": merged["field"].astype(int).to_numpy(),
            "Metadata_TimePoint": merged["timepoint"].astype(int).to_numpy(),
            "Metadata_ObjectNumber": merged["object_id"].astype(int).to_numpy(),
        },
        index=merged.index,
    )
    for key, value in _well_annotations(layout, well).items():
        meta[key] = value
    keys = list(zip(merged["field"], merged["timepoint"], merged["object_id"], strict=True))
    meta["Metadata_QCExcluded"] = (
        [bool(excluded.get(k, False)) for k in keys] if not excluded.empty else False
    )
    return pd.concat([meta, merged.drop(columns=KEY)], axis=1)


def iter_single_cells(
    run: Path,
    channel_map: dict[str, str] | None = None,
    *,
    compartments: Sequence[str] = COMPARTMENTS,
    radial_bins: int = 3,
) -> Iterator[pd.DataFrame]:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    run = Path(run)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    plate = str(
        (manifest.get("plate") or {}).get("barcode")
        or (manifest.get("plate") or {}).get("name")
        or run.name
    )
    path = run_layout_file(run)
    layout = load_plate_layout(path) if path is not None else None
    prefixes = channel_prefixes_of(run)
    for well_dir in sorted(
        p for p in (run / "wells").iterdir() if (p / "objects.parquet").exists()
    ):
        frame = well_single_cells(
            well_dir,
            plate,
            prefixes,
            channel_map or {},
            compartments=compartments,
            layout=layout,
            radial_bins=radial_bins,
        )
        if not frame.empty:
            yield frame


def write_parquet_stream(frames: Iterator[pd.DataFrame], target: Path) -> int:
    """Write data frames with possibly different columns into one Parquet file (union of the
    columns, missing ones null), keeping one frame in memory at a time (parts written first)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = target.with_name(target.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    parts = []
    for i, frame in enumerate(frames):
        part = parts_dir / f"{i:06d}.parquet"
        frame.to_parquet(part, index=False)
        parts.append(part)
    if not parts:
        pd.DataFrame().to_parquet(target)
        parts_dir.rmdir()
        return 0
    schema = pa.unify_schemas(
        [pq.read_schema(p).remove_metadata() for p in parts], promote_options="permissive"
    )
    names = sorted(
        schema.names, key=lambda n: (not n.startswith("Metadata_"), schema.names.index(n))
    )
    schema = pa.schema([schema.field(n) for n in names])
    rows = 0
    temporary = target.with_name(target.name + ".part")
    with pq.ParquetWriter(temporary, schema) as writer:
        for part in parts:
            table = pq.read_table(part)
            for column in schema.names:
                if column not in table.column_names:
                    table = table.append_column(
                        column, pa.nulls(len(table), schema.field(column).type)
                    )
            writer.write_table(table.select(schema.names).cast(schema))
            rows += len(table)
            part.unlink()
    parts_dir.rmdir()
    temporary.replace(target)
    return rows


def write_single_cells(
    run: Path, out: Path | None = None, channel_map: dict[str, str] | None = None, **options: Any
) -> tuple[Path, int]:
    """``<run>/profiling/single_cell.parquet`` (or ``out``); returns (path, cells)."""
    target = Path(out) if out is not None else Path(run) / "profiling" / "single_cell.parquet"
    return target, write_parquet_stream(
        iter_single_cells(Path(run), channel_map, **options), target
    )


def feature_columns(frame: pd.DataFrame, compartments: Sequence[str] = COMPARTMENTS) -> list[str]:
    """CellProfiler-style feature columns (compartment prefixes; not Metadata_)."""
    starts = tuple(f"{compartment(c)}_" for c in compartments)
    return [c for c in frame.columns if str(c).startswith(starts)]
