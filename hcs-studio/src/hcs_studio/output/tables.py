"""Per-well and plate-level tables of a run: objects, fields, QC flags, per-well summary.

Per-object tables can be very large (10⁷ rows), so the plate-wide Parquet file is written
by streaming the per-well files with a unified schema, never by loading them all at once.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Columns added to every table row to say where it came from.
CONTEXT_COLUMNS = ("well", "row", "col", "field", "timepoint")


def write_well_tables(well_dir: Path, tables: dict[str, list[pd.DataFrame]]) -> dict[str, str]:
    """Write one well's tables; returns {table name: relative file name}."""
    written = {}
    for name, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        if name == "objects":
            path = well_dir / "objects.parquet"
            frame.to_parquet(path.with_name(path.name + ".part"), index=False)
        else:
            path = well_dir / ("coloc.csv" if name == "coloc" else f"{name}.csv")
            frame.to_csv(path.with_name(path.name + ".part"), index=False)
        path.with_name(path.name + ".part").replace(path)
        written[name] = path.name
    return written


#: Annotations written by table steps next to a well's objects.parquet (same row order).
ANNOTATIONS_FILE = "objects_annotations.parquet"
_KEY = ("well", "field", "timepoint", "object_type", "object_id")


def _annotation_columns(well_dir: Path) -> list[str]:
    import pyarrow.parquet as pq

    path = well_dir / ANNOTATIONS_FILE
    return [c for c in pq.read_schema(path).names if c not in _KEY] if path.exists() else []


def merge_object_tables(well_dirs: list[Path], target: Path) -> int:
    """Stream wells/*/objects.parquet (with any table-step annotations) into one file with
    the union of all columns. One well at a time is in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dirs = [d for d in well_dirs if (d / "objects.parquet").exists()]
    if not dirs:
        return 0
    schemas = []
    for d in dirs:
        schema = pq.read_schema(d / "objects.parquet")
        if (d / ANNOTATIONS_FILE).exists():
            extra = pq.read_schema(d / ANNOTATIONS_FILE)
            for name in _annotation_columns(d):
                if name not in schema.names:
                    schema = schema.append(extra.field(name))
        schemas.append(schema.remove_metadata())
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    rows = 0
    part = target.with_name(target.name + ".part")
    with pq.ParquetWriter(part, schema) as writer:
        for d in dirs:
            annotations = None
            if (d / ANNOTATIONS_FILE).exists():
                annotations = pq.read_table(d / ANNOTATIONS_FILE, columns=_annotation_columns(d))
            offset = 0
            for batch in pq.ParquetFile(d / "objects.parquet").iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch])
                if annotations is not None and annotations.num_rows >= offset + len(table):
                    piece = annotations.slice(offset, len(table))
                    for name in piece.column_names:
                        if name not in table.column_names:
                            table = table.append_column(name, piece.column(name))
                offset += len(table)
                for column in schema.names:
                    if column not in table.column_names:
                        table = table.append_column(
                            column, pa.nulls(len(table), schema.field(column).type)
                        )
                writer.write_table(table.select(schema.names).cast(schema))
                rows += len(table)
    part.replace(target)
    return rows


def qc_flags(fields: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Flag out-of-focus, saturated and edge fields, relative to the plate."""
    flags = fields.copy()
    median_focus = float(flags["focus_score"].median()) if "focus_score" in flags else np.nan
    flags["out_of_focus"] = flags.get("focus_score", np.nan) < params.focus_fraction * median_focus
    flags["saturated"] = flags.get("saturated_fraction", 0) > params.saturation_fraction
    flags["edge_field"] = False
    if {"x_um", "y_um"}.issubset(flags.columns):
        for _, group in flags.groupby("well"):
            positions = group.drop_duplicates("field")
            if len(positions) < 9 or positions[["x_um", "y_um"]].isna().any().any():
                continue
            x, y = positions.x_um, positions.y_um
            edge = (
                np.isclose(x, x.min())
                | np.isclose(x, x.max())
                | np.isclose(y, y.min())
                | np.isclose(y, y.max())
            )
            edge_fields = set(positions.field[np.asarray(edge, dtype=bool)])
            flags.loc[group.index, "edge_field"] = group.field.isin(edge_fields)
    flags["excluded"] = (params.exclude_out_of_focus & flags.out_of_focus) | (
        params.exclude_edge_fields & flags.edge_field
    )
    return flags


def well_summary(
    objects: pd.DataFrame | None,
    flags: pd.DataFrame | None,
    coloc: pd.DataFrame | None,
    params: Any,
    analysis: dict[str, dict[str, Any]] | None = None,
    field_rects: dict[str, dict[int, tuple[float, float, float, float]]] | None = None,
) -> pd.DataFrame:
    """One row per well: counts, density, feature means/medians/SDs, colocalization, and
    the values of table steps (``analysis``, by well).

    Stitched wells (``field_rects``: well → field → (y0, x0, y1, x1) µm) count each object
    once: objects marked ``in_overlap_duplicate`` are left out of counts and statistics, and
    densities use the area the fields cover together (overlaps once).
    """
    rows: dict[str, dict[str, Any]] = {}
    field_rects = field_rects or {}
    unique_counts: dict[tuple[str, str], int] = {}
    if objects is not None and "in_overlap_duplicate" in objects and flags is not None:
        excluded_fields = set(map(tuple, flags[flags.excluded][["well", "field"]].values.tolist()))
        keep = ~objects["in_overlap_duplicate"].fillna(False).astype(bool)
        keep &= np.asarray(
            [
                (w, f) not in excluded_fields
                for w, f in zip(objects.well, objects.field, strict=True)
            ],
            dtype=bool,
        )
        sizes = objects[keep].groupby(["well", "object_type"]).size()
        for key, n in sizes.to_dict().items():
            parts: tuple[Any, ...] = tuple(key)  # type: ignore[arg-type]
            unique_counts[(str(parts[0]), str(parts[1]))] = int(n)
    if flags is not None and not flags.empty:
        count_columns = [c for c in flags.columns if c.startswith("n_")]
        for well, group in flags.groupby("well"):
            used = group[~group.excluded]
            row = rows.setdefault(str(well), {"well": well})
            row["fields"] = int(group.field.nunique())
            row["fields_excluded"] = int(group[group.excluded].field.nunique())
            row["fields_out_of_focus"] = int(group[group.out_of_focus].field.nunique())
            row["fields_saturated"] = int(group[group.saturated].field.nunique())
            area = float(used["area_mm2"].sum()) if "area_mm2" in used else np.nan
            rects = field_rects.get(str(well))
            if rects:
                from hcs_studio.analysis.stitching import union_area

                chosen = [rects[int(f)] for f in used.field.unique() if int(f) in rects]
                area = union_area(chosen) / 1e6 if chosen else np.nan
            row["area_mm2"] = area
            for column in count_columns:
                total = unique_counts.get((str(well), column[2:]), int(used[column].sum()))
                row[f"{column[2:]}_count"] = total
                row[f"{column[2:]}_per_mm2"] = total / area if area and area > 0 else np.nan
        count_label = str(flags["count_labels"].iloc[0]) if "count_labels" in flags else ""
        key = f"{count_label}_count"
        counts = pd.Series({w: r.get(key, np.nan) for w, r in rows.items()}, dtype=float)
        median = float(counts.median()) if counts.notna().any() else np.nan
        for well, row in rows.items():
            row["low_count"] = bool(counts.get(well, np.nan) < params.low_count_fraction * median)
    if objects is not None and not objects.empty:
        excluded = set()
        if flags is not None and not flags.empty:
            excluded = set(map(tuple, flags[flags.excluded][["well", "field"]].to_numpy().tolist()))
        if excluded:
            kept = [
                (w, f) not in excluded for w, f in zip(objects.well, objects.field, strict=True)
            ]
            objects = objects[kept]
        duplicate = objects.get("in_overlap_duplicate")
        if duplicate is not None:
            objects = objects[~duplicate.fillna(False).astype(bool)]
        skip = {
            "object_id",
            "parent_id",
            "row",
            "col",
            "field",
            "timepoint",
            "well_x_um",
            "well_y_um",
        } | {
            c
            for c in objects.columns
            if c.startswith(("centroid_", "gate_", "class_", "cluster_", "spatial_"))
        }
        numeric = [c for c in objects.select_dtypes("number").columns if c not in skip]
        for keys, group in objects.groupby(["well", "object_type"]):
            well, kind = str(keys[0]), str(keys[1])
            row = rows.setdefault(str(well), {"well": well})
            row[f"{kind}_objects_measured"] = len(group)
            stats = group[numeric].agg(["mean", "median", "std"])
            for column in numeric:
                if group[column].notna().any():
                    for stat in ("mean", "median", "std"):
                        row[f"{kind}_{column}_{stat}"] = float(stats.loc[stat, column])  # type: ignore[arg-type]
    if coloc is not None and not coloc.empty:
        for keys, group in coloc.groupby(["well", "channel_a", "channel_b"]):
            well, a, b = (str(k) for k in keys)
            row = rows.setdefault(str(well), {"well": well})
            for metric in ("pearson", "m1", "m2", "tm1", "tm2", "icq"):
                row[f"coloc_{a}_{b}_{metric}_median"] = float(group[metric].median())
    for well, values in (analysis or {}).items():
        rows.setdefault(str(well), {"well": well}).update(values)
    return pd.DataFrame(list(rows.values())).sort_values("well") if rows else pd.DataFrame()


def _field_rects(well_dirs: list[Path]) -> dict[str, dict[int, tuple[float, float, float, float]]]:
    """Stitched field rectangles (y0, x0, y1, x1) in µm per well, from well_summary.json."""
    out = {}
    for d in well_dirs:
        try:
            data = json.loads((d / "well_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        offsets = (data.get("stitching") or {}).get("field_offsets_um")
        size = data.get("field_size_um")
        if offsets and size:
            out[str(data.get("well", d.name))] = {
                int(f): (y, x, y + size[0], x + size[1]) for f, (y, x) in offsets.items()
            }
    return out


def aggregate_run(
    folder: Path,
    params: Any,
    table_modules: Sequence[Any] = (),
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build tables/ for a finished run; returns per-well summary rows by well name.

    ``table_modules`` (table steps) run first; their annotations and per-well values are
    merged in, and ``tables/analysis_summary.json`` records what they did.
    """
    from hcs_studio.analysis.table_steps import run_table_steps

    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    well_dirs = sorted(p for p in (folder / "wells").iterdir() if p.is_dir())
    analysis = run_table_steps(folder, list(table_modules), log)
    record = tables / "analysis_summary.json"
    if table_modules:
        payload = {"steps": analysis.steps, "columns": analysis.columns}
        record.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        record.unlink(missing_ok=True)
    n_objects = merge_object_tables(well_dirs, tables / "per_object_all.parquet")

    def gather(name: str) -> pd.DataFrame | None:
        parts = [pd.read_csv(d / name) for d in well_dirs if (d / name).exists()]
        return pd.concat(parts, ignore_index=True) if parts else None

    fields = gather("fields.csv")
    coloc = gather("coloc.csv")
    flags = None
    if fields is not None:
        flags = qc_flags(fields, params)
        fields.to_csv(tables / "per_field.csv", index=False)
    objects = None
    if n_objects:
        import pyarrow.parquet as pq

        # Summary statistics need numbers only; read the plate file column-wise.
        objects = pq.read_table(tables / "per_object_all.parquet").to_pandas()
    summary = well_summary(
        objects, flags, coloc, params, analysis.per_well, _field_rects(well_dirs)
    )
    if flags is not None:
        well_flags = summary.set_index("well")["low_count"] if "low_count" in summary else None
        flags["low_count_well"] = flags.well.map(well_flags) if well_flags is not None else False
        flags.to_csv(tables / "qc_flags.csv", index=False)
    if coloc is not None:
        coloc.to_csv(tables / "coloc_all.csv", index=False)
    if not summary.empty:
        summary.to_csv(tables / "per_well_summary.csv", index=False)
    return {str(r["well"]): r for r in summary.to_dict("records")} if not summary.empty else {}
