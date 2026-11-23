"""Time-course tables (Phase 22a): readouts per well and time point, on real times.

``build_time_course(folder)`` turns a run of a time-lapse plate into one row per (well, time
point) — the same statistics as ``per_well_summary.csv`` (object counts and per-object feature
statistics, from ``object_statistics``; per-field values such as ``confluence_pct`` averaged
over the well's fields), plus:

- ``time_s`` / ``time_h``: since the first time point, from the instrument's timestamps (or the
  user's time interval) recorded in the run manifest — never frame numbers; a run without times
  gets no time columns and a note;
- ``hours_since_treatment``: from the treatment time (``treatment_time``) when known.

``assemble([...])`` joins the time courses or per-well summaries of several runs — a plate
imaged repeatedly as separate acquisitions (scheduler re-imaging) — on their acquisition times.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTEXT = ("well", "row", "col", "field", "timepoint", "x_um", "y_um", "field_y_um", "field_x_um")


def parse_time(text: str | None) -> datetime | None:
    """ISO 8601 → naive local datetime (aware times are converted to this computer's local time;
    instruments write local times without a zone)."""
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value.astimezone().replace(tzinfo=None) if value.tzinfo else value


def dispensed_at(barcode: str) -> str | None:
    """The latest ``plate.dispensed`` event of a barcode in the registry (None without one)."""
    if not barcode:
        return None
    try:
        from sqlalchemy import select

        from hcs_studio.db.automation import LabEventRecord
        from hcs_studio.db.engine import session_scope

        with session_scope() as session:
            rows = session.scalars(
                select(LabEventRecord)
                .where(LabEventRecord.type == "plate.dispensed",
                       LabEventRecord.barcode == barcode,
                       LabEventRecord.status == "accepted")
                .order_by(LabEventRecord.logged_at.desc())
            ).all()  # fmt: skip
            for row in rows:
                payload = (row.message or {}).get("payload") or {}
                if payload.get("dispensed_at"):
                    return str(payload["dispensed_at"])
    except Exception:  # no registry, an older schema: the other sources still apply
        return None
    return None


def treatment_time(
    manifest: dict[str, Any], layout: Any = None, override: str | None = None
) -> tuple[datetime | None, str]:
    """When the compounds were added: ``override``, else the layout's ``treated_at``, else the
    plate's ``plate.dispensed`` event in the registry. (time, source)."""
    if override:
        return parse_time(override), "given"
    if layout is not None and str(getattr(layout, "treated_at", "") or ""):
        return parse_time(layout.treated_at), "plate layout"
    barcode = str((manifest.get("plate") or {}).get("barcode") or "")
    event = dispensed_at(barcode)
    if event:
        return parse_time(event), "plate.dispensed event"
    return None, ""


def _layout(folder: Path) -> Any:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    path = run_layout_file(folder)
    return load_plate_layout(path) if path is not None else None


def build_time_course(
    folder: Path,
    excluded: set[tuple[str, int]] | None = None,
    treated_at: str | None = None,
) -> pd.DataFrame | None:
    """Readouts per well and time point of a time-lapse run (None for a single time point)."""
    from hcs_studio.analysis.table_steps import read_well_objects
    from hcs_studio.output.tables import object_statistics

    folder = Path(folder)
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    plate = manifest.get("plate") or {}
    n_t = int(plate.get("timepoints") or 1)
    if n_t < 2:
        return None
    rows: list[dict[str, Any]] = []
    for well_dir in sorted(p for p in (folder / "wells").iterdir() if p.is_dir()):
        per_t: dict[int, dict[str, Any]] = {}
        if (well_dir / "objects.parquet").exists():
            objects = read_well_objects(well_dir)
            for t, group in objects.groupby("timepoint"):
                stats, _ = object_statistics(group, excluded or set())
                per_t.setdefault(int(str(t)), {}).update(stats.get(well_dir.name, {}))
        if (well_dir / "fields.csv").exists():
            fields = pd.read_csv(well_dir / "fields.csv")
            if "timepoint" in fields:
                keep = fields[[c for c in fields.columns if c not in CONTEXT or c == "timepoint"]]
                numeric = keep.select_dtypes("number")
                for t, group in numeric.groupby(keep["timepoint"]):
                    means = group.drop(columns="timepoint").mean(numeric_only=True)
                    values = {str(k): float(v) for k, v in means.items()}
                    per_t.setdefault(int(str(t)), {}).update(values)
        for t in range(n_t):
            values = dict(per_t.get(t, {}))
            values.pop("well", None)
            rows.append({"well": well_dir.name, "timepoint": t, **values})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    times = plate.get("time_points_s")
    if times and len(times) == n_t:
        table.insert(2, "time_s", [float(times[int(t)]) for t in table["timepoint"]])
        table.insert(3, "time_h", table["time_s"] / 3600)
        layout = _layout(folder)
        treated, _source = treatment_time(manifest, layout, treated_at)
        started = parse_time(plate.get("acquired_at"))
        if treated is not None and started is not None:
            offset = (started - treated).total_seconds() / 3600
            table.insert(4, "hours_since_treatment", table["time_h"] + offset)
    layout = _layout(folder)
    if layout is not None:
        from hcs_studio.io.layout_io import annotation_rows

        annotation = annotation_rows(layout)
        if annotation:
            extra = pd.DataFrame.from_dict(annotation, orient="index")
            extra.index.name = "well"
            table = table.merge(extra.reset_index(), on="well", how="left")
    return table


def time_info(folder: Path, treated_at: str | None = None) -> dict[str, Any]:
    """Times of a run for reports: source, start, treatment time and its source."""
    manifest = json.loads((Path(folder) / "run_manifest.json").read_text(encoding="utf-8"))
    plate = manifest.get("plate") or {}
    treated, source = treatment_time(manifest, _layout(Path(folder)), treated_at)
    return {
        "timepoints": plate.get("timepoints", 1),
        "time_points_s": plate.get("time_points_s"),
        "time_source": plate.get("time_source", ""),
        "acquired_at": plate.get("acquired_at"),
        "treated_at": treated.isoformat() if treated else None,
        "treatment_source": source,
    }


def _identity(manifest: dict[str, Any]) -> str:
    """The plate a run is of: its barcode, else its name."""
    plate = manifest.get("plate") or {}
    return str(plate.get("barcode") or plate.get("name") or "")


def assemble(folders: list[Path], treated_at: str | None = None) -> pd.DataFrame:
    """One time course from several runs of the same plate (separate acquisitions), ordered by
    acquisition time; each run contributes its time course (time-lapse runs) or its per-well
    summary (single time point). Runs of different plates (barcode, else plate name) are refused."""
    runs = []
    for folder in folders:
        manifest = json.loads((Path(folder) / "run_manifest.json").read_text(encoding="utf-8"))
        plate = manifest.get("plate") or {}
        started = parse_time(plate.get("acquired_at"))
        if started is None:
            raise ValueError(f"{Path(folder).name}: no acquisition time in the run manifest")
        runs.append((started, Path(folder), manifest))
    plates = {_identity(m) for _, _, m in runs}
    if len(plates) > 1:
        raise ValueError(f"The runs are of different plates: {sorted(plates)}")
    runs.sort(key=lambda r: r[0])
    first = runs[0][0]
    parts = []
    for index, (started, folder, _manifest) in enumerate(runs):
        course = build_time_course(folder, treated_at=treated_at)
        if course is None:
            summary = pd.read_csv(folder / "tables" / "per_well_summary.csv")
            front = pd.DataFrame({"well": summary["well"], "timepoint": 0, "time_s": 0.0})
            course = pd.concat([front, summary.drop(columns="well")], axis=1)
        offset = (started - first).total_seconds()
        course = course.copy()
        course["time_s"] = course["time_s"].astype(float) + offset
        course["time_h"] = course["time_s"] / 3600
        course["acquisition"] = index
        course["run"] = folder.name
        parts.append(course)
    table = pd.concat(parts, ignore_index=True)
    treated, _ = treatment_time(runs[0][2], _layout(runs[0][1]), treated_at)
    if treated is not None:
        table["hours_since_treatment"] = table["time_h"] + (first - treated).total_seconds() / 3600
    order = ["well", "acquisition", "timepoint", "time_s", "time_h"]
    order += [c for c in ("hours_since_treatment",) if c in table]
    table["timepoint"] = table.groupby("well").cumcount()
    return (
        table[order + [c for c in table.columns if c not in order]]
        .sort_values(["well", "time_s"])
        .reset_index(drop=True)
    )


def readout_matrix(table: pd.DataFrame, readout: str) -> pd.DataFrame:
    """Wells × time (h) matrix of one readout (NaN where a time point is missing)."""
    time = "hours_since_treatment" if "hours_since_treatment" in table else "time_h"
    matrix = table.pivot_table(index="well", columns=time, values=readout, aggfunc="mean")
    matrix.columns = [round(float(c), 4) if math.isfinite(float(c)) else c for c in matrix.columns]
    return matrix.astype(float).replace([np.inf, -np.inf], np.nan)
