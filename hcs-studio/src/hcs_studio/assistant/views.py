"""Curated views of the registry: the only data the lab assistant can see (Phase 28.2–28.3).

The assistant never reads the registry tables. :func:`build_snapshot` copies a documented,
privacy-filtered subset into a separate SQLite file, which the assistant then queries through a
read-only connection (:mod:`hcs_studio.assistant.sql`). In the snapshot:

- file paths, folders and user names are left out; runs, plates and experiments are identified
  by their registry IDs;
- free-text notes (experiment purpose and notes, plate notes) are left out unless
  ``Privacy.include_notes`` is set for the question;
- compound structures (SMILES), suppliers and lots are never included;
- with ``Privacy.compound_ids_only`` every compound name is replaced by a code (``CPD-0007``),
  shuffled per snapshot; the app maps codes back to names on screen only
  (:meth:`Pseudonyms.reveal`).

Add a view here (with its documentation) before a tool may use new data.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_FORMAT = "hcs-studio-assistant-views/1"


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # INTEGER | REAL | TEXT
    description: str


@dataclass(frozen=True)
class View:
    name: str
    description: str
    columns: tuple[Column, ...]
    #: free-text columns, filled only with ``Privacy.include_notes``
    notes: tuple[str, ...] = ()


def _c(name: str, type_: str, description: str) -> Column:
    return Column(name, type_, description)


VIEWS: tuple[View, ...] = (
    View(
        "experiments",
        "Experiments (groups of plates).",
        (
            _c("experiment_id", "INTEGER", "registry ID"),
            _c("name", "TEXT", "experiment name"),
            _c("project", "TEXT", "project name"),
            _c("assay", "TEXT", "assay name"),
            _c("start_date", "TEXT", "ISO date"),
            _c("end_date", "TEXT", "ISO date"),
            _c("purpose", "TEXT", "free text (only when notes are included)"),
            _c("notes", "TEXT", "free text (only when notes are included)"),
        ),
        notes=("purpose", "notes"),
    ),
    View(
        "plates",
        "Assay plates.",
        (
            _c("plate_id", "INTEGER", "registry ID"),
            _c("barcode", "TEXT", "plate barcode"),
            _c("name", "TEXT", "plate name"),
            _c("plate_type", "INTEGER", "number of wells (96, 384, 1536)"),
            _c("assay", "TEXT", "assay name"),
            _c("experiment_id", "INTEGER", "experiments.experiment_id"),
            _c("cell_lines", "TEXT", "cell lines on the plate, comma separated"),
            _c("seeded_at", "TEXT", "ISO time (UTC)"),
            _c("treated_at", "TEXT", "ISO time (UTC)"),
            _c("notes", "TEXT", "free text (only when notes are included)"),
        ),
        notes=("notes",),
    ),
    View(
        "runs",
        "Analysis runs of plates (a plate can have several runs).",
        (
            _c("run_id", "INTEGER", "registry ID"),
            _c("plate_id", "INTEGER", "plates.plate_id"),
            _c("name", "TEXT", "run name"),
            _c("pipeline_name", "TEXT", "pipeline name"),
            _c("pipeline_hash", "TEXT", "pipeline fingerprint (same hash = same processing)"),
            _c("software_version", "TEXT", "HCS Studio version"),
            _c("status", "TEXT", "completed | completed_with_failures | failed | …"),
            _c("started", "TEXT", "ISO time (UTC)"),
            _c("finished", "TEXT", "ISO time (UTC)"),
            _c("instrument", "TEXT", "microscope of the acquisition"),
            _c("image_qc_passed", "INTEGER", "1 = plate passed image QC, 0 = failed, NULL = no QC"),
        ),
    ),
    View(
        "wells",
        "Plate layout: what is in each well. Concentrations in mol/L (M).",
        (
            _c("plate_id", "INTEGER", "plates.plate_id"),
            _c("well", "TEXT", "well name, e.g. B07"),
            _c("cell_line", "TEXT", "cell line"),
            _c(
                "role",
                "TEXT",
                "sample | negative_control | positive_control | untreated | blank "
                "| empty | excluded | unstained",
            ),
            _c(
                "n_compounds", "INTEGER", "0 = vehicle/untreated, 1 = single agent, 2 = combination"
            ),
            _c("compound_1", "TEXT", "first compound (alphabetical), NULL if none"),
            _c("concentration_1_m", "REAL", "concentration of compound_1 in M"),
            _c("compound_2", "TEXT", "second compound, NULL if none"),
            _c("concentration_2_m", "REAL", "concentration of compound_2 in M"),
        ),
    ),
    View(
        "well_results",
        "Per-well results of runs (one row per run, well and readout).",
        (
            _c("run_id", "INTEGER", "runs.run_id"),
            _c("well", "TEXT", "well name"),
            _c("readout", "TEXT", "readout ID (see readouts)"),
            _c("value", "REAL", "the well's value"),
            _c("n_objects", "INTEGER", "objects measured in the well"),
            _c("qc_flag", "TEXT", "'' = ok, else the QC flag (e.g. qc_failed)"),
        ),
    ),
    View(
        "readouts",
        "Readouts present in well_results, with display names and units.",
        (
            _c("readout", "TEXT", "readout ID"),
            _c("display", "TEXT", "display name"),
            _c("unit", "TEXT", "unit"),
            _c("level", "TEXT", "object | field | well | fit"),
            _c("direction", "TEXT", "up | down | either | none (which way is 'more effect')"),
            _c("description", "TEXT", "what it measures"),
            _c("n_runs", "INTEGER", "runs that have it"),
        ),
    ),
    View(
        "dose_response",
        "Dose-response fits (one per run, readout, cell line and compound). Concentrations in M.",
        (
            _c("run_id", "INTEGER", "runs.run_id"),
            _c("readout", "TEXT", "readout ID"),
            _c("cell_line", "TEXT", "cell line"),
            _c("compound", "TEXT", "compound"),
            _c("model", "TEXT", "4PL | 5PL | 3PL | …"),
            _c("ec50_m", "REAL", "EC50/IC50 in M"),
            _c("ec50_ci_low_m", "REAL", "95 % CI low"),
            _c("ec50_ci_high_m", "REAL", "95 % CI high"),
            _c("log_ec50", "REAL", "log10 EC50 (M)"),
            _c("abs_ic50_m", "REAL", "absolute IC50 (50 % effect) in M"),
            _c("hill", "REAL", "Hill slope"),
            _c("top", "REAL", "top plateau"),
            _c("bottom", "REAL", "bottom plateau"),
            _c("emax", "REAL", "maximal effect"),
            _c("auc", "REAL", "area under the curve"),
            _c("r2", "REAL", "R²"),
            _c("n_points", "INTEGER", "points fitted"),
            _c("flags", "TEXT", "fit warnings"),
        ),
    ),
    View(
        "combination_scores",
        "Drug-combination synergy scores (one per run, readout, cell line, pair and model).",
        (
            _c("run_id", "INTEGER", "runs.run_id"),
            _c("readout", "TEXT", "readout ID"),
            _c("cell_line", "TEXT", "cell line"),
            _c("drug_a", "TEXT", "first drug"),
            _c("drug_b", "TEXT", "second drug"),
            _c("model", "TEXT", "bliss | hsa | loewe | zip"),
            _c("mean", "REAL", "mean synergy score over the matrix (> 0 = synergy)"),
            _c("max_3x3", "REAL", "most synergistic 3×3 window"),
            _c("p", "REAL", "p value"),
            _c("label", "TEXT", "synergistic | additive | antagonistic"),
        ),
    ),
    View(
        "plate_qc",
        "Plate QC per run and readout (controls).",
        (
            _c("run_id", "INTEGER", "runs.run_id"),
            _c("readout", "TEXT", "readout ID"),
            _c("z_prime", "REAL", "Z′ factor"),
            _c("ssmd", "REAL", "SSMD"),
            _c("signal_window", "REAL", "signal window"),
            _c("cv_vehicle_pct", "REAL", "CV of vehicle wells, %"),
            _c("cv_positive_pct", "REAL", "CV of positive controls, %"),
            _c("vehicle_mean", "REAL", "mean of vehicle wells"),
            _c("positive_mean", "REAL", "mean of positive controls"),
            _c("n_vehicle", "INTEGER", "vehicle wells"),
            _c("n_positive", "INTEGER", "positive-control wells"),
            _c("flag", "TEXT", "QC flag"),
        ),
    ),
    View(
        "compounds",
        "Compounds known to the registry (no structures).",
        (
            _c("compound", "TEXT", "compound name (or code)"),
            _c("moa", "TEXT", "mechanism of action"),
            _c("target", "TEXT", "target"),
        ),
    ),
    View(
        "cell_lines",
        "Cell lines.",
        (
            _c("cell_line", "TEXT", "name"),
            _c("rrid", "TEXT", "Cellosaurus RRID"),
            _c("cancer_type", "TEXT", "cancer type"),
        ),
    ),
)

VIEW_NAMES = tuple(v.name for v in VIEWS)


@dataclass(frozen=True)
class Privacy:
    """What may leave the app for one question."""

    compound_ids_only: bool = False
    include_notes: bool = False


class Pseudonyms:
    """Compound names ↔ codes for ``compound_ids_only``. Codes are shuffled per snapshot so
    their order says nothing about the names."""

    def __init__(self, names: Iterable[str] = (), seed: int | None = None) -> None:
        unique = sorted({n for n in names if n})
        random.Random(seed).shuffle(unique)
        self.code = {name: f"CPD-{i + 1:04d}" for i, name in enumerate(unique)}
        self.name = {c: n for n, c in self.code.items()}

    def __call__(self, name: str | None) -> str | None:
        if not name:
            return name
        return self.code.get(name, name)

    def reveal(self, text: str) -> str:
        """Codes in ``text`` replaced by names (on screen only)."""
        if not self.name:
            return text
        return re.sub(r"CPD-\d{4}", lambda m: self.name.get(m.group(0), m.group(0)), text)

    def to_json(self) -> dict[str, str]:
        return dict(self.name)


@dataclass
class Snapshot:
    """A built snapshot: the SQLite file and how it was made."""

    path: Path
    privacy: Privacy
    pseudonyms: Pseudonyms = field(default_factory=Pseudonyms)
    counts: dict[str, int] = field(default_factory=dict)
    #: run_id → the run folder on this computer (never sent; used by the manifest/image tools)
    run_folders: dict[int, str] = field(default_factory=dict)
    built: str = ""


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _compounds(raw: Any) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and item:
            conc = item[1] if len(item) > 1 else None
            out.append((str(item[0]), float(conc) if conc is not None else None))
    return sorted(out, key=lambda c: c[0])


def _rows(session: Any, privacy: Privacy) -> tuple[dict[str, list[tuple[Any, ...]]], Snapshot]:
    """The views' rows from the registry (names not yet pseudonymised)."""
    from sqlalchemy import func, select

    from hcs_studio.db.catalog import (
        Acquisition,
        CellLine,
        CombinationScore,
        Compound,
        DoseResponseFit,
        Experiment,
        Plate,
        PlateQCRecord,
        PlateWell,
        Run,
        WellResult,
    )

    rows: dict[str, list[tuple[Any, ...]]] = {v.name: [] for v in VIEWS}
    notes = privacy.include_notes
    for exp in session.scalars(select(Experiment).order_by(Experiment.id)):
        rows["experiments"].append(
            (
                exp.id,
                exp.name,
                exp.project.name if exp.project else None,
                exp.assay,
                _iso(exp.start_date),
                _iso(exp.end_date),
                exp.purpose if notes else None,
                exp.notes if notes else None,
            )
        )
    for plate in session.scalars(select(Plate).order_by(Plate.id)):
        rows["plates"].append(
            (
                plate.id,
                plate.barcode,
                plate.name,
                plate.rows * plate.cols or None,
                plate.assay,
                plate.experiment_id,
                ", ".join(sorted(c.name for c in plate.cell_lines)),
                _iso(plate.seeded_at),
                _iso(plate.treated_at),
                plate.notes if notes else None,
            )
        )
    folders: dict[int, str] = {}
    for run in session.scalars(select(Run).order_by(Run.id)):
        instruments = sorted({a.instrument for a in run.acquisitions if a.instrument})
        passed = (run.image_qc or {}).get("passed")
        rows["runs"].append(
            (
                run.id,
                run.plate_id,
                run.name,
                run.pipeline_name,
                run.pipeline_hash,
                run.software_version,
                run.status,
                _iso(run.started),
                _iso(run.finished),
                ", ".join(instruments),
                None if passed is None else int(bool(passed)),
            )
        )
        folders[run.id] = run.folder
    for well in session.scalars(select(PlateWell).order_by(PlateWell.plate_id, PlateWell.id)):
        compounds = _compounds(well.compounds)
        first = compounds[0] if compounds else (None, None)
        second = compounds[1] if len(compounds) > 1 else (None, None)
        rows["wells"].append(
            (well.plate_id, well.well, well.cell_line, well.role, len(compounds), *first, *second)
        )
    rows["well_results"] = [
        tuple(r)
        for r in session.execute(
            select(
                WellResult.run_id,
                WellResult.well,
                WellResult.readout_id,
                WellResult.value,
                WellResult.n_objects,
                WellResult.qc_flag,
            ).order_by(WellResult.run_id, WellResult.id)
        )
    ]
    from hcs_studio.readouts import resolve

    for readout_id, n_runs in session.execute(
        select(WellResult.readout_id, func.count(func.distinct(WellResult.run_id))).group_by(
            WellResult.readout_id
        )
    ):
        spec = resolve(readout_id)
        rows["readouts"].append(
            (
                readout_id,
                spec.display if spec else readout_id,
                spec.unit if spec else "",
                spec.spec.level if spec else "",
                spec.spec.direction if spec else "",
                spec.spec.description if spec else "",
                n_runs,
            )
        )
    for fit in session.scalars(select(DoseResponseFit).order_by(DoseResponseFit.id)):
        rows["dose_response"].append(
            (
                fit.run_id,
                fit.readout_id,
                fit.cell_line,
                fit.compound,
                fit.model,
                fit.ec50_m,
                fit.ec50_ci_low_m,
                fit.ec50_ci_high_m,
                fit.log_ec50,
                fit.abs_ic50_m,
                fit.hill,
                fit.top,
                fit.bottom,
                fit.emax,
                fit.auc,
                fit.r2,
                fit.n_points,
                fit.flags,
            )
        )
    for s in session.scalars(select(CombinationScore).order_by(CombinationScore.id)):
        rows["combination_scores"].append(
            (
                s.run_id,
                s.readout_id,
                s.cell_line,
                s.drug_a,
                s.drug_b,
                s.model,
                s.mean,
                s.max_3x3,
                s.p,
                s.label,
            )
        )
    for q in session.scalars(select(PlateQCRecord).order_by(PlateQCRecord.id)):
        rows["plate_qc"].append(
            (
                q.run_id,
                q.readout_id,
                q.z_prime,
                q.ssmd,
                q.signal_window,
                q.cv_vehicle_pct,
                q.cv_positive_pct,
                q.vehicle_mean,
                q.positive_mean,
                q.n_vehicle,
                q.n_positive,
                q.flag,
            )
        )
    for c in session.scalars(select(Compound).order_by(Compound.name)):
        rows["compounds"].append((c.name, c.moa, c.target))
    for line in session.scalars(select(CellLine).order_by(CellLine.name)):
        rows["cell_lines"].append((line.name, line.rrid, line.cancer_type))
    del Acquisition  # imported for the relationship only
    return rows, Snapshot(path=Path(), privacy=privacy, run_folders=folders)


#: (view, column) pairs holding compound names
COMPOUND_COLUMNS = {
    ("wells", "compound_1"),
    ("wells", "compound_2"),
    ("dose_response", "compound"),
    ("combination_scores", "drug_a"),
    ("combination_scores", "drug_b"),
    ("compounds", "compound"),
}


def _view(name: str) -> View:
    return next(v for v in VIEWS if v.name == name)


def build_snapshot(
    path: Path, privacy: Privacy = Privacy(), session: Any = None, seed: int | None = None
) -> Snapshot:
    """Write the curated views of the registry to the SQLite file ``path`` (replaced)."""
    from hcs_studio.db import session_scope

    if session is None:
        with session_scope() as own:
            rows, snapshot = _rows(own, privacy)
    else:
        rows, snapshot = _rows(session, privacy)
    names: set[str] = set()
    if privacy.compound_ids_only:
        for view_name, column in COMPOUND_COLUMNS:
            at = [c.name for c in _view(view_name).columns].index(column)
            names.update(str(r[at]) for r in rows[view_name] if r[at])
    pseudonyms = Pseudonyms(names, seed=seed) if privacy.compound_ids_only else Pseudonyms()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    db = sqlite3.connect(partial)
    try:
        for view in VIEWS:
            columns = [c.name for c in view.columns]
            db.execute(
                f'CREATE TABLE "{view.name}" ('
                + ", ".join(f'"{c.name}" {c.type}' for c in view.columns)
                + ")"
            )
            coded = {c for v, c in COMPOUND_COLUMNS if v == view.name}
            data = rows[view.name]
            if coded and privacy.compound_ids_only:
                positions = [columns.index(c) for c in coded]
                data = [
                    tuple(pseudonyms(v) if i in positions else v for i, v in enumerate(r))
                    for r in data
                ]
            if view.name == "compounds" and privacy.compound_ids_only:
                # codes only: names that never occur in results stay out entirely
                data = [r for r in data if str(r[0]).startswith("CPD-")]
            db.executemany(
                f'INSERT INTO "{view.name}" VALUES ({", ".join("?" * len(columns))})', data
            )
            snapshot.counts[view.name] = len(data)
        db.execute("CREATE TABLE _snapshot (key TEXT, value TEXT)")
        db.execute("INSERT INTO _snapshot VALUES ('format', ?)", (SNAPSHOT_FORMAT,))
        db.commit()
    finally:
        db.close()
    partial.replace(path)
    snapshot.path = path
    snapshot.pseudonyms = pseudonyms
    snapshot.built = datetime.now().astimezone().isoformat(timespec="seconds")
    return snapshot


def schema_text(privacy: Privacy = Privacy(), views: Sequence[View] = VIEWS) -> str:
    """The documentation of the views (what the assistant is told; stable for caching)."""
    lines = []
    for view in views:
        lines.append(f"{view.name}: {view.description}")
        for column in view.columns:
            if column.name in view.notes and not privacy.include_notes:
                continue
            lines.append(f"  - {column.name} {column.type}: {column.description}")
    extra = []
    if privacy.compound_ids_only:
        extra.append(
            "Compound names are replaced by codes (CPD-0001 …); use the codes as given. "
            "The user sees the real names."
        )
    if not privacy.include_notes:
        extra.append("Free-text notes are not available.")
    return "\n".join(lines + extra)


def schema_json(privacy: Privacy = Privacy()) -> list[dict[str, Any]]:
    return [
        {
            "view": v.name,
            "description": v.description,
            "columns": [
                {"name": c.name, "type": c.type, "description": c.description}
                for c in v.columns
                if privacy.include_notes or c.name not in v.notes
            ],
        }
        for v in VIEWS
    ]


_PATHLIKE = re.compile(r"(^[A-Za-z]:[\\/])|(^\\\\)|(^/)|(^~[\\/])|([\\/][^\\/\s]+[\\/])")


def scrub(value: Any, pseudonyms: Pseudonyms | None = None, depth: int = 0) -> Any:
    """A JSON value with path-like strings replaced by ``[path]`` (manifests, QC summaries),
    compound names coded when ``pseudonyms`` has codes, and nesting capped."""
    if depth > 8:
        return "[…]"
    if isinstance(value, dict):
        return {
            str(k): scrub(v, pseudonyms, depth + 1)
            for k, v in value.items()
            if not _sensitive_key(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [scrub(v, pseudonyms, depth + 1) for v in value]
    if isinstance(value, str):
        if _PATHLIKE.search(value) and not value.startswith("registry:"):
            return "[path]"
        if pseudonyms is not None and pseudonyms.code:
            for name, code in pseudonyms.code.items():
                if name in value:
                    value = value.replace(name, code)
        return value
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return scrub(json.loads(json.dumps(value, default=str)), pseudonyms, depth + 1)


#: keys never sent (who ran it, where, free text)
SENSITIVE_KEYS = (
    "path",
    "folder",
    "file",
    "user",
    "operator",
    "actor",
    "owner",
    "host",
    "hostname",
    "notes",
    "note",
    "comment",
    "email",
    "smiles",
    "rooted",
    "source",
    "root",
    "dir",
)


def _sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(
        lower == k or lower.endswith("_" + k) or lower.startswith(k + "_") for k in SENSITIVE_KEYS
    )
