"""Index run folders and raw plate folders into the registry (Phase 17a).

Re-running a scan is safe: plates, acquisitions and runs have stable keys (barcode or name +
rooted raw path; rooted raw path + time point; rooted run folder), and a run whose files
did not change since the last scan (its *fingerprint*) is skipped. A changed run replaces
its own results only. Nothing in the scanned folders is modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from hcs_studio.db.catalog import (
    Acquisition,
    CellLine,
    CombinationScore,
    Compound,
    DoseResponseFit,
    Plate,
    PlateQCRecord,
    PlateWell,
    Run,
    WellResult,
)
from hcs_studio.paths import to_rooted

log = logging.getLogger(__name__)
MANIFEST = "run_manifest.json"
#: Sub-folders of a run that never contain other runs (skipped while walking).
RUN_PARTS = {"wells", "tables", "state", "logs", "pharmacology", "advanced", "figures", "qc"}
PHARM_FILES = ("dose_response_fits.csv", "combination_scores.csv", "plate_qc.csv")


@dataclass
class ScanReport:
    runs_added: list[str] = field(default_factory=list)
    runs_updated: list[str] = field(default_factory=list)
    runs_unchanged: list[str] = field(default_factory=list)
    plates_added: list[str] = field(default_factory=list)
    raw_acquisitions_added: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    #: Numeric per-well columns not in the readout registry (stored all the same).
    unregistered: set[str] = field(default_factory=set)

    @property
    def changed(self) -> bool:
        return bool(
            self.runs_added or self.runs_updated or self.plates_added or self.raw_acquisitions_added
        )

    def text(self) -> str:
        lines = [
            f"Runs: {len(self.runs_added)} added, {len(self.runs_updated)} updated, "
            f"{len(self.runs_unchanged)} unchanged",
            f"Plates added: {len(self.plates_added)}; raw acquisitions added: "
            f"{len(self.raw_acquisitions_added)}",
        ]
        if self.unregistered:
            names = ", ".join(sorted(self.unregistered)[:10])
            lines.append(f"Readouts not in the registry (stored as they are): {names}")
        lines += [f"ERROR {path}: {message}" for path, message in self.errors.items()]
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "runs_added": len(self.runs_added),
            "runs_updated": len(self.runs_updated),
            "runs_unchanged": len(self.runs_unchanged),
            "plates_added": len(self.plates_added),
            "raw_acquisitions_added": len(self.raw_acquisitions_added),
            "errors": len(self.errors),
        }


# ------------------------------------------------------------------- finding


def find_run_folders(paths: Iterable[Path]) -> Iterator[Path]:
    """Run folders (with ``run_manifest.json``) at or below each path."""
    for path in paths:
        path = Path(path)
        if (path / MANIFEST).is_file():
            yield path
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            if MANIFEST in filenames:
                yield Path(dirpath)
                dirnames[:] = []  # a run folder contains no other runs
                continue
            dirnames[:] = sorted(
                d for d in dirnames if d not in RUN_PARTS and not d.startswith(".")
            )


def _rooted(path: str | Path) -> str:
    return str(to_rooted(Path(path).absolute()))


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def fingerprint(folder: Path) -> str:
    """Changes when the manifest, the per-well table, the layout, pharmacology or image QC
    change."""
    digest = hashlib.sha256()
    files = [
        folder / MANIFEST,
        folder / "tables" / "per_well_summary.csv",
        folder / "plate_layout.json",
        *(folder / "pharmacology" / name for name in PHARM_FILES),
        folder / "qc" / "plate_qc.json",
        folder / "qc" / "well_qc.csv",
    ]
    for file in files:
        try:
            stat = file.stat()
        except OSError:
            digest.update(f"{file.name}:missing".encode())
            continue
        digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


# ------------------------------------------------------------------ indexing


def _get_or_create(
    session: Session, model: Any, defaults: dict[str, Any] | None = None, **keys: Any
) -> tuple[Any, bool]:
    found = session.scalars(select(model).filter_by(**keys)).first()
    if found is not None:
        return found, False
    obj = model(**keys, **(defaults or {}))
    session.add(obj)
    session.flush()
    return obj, True


def _plate_for(session: Session, manifest: dict[str, Any], report: ScanReport) -> Plate:
    info = manifest.get("plate", {})
    barcode = str(info.get("barcode") or "")
    raw = info.get("rooted_path") or (_rooted(info["path"]) if info.get("path") else "")
    key = f"barcode:{barcode}" if barcode else f"name:{info.get('name', '')}|{raw}"
    found, created = _get_or_create(session, Plate, key=key)
    plate: Plate = found
    plate.barcode = barcode
    plate.name = str(info.get("name") or barcode)
    plate.rows = int(info.get("rows") or plate.rows or 0)
    plate.cols = int(info.get("cols") or plate.cols or 0)
    if created:
        report.plates_added.append(plate.barcode or plate.name)
    return plate


def _acquisition_for(
    session: Session, plate: Plate, manifest: dict[str, Any]
) -> Acquisition | None:
    info = manifest.get("plate", {})
    raw = info.get("rooted_path") or (_rooted(info["path"]) if info.get("path") else "")
    if not raw:
        return None
    found, _ = _get_or_create(
        session,
        Acquisition,
        key=f"{raw}|t",
        defaults={"plate_id": plate.id, "raw_path": raw},
    )
    acquisition: Acquisition = found
    acquisition.instrument = acquisition.instrument or str(info.get("vendor") or "")
    acquisition.plate_id = plate.id
    return acquisition


def _layout_wells(session: Session, plate: Plate, folder: Path) -> None:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file
    from hcs_studio.model.plate import well_name
    from hcs_studio.pharmacology.normalize import drug_treatments

    path = run_layout_file(folder)
    if path is None:
        return
    try:
        layout = load_plate_layout(path)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("Layout of %s not read: %s", folder, exc)
        return
    from hcs_studio.io.layout_store import layout_sha256

    plate.layout_version = layout.version or None
    plate.layout_sha256 = layout_sha256(layout)
    plate.assay = plate.assay or (layout.assay or "")
    session.execute(delete(PlateWell).where(PlateWell.plate_id == plate.id))
    lines: dict[str, CellLine] = {}
    rows = []
    for (row, col), annotation in sorted(layout.wells.items()):
        treatments = [[name, conc] for name, conc, _ in drug_treatments(annotation)]
        rows.append(
            {
                "plate_id": plate.id,
                "well": well_name(row, col),
                "cell_line": annotation.cell_line,
                "role": annotation.role,
                "treatment": annotation.treatment,
                "compounds": treatments,
            }
        )
        if annotation.cell_line and annotation.cell_line not in lines:
            lines[annotation.cell_line] = _get_or_create(
                session, CellLine, name=annotation.cell_line
            )[0]
        for name, _ in treatments:
            _get_or_create(session, Compound, name=name)
    if rows:
        session.execute(insert(PlateWell), rows)
    plate.cell_lines = sorted(lines.values(), key=lambda c: c.name)


def _image_qc(folder: Path) -> dict[str, Any]:
    """The run's image QC summary (``qc/plate_qc.json``), or {}."""
    path = folder / "qc" / "plate_qc.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys = ("passed", "reasons", "fields", "excluded_fields", "wells", "failed_wells",
            "flag_counts", "compound_flags", "rules")  # fmt: skip
    return {k: data.get(k) for k in keys}


def _well_results(session: Session, run: Run, folder: Path, report: ScanReport) -> int:
    from hcs_studio.io.layout_io import is_annotation_column
    from hcs_studio.readouts import resolve

    table = folder / "tables" / "per_well_summary.csv"
    if not table.exists():
        return 0
    frame = pd.read_csv(table)
    if "well" not in frame:
        return 0
    numeric = [
        str(c)
        for c in frame.select_dtypes(include=["number", "bool"]).columns
        if not is_annotation_column(str(c), None) and str(c) != "qc_failed"
    ]
    readouts = {c: resolve(c, "well") for c in numeric}
    flags = pd.Series("", index=frame.index, dtype=object)
    if "low_count" in frame:
        flags[frame["low_count"].fillna(False).astype(bool)] = "low_count"
    if "fields" in frame:
        flags[frame["fields"].fillna(0) == 0] = "no_fields"
    if "qc_failed" in frame:
        flags[frame["qc_failed"].fillna(False).astype(bool)] = "qc_failed"
    if "role" in frame:
        flags[frame["role"].astype(str) == "excluded"] = "excluded"
    report.unregistered |= {c for c, r in readouts.items() if r is None}
    rows: list[dict[str, Any]] = []
    for column, readout in readouts.items():
        labels = readout.values.get("labels") if readout is not None else None
        counts = frame.get(f"{labels}_count") if labels else None
        for i, value in enumerate(frame[column].to_numpy()):
            rows.append(
                {
                    "run_id": run.id,
                    "well": str(frame["well"].iat[i]),
                    "readout_id": column,
                    "value": _float(value),
                    "n_objects": _int(counts.iat[i]) if counts is not None else None,
                    "qc_flag": flags.iat[i],
                }
            )
    for start in range(0, len(rows), 20000):
        session.execute(insert(WellResult), rows[start : start + 20000])
    return len(rows)


def _pharmacology(session: Session, run: Run, folder: Path) -> None:
    pharm = folder / "pharmacology"
    fits = pharm / "dose_response_fits.csv"
    if fits.exists():
        rows = []
        for r in pd.read_csv(fits).to_dict("records"):
            extra = {
                k: v
                for k, v in r.items()
                if str(k).startswith(("GR", "GI", "GEC", "h_GR")) and _float(v) is not None
            }
            rows.append(
                {
                    "run_id": run.id,
                    "readout_id": str(r.get("readout", "")),
                    "cell_line": str(r.get("cell_line") or ""),
                    "compound": str(r.get("compound") or ""),
                    "unit": str(r.get("unit") or ""),
                    "model": str(r.get("model") or ""),
                    "ec50_m": _float(r.get("ec50")),
                    "ec50_ci_low_m": _float(r.get("ec50_ci_low")),
                    "ec50_ci_high_m": _float(r.get("ec50_ci_high")),
                    "log_ec50": _float(r.get("log_ec50")),
                    "abs_ic50_m": _float(r.get("abs_ic50")),
                    "hill": _float(r.get("hill")),
                    "top": _float(r.get("top")),
                    "bottom": _float(r.get("bottom")),
                    "emax": _float(r.get("emax_fitted")),
                    "auc": _float(r.get("auc")),
                    "r2": _float(r.get("r2")),
                    "n_points": _int(r.get("n_points")),
                    "flags": "" if pd.isna(r.get("flags")) else str(r.get("flags")),
                    "extra": {str(k): float(v) for k, v in extra.items()},
                }
            )
        if rows:
            session.execute(insert(DoseResponseFit), rows)
    combos = pharm / "combination_scores.csv"
    if combos.exists():
        rows = []
        for r in pd.read_csv(combos).to_dict("records"):
            for model in ("bliss", "hsa", "loewe", "zip"):
                if f"{model}_mean" not in r:
                    continue
                rows.append(
                    {
                        "run_id": run.id,
                        "readout_id": str(r.get("readout", "")),
                        "cell_line": str(r.get("cell_line") or ""),
                        "drug_a": str(r.get("drug_a") or ""),
                        "drug_b": str(r.get("drug_b") or ""),
                        "model": model,
                        "mean": _float(r.get(f"{model}_mean")),
                        "max_3x3": _float(r.get(f"{model}_max_3x3")),
                        "p": _float(r.get(f"{model}_p")),
                        "label": ""
                        if pd.isna(r.get(f"{model}_label"))
                        else str(r.get(f"{model}_label")),
                    }
                )
        if rows:
            session.execute(insert(CombinationScore), rows)
    qc = pharm / "plate_qc.csv"
    if qc.exists():
        rows = [
            {
                "run_id": run.id,
                "readout_id": str(r.get("readout", "")),
                "z_prime": _float(r.get("z_prime")),
                "ssmd": _float(r.get("ssmd")),
                "signal_window": _float(r.get("signal_window")),
                "cv_vehicle_pct": _float(r.get("cv_vehicle_pct")),
                "cv_positive_pct": _float(r.get("cv_positive_pct")),
                "vehicle_mean": _float(r.get("vehicle_mean")),
                "positive_mean": _float(r.get("positive_mean")),
                "n_vehicle": _int(r.get("n_vehicle")),
                "n_positive": _int(r.get("n_positive")),
                "flag": "" if pd.isna(r.get("flag")) else str(r.get("flag")),
            }
            for r in pd.read_csv(qc).to_dict("records")
        ]
        if rows:
            session.execute(insert(PlateQCRecord), rows)


def index_run(session: Session, folder: Path, report: ScanReport, force: bool = False) -> Run:
    folder = Path(folder).absolute()
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    key = _rooted(folder)
    print_ = fingerprint(folder)
    run = session.scalars(select(Run).where(Run.folder == key)).first()
    if run is not None and run.fingerprint == print_ and not force:
        report.runs_unchanged.append(key)
        return run
    is_new = run is None
    if run is None:
        run = Run(folder=key)
        session.add(run)
        session.flush()
    else:  # the run changed: replace its results
        for model in (WellResult, DoseResponseFit, CombinationScore, PlateQCRecord):
            session.execute(delete(model).where(model.run_id == run.id))
    plate = _plate_for(session, manifest, report)
    acquisition = _acquisition_for(session, plate, manifest)
    software = manifest.get("software", {})
    pipeline = manifest.get("pipeline", {})
    history = manifest.get("history") or []
    run.name = folder.name
    run.plate_id = plate.id
    run.pipeline_name = str(pipeline.get("name", ""))
    run.pipeline_hash = str(manifest.get("pipeline_hash", ""))
    run.software_version = str(software.get("version", ""))
    run.git_commit = str(software.get("git_commit") or "")
    run.models = list(manifest.get("models") or [])
    run.status = str(manifest.get("status", ""))
    run.started = _parse_time(
        manifest.get("started") or (history[0].get("started") if history else None)
    )
    run.finished = _parse_time(manifest.get("finished"))
    run.actor = str(manifest.get("actor", ""))
    run.fingerprint = print_
    run.image_qc = _image_qc(folder)
    run.indexed_at = datetime.now().astimezone()
    if acquisition is not None and acquisition not in run.acquisitions:
        run.acquisitions.append(acquisition)
    _layout_wells(session, plate, folder)
    _well_results(session, run, folder, report)
    _pharmacology(session, run, folder)
    (report.runs_added if is_new else report.runs_updated).append(key)
    return run


def index_raw_plate(session: Session, path: Path, report: ScanReport) -> Acquisition | None:
    """Register a raw plate folder (no analysis yet) as a plate + acquisition."""
    from hcs_studio.io.base import open_plate

    plate_info = open_plate(path)
    raw = _rooted(plate_info.path)
    manifest = {
        "plate": {
            "name": plate_info.name,
            "barcode": plate_info.barcode,
            "rows": plate_info.rows,
            "cols": plate_info.cols,
            "vendor": plate_info.vendor,
            "rooted_path": raw,
        }
    }
    plate = _plate_for(session, manifest, report)
    existing = session.scalars(select(Acquisition).where(Acquisition.key == f"{raw}|t")).first()
    acquisition = _acquisition_for(session, plate, manifest)
    if existing is None and acquisition is not None:
        report.raw_acquisitions_added.append(raw)
    return acquisition


def scan(
    session: Session,
    paths: Iterable[Path],
    *,
    raw: bool = False,
    force: bool = False,
) -> ScanReport:
    """Index every run folder under ``paths`` (and, with ``raw``, the raw plates given)."""
    report = ScanReport()
    paths = [Path(p) for p in paths]
    for folder in find_run_folders(paths):
        try:
            with session.begin_nested():
                index_run(session, folder, report, force=force)
        except Exception as exc:
            report.errors[str(folder)] = f"{type(exc).__name__}: {exc}"
            log.exception("Indexing %s failed", folder)
    if raw:
        from hcs_studio.io.base import detect_reader

        for path in paths:
            if (path / MANIFEST).exists() or detect_reader(path) is None:
                continue
            try:
                with session.begin_nested():
                    index_raw_plate(session, path, report)
            except Exception as exc:
                report.errors[str(path)] = f"{type(exc).__name__}: {exc}"
    return report
