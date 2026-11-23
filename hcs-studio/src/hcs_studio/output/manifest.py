"""Run provenance: everything needed to know how an output was produced."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio import __version__
from hcs_studio.model.plate import Plate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Commit of the source checkout the software runs from, if it is one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def describe_inputs(files: list[Path], checksums: bool) -> list[dict[str, Any]]:
    entries = []
    for file in sorted(set(files)):
        stat = file.stat()
        entry: dict[str, Any] = {
            "path": str(file),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        }
        if checksums:
            entry["sha256"] = sha256_file(file)
        entries.append(entry)
    return entries


def build_manifest(
    plate: Plate,
    pipeline: dict[str, Any],
    *,
    status: str,
    started: datetime,
    wells: list[str],
    inputs: list[dict[str, Any]],
    fitted: list[dict[str, Any]],
    failures: dict[str, str],
    overrides: dict[str, Any],
    checksums: bool,
) -> dict[str, Any]:
    return {
        "software": {
            "name": "hcs-studio",
            "version": __version__,
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "status": status,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat() if status != "running" else None,
        "plate": {
            "name": plate.name,
            "path": str(plate.path),
            "rooted_path": rooted_text(plate.path),
            "vendor": plate.vendor,
            "barcode": plate.barcode,
            "rows": plate.rows,
            "cols": plate.cols,
            "metadata": asdict(plate.metadata),
            "metadata_overrides": overrides,
            "reader_notes": list(plate.notes),
            "source": plate.source,
            **_times(plate),
        },
        "pipeline": pipeline,
        "fitted_state": fitted,
        "models": [],
        "wells": wells,
        "failed_wells": failures,
        "inputs": inputs,
        "input_checksums": "sha256" if checksums else "not computed (size and time only)",
    }


def _times(plate: Any) -> dict[str, Any]:
    """Time points of the plate (seconds since the first), their source and the start."""
    from hcs_studio.model.plate import plate_times

    times, source = plate_times(plate)
    return {
        "timepoints": plate.shape_tczyx[0] if plate.shape_tczyx else 1,
        "time_points_s": list(times) if times is not None else None,
        "time_source": source,
        "acquired_at": plate.acquired_at,
    }


def rooted_text(path: str | Path) -> str:
    """``ROOT:relative/path`` for the manifest (``ABSOLUTE:…`` outside every data root);
    an unreadable settings file never stops a run."""
    from hcs_studio.paths import to_rooted

    try:
        return str(to_rooted(path))
    except Exception:  # settings unreadable: keep the run going
        return f"ABSOLUTE:{path}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
