"""Image QC of a plate from raw data, and the ``qc/`` files (Phase 18).

Fields are read one at a time (first timepoint), so memory stays at one field per worker.
Files: ``qc/field_qc.parquet`` (metrics, flags, severity, exclusion and reasons per field),
``qc/well_qc.csv``, ``qc/plate_qc.json`` (decision, counts, rules name/version/SHA-256,
provenance) and ``qc/rules.yaml`` (the rules used).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio import __version__
from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.plate import Plate, Well
from hcs_studio.qc.decide import QcResult, apply_rules
from hcs_studio.qc.metrics import MetricSettings, field_metrics
from hcs_studio.qc.rules import QcRules, save_rules

QC_FOLDER = "qc"
Progress = Callable[[int, int], None]


def metric_settings(plate: Plate, rules: QcRules) -> MetricSettings:
    return MetricSettings(
        content_channel=rules.settings.content_channel,
        nucleus_diameter_um=rules.settings.nucleus_diameter_um,
        pixel_size_um=plate.metadata.pixel_size_um,
        bit_depth=plate.metadata.bit_depth,
        brightfield_channel=rules.settings.brightfield_channel,
    )


def _field_row(well: Well, index: int, settings: MetricSettings) -> dict[str, Any]:
    f = well.fields[index]
    data = np.asarray(f.load()[0])  # (C, Z, Y, X) of the first timepoint
    row: dict[str, Any] = {
        "well": well.name,
        "row": well.row,
        "col": well.col,
        "field": f.index,  # the reader's field number
        "x_um": f.x_um,
        "y_um": f.y_um,
    }
    row.update(field_metrics(data, settings))
    return row


def measure_plate(
    plate: Plate,
    rules: QcRules,
    wells: Sequence[tuple[int, int]] | None = None,
    *,
    workers: int = 1,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Metrics of every field of the selected wells (all wells by default). An exception
    raised by ``progress`` (e.g. a cancellation) stops the remaining fields."""
    settings = metric_settings(plate, rules)
    channels = len(plate.metadata.channels) or (plate.shape_tczyx or (0, 0))[1]
    if channels and settings.content_channel > channels:
        raise ValueError(
            f"Content channel {settings.content_channel} does not exist ({channels} channels)"
        )
    selected = [plate.wells[w] for w in (wells if wells is not None else sorted(plate.wells))]
    jobs = [(well, i) for well in selected for i in range(len(well.fields))]
    if not jobs:
        raise ValueError("No fields to check")
    rows: list[dict[str, Any]] = []
    done = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for row in pool.map(lambda job: _field_row(job[0], job[1], settings), jobs):
            rows.append(row)
            done += 1
            if progress is not None:
                progress(done, len(jobs))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return pd.DataFrame(rows)


def write_qc(
    result: QcResult,
    folder: Path,
    rules: QcRules,
    *,
    plate: Plate | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write the ``qc/`` files into ``folder`` (a run folder or any output folder)."""
    from hcs_studio.output.manifest import rooted_text

    qc = Path(folder) / QC_FOLDER
    qc.mkdir(parents=True, exist_ok=True)
    files = [qc / "field_qc.parquet", qc / "well_qc.csv", qc / "plate_qc.json", qc / "rules.yaml"]
    result.fields.to_parquet(files[0], index=False)
    result.wells.to_csv(files[1], index=False)
    summary = {
        **result.plate,
        "rules": rules.describe(),
        "software": f"hcs-studio {__version__}",
        "written": datetime.now(UTC).isoformat(),
        **(dict(extra) if extra else {}),
    }
    if plate is not None:
        summary["plate"] = {
            "name": plate.name,
            "barcode": plate.barcode,
            "path": str(plate.path),
            "rooted_path": rooted_text(plate.path),
            "vendor": plate.vendor,
        }
    files[2].write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    save_rules(rules, files[3])
    return files


def run_qc(
    plate: Plate,
    rules: QcRules,
    output: Path,
    *,
    wells: Sequence[tuple[int, int]] | None = None,
    roles: Mapping[str, str] | None = None,
    z_prime: float | None = None,
    layout: PlateLayout | None = None,
    workers: int = 1,
    progress: Progress | None = None,
) -> QcResult:
    """Measure, flag and decide, then write ``<output>/qc/``. Refuses outputs in raw data."""
    raw = (plate.path if plate.path.is_dir() else plate.path.parent).resolve()
    target = Path(output).resolve()
    if target == raw or raw in target.parents:
        raise ValueError(
            f"The output folder {target} is inside the raw data folder {raw}; "
            "choose a location outside it"
        )
    started = datetime.now(UTC)
    fields = measure_plate(plate, rules, wells, workers=workers, progress=progress)
    result = apply_rules(fields, rules, roles=roles, z_prime=z_prime, layout=layout)
    seconds = (datetime.now(UTC) - started).total_seconds()
    write_qc(result, target, rules, plate=plate, extra={"seconds": round(seconds, 2)})
    return result


def load_qc(folder: Path) -> QcResult:
    """Read the ``qc/`` files of a folder back."""
    qc = Path(folder) / QC_FOLDER
    fields = pd.read_parquet(qc / "field_qc.parquet")
    wells = pd.read_csv(qc / "well_qc.csv", keep_default_na=False, na_values=[""])
    plate = json.loads((qc / "plate_qc.json").read_text(encoding="utf-8"))
    return QcResult(fields, wells, plate)
