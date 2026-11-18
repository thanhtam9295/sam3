"""Where a panel's pixels come from: a field of a finished run.

Sources: ``corrected`` (the saved linear image), ``enhanced`` (display-only, if saved),
``raw`` (the field read again from the raw data) and ``step:<n>`` (the raw field processed
by the run's pipeline up to and including image step n, with the run's fitted plate-level
estimates) — e.g. after AF correction, deconvolution, projection or background removal.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.compare.spec import PanelSpec
from hcs_studio.model.plate import parse_well_name
from hcs_studio.processing.base import ImageData


@dataclass(frozen=True)
class PanelImage:
    image: ImageData  # (C, Z, Y, X) float32 with labels
    field: int
    source: str
    files: tuple[str, ...]  # the files the pixels come from
    steps: tuple[str, ...]  # processing applied


def well_summary(run: Path, well: str) -> dict[str, Any]:
    path = Path(run) / "wells" / well / "well_summary.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def field_numbers(run: Path, well: str) -> list[int]:
    return [int(f["field"]) for f in well_summary(run, well).get("fields", [])]


def representative_field(run: Path, well: str, readout: str | None) -> int:
    """The field whose readout is closest to the well's median (``readout``: a column of
    the well's ``fields.csv``, or an object feature, by its median per field; empty: the
    object count). The first field when nothing can be measured."""
    fields = field_numbers(run, well)
    if not fields:
        raise ValueError(f"{well} has no fields in {Path(run).name}")
    well_dir = Path(run) / "wells" / well
    values: pd.Series | None = None
    table = well_dir / "fields.csv"
    if readout and table.exists():
        frame = pd.read_csv(table)
        if readout in frame and "field" in frame:
            values = frame.groupby("field")[readout].median()
    objects = well_dir / "objects.parquet"
    if values is None and objects.exists():
        frame = pd.read_parquet(objects)
        if "field" in frame:
            if readout and readout in frame:
                values = frame.groupby("field")[readout].median()
            else:
                values = frame.groupby("field").size().astype(float)
    if values is None or values.dropna().empty:
        return fields[0]
    values = values.dropna()
    median = float(values.median())
    best: Any = (values - median).abs().idxmin()
    return int(best)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: Loaded panels, least recently used first; bounded by bytes, not count (real fields
#: with Z stacks can be hundreds of MB each).
_CACHE: OrderedDict[tuple[str, str, int, str, int], PanelImage] = OrderedDict()
_LOCK = threading.Lock()
CACHE_LIMIT_BYTES = 2 * 1024**3


def _nbytes(panel: PanelImage) -> int:
    return int(panel.image.data.nbytes + sum(ls.data.nbytes for ls in panel.image.labels))


def cache_bytes() -> int:
    with _LOCK:
        return sum(_nbytes(p) for p in _CACHE.values())


def _cached(run: str, well: str, field: int, source: str, timepoint: int) -> PanelImage:
    key = (run, well, field, source, timepoint)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    panel = _load(Path(run), well, field, source, timepoint)
    with _LOCK:
        _CACHE[key] = panel
        total = sum(_nbytes(p) for p in _CACHE.values())
        while total > CACHE_LIMIT_BYTES and len(_CACHE) > 1:
            _, dropped = _CACHE.popitem(last=False)
            total -= _nbytes(dropped)
    return panel


def load_panel(panel: PanelSpec, representative_readout: str | None = None) -> PanelImage:
    """The panel's field (cached). Raises ValueError with a readable message."""
    field = panel.field
    if field is None:
        field = (
            representative_field(Path(panel.run), panel.well, representative_readout)
            if representative_readout is not None
            else field_numbers(Path(panel.run), panel.well)[0]
        )
    return _cached(str(Path(panel.run)), panel.well, int(field), panel.source, panel.timepoint)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _load(run: Path, well: str, field: int, source: str, timepoint: int) -> PanelImage:
    from hcs_studio.output.load import load_field_output

    summary = well_summary(run, well)
    entry = next((f for f in summary["fields"] if int(f["field"]) == field), None)
    if entry is None:
        raise ValueError(f"{well} has no field {field} in {run.name}")
    files = entry.get("files", {})
    if source in ("corrected", "enhanced"):
        image = load_field_output(run, well, field, source.capitalize())
        key = (
            "enhanced_display_only"
            if source == "enhanced" and "enhanced_display_only" in files
            else "corrected"
        )
        used = [str(run / "wells" / well / files[key])]
        used += [str(run / "wells" / well / v) for k, v in files.items() if k.startswith("mask_")]
        steps = tuple(entry.get("steps_enhanced" if key != "corrected" else "steps_corrected", []))
        return PanelImage(image, field, source, tuple(used), steps)
    if source == "raw" or source.startswith("step:"):
        return _recompute(run, well, field, source, timepoint)
    raise ValueError(f"Unknown image source {source!r} (raw, corrected, enhanced or step:<n>)")


def _recompute(run: Path, well: str, field: int, source: str, timepoint: int) -> PanelImage:
    from hcs_studio.io.base import reopen_plate
    from hcs_studio.model.layout import load_plate_layout, run_layout_file
    from hcs_studio.output.load import run_metadata
    from hcs_studio.pipeline.executor import PipelineRunner
    from hcs_studio.pipeline.schema import Pipeline

    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    source_info = manifest["plate"].get("source")
    if not source_info:
        raise ValueError("this run does not record how to reopen its raw data")
    try:
        plate = reopen_plate(source_info)
    except Exception as exc:
        raise ValueError(f"the raw data of {run.name} could not be opened: {exc}") from None
    from dataclasses import replace

    plate = replace(plate, metadata=run_metadata(run))
    key = parse_well_name(well)
    if key not in plate.wells:
        raise ValueError(f"{well} is not in the raw data")
    item = next((f for f in plate.wells[key].fields if f.index == field), None)
    if item is None:
        raise ValueError(f"{well} has no field {field} in the raw data")
    layout_file = run_layout_file(run)
    layout = dict(load_plate_layout(layout_file).wells) if layout_file else {}
    pipeline = Pipeline.load(run / "pipeline.yaml")
    runner = PipelineRunner(plate, pipeline, layout)
    raw = runner.load(item, timepoint)
    files = tuple(str(p) for p in item.sources) or (str(plate.path),)
    if source == "raw":
        return PanelImage(raw, field, source, files, ())
    try:
        upto = int(source.split(":", 1)[1])
    except ValueError:
        raise ValueError(f"Bad source {source!r}: use step:<number>") from None
    if not 1 <= upto <= len(runner.modules):
        raise ValueError(
            f"The pipeline has {len(runner.modules)} image steps; step {upto} does not exist"
        )
    state_file = run / "state" / "fitted.pkl"
    if any(m.needs_fit for m in runner.modules[:upto]):
        if not state_file.exists():
            raise ValueError("this run has no saved plate-level estimates (state/fitted.pkl)")
        runner.set_states(pickle.loads(state_file.read_bytes())["states"])
    else:
        runner.set_states([None] * len(runner.modules))
    result = runner.process(raw, upto=upto, _fit=False)
    return PanelImage(result.final, field, source, files, tuple(result.final.history))


def step_names(run: Path) -> list[str]:
    """``step:<n>`` sources with their titles, e.g. ('step:2', '2. Deconvolution')."""
    from hcs_studio.pipeline.schema import Pipeline

    pipeline = Pipeline.load(Path(run) / "pipeline.yaml")
    return [f"{n}. {s.module}" for n, s in enumerate(pipeline.image_steps(), start=1)]


def plane(image: ImageData, z: int | str) -> np.ndarray:
    """(C, Y, X) at a Z plane, or the maximum over Z."""
    data = image.data
    if z == "max" or data.shape[1] == 1:
        return np.asarray(data.max(axis=1), dtype=np.float32)
    k = int(z)
    if not 0 <= k < data.shape[1]:
        raise ValueError(f"Z plane {k + 1} does not exist ({data.shape[1]} planes)")
    return np.asarray(data[:, k], dtype=np.float32)
