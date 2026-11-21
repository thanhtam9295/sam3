"""Runs a pipeline on fields of a plate and writes a new, self-describing run folder."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.analysis.stitching import (
    StitchWell,
    float_mosaic,
    mosaic,
    owner_fields,
    stitch,
)
from hcs_studio.io.base import reopen_plate
from hcs_studio.io.sidecar import load_overrides
from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.layout import WellAnnotation, load_plate_layout, processing_signature
from hcs_studio.model.plate import Field, ImageMetadata, Plate, parse_well_name, well_name
from hcs_studio.output.manifest import build_manifest, describe_inputs, write_json
from hcs_studio.output.tables import aggregate_run, write_well_tables
from hcs_studio.output.writers import (
    write_labels,
    write_ome_tiff,
    write_parent_links,
    write_png,
)
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import (
    ImageData,
    MeasureContext,
    MeasurementModule,
    Module,
    ModuleError,
    block_reduce_mean,
)
from hcs_studio.utils.display import (
    OUTLINE_COLOURS,
    channel_colormaps,
    make_thumbnail,
    outline_overlay,
)
from hcs_studio.utils.resources import peak_rss_bytes

_UNFITTED = object()
Progress = Callable[[int, int, str], None]
Events = Callable[[dict[str, Any]], None]


class RunCancelled(Exception):
    """The user cancelled the run."""


@dataclass(frozen=True)
class ProcessResult:
    #: Output of the last step.
    final: ImageData
    #: Last image whose intensities are still linear: the data to measure on.
    corrected: ImageData
    #: Tables from measurement steps (pandas DataFrames), by name.
    tables: dict[str, Any] = field(default_factory=dict)


class _Context:
    def __init__(self, runner: PipelineRunner, step: int) -> None:
        self._runner, self._step = runner, step
        self.plate = runner.plate
        self.layout = runner.layout

    def sample(
        self, max_fields: int, wells: Sequence[tuple[int, int]] | None = None
    ) -> Iterator[ImageData]:
        runner = self._runner
        keys = sorted(wells) if wells is not None else sorted(self.plate.wells)
        fields = [f for k in keys if k in self.plate.wells for f in self.plate.wells[k].fields]
        limit = min(max_fields, runner.fit_field_limit or max_fields)
        if len(fields) > limit:
            picks = np.linspace(0, len(fields) - 1, limit).round().astype(int)
            fields = [fields[i] for i in sorted(set(picks.tolist()))]
        for item in fields:
            runner.check_cancelled()
            try:
                image = runner.load(item)
            except Exception as exc:  # an unreadable field must not stop the whole plate
                runner.fit_skipped.append(
                    f"{item.sources[0] if item.sources else item.index}: {exc}"
                )
                continue
            yield runner.process(image, upto=self._step, _fit=False).final


def _concat(tables: dict[str, list[Any]]) -> dict[str, Any]:
    import pandas as pd

    return {
        name: pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        for name, parts in tables.items()
    }


class PipelineRunner:
    """Holds the modules of one pipeline for one plate, with their fitted state."""

    def __init__(
        self,
        plate: Plate,
        pipeline: Pipeline,
        layout: dict[tuple[int, int], WellAnnotation] | None = None,
        *,
        fit_field_limit: int | None = None,
        cancel: threading.Event | None = None,
        fit_cache: dict[str, Any] | None = None,
    ) -> None:
        self.plate = plate
        self.pipeline = pipeline
        self.layout = layout or {}
        #: Cap on fields sampled per fit (used for fast interactive previews).
        self.fit_field_limit = fit_field_limit
        self._cancel = cancel
        #: Optional shared cache of fitted state, keyed by plate + the steps that produced it.
        self._fit_cache = fit_cache
        #: Shared lock serialising GPU and exclusive steps across worker processes.
        self.gpu_lock: Any = None
        #: Fields that plate-level estimates skipped because they could not be read.
        self.fit_skipped: list[str] = []
        #: Seconds spent in each step ("<number> <module>") since the last reset.
        self.timings: dict[str, float] = {}
        #: Steps applied to each field (table steps run later, on the run's tables).
        self.steps = pipeline.image_steps()
        self.modules: list[Module] = [s.create() for s in self.steps]
        self.states: list[Any] = [_UNFITTED] * len(self.modules)
        metadata = plate.metadata
        for number, module in enumerate(self.modules, start=1):
            try:
                module.validate(metadata)
            except ModuleError as exc:
                raise ModuleError(f"Step {number} ({module.title}): {exc}") from None
            metadata = module.output_metadata(metadata)
        self.output_metadata: ImageMetadata = metadata

    def check_cancelled(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise RunCancelled

    def load(self, item: Field, timepoint: int = 0) -> ImageData:
        """Read one timepoint of a field into memory as float32 (C, Z, Y, X)."""
        return ImageData.from_array(np.asarray(item.load()[timepoint]), self.plate.metadata)

    def fit(self, upto: int | None = None) -> None:
        """Fit every step that needs plate-level state, in order."""
        end = len(self.modules) if upto is None else upto
        for i, module in enumerate(self.modules[:end]):
            if self.states[i] is not _UNFITTED:
                continue
            if not module.needs_fit:
                self.states[i] = None
                continue
            key = self._fit_key(i)
            if self._fit_cache is not None and key in self._fit_cache:
                self.states[i] = self._fit_cache[key]
                continue
            self.check_cancelled()
            try:
                self.states[i] = module.fit(_Context(self, i))
            except ModuleError as exc:
                raise ModuleError(f"Step {i + 1} ({module.title}): {exc}") from None
            if self._fit_cache is not None:
                self._fit_cache[key] = self.states[i]

    def _fit_key(self, step: int) -> str:
        """Identifies fitted state: same plate, metadata, layout and steps up to ``step``."""
        steps = [s.model_dump(mode="json") for s in self.steps[: step + 1]]
        return json.dumps(
            [
                str(self.plate.path),
                repr(self.plate.metadata),
                processing_signature(self.layout),
                self.fit_field_limit,
                steps,
            ],
            sort_keys=True,
            default=str,
        )

    def set_states(self, states: list[Any]) -> None:
        """Use fitted states computed elsewhere (e.g. by the coordinating process)."""
        if len(states) != len(self.modules):
            raise ValueError("Fitted states do not match the pipeline's steps")
        self.states = list(states)

    def process(
        self, image: ImageData, upto: int | None = None, _fit: bool = True
    ) -> ProcessResult:
        """Apply steps ``[0, upto)`` to an image."""
        end = len(self.modules) if upto is None else upto
        if _fit:
            self.fit(end)
        raw = corrected = image
        tables: dict[str, list[Any]] = {}
        for i, module in enumerate(self.modules[:end]):
            self.check_cancelled()
            started = time.perf_counter()
            try:
                if isinstance(module, MeasurementModule):
                    context = MeasureContext(raw=raw, corrected=corrected, labels=image.labels)
                    context.check_shapes()
                    for name, table in module.measure(context).items():
                        tables.setdefault(name, []).append(table)
                    continue  # (the finally clause below still records the time)
                if self.gpu_lock is not None and (module.uses_gpu() or module.exclusive):
                    with self.gpu_lock:
                        image = module.apply(image, self.states[i])
                else:
                    image = module.apply(image, self.states[i])
            except ModuleError as exc:
                raise ModuleError(f"Step {i + 1} ({module.title}): {exc}") from None
            finally:
                key = f"{i + 1} {module.name}"
                self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - started
            if image.linear:
                corrected = image
        return ProcessResult(final=image, corrected=corrected, tables=_concat(tables))

    def fitted_summary(self) -> list[dict[str, Any]]:
        summary = []
        for i, module in enumerate(self.modules):
            state = self.states[i]
            if module.needs_fit and state is not _UNFITTED and state is not None:
                summary.append(
                    {"step": i + 1, "module": module.name, **module.describe_state(state)}
                )
        return summary


# --------------------------------------------------------------------- wells


def parse_wells(text: str, plate: Plate) -> list[tuple[int, int]]:
    """'all', 'A01,B03' or rectangular ranges 'A01-H12' → wells that exist on the plate."""
    if text.strip().lower() in ("", "all"):
        return sorted(plate.wells)
    selected: set[tuple[int, int]] = set()
    for part in text.split(","):
        ends = [p for p in part.strip().split("-") if p]
        if len(ends) not in (1, 2):
            raise ValueError(f"Cannot read well selection {part!r}")
        (r0, c0), (r1, c1) = parse_well_name(ends[0]), parse_well_name(ends[-1])
        for row in range(min(r0, r1), max(r0, r1) + 1):
            for col in range(min(c0, c1), max(c0, c1) + 1):
                selected.add((row, col))
    wells = sorted(w for w in selected if w in plate.wells)
    if not wells:
        raise ValueError(f"None of the selected wells ({text}) contain images on this plate")
    return wells


# ----------------------------------------------------------------------- run


def _qc_params(modules: list[Module]) -> Any:
    """QC settings of the pipeline's QC step, or the defaults."""
    from hcs_studio.analysis.measure import FieldQc, QcParams

    for module in modules:
        if isinstance(module, FieldQc):
            return module.params
    return QcParams()


@dataclass
class RunResult:
    folder: Path
    fields_written: int = 0
    failed_wells: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False
    #: Wells skipped because a previous run already completed them (resume).
    skipped_wells: list[str] = field(default_factory=list)


def _plate_folder(plate: Plate) -> Path:
    return plate.path if plate.path.is_dir() else plate.path.parent


def make_run_folder(plate: Plate, pipeline: Pipeline, output_root: Path | None) -> Path:
    """Create ``<root>/<Plate>__<Pipeline>__<timestamp>``; never inside the raw data."""
    raw = _plate_folder(plate).resolve()
    root = (output_root if output_root is not None else raw.parent).resolve()
    if root == raw or raw in root.parents:
        raise ValueError(
            f"The output folder {root} is inside the raw data folder {raw}; "
            "choose a location outside it"
        )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", plate.name).strip("._") or "plate"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = root / f"{safe}__{pipeline.name}__{stamp}"
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = root / f"{safe}__{pipeline.name}__{stamp}-{suffix}"
    (folder / "wells").mkdir(parents=True)
    (folder / "logs").mkdir()
    return folder


def write_run_layout(
    folder: Path, layout: PlateLayout, source: Path | None = None
) -> dict[str, Any]:
    """Write the layout a run uses (``plate_layout.json`` and a long CSV for people, plus
    the imported file as given) and return its record for the manifest."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.io.layout_store import layout_sha256

    save_layout(layout, folder / "plate_layout.json")
    save_layout(layout, folder / "plate_layout.csv")
    record: dict[str, Any] = {
        "file": "plate_layout.json",
        "version": layout.version or None,
        "sha256": layout_sha256(layout),
        "wells": len(layout.wells),
    }
    if source is not None and Path(source).exists():
        copy = folder / f"plate_layout_source{Path(source).suffix.lower()}"
        if Path(source).resolve() != copy.resolve():
            shutil.copyfile(source, copy)
        record["source"] = {"path": str(source), "copy": copy.name}
    return record


def check_disk_space(path: Path, needed: int) -> None:
    """Refuse to start when the output disk clearly cannot hold the run's outputs."""
    from hcs_studio.utils.resources import free_disk_bytes

    free = free_disk_bytes(Path(path))
    if free is not None and needed > free:
        raise ValueError(
            f"The run needs about {needed / 1024**3:.1f} GB but only {free / 1024**3:.1f} GB "
            f"is free on the disk of {path}. Free space, choose another output folder, or do "
            "not save corrected images (masks, previews and tables are still written)."
        )


def pipeline_hash(
    plate: Plate,
    pipeline: Pipeline,
    layout: dict[tuple[int, int], WellAnnotation] | PlateLayout,
) -> str:
    """Identifies what a run's wells contain: image steps, plate, metadata (with overrides)
    and layout. Table steps and QC rules are left out: they are recomputed from the tables
    every time."""
    data = pipeline.to_dict()
    data.pop("qc", None)  # QC rules decide exclusions in the tables, not what wells contain
    data["steps"] = [
        step
        for step, parsed in zip(data["steps"], pipeline.steps, strict=True)
        if not parsed.is_table_step() and not parsed.is_run_step()
    ]
    payload = json.dumps(
        [
            data,
            str(Path(plate.path).resolve()),
            repr(plate.metadata),
            processing_signature(layout),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def well_is_complete(folder: Path, name: str, run_hash: str) -> bool:
    """A well is complete when its summary exists, matches the run and recorded no error."""
    path = folder / "wells" / name / "well_summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("pipeline_hash") == run_hash and not data.get("error")


@dataclass(frozen=True)
class RunPlan:
    """Everything a worker process needs to process wells; picklable."""

    folder: Path
    plate_source: dict[str, Any]
    metadata: ImageMetadata
    pipeline: dict[str, Any]
    layout: dict[tuple[int, int], WellAnnotation]
    states: list[Any]
    run_hash: str
    field_filter: dict[tuple[int, int], list[int]] | None = None
    save_images: bool = True
    max_memory_bytes: float | None = None
    image_dtype: str = "float32"


@dataclass
class WellOutcome:
    well: str
    fields_written: int = 0
    inputs: list[str] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False
    #: The run cannot continue (e.g. the output disk is full).
    fatal: bool = False


def process_well(
    plan: RunPlan,
    runner: PipelineRunner,
    key: tuple[int, int],
    emit: Events,
) -> WellOutcome:
    """Process one well and write its outputs. ``well_summary.json`` is written last."""
    plate = runner.plate
    name = well_name(*key)
    well_dir = plan.folder / "wells" / name
    if well_dir.exists():  # left over from an interrupted or failed attempt
        if plan.folder.resolve() not in well_dir.resolve().parents:
            raise RuntimeError(f"Refusing to clean {well_dir}: not inside the run folder")
        shutil.rmtree(well_dir)
    outcome = WellOutcome(well=name)
    n_timepoints = plate.shape_tczyx[0] if plate.shape_tczyx else 1
    colormaps = channel_colormaps(runner.output_metadata.channels)
    wanted = plan.field_filter.get(key) if plan.field_filter else None
    items = [f for f in plate.wells[key].fields if wanted is None or f.index in wanted]
    written: list[dict[str, Any]] = []
    well_warnings: list[str] = []
    well_tables: dict[str, list[Any]] = {}
    stitcher = next((m for m in runner.modules if isinstance(m, StitchWell)), None)
    small: dict[int, Any] = {}  # downsampled fields for stitching
    linear_tiles: dict[int, Any] = {}  # downsampled corrected fields for the mosaic image
    previews: dict[int, Any] = {}
    field_shape: tuple[int, ...] | None = None
    runner.timings = {}
    clock: dict[str, float] = {"read": 0.0, "write": 0.0}
    well_started = time.perf_counter()
    emit({"type": "well_started", "well": name, "fields": len(items)})
    try:
        for item in items:
            runner.check_cancelled()
            finals, correcteds = [], []
            raw_first: ImageData | None = None
            for t in range(n_timepoints):
                started = time.perf_counter()
                raw_image = runner.load(item, t)
                clock["read"] += time.perf_counter() - started
                if t == 0:
                    raw_first = raw_image
                processed = runner.process(raw_image)
                finals.append(processed.final)
                correcteds.append(processed.corrected)
                for table_name, table in processed.tables.items():
                    table = table.copy()
                    context: dict[str, Any] = {
                        "well": name,
                        "row": key[0] + 1,
                        "col": key[1] + 1,
                        "field": item.index,
                        "timepoint": t,
                    }
                    if table_name == "fields":
                        context.update(x_um=item.x_um, y_um=item.y_um)
                    for position, (column, value) in enumerate(context.items()):
                        table.insert(position, column, value)
                    well_tables.setdefault(table_name, []).append(table)
            assert raw_first is not None
            raw_colormaps = channel_colormaps(plate.metadata.channels)
            started = time.perf_counter()
            record, previews[item.index] = _write_field(
                well_dir,
                item.index,
                finals,
                correcteds,
                colormaps,
                raw_first,
                raw_colormaps,
                save_images=plan.save_images,
                image_dtype=plan.image_dtype,
            )
            for warning in _field_warnings(record):
                if warning not in well_warnings:
                    well_warnings.append(warning)
                    emit({"type": "log", "level": "warning", "text": f"{name}: {warning}"})
            clock["write"] += time.perf_counter() - started
            written.append(record)
            field_shape = finals[0].data.shape[-2:]
            if stitcher is not None:
                plane = correcteds[0].data[stitcher.params.channel - 1].max(axis=0)
                small[item.index] = block_reduce_mean(plane, stitcher.params.image_max_size)
                if stitcher.params.save_mosaic:
                    factor = stitcher.params.mosaic_downsample
                    linear_tiles[item.index] = np.stack(
                        [downsample(channel.max(axis=0), factor) for channel in correcteds[0].data]
                    )
            outcome.inputs.extend(str(p) for p in item.sources)
            outcome.fields_written += 1
            emit({"type": "field_done", "well": name, "field": item.index})
    except RunCancelled:
        outcome.cancelled = True
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            outcome.error = f"the output disk is full ({exc}); free space, then resume the run"
            outcome.fatal = True
        else:
            outcome.error = f"{type(exc).__name__}: {exc}"
    except MemoryError:
        outcome.error = (
            "out of memory: use fewer worker processes, set a memory limit, or process "
            "fewer channels / Z planes at once"
        )
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        import traceback

        emit(
            {
                "type": "log",
                "level": "error",
                "text": f"Well {name} failed:\n{traceback.format_exc()}",
            }
        )
    if outcome.cancelled:
        if well_dir.exists():
            shutil.rmtree(well_dir)  # incomplete: redone on resume
        return outcome
    extra: dict[str, Any] = {}
    if well_warnings:
        extra["warnings"] = well_warnings
    pixel = runner.output_metadata.pixel_size_um
    if field_shape is not None and pixel is not None:
        shape = field_shape
        size_um = (shape[0] * pixel, shape[1] * pixel)
        extra["field_size_um"] = list(size_um)
        if stitcher is not None and outcome.error is None:
            try:
                extra["stitching"] = _stitch_well(
                    stitcher,
                    items,
                    size_um,
                    small,
                    previews,
                    well_dir,
                    well_tables,
                    linear_tiles,
                    runner.output_metadata,
                )
            except Exception as exc:  # stitching must not lose the well's measurements
                outcome.error = f"stitching failed: {type(exc).__name__}: {exc}"
    table_files: dict[str, str] = {}
    if well_tables:
        well_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        table_files = write_well_tables(well_dir, well_tables)
        clock["write"] += time.perf_counter() - started
    peak = peak_rss_bytes()
    extra["timing"] = {
        "total_s": round(time.perf_counter() - well_started, 3),
        "read_s": round(clock["read"], 3),
        "write_s": round(clock["write"], 3),
        "steps_s": {k: round(v, 3) for k, v in runner.timings.items()},
        "worker_peak_memory_mb": round(peak / 1024**2) if peak else None,
    }
    well_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        well_dir / "well_summary.json",
        {
            "well": name,
            "pipeline_hash": plan.run_hash,
            "tables": table_files,
            "fields": written,
            "inputs": outcome.inputs,
            "measurement_note": "Measure intensities on the *_corrected files only; "
            "*_enhanced files are for viewing and segmentation.",
            "error": outcome.error,
            **extra,
        },
    )
    return outcome


def _stitch_well(
    stitcher: StitchWell,
    items: list[Field],
    size_um: tuple[float, float],
    small: dict[int, Any],
    previews: dict[int, Any],
    well_dir: Path,
    well_tables: dict[str, list[Any]],
    linear_tiles: dict[int, Any] | None = None,
    metadata: ImageMetadata | None = None,
) -> dict[str, Any]:
    """Place the well's fields, write the mosaic and add well coordinates to the tables."""
    import pandas as pd

    positions = {f.index: (f.x_um, f.y_um) for f in items}
    if any(x is None or y is None for x, y in positions.values()):
        return {"notes": ["stage positions are unknown: fields were not stitched"]}
    images = {i: small[i] for i in positions}
    first = next(iter(images.values()))
    image_scale = size_um[0] / first.shape[0]
    result = stitch(
        {i: (float(x), float(y)) for i, (x, y) in positions.items()},  # type: ignore[arg-type]
        size_um,
        images,
        image_scale,
        orientation=stitcher.orientation(),
        refine=stitcher.params.refine,
        max_correction_fraction=stitcher.params.max_correction_fraction,
        min_correlation=stitcher.params.min_correlation,
    )
    tile = next(iter(previews.values()))
    canvas, mosaic_scale = mosaic(previews, result, size_um[0] / tile.shape[0])
    write_png(well_dir / "previews" / "mosaic.png", canvas)
    offsets = result.offsets_um
    if "objects" in well_tables:
        frame = pd.concat(well_tables["objects"], ignore_index=True)
        dy = frame.field.map(lambda f: offsets.get(int(f), (np.nan, np.nan))[0])
        dx = frame.field.map(lambda f: offsets.get(int(f), (np.nan, np.nan))[1])
        frame["well_y_um"] = dy.to_numpy(float) + frame["centroid_y_um"].to_numpy(float)
        frame["well_x_um"] = dx.to_numpy(float) + frame["centroid_x_um"].to_numpy(float)
        owners = owner_fields(frame[["well_y_um", "well_x_um"]].to_numpy(), result, size_um)
        frame["in_overlap_duplicate"] = (owners != frame.field.to_numpy()) & (owners >= 0)
        well_tables["objects"] = [frame]
    if "fields" in well_tables:
        frame = pd.concat(well_tables["fields"], ignore_index=True)
        frame["field_y_um"] = frame.field.map(lambda f: offsets[int(f)][0])
        frame["field_x_um"] = frame.field.map(lambda f: offsets[int(f)][1])
        well_tables["fields"] = [frame]
    extra: dict[str, Any] = {}
    if linear_tiles and metadata is not None:
        factor = stitcher.params.mosaic_downsample
        tile = next(iter(linear_tiles.values()))
        scale = size_um[0] / tile.shape[1]
        image = float_mosaic(linear_tiles, result, scale)
        pixel = metadata.pixel_size_um * factor if metadata.pixel_size_um else None
        write_ome_tiff(
            well_dir / "mosaic.ome.tif",
            image[None, :, None],
            replace(metadata, pixel_size_um=pixel, z_step_um=None),
        )
        extra = {"mosaic_image": "mosaic.ome.tif", "mosaic_image_um_per_px": scale}
    return {
        **extra,
        "orientation": list(result.orientation),
        "refined_pairs": result.refined_pairs,
        "notes": result.notes,
        "field_offsets_um": {str(i): list(p) for i, p in sorted(offsets.items())},
        "mosaic": "previews/mosaic.png",
        "mosaic_um_per_px": mosaic_scale,
    }


def _write_field(
    well_dir: Path,
    index: int,
    finals: list[ImageData],
    correcteds: list[ImageData],
    colormaps: list[str],
    raw: ImageData,
    raw_colormaps: list[str],
    *,
    save_images: bool = True,
    image_dtype: str = "float32",
) -> tuple[dict[str, Any], Any]:
    """Write one field: images, masks, and previews of each stage plus outline overlays.

    Returns the field's record for ``well_summary.json`` and its enhanced RGB preview."""
    stem = f"f{index:03d}"
    (well_dir / "enhanced").mkdir(parents=True, exist_ok=True)
    (well_dir / "previews").mkdir(exist_ok=True)
    corrected = np.stack([c.data for c in correcteds])
    files: dict[str, Any] = {}
    saved_as: dict[str, Any] = {}
    if save_images:
        stored, saved_as = to_storage_dtype(corrected, image_dtype, raw.metadata.bit_depth)
        write_ome_tiff(
            well_dir / "enhanced" / f"{stem}_corrected.ome.tif", stored, correcteds[0].metadata
        )
        files["corrected"] = f"enhanced/{stem}_corrected.ome.tif"
    final = np.stack([f.data for f in finals])
    if save_images and not finals[0].linear:
        write_ome_tiff(
            well_dir / "enhanced" / f"{stem}_enhanced.ome.tif", final, finals[0].metadata
        )
        files["enhanced_display_only"] = f"enhanced/{stem}_enhanced.ome.tif"
    preview = make_thumbnail(final, colormaps, max_size=1024)
    write_png(
        well_dir / "previews" / f"{stem}_raw.png",
        make_thumbnail(raw.data[None], raw_colormaps, max_size=1024),
    )
    write_png(
        well_dir / "previews" / f"{stem}_corrected.png",
        make_thumbnail(
            corrected[:1], channel_colormaps(correcteds[0].metadata.channels), max_size=1024
        ),
    )
    files["preview_raw"] = f"previews/{stem}_raw.png"
    files["preview_corrected"] = f"previews/{stem}_corrected.png"
    label_sets = finals[0].labels
    if label_sets:
        (well_dir / "masks").mkdir(exist_ok=True)
    middle = final.shape[2] // 2
    overlays: dict[str, str] = {}
    for number, ls in enumerate(label_sets):
        colour = OUTLINE_COLOURS[number % len(OUTLINE_COLOURS)]
        overlay = outline_overlay(preview.shape[:2], ls.data[middle], colour)
        name = f"previews/{stem}_outline_{ls.name}.png"
        write_png(well_dir / name, overlay)
        overlays[ls.name] = name
    files["overlays"] = overlays
    objects: dict[str, Any] = {}
    for ls in label_sets:
        stack = np.stack([f.label_set(ls.name).data for f in finals])
        mask_file = f"masks/{stem}_{ls.name}.ome.tif"
        write_labels(well_dir / mask_file, stack, finals[0].metadata, ls.name)
        files[f"mask_{ls.name}"] = mask_file
        objects[ls.name] = {"count": ls.count, "source": ls.source, "parent": ls.parent}
        if ls.parent is not None and ls.parent_ids is not None:
            links = f"masks/{stem}_{ls.name}_parents.csv"
            write_parent_links(well_dir / links, ls.parent_ids, ls.parent)
            files[f"parents_{ls.name}"] = links
    write_png(well_dir / "previews" / f"{stem}.png", preview)
    files["preview"] = f"previews/{stem}.png"
    record = {
        "field": index,
        "files": files,
        "steps_corrected": list(correcteds[0].history),
        "steps_enhanced": list(finals[0].history),
        "reports": [list(f.reports) for f in finals],
        "objects": objects,
    }
    if saved_as:
        record["corrected_saved_as"] = saved_as
    return record, preview


# ------------------------------------------------------------ worker processes

_worker: dict[str, Any] = {}


def _worker_init(plan: RunPlan, events: Any, cancel: Any, gpu_lock: Any, workers: int) -> None:
    """Runs once in each worker process: rebuild the plate and the pipeline."""
    from hcs_studio.processing import deconvolution

    deconvolution.MEMORY_SHARE = 1.0 / max(1, workers)
    deconvolution.MEMORY_LIMIT = plan.max_memory_bytes
    plate = replace(reopen_plate(plan.plate_source), metadata=plan.metadata)
    runner = PipelineRunner(
        plate, Pipeline.model_validate(plan.pipeline), plan.layout, cancel=cancel
    )
    runner.set_states(plan.states)
    runner.gpu_lock = gpu_lock
    _worker.update(plan=plan, runner=runner, events=events)


def _worker_well(key: tuple[int, int]) -> WellOutcome:
    events = _worker["events"]
    return process_well(_worker["plan"], _worker["runner"], key, events.put)


def worker_memory_bytes(plate: Plate) -> int:
    """Estimated peak memory of one worker: about 40 × a field as float32, plus 600 MB."""
    shape = plate.shape_tczyx or (1, 1, 1, 1024, 1024)
    return 40 * 4 * int(np.prod(shape[1:])) + 600 * 1024**2


def auto_workers(plate: Plate, modules: list[Module], max_memory_bytes: float | None = None) -> int:
    """Worker count from CPU cores and memory (and an optional run memory limit)."""
    if any(m.exclusive for m in modules):
        return 1
    cores = max(1, (os.cpu_count() or 2) - 1)
    from hcs_studio.processing.deconvolution import _available_ram

    ram = 0.6 * (_available_ram() or 8 * 1024**3)
    if max_memory_bytes is not None:
        ram = min(ram, max_memory_bytes)
    return max(1, min(cores, int(ram // worker_memory_bytes(plate)), len(plate.wells)))


def downsample(plane: NDArray[Any], factor: int) -> NDArray[np.float32]:
    """Block-average a 2D image by an integer factor (edges cropped)."""
    if factor <= 1:
        return np.asarray(plane, dtype=np.float32)
    h, w = (plane.shape[0] // factor) * factor, (plane.shape[1] // factor) * factor
    blocks = plane[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def to_storage_dtype(
    data: NDArray[Any], image_dtype: str, bit_depth: int | None
) -> tuple[NDArray[Any], dict[str, Any]]:
    """Corrected images as written: float32, or rounded to the camera's bit depth
    ("original"; values outside the range are clipped and counted)."""
    if image_dtype == "float32":
        return data, {}
    if image_dtype != "original":
        raise ValueError(f"Unknown image type {image_dtype!r} (float32 or original)")
    bits = bit_depth or 16
    dtype = np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32
    top = 2**bits - 1
    rounded = np.rint(data)
    clipped = int(np.count_nonzero((rounded < 0) | (rounded > top)))
    stored = np.clip(rounded, 0, top).astype(dtype)
    return stored, {
        "dtype": np.dtype(dtype).name,
        "bit_depth": bits,
        "bit_depth_assumed": bit_depth is None,
        "clipped_pixels": clipped,
    }


def _field_warnings(record: dict[str, Any]) -> list[str]:
    """Warnings from the steps' reports of one field (e.g. a background radius check)."""
    found: list[str] = []
    for reports in record.get("reports", []):
        for report in reports:
            for warning in report.get("warnings", []) if isinstance(report, dict) else []:
                if warning not in found:
                    found.append(str(warning))
    saved = record.get("corrected_saved_as") or {}
    if saved.get("clipped_pixels"):
        found.append(
            f"{saved['clipped_pixels']} pixel(s) outside the {saved['bit_depth']}-bit range "
            "were clipped when saving; measurements use the unclipped values"
        )
    return found


def estimate_output_bytes(
    plate: Plate,
    n_fields: int,
    metadata: ImageMetadata,
    save_images: bool,
    bytes_per_value: int = 4,
) -> int:
    """Rough size of a run's outputs: corrected images (float32, if saved), masks and
    previews; tables are small in comparison."""
    t, c, z, y, x = plate.shape_tczyx or (1, 1, 1, 1024, 1024)
    per_field = 3 * 1024**2  # previews and outlines
    per_field += 2 * z * y * x  # masks (compressed int32, a rough allowance)
    if save_images:
        per_field += bytes_per_value * t * max(c, len(metadata.channels)) * z * y * x
    return int(n_fields * per_field * 1.1)


# -------------------------------------------------------------------- orchestration


def run_pipeline(
    plate: Plate,
    pipeline: Pipeline,
    *,
    output_root: Path | None = None,
    wells: list[tuple[int, int]] | None = None,
    fields: dict[tuple[int, int], list[int]] | None = None,
    layout: dict[tuple[int, int], WellAnnotation] | PlateLayout | None = None,
    layout_path: Path | None = None,
    checksums: bool = True,
    progress: Progress | None = None,
    events: Events | None = None,
    cancel: Any = None,
    pause: Any = None,
    workers: int | None = 1,
    resume: Path | None = None,
    save_images: bool = True,
    max_memory_gb: float | None = None,
    image_dtype: str = "float32",
) -> RunResult:
    """Process ``wells`` (default: all) and write the run folder. Raw data is only read.

    ``workers``: 1 runs in this process; more uses that many worker processes; ``None``
    chooses from CPU cores and memory. ``resume``: continue an earlier run folder, skipping
    wells it completed. ``cancel`` / ``pause``: events (threading or multiprocessing).
    ``save_images``: write corrected (and enhanced) OME-TIFFs; masks, previews and tables
    are always written. ``max_memory_gb``: RAM the whole run may use (all workers).
    ``image_dtype``: corrected images as float32, or "original" (the camera's bit depth,
    rounded and clipped; measurements always use the float values).
    """
    if image_dtype not in ("float32", "original"):
        raise ValueError(f"Unknown image type {image_dtype!r} (float32 or original)")
    value_bytes = 4 if image_dtype == "float32" else 2
    max_memory = max_memory_gb * 1024**3 if max_memory_gb else None
    started = datetime.now(UTC)
    if layout is None and layout_path is not None:
        layout = load_plate_layout(Path(layout_path))
    document = (
        layout
        if isinstance(layout, PlateLayout)
        else PlateLayout(rows=plate.rows, cols=plate.cols, wells=layout or {})
    )
    layout = dict(document.wells)  # plain dict: picklable for worker processes
    runner = PipelineRunner(plate, pipeline, layout)
    selected = wells if wells is not None else sorted(plate.wells)
    missing = [well_name(*w) for w in selected if w not in plate.wells]
    if missing:
        raise ValueError(f"Wells without images on this plate: {', '.join(missing)}")
    run_hash = pipeline_hash(plate, pipeline, layout)
    history: list[dict[str, Any]] = []
    if resume is not None:
        folder = Path(resume)
        try:
            previous = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError(f"{folder} is not an HCS Studio run folder") from None
        if previous.get("pipeline_hash") != run_hash:
            raise ValueError(
                "This run folder was made with a different pipeline, plate, metadata or "
                "layout; start a new run instead of resuming"
            )
        started = datetime.fromisoformat(previous["started"])
        history = list(previous.get("history", []))
        (folder / "logs").mkdir(exist_ok=True)
    else:
        n_fields = sum(
            len(
                [
                    f
                    for f in plate.wells[w].fields
                    if not fields or w not in fields or f.index in fields[w]
                ]
            )
            for w in selected
        )
        raw = _plate_folder(plate)
        check_disk_space(
            output_root if output_root is not None else raw.parent,
            estimate_output_bytes(
                plate, n_fields, runner.output_metadata, save_images, value_bytes
            ),
        )
        folder = make_run_folder(plate, pipeline, output_root)
    result = RunResult(folder=folder)
    history.append({"started": datetime.now(UTC).isoformat(), "wells_requested": len(selected)})

    log = logging.getLogger(f"hcs_studio.run.{folder.name}.{id(result)}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(folder / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)

    try:
        overrides = load_overrides(plate.path)
    except (OSError, ValueError):
        overrides = {}

    todo = [
        w for w in selected if not (resume and well_is_complete(folder, well_name(*w), run_hash))
    ]
    result.skipped_wells = [well_name(*w) for w in selected if w not in todo]
    total = sum(
        len(
            [
                f
                for f in plate.wells[w].fields
                if not fields or w not in fields or f.index in fields[w]
            ]
        )
        for w in todo
    )
    done = 0
    started_clock = time.monotonic()

    def emit(event: dict[str, Any]) -> None:
        nonlocal done
        kind = event.get("type")
        if kind == "field_done":
            done += 1
            elapsed = time.monotonic() - started_clock
            event = {
                **event,
                "done": done,
                "total": total,
                "eta_s": elapsed / done * (total - done) if done else None,
            }
        elif kind == "log":
            (log.error if event.get("level") == "error" else log.info)(event["text"])
        if events is not None:
            events(event)
        if progress is not None:
            if kind == "field_done":
                progress(done, total, f"{event['well']} field {event['field']}")
            elif kind == "status":
                progress(done, total, event["text"])

    def manifest(status: str) -> None:
        all_inputs: list[Path] = []
        all_wells = sorted(
            {
                *(well_name(*w) for w in selected),
                *(p.name for p in (folder / "wells").iterdir() if p.is_dir()),
            }
        )
        for name in all_wells:
            summary = folder / "wells" / name / "well_summary.json"
            if summary.exists():
                data = json.loads(summary.read_text(encoding="utf-8"))
                all_inputs.extend(Path(p) for p in data.get("inputs", []))
        record = build_manifest(
            plate,
            pipeline.to_dict(),
            status=status,
            started=started,
            wells=all_wells,
            inputs=describe_inputs(all_inputs, checksums) if status != "running" else [],
            fitted=runner.fitted_summary(),
            failures=result.failed_wells,
            overrides=overrides,
            checksums=checksums,
        )
        record["pipeline_hash"] = run_hash
        record["history"] = history
        if layout_record:
            record["layout"] = layout_record
        record["options"] = {
            "save_images": save_images,
            "image_dtype": image_dtype,
            "max_memory_gb": max_memory_gb,
            "workers_requested": workers,
        }
        if performance:
            record["performance"] = performance
        if qc_record:
            record["image_qc"] = {
                key: qc_record.get(key)
                for key in ("passed", "reasons", "fields", "excluded_fields", "failed_wells",
                            "rules", "seconds")
            }  # fmt: skip
        write_json(folder / "run_manifest.json", record)

    status = "failed"
    performance: dict[str, Any] = {}
    layout_record: dict[str, Any] = {}
    qc_record: dict[str, Any] = {}
    phase_started = time.monotonic()
    try:
        (folder / "pipeline.yaml").write_text(pipeline.to_yaml(), encoding="utf-8")
        if document.wells:
            layout_record = write_run_layout(folder, document, layout_path)
        manifest("running")
        log.info(
            "Run %s: plate %s, pipeline %s, %d wells (%d already complete)",
            "resumed" if resume else "started",
            plate.name,
            pipeline.name,
            len(selected),
            len(result.skipped_wells),
        )
        emit(
            {
                "type": "run_started",
                "folder": str(folder),
                "wells": [well_name(*w) for w in todo],
                "skipped": result.skipped_wells,
                "total": total,
            }
        )
        if pipeline.qc is not None:
            qc_record = image_qc_stage(
                plate,
                pipeline.qc,
                folder,
                selected,
                document,
                emit,
                cancel,
                workers=max(1, min(8, workers or (os.cpu_count() or 1))),
            )
            if not qc_record.get("passed", True) and pipeline.qc.plate.stop_run:
                raise QcStop("; ".join(qc_record.get("reasons", [])))
        emit({"type": "status", "text": "Estimating plate-level corrections"})
        fit_started = time.monotonic()
        _fit_or_load(runner, folder, run_hash, cancel)
        fit_seconds = time.monotonic() - fit_started
        for entry in runner.fitted_summary():
            log.info("Fitted step %s (%s): %s", entry["step"], entry["module"], entry)
        for skipped in runner.fit_skipped:
            log.warning("Plate-level estimate skipped an unreadable field: %s", skipped)
            emit(
                {"type": "log", "level": "warning", "text": f"Skipped unreadable field: {skipped}"}
            )
        plan = RunPlan(
            folder,
            plate.source,
            plate.metadata,
            pipeline.to_dict(),
            layout,
            runner.states,
            run_hash,
            fields,
            save_images=save_images,
            max_memory_bytes=max_memory,
            image_dtype=image_dtype,
        )
        if resume is not None:
            check_disk_space(
                folder,
                estimate_output_bytes(
                    plate, total, runner.output_metadata, save_images, value_bytes
                ),
            )
        count = (
            workers
            if workers is not None
            else auto_workers(plate, runner.modules, max_memory_bytes=max_memory)
        )
        if max_memory is not None and count * worker_memory_bytes(plate) > max_memory:
            log.warning(
                "%d worker(s) may need about %.1f GB, more than the %.1f GB limit",
                count,
                count * worker_memory_bytes(plate) / 1024**3,
                max_memory / 1024**3,
            )
        if count > 1 and any(m.exclusive for m in runner.modules):
            log.info("A step needs exclusive use of the machine (SAM 3): using one worker")
            count = 1
        count = max(1, min(count, len(todo) or 1))
        if count > 1 and not plate.source:
            log.info("The plate cannot be reopened in worker processes: using one worker")
            count = 1
        log.info("Processing %d wells with %d worker(s)", len(todo), count)
        emit({"type": "status", "text": f"Processing {len(todo)} wells with {count} worker(s)"})

        stop = threading.Event()  # set when the run cannot continue (disk full)
        fatal: list[str] = []

        def finished(outcome: WellOutcome) -> None:
            result.fields_written += outcome.fields_written
            if outcome.error:
                result.failed_wells[outcome.well] = outcome.error
            if outcome.fatal and outcome.error:
                fatal.append(outcome.error)
                stop.set()
                log.error("Stopping the run: %s", outcome.error)
            state = "cancelled" if outcome.cancelled else ("failed" if outcome.error else "done")
            emit(
                {
                    "type": "well_finished",
                    "well": outcome.well,
                    "status": state,
                    "error": outcome.error,
                    "fields": outcome.fields_written,
                }
            )
            if state == "done":
                log.info("Well %s done (%d fields)", outcome.well, outcome.fields_written)

        if count == 1:
            from hcs_studio.processing import deconvolution

            runner._cancel = cancel
            previous_limit, deconvolution.MEMORY_LIMIT = deconvolution.MEMORY_LIMIT, max_memory
            try:
                for key in todo:
                    if stop.is_set() or _wait_while_paused(pause, cancel):
                        break
                    outcome = process_well(plan, runner, key, emit)
                    finished(outcome)
                    if outcome.cancelled:
                        break
            finally:
                deconvolution.MEMORY_LIMIT = previous_limit
        else:
            _run_parallel(plan, todo, count, emit, finished, cancel, pause, stop)
        if cancel is not None and cancel.is_set():
            raise RunCancelled
        if stop.is_set():
            raise OSError(errno.ENOSPC, f"Run stopped: {fatal[0]}")
        wells_seconds = time.monotonic() - fit_started - fit_seconds
        emit({"type": "status", "text": "Summarising tables"})
        table_modules = [s.create() for s in pipeline.table_steps()]
        summary_started = time.monotonic()
        summarise_run(folder, runner.modules, table_modules, log)
        performance = run_performance(
            folder,
            [well_name(*w) for w in todo],
            workers=count,
            fit_s=fit_seconds,
            wells_s=wells_seconds,
            tables_s=time.monotonic() - summary_started,
            total_s=time.monotonic() - phase_started,
        )
        if qc_record:
            performance["wall_s"]["image_qc"] = qc_record.get("seconds")
        log.info("Performance: %s", json.dumps(performance))
        status = "completed_with_failures" if result.failed_wells else "completed"
    except RunCancelled:
        result.cancelled = True
        status = "cancelled"
        log.warning("Run cancelled by the user")
    except QcStop as exc:
        status = "qc_failed"
        log.warning("Run stopped: the plate failed image QC (%s)", exc)
    except Exception:
        log.exception("Run failed")
        history[-1]["finished"] = datetime.now(UTC).isoformat()
        history[-1]["status"] = "failed"
        manifest("failed")
        raise
    finally:
        log.removeHandler(handler)
        handler.close()
    history[-1]["finished"] = datetime.now(UTC).isoformat()
    history[-1]["status"] = status
    manifest(status)
    if pipeline.run_steps() and status in ("completed", "completed_with_failures"):
        emit({"type": "status", "text": "Run steps (pharmacology, figures, reports)"})
        step_log = _run_log(folder, "steps")
        try:
            for record in finish_run_steps(folder, pipeline.run_steps(), step_log):
                if record["status"] != "completed":
                    emit(
                        {
                            "type": "log",
                            "level": "warning",
                            "text": f"Run step {record['module']} failed: {record.get('error')}",
                        }
                    )
        finally:
            for h in list(step_log.handlers):
                step_log.removeHandler(h)
                h.close()
    emit(
        {
            "type": "finished",
            "status": status,
            "folder": str(folder),
            "failed": result.failed_wells,
            "fields": result.fields_written,
        }
    )
    emit({"type": "status", "text": "Finished" if not result.cancelled else "Cancelled"})
    return result


def run_performance(
    folder: Path,
    wells: list[str],
    *,
    workers: int,
    fit_s: float,
    wells_s: float,
    tables_s: float,
    total_s: float,
) -> dict[str, Any]:
    """Where the time went in this attempt: wall-clock phases, and the time summed over all
    fields per step (CPU-seconds across workers), from each well's ``timing``."""
    steps: dict[str, float] = {}
    read = write = 0.0
    fields = 0
    peaks = []
    for name in wells:
        try:
            data = json.loads(
                (folder / "wells" / name / "well_summary.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        timing = data.get("timing") or {}
        fields += len(data.get("fields", []))
        read += timing.get("read_s", 0.0)
        write += timing.get("write_s", 0.0)
        for key, value in (timing.get("steps_s") or {}).items():
            steps[key] = steps.get(key, 0.0) + value
        if timing.get("worker_peak_memory_mb"):
            peaks.append(timing["worker_peak_memory_mb"])
    output_bytes = sum(f.stat().st_size for f in (folder / "wells").rglob("*") if f.is_file())
    return {
        "workers": workers,
        "wells": len(wells),
        "fields": fields,
        "wall_s": {
            "fit": round(fit_s, 1),
            "wells": round(wells_s, 1),
            "tables_and_report": round(tables_s, 1),
            "total": round(total_s, 1),
        },
        "fields_per_minute": round(60 * fields / wells_s, 1) if wells_s > 0 else None,
        "summed_s": {
            "read": round(read, 1),
            "write": round(write, 1),
            **{k: round(v, 1) for k, v in sorted(steps.items())},
        },
        "worker_peak_memory_mb": max(peaks) if peaks else None,
        "output_gb": round(output_bytes / 1024**3, 3),
    }


def _run_log(folder: Path, name: str) -> logging.Logger:
    """A logger that appends to the run's ``logs/run.log``."""
    log = logging.getLogger(f"hcs_studio.{name}.{folder.name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    (folder / "logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(folder / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log


def finish_run_steps(folder: Path, steps: list[Any], log: logging.Logger) -> list[dict[str, Any]]:
    """Run the run steps, record them in the manifest and refresh ``index.html``."""
    from hcs_studio.analysis.run_steps import run_run_steps

    records = run_run_steps(folder, steps, log)
    manifest_path = folder / "run_manifest.json"
    with contextlib.suppress(OSError, ValueError):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_steps"] = records
        write_json(manifest_path, manifest)
    try:
        from hcs_studio.output.report import write_report

        write_report(folder)
    except Exception:
        log.exception("The HTML report could not be rewritten after the run steps")
    return records


def summarise_run(
    folder: Path, image_modules: list[Module], table_modules: list[Module], log: logging.Logger
) -> None:
    """Run table steps, build tables/, update each well's summary and write the report."""
    summaries = aggregate_run(folder, _qc_params(image_modules), table_modules, log.info)
    for name, summary in summaries.items():
        path = folder / "wells" / name / "well_summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["summary"] = summary
            write_json(path, data)
    try:
        from hcs_studio.output.report import write_report

        write_report(folder)
    except Exception:  # the report must never make a finished run fail
        log.exception("The HTML report could not be written")


def analyze_run(
    folder: Path, table_steps: Pipeline | list[Any], *, include_run_steps: bool = True
) -> dict[str, Any]:
    """Apply table steps (gates, classifiers, clustering, spatial statistics) to a finished
    run without reprocessing images. ``table_steps``: a pipeline (its table steps are used)
    or a list of ``PipelineStep``. With ``include_run_steps`` the run pipeline's own table
    steps run first. Replaces any earlier analysis of the run; returns the record added to
    the manifest's ``analyses``."""
    folder = Path(folder)
    manifest_path = folder / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_pipeline_def = Pipeline.model_validate(manifest["pipeline"])
    except (OSError, ValueError, KeyError):
        raise ValueError(f"{folder} is not an HCS Studio run folder") from None
    if isinstance(table_steps, Pipeline):
        steps = table_steps.table_steps()
        extra_run_steps = table_steps.run_steps()
    else:
        given = list(table_steps)
        steps = [s for s in given if s.is_table_step()]
        extra_run_steps = [s for s in given if s.is_run_step()]
        if len(steps) + len(extra_run_steps) != len(given):
            raise ValueError(
                "Only table steps and run steps (pharmacology, figures, reports) can be "
                "applied to a finished run"
            )
    if include_run_steps:
        steps = run_pipeline_def.table_steps() + steps
    image_modules = [s.create() for s in run_pipeline_def.image_steps()]
    table_modules = [s.create() for s in steps]
    log = logging.getLogger(f"hcs_studio.analyze.{folder.name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    (folder / "logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(folder / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    try:
        log.info("Analysis: applying %d table step(s) to the finished run", len(steps))
        summarise_run(folder, image_modules, table_modules, log)
    finally:
        log.removeHandler(handler)
        handler.close()
    record: dict[str, Any] = {
        "finished": datetime.now(UTC).isoformat(),
        "steps": [s.model_dump(mode="json") for s in steps],
    }
    manifest.setdefault("analyses", []).append(record)
    write_json(manifest_path, manifest)
    if extra_run_steps:
        step_log = _run_log(folder, "analyze-steps")
        try:
            record["run_steps"] = finish_run_steps(folder, extra_run_steps, step_log)
        finally:
            for h in list(step_log.handlers):
                step_log.removeHandler(h)
                h.close()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["analyses"][-1] = record
        write_json(manifest_path, manifest)
    return record


def annotate_run(folder: Path, layout: PlateLayout, source: Path | None = None) -> dict[str, Any]:
    """Give a finished run a new or corrected plate layout without reprocessing images:
    the layout is written to the run folder, the tables (annotation columns, table steps)
    and the report are rebuilt, and the manifest records the change. Refused when the
    unstained or blank wells differ from those the images were processed with."""
    from hcs_studio.model.layout import run_layout_file

    folder = Path(folder)
    manifest_path = folder / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_pipeline_def = Pipeline.model_validate(manifest["pipeline"])
    except (OSError, ValueError, KeyError):
        raise ValueError(f"{folder} is not an HCS Studio run folder") from None
    previous_file = run_layout_file(folder)
    previous = load_plate_layout(previous_file) if previous_file else None
    if processing_signature(previous) != processing_signature(layout):
        raise ValueError(
            "The new layout marks different unstained or blank wells than the images were "
            "processed with (they drive autofluorescence subtraction); run the plate again"
        )
    record = write_run_layout(folder, layout, source)
    log = logging.getLogger(f"hcs_studio.annotate.{folder.name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    (folder / "logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(folder / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    try:
        log.info(
            "Annotation: layout %s applied (%d wells)", record["sha256"][:12], len(layout.wells)
        )
        summarise_run(
            folder,
            [s.create() for s in run_pipeline_def.image_steps()],
            [s.create() for s in run_pipeline_def.table_steps()],
            log,
        )
    finally:
        log.removeHandler(handler)
        handler.close()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("layout"):
        manifest.setdefault("layout_history", []).append(manifest["layout"])
    manifest["layout"] = {**record, "applied": datetime.now(UTC).isoformat()}
    write_json(manifest_path, manifest)
    return record


class QcStop(Exception):
    """The plate failed image QC and the rules stop the run (``plate.stop_run``)."""


def image_qc_stage(
    plate: Plate,
    rules: Any,
    folder: Path,
    wells: list[tuple[int, int]],
    layout: PlateLayout,
    emit: Events,
    cancel: Any,
    workers: int,
) -> dict[str, Any]:
    """Image QC of the run's wells from the raw data, before any field is processed
    (Phase 18). Writes ``<run>/qc/``; a resumed run reuses it when the rules are the same."""
    from hcs_studio.qc.decide import apply_rules
    from hcs_studio.qc.run import measure_plate, write_qc

    existing = folder / "qc" / "plate_qc.json"
    if existing.exists():
        try:
            previous = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("rules", {}).get("sha256") == rules.sha256():
            emit({"type": "status", "text": "Image QC: using the results of the earlier attempt"})
            return dict(previous)
    n_fields = sum(len(plate.wells[w].fields) for w in wells)
    emit({"type": "qc_started", "fields": n_fields})
    emit({"type": "status", "text": f"Image QC of {n_fields} fields"})
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        if cancel is not None and cancel.is_set():
            raise RunCancelled
        if done == total or done % 50 == 0:
            emit({"type": "status", "text": f"Image QC: {done}/{total} fields"})

    fields = measure_plate(plate, rules, wells, workers=workers, progress=progress)
    result = apply_rules(fields, rules, layout=layout if layout.wells else None)
    seconds = round(time.monotonic() - started, 2)
    write_qc(result, folder, rules, plate=plate, extra={"seconds": seconds})
    summary = {**result.plate, "rules": rules.describe(), "seconds": seconds}
    verdict = "passed" if summary["passed"] else "FAILED: " + "; ".join(summary["reasons"])
    text = (
        f"Image QC {verdict} — {summary['excluded_fields']} of {summary['fields']} fields "
        f"excluded, {summary['failed_wells']} wells failed ({seconds:.0f} s)"
    )
    emit({"type": "qc_finished", "passed": summary["passed"], "text": text, "summary": summary})
    emit({"type": "log", "level": "info", "text": text})
    return summary


def _fit_or_load(runner: PipelineRunner, folder: Path, run_hash: str, cancel: Any) -> None:
    """Fit plate-level steps, or reuse the estimates saved by an earlier attempt of this run."""
    state_file = folder / "state" / "fitted.pkl"
    if state_file.exists():
        try:
            saved = pickle.loads(state_file.read_bytes())
            if saved.get("hash") == run_hash:
                runner.set_states(saved["states"])
                return
        except (OSError, ValueError, pickle.UnpicklingError, KeyError, EOFError):
            pass
    runner._cancel = cancel
    runner.fit()
    state_file.parent.mkdir(exist_ok=True)
    tmp = state_file.with_name("fitted.pkl.part")
    tmp.write_bytes(pickle.dumps({"hash": run_hash, "states": runner.states}))
    tmp.replace(state_file)


def _wait_while_paused(pause: Any, cancel: Any) -> bool:
    """Block while paused. Returns True if cancelled meanwhile."""
    while pause is not None and pause.is_set():
        if cancel is not None and cancel.is_set():
            return True
        time.sleep(0.2)
    return cancel is not None and cancel.is_set()


def _run_parallel(
    plan: RunPlan,
    todo: list[tuple[int, int]],
    count: int,
    emit: Events,
    finished: Callable[[WellOutcome], None],
    cancel: Any,
    pause: Any,
    stop: threading.Event | None = None,
) -> None:
    """Process wells in ``count`` worker processes; at most ``count`` wells in flight."""
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    events = manager.Queue()
    worker_cancel = manager.Event()
    gpu_lock = manager.Lock()
    pending = list(todo)
    running: dict[Future[WellOutcome], tuple[int, int]] = {}

    def drain() -> None:
        while True:
            try:
                emit(events.get_nowait())
            except queue.Empty:
                return

    try:
        with ProcessPoolExecutor(
            max_workers=count,
            mp_context=context,
            initializer=_worker_init,
            initargs=(plan, events, worker_cancel, gpu_lock, count),
        ) as pool:
            while pending or running:
                if (cancel is not None and cancel.is_set()) or (stop is not None and stop.is_set()):
                    worker_cancel.set()
                    pending.clear()
                paused = pause is not None and pause.is_set()
                while pending and len(running) < count and not paused:
                    key = pending.pop(0)
                    running[pool.submit(_worker_well, key)] = key
                if not running:
                    drain()
                    time.sleep(0.2)
                    continue
                complete, _ = wait(list(running), timeout=0.2, return_when=FIRST_COMPLETED)
                drain()
                for future in complete:
                    key = running.pop(future)
                    try:
                        finished(future.result())
                    except Exception as exc:  # a worker process died
                        finished(WellOutcome(well=well_name(*key), error=f"worker failed: {exc}"))
            drain()
    finally:
        manager.shutdown()
