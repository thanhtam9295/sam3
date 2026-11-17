"""Runs a pipeline on fields of a plate and writes a new, self-describing run folder."""

from __future__ import annotations

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

from hcs_studio.io.base import reopen_plate
from hcs_studio.io.sidecar import load_overrides
from hcs_studio.model.layout import WellAnnotation
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
)
from hcs_studio.utils.display import add_outlines, channel_colormaps, make_thumbnail

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
        self.modules: list[Module] = [s.create() for s in pipeline.enabled_steps()]
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
        steps = [s.model_dump(mode="json") for s in self.pipeline.enabled_steps()[: step + 1]]
        return json.dumps(
            [
                str(self.plate.path),
                repr(self.plate.metadata),
                sorted((k, repr(v)) for k, v in self.layout.items()),
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
            try:
                if isinstance(module, MeasurementModule):
                    context = MeasureContext(raw=raw, corrected=corrected, labels=image.labels)
                    context.check_shapes()
                    for name, table in module.measure(context).items():
                        tables.setdefault(name, []).append(table)
                    continue
                if self.gpu_lock is not None and (module.uses_gpu() or module.exclusive):
                    with self.gpu_lock:
                        image = module.apply(image, self.states[i])
                else:
                    image = module.apply(image, self.states[i])
            except ModuleError as exc:
                raise ModuleError(f"Step {i + 1} ({module.title}): {exc}") from None
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


def pipeline_hash(
    plate: Plate, pipeline: Pipeline, layout: dict[tuple[int, int], WellAnnotation]
) -> str:
    """Identifies what a run computes: pipeline, plate, metadata (with overrides) and layout."""
    payload = json.dumps(
        [
            pipeline.to_dict(),
            str(Path(plate.path).resolve()),
            repr(plate.metadata),
            sorted((f"{k}", repr(v)) for k, v in layout.items()),
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


@dataclass
class WellOutcome:
    well: str
    fields_written: int = 0
    inputs: list[str] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False


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
    well_tables: dict[str, list[Any]] = {}
    emit({"type": "well_started", "well": name, "fields": len(items)})
    try:
        for item in items:
            runner.check_cancelled()
            finals, correcteds = [], []
            for t in range(n_timepoints):
                processed = runner.process(runner.load(item, t))
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
            written.append(_write_field(well_dir, item.index, finals, correcteds, colormaps))
            outcome.inputs.extend(str(p) for p in item.sources)
            outcome.fields_written += 1
            emit({"type": "field_done", "well": name, "field": item.index})
    except RunCancelled:
        outcome.cancelled = True
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
    table_files: dict[str, str] = {}
    if well_tables:
        well_dir.mkdir(parents=True, exist_ok=True)
        table_files = write_well_tables(well_dir, well_tables)
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
        },
    )
    return outcome


def _write_field(
    well_dir: Path,
    index: int,
    finals: list[ImageData],
    correcteds: list[ImageData],
    colormaps: list[str],
) -> dict[str, Any]:
    stem = f"f{index:03d}"
    (well_dir / "enhanced").mkdir(parents=True, exist_ok=True)
    (well_dir / "previews").mkdir(exist_ok=True)
    corrected = np.stack([c.data for c in correcteds])
    write_ome_tiff(
        well_dir / "enhanced" / f"{stem}_corrected.ome.tif", corrected, correcteds[0].metadata
    )
    files = {"corrected": f"enhanced/{stem}_corrected.ome.tif"}
    final = np.stack([f.data for f in finals])
    if not finals[0].linear:
        write_ome_tiff(
            well_dir / "enhanced" / f"{stem}_enhanced.ome.tif", final, finals[0].metadata
        )
        files["enhanced_display_only"] = f"enhanced/{stem}_enhanced.ome.tif"
    preview = make_thumbnail(final, colormaps, max_size=1024)
    label_sets = finals[0].labels
    if label_sets:
        (well_dir / "masks").mkdir(exist_ok=True)
        middle = final.shape[2] // 2
        preview = add_outlines(preview, [ls.data[middle] for ls in label_sets])
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
    return {
        "field": index,
        "files": files,
        "steps_corrected": list(correcteds[0].history),
        "steps_enhanced": list(finals[0].history),
        "reports": [list(f.reports) for f in finals],
        "objects": objects,
    }


# ------------------------------------------------------------ worker processes

_worker: dict[str, Any] = {}


def _worker_init(plan: RunPlan, events: Any, cancel: Any, gpu_lock: Any, workers: int) -> None:
    """Runs once in each worker process: rebuild the plate and the pipeline."""
    from hcs_studio.processing import deconvolution

    deconvolution.MEMORY_SHARE = 1.0 / max(1, workers)
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


def auto_workers(plate: Plate, modules: list[Module]) -> int:
    """Worker count from CPU cores and memory: about 40 × the field size per worker."""
    if any(m.exclusive for m in modules):
        return 1
    cores = max(1, (os.cpu_count() or 2) - 1)
    from hcs_studio.processing.deconvolution import _available_ram

    ram = _available_ram() or 8 * 1024**3
    shape = plate.shape_tczyx or (1, 1, 1, 1024, 1024)
    per_worker = 40 * 4 * int(np.prod(shape[1:])) + 600 * 1024**2
    return max(1, min(cores, int(0.6 * ram // per_worker), len(plate.wells)))


# -------------------------------------------------------------------- orchestration


def run_pipeline(
    plate: Plate,
    pipeline: Pipeline,
    *,
    output_root: Path | None = None,
    wells: list[tuple[int, int]] | None = None,
    fields: dict[tuple[int, int], list[int]] | None = None,
    layout: dict[tuple[int, int], WellAnnotation] | None = None,
    layout_path: Path | None = None,
    checksums: bool = True,
    progress: Progress | None = None,
    events: Events | None = None,
    cancel: Any = None,
    pause: Any = None,
    workers: int | None = 1,
    resume: Path | None = None,
) -> RunResult:
    """Process ``wells`` (default: all) and write the run folder. Raw data is only read.

    ``workers``: 1 runs in this process; more uses that many worker processes; ``None``
    chooses from CPU cores and memory. ``resume``: continue an earlier run folder, skipping
    wells it completed. ``cancel`` / ``pause``: events (threading or multiprocessing).
    """
    started = datetime.now(UTC)
    layout = layout or {}
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
        write_json(folder / "run_manifest.json", record)

    status = "failed"
    try:
        (folder / "pipeline.yaml").write_text(pipeline.to_yaml(), encoding="utf-8")
        if layout_path is not None:
            shutil.copyfile(layout_path, folder / "plate_layout.csv")
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
        emit({"type": "status", "text": "Estimating plate-level corrections"})
        _fit_or_load(runner, folder, run_hash, cancel)
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
        )
        count = workers if workers is not None else auto_workers(plate, runner.modules)
        if count > 1 and any(m.exclusive for m in runner.modules):
            log.info("A step needs exclusive use of the machine (SAM 3): using one worker")
            count = 1
        count = max(1, min(count, len(todo) or 1))
        if count > 1 and not plate.source:
            log.info("The plate cannot be reopened in worker processes: using one worker")
            count = 1
        log.info("Processing %d wells with %d worker(s)", len(todo), count)
        emit({"type": "status", "text": f"Processing {len(todo)} wells with {count} worker(s)"})

        def finished(outcome: WellOutcome) -> None:
            result.fields_written += outcome.fields_written
            if outcome.error:
                result.failed_wells[outcome.well] = outcome.error
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
            runner._cancel = cancel
            for key in todo:
                if _wait_while_paused(pause, cancel):
                    break
                outcome = process_well(plan, runner, key, emit)
                finished(outcome)
                if outcome.cancelled:
                    break
        else:
            _run_parallel(plan, todo, count, emit, finished, cancel, pause)
        if cancel is not None and cancel.is_set():
            raise RunCancelled
        emit({"type": "status", "text": "Summarising tables"})
        summaries = aggregate_run(folder, _qc_params(runner.modules))
        for name, summary in summaries.items():
            path = folder / "wells" / name / "well_summary.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                data["summary"] = summary
                write_json(path, data)
        status = "completed_with_failures" if result.failed_wells else "completed"
    except RunCancelled:
        result.cancelled = True
        status = "cancelled"
        log.warning("Run cancelled by the user")
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
                if cancel is not None and cancel.is_set():
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
