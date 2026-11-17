"""Runs a pipeline on fields of a plate and writes a new, self-describing run folder."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.io.sidecar import load_overrides
from hcs_studio.model.layout import WellAnnotation
from hcs_studio.model.plate import Field, ImageMetadata, Plate, parse_well_name, well_name
from hcs_studio.output.manifest import build_manifest, describe_inputs, write_json
from hcs_studio.output.writers import (
    write_labels,
    write_ome_tiff,
    write_parent_links,
    write_png,
)
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import ImageData, Module, ModuleError
from hcs_studio.utils.display import add_outlines, channel_colormaps, make_thumbnail

_UNFITTED = object()
Progress = Callable[[int, int, str], None]


class RunCancelled(Exception):
    """The user cancelled the run."""


@dataclass(frozen=True)
class ProcessResult:
    #: Output of the last step.
    final: ImageData
    #: Last image whose intensities are still linear: the data to measure on.
    corrected: ImageData


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
            yield runner.process(runner.load(item), upto=self._step, _fit=False).final


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

    def process(
        self, image: ImageData, upto: int | None = None, _fit: bool = True
    ) -> ProcessResult:
        """Apply steps ``[0, upto)`` to an image."""
        end = len(self.modules) if upto is None else upto
        if _fit:
            self.fit(end)
        corrected = image
        for i, module in enumerate(self.modules[:end]):
            self.check_cancelled()
            try:
                image = module.apply(image, self.states[i])
            except ModuleError as exc:
                raise ModuleError(f"Step {i + 1} ({module.title}): {exc}") from None
            if image.linear:
                corrected = image
        return ProcessResult(final=image, corrected=corrected)

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


@dataclass
class RunResult:
    folder: Path
    fields_written: int = 0
    failed_wells: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False


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


def run_pipeline(
    plate: Plate,
    pipeline: Pipeline,
    *,
    output_root: Path | None = None,
    wells: list[tuple[int, int]] | None = None,
    layout: dict[tuple[int, int], WellAnnotation] | None = None,
    layout_path: Path | None = None,
    checksums: bool = True,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> RunResult:
    """Process ``wells`` (default: all) and write the run folder. Raw data is only read."""
    started = datetime.now(UTC)
    runner = PipelineRunner(plate, pipeline, layout, cancel=cancel)
    selected = wells if wells is not None else sorted(plate.wells)
    missing = [well_name(*w) for w in selected if w not in plate.wells]
    if missing:
        raise ValueError(f"Wells without images on this plate: {', '.join(missing)}")
    folder = make_run_folder(plate, pipeline, output_root)
    result = RunResult(folder=folder)

    log = logging.getLogger(f"hcs_studio.run.{folder.name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(folder / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)

    try:
        overrides = load_overrides(plate.path)
    except (OSError, ValueError):
        overrides = {}
    well_names = [well_name(*w) for w in selected]
    inputs: list[Path] = []

    def manifest(status: str) -> None:
        write_json(
            folder / "run_manifest.json",
            build_manifest(
                plate,
                pipeline.to_dict(),
                status=status,
                started=started,
                wells=well_names,
                inputs=describe_inputs(inputs, checksums) if status != "running" else [],
                fitted=runner.fitted_summary(),
                failures=result.failed_wells,
                overrides=overrides,
                checksums=checksums,
            ),
        )

    total = sum(len(plate.wells[w].fields) for w in selected)
    done = 0

    def report(message: str) -> None:
        if progress is not None:
            progress(done, total, message)

    try:
        (folder / "pipeline.yaml").write_text(pipeline.to_yaml(), encoding="utf-8")
        if layout_path is not None:
            shutil.copyfile(layout_path, folder / "plate_layout.csv")
        manifest("running")
        log.info(
            "Run started: plate %s, pipeline %s, %d wells", plate.name, pipeline.name, len(selected)
        )
        report("Estimating plate-level corrections")
        runner.fit()
        for entry in runner.fitted_summary():
            log.info("Fitted step %s (%s): %s", entry["step"], entry["module"], entry)

        n_timepoints = plate.shape_tczyx[0] if plate.shape_tczyx else 1
        colormaps = channel_colormaps(runner.output_metadata.channels)
        for key in selected:
            name = well_name(*key)
            well_dir = folder / "wells" / name
            written: list[dict[str, Any]] = []
            try:
                for item in plate.wells[key].fields:
                    runner.check_cancelled()
                    report(f"{name} field {item.index}")
                    finals, correcteds = [], []
                    for t in range(n_timepoints):
                        processed = runner.process(runner.load(item, t))
                        finals.append(processed.final)
                        correcteds.append(processed.corrected)
                    stem = f"f{item.index:03d}"
                    (well_dir / "enhanced").mkdir(parents=True, exist_ok=True)
                    (well_dir / "previews").mkdir(exist_ok=True)
                    corrected = np.stack([c.data for c in correcteds])
                    write_ome_tiff(
                        well_dir / "enhanced" / f"{stem}_corrected.ome.tif",
                        corrected,
                        correcteds[0].metadata,
                    )
                    files = {"corrected": f"enhanced/{stem}_corrected.ome.tif"}
                    final = np.stack([f.data for f in finals])
                    if not finals[0].linear:
                        write_ome_tiff(
                            well_dir / "enhanced" / f"{stem}_enhanced.ome.tif",
                            final,
                            finals[0].metadata,
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
                        objects[ls.name] = {
                            "count": ls.count,
                            "source": ls.source,
                            "parent": ls.parent,
                        }
                        if ls.parent is not None and ls.parent_ids is not None:
                            links = f"masks/{stem}_{ls.name}_parents.csv"
                            write_parent_links(well_dir / links, ls.parent_ids, ls.parent)
                            files[f"parents_{ls.name}"] = links
                    write_png(well_dir / "previews" / f"{stem}.png", preview)
                    files["preview"] = f"previews/{stem}.png"
                    written.append(
                        {
                            "field": item.index,
                            "files": files,
                            "steps_corrected": list(correcteds[0].history),
                            "steps_enhanced": list(finals[0].history),
                            "reports": [list(f.reports) for f in finals],
                            "objects": objects,
                        }
                    )
                    inputs.extend(item.sources)
                    done += 1
                    result.fields_written += 1
            except RunCancelled:
                raise
            except Exception as exc:
                result.failed_wells[name] = f"{type(exc).__name__}: {exc}"
                log.exception("Well %s failed", name)
                done = sum(len(plate.wells[w].fields) for w in selected[: selected.index(key) + 1])
            if written:
                write_json(
                    well_dir / "well_summary.json",
                    {
                        "well": name,
                        "fields": written,
                        "measurement_note": "Measure intensities on the *_corrected files only; "
                        "*_enhanced files are for viewing and segmentation.",
                        "error": result.failed_wells.get(name),
                    },
                )
        status = "completed_with_failures" if result.failed_wells else "completed"
    except RunCancelled:
        result.cancelled = True
        status = "cancelled"
        log.warning("Run cancelled by the user")
    except Exception:
        log.exception("Run failed")
        manifest("failed")
        raise
    finally:
        log.removeHandler(handler)
        handler.close()
    manifest(status)
    report("Finished" if not result.cancelled else "Cancelled")
    return result
