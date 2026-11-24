"""Service functions for the state changes that exist today (Phases 6–16).

Each wraps a core function, describes the object it changes (``ctx.note``) and is recorded
as an event. Run folders are identified by their rooted path (``ROOT:relative``).
Read-only work (previews, re-analysis without writing) does not go through services.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hcs_studio.services import ServiceContext, service

if TYPE_CHECKING:
    from hcs_studio.analysis.classifier import TrainedClassifier
    from hcs_studio.io.layout_store import LayoutStore
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.plate import Plate
    from hcs_studio.pharmacology.analysis import PharmOptions, PharmResult
    from hcs_studio.pipeline.executor import RunResult
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings
    from hcs_studio.segmentation.sam3_backend import Sam3Settings


def object_id(path: str | Path) -> str:
    """A run folder or file as stored in events: its rooted path."""
    from hcs_studio.output.manifest import rooted_text

    return rooted_text(Path(path).absolute())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- layouts


@service("layout.save", object_type="layout")
def save_layout_version(
    ctx: ServiceContext, store: LayoutStore, layout: PlateLayout, note: str = ""
) -> PlateLayout:
    """Save a new version of a plate's layout (versions are never overwritten)."""
    saved = store.save(layout, note or ctx.reason)
    ctx.note(
        object_id=f"{store.key}/v{saved.version:04d}",
        before={"version": layout.version or None},
        after={"version": saved.version, "wells": len(saved.wells), "note": note},
    )
    return saved


@service("run.annotate", object_type="run")
def annotate_run(
    ctx: ServiceContext, folder: Path, layout: PlateLayout, source: Path | None = None
) -> dict[str, Any]:
    """Apply a layout to a finished run (tables and report rebuilt)."""
    from hcs_studio.io.layout_store import layout_sha256
    from hcs_studio.pipeline import executor

    before = _manifest_layout(folder)
    record = executor.annotate_run(folder, layout, source)
    ctx.note(
        object_id=object_id(folder),
        before=before,
        after={"sha256": layout_sha256(layout), "wells": len(layout.wells)},
    )
    return record


def _manifest_layout(folder: Path) -> dict[str, Any] | None:
    import json

    try:
        manifest = json.loads((Path(folder) / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    layout = manifest.get("layout")
    return {"sha256": layout.get("sha256"), "wells": layout.get("wells")} if layout else None


# ---------------------------------------------------------------------- runs


@service("run.start", object_type="run")
def run_pipeline(ctx: ServiceContext, plate: Plate, pipeline: Pipeline, **kwargs: Any) -> RunResult:
    """Run a pipeline in this process (the command line); queued runs use the job queue."""
    from hcs_studio.pipeline import executor

    result = executor.run_pipeline(plate, pipeline, **kwargs)
    ctx.note(
        object_id=object_id(result.folder),
        after={
            "plate": plate.name,
            "pipeline": pipeline.name,
            "resume": bool(kwargs.get("resume")),
            "failed_wells": sorted(result.failed_wells),
        },
    )
    return result


@service("run.analyze", object_type="run")
def analyze_run(
    ctx: ServiceContext,
    folder: Path,
    table_steps: Pipeline | list[Any],
    *,
    include_run_steps: bool = True,
) -> dict[str, Any]:
    """Apply table and run steps to a finished run (replaces any earlier analysis)."""
    from hcs_studio.pipeline import executor

    record = executor.analyze_run(folder, table_steps, include_run_steps=include_run_steps)
    ctx.note(
        object_id=object_id(folder),
        after={
            "steps": [s.get("module") for s in record.get("steps", [])],
            "run_steps": [s.get("module") for s in record.get("run_steps", [])],
        },
    )
    return record


@service("run.report", object_type="run")
def write_report(ctx: ServiceContext, folder: Path) -> Path:
    from hcs_studio.output.report import write_report as write

    path = write(folder)
    ctx.note(object_id=object_id(folder), after={"report": path.name})
    return path


# -------------------------------------------------------------- pharmacology


def _pharm_needs_reason(
    ctx: ServiceContext, folders: Any, options: PharmOptions, *args: Any, **kw: Any
) -> bool:
    return bool(kw.get("write", True)) and bool(options.excluded_wells)


@service("pharmacology.write", object_type="run", reason_required=_pharm_needs_reason)
def run_pharmacology(
    ctx: ServiceContext,
    folders: list[Path],
    options: PharmOptions,
    output: Path | None = None,
    *,
    write: bool = True,
    refresh_reports: bool = True,
) -> PharmResult:
    """Analyse runs and write ``pharmacology/``; manual exclusions need a reason."""
    from hcs_studio.pharmacology import analysis

    result = analysis.run_pharmacology(
        folders, options, output, write=write, refresh_reports=refresh_reports
    )
    ctx.note(
        object_id=object_id(folders[0]) if folders else "",
        after={
            "runs": [object_id(f) for f in folders],
            "readouts": list(options.readouts),
            "normalization": options.normalization,
            "excluded_wells": [list(w) for w in options.excluded_wells],
            "included_wells": [list(w) for w in options.included_wells],
            "output": object_id(output) if output else "",
        },
    )
    return result


@service("run.profile", object_type="run")
def run_profiling(
    ctx: ServiceContext, runs: list[Path], options: Any = None, output: Path | None = None
) -> Any:
    """PCA/UMAP profiles of runs (``advanced/`` of the first run, whose report is refreshed)."""
    from hcs_studio.analysis.profiling import run_profiling as profile
    from hcs_studio.output.report import write_report as write

    result = profile(runs, options, output)
    if output is None and runs:
        write(Path(runs[0]))  # the report shows the profiles of the first run
    ctx.note(
        object_id=object_id(runs[0]) if runs else "",
        after={"runs": [object_id(r) for r in runs], "output": str(result.folder)},
    )
    return result


# ---------------------------------------------------------------- image QC


@service("qc.run", object_type="plate")
def run_image_qc(ctx: ServiceContext, plate: Plate, rules: Any, output: Path, **kwargs: Any) -> Any:
    """Image QC of a plate from raw data; writes ``<output>/qc/`` (Phase 18)."""
    from hcs_studio.qc.run import run_qc

    result = run_qc(plate, rules, Path(output), **kwargs)
    ctx.note(
        object_id=object_id(plate.path),
        after={
            "output": object_id(output),
            "rules": rules.describe(),
            "passed": result.plate["passed"],
            "fields": result.plate["fields"],
            "excluded_fields": result.plate["excluded_fields"],
            "failed_wells": result.plate["failed_wells"],
        },
    )
    return result


@service("qc.review", object_type="qc", reason_required=True)
def review_qc(
    ctx: ServiceContext,
    folder: Path,
    level: str,
    well: str,
    action: str,
    field: int | None = None,
) -> Any:
    """Accept or override a QC decision for a field or a well (a reason is required); the
    run's tables and report follow."""
    from hcs_studio.qc.review import Override, apply_override, load_fields

    fields = load_fields(Path(folder))
    if level == "field":
        rows = fields[(fields["well"] == well) & (fields["field"] == field)]
        before = {"excluded": bool(rows["excluded"].any()), "flags": ";".join(rows["flags"])}
    else:
        before = {"excluded_fields": int(fields.loc[fields["well"] == well, "excluded"].sum())}
    override = Override.model_validate(
        {
            "level": level,
            "well": well,
            "field": field,
            "action": action,
            "reason": ctx.reason,
            "actor": ctx.actor.id,
        }
    )
    result = apply_override(Path(folder), override)
    target = f"{object_id(folder)}#{well}" + (f"/f{field}" if field is not None else "")
    wells = result.wells.set_index("well")
    ctx.note(
        object_id=target,
        before=before,
        after={
            "action": action,
            "well_failed": bool(wells.loc[well, "failed"]) if well in wells.index else None,
            "plate_passed": result.plate["passed"],
        },
    )
    return result


# -------------------------------------------------------------------- models


@service("model.save_classifier", object_type="model")
def save_classifier(ctx: ServiceContext, classifier: TrainedClassifier, path: Path) -> Path:
    from hcs_studio.analysis import classifier as core

    core.save_classifier(classifier, path)
    ctx.note(
        object_id=object_id(path),
        after={"sha256": file_sha256(Path(path)), "classes": list(classifier.classes)},
    )
    return Path(path)


@service("model.fine_tune_cellpose", object_type="model")
def fine_tune_cellpose(ctx: ServiceContext, folder: Path, **kwargs: Any) -> tuple[Path, Any]:
    from hcs_studio.segmentation.cellpose_backend import fine_tune

    path, history = fine_tune(folder, **kwargs)
    ctx.note(
        object_id=object_id(path),
        after={"training_data": object_id(folder), **{k: str(v) for k, v in kwargs.items()}},
    )
    return path, history


# ------------------------------------------------------------------ settings


@service("settings.ilastik", object_type="settings")
def save_ilastik_settings(ctx: ServiceContext, settings: IlastikSettings) -> Path:
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings as Settings

    before = asdict(Settings.load())
    path = settings.save()
    ctx.note(object_id="ilastik", before=before, after=asdict(settings))
    return path


@service("settings.sam3", object_type="settings")
def save_sam3_settings(ctx: ServiceContext, settings: Sam3Settings) -> None:
    from hcs_studio.segmentation.sam3_backend import Sam3Settings as Settings

    before = asdict(Settings.load())
    settings.save()
    ctx.note(object_id="sam3", before=before, after=asdict(settings))


@service("settings.app", object_type="settings")
def save_app_settings(ctx: ServiceContext, settings: Any) -> Path:
    from hcs_studio.config import load_settings, save_settings

    before = load_settings().model_dump(mode="json")
    path = save_settings(settings)
    ctx.note(object_id="settings.json", before=before, after=settings.model_dump(mode="json"))
    return path


# ---------------------------------------------------------------------- jobs


@service("run.submit", object_type="job")
def submit_job(ctx: ServiceContext, queue: Any, job: Any, priority: str = "interactive") -> str:
    """Queue a pipeline run (``queue``: a ``JobQueue``)."""
    from hcs_studio.pipeline.queue import JobSpec

    job_id = str(queue.submit(JobSpec.for_job(job, priority=priority, owner=ctx.actor)))
    ctx.note(
        object_id=job_id,
        after={
            "label": job.label,
            "plate": str(job.plate_source.get("path", "")),
            "resume": job.resume or "",
            "wells": len(job.wells or []),
            "priority": priority,
        },
    )
    return job_id


@service("run.cancel", object_type="job")
def cancel_job(ctx: ServiceContext, queue: Any, job_id: str) -> None:
    before = queue.status(job_id).status
    queue.cancel(job_id)
    ctx.note(object_id=job_id, before={"status": before})


@service("run.retry", object_type="job")
def retry_job(ctx: ServiceContext, queue: Any, job_id: str) -> str | None:
    """Queue the failed wells of a finished job again (in the same run folder)."""
    new = queue.retry(job_id)
    ctx.note(object_id=job_id, after={"retry_job": new})
    return new  # type: ignore[no-any-return]


@service("run.resume", object_type="job")
def resume_job(ctx: ServiceContext, queue: Any, job_id: str) -> str | None:
    new = queue.resume(job_id)
    ctx.note(object_id=job_id, after={"resume_job": new})
    return new  # type: ignore[no-any-return]


@service("screen.score", object_type="screen")
def run_screen(
    ctx: ServiceContext, folders: list[Path], config: Any, output: Path, gallery: int = 0
) -> Any:
    """Score a screen and write its hit list, report and (``gallery`` > 0) hit gallery; manual
    decisions already made are kept."""
    from hcs_studio.screening.screen import run_screen as score

    result = score([Path(f) for f in folders], config, Path(output), gallery=gallery)
    ctx.note(
        object_id=object_id(output), after={"plates": len(result.plates), "hits": len(result.hits)}
    )
    return result


@service("screen.review", object_type="screen", reason_required=True)
def review_hit(
    ctx: ServiceContext, output: Path, compound: str, action: Literal["include", "exclude"]
) -> Any:
    """Include or exclude a compound from the hits (a reason is required); the hit list is
    rewritten from the screen's runs and configuration."""
    import json

    from hcs_studio.screening.screen import Override, ScreenConfig, add_override
    from hcs_studio.screening.screen import run_screen as score

    output = Path(output)
    summary = json.loads((output / "screen.json").read_text(encoding="utf-8"))
    hits = output / "hits.csv"
    before = compound in hits.read_text(encoding="utf-8") if hits.exists() else None
    add_override(
        output,
        Override(compound=compound, action=action, reason=ctx.reason or "", actor=ctx.actor.id),
    )
    config = ScreenConfig.model_validate(summary["config"])
    result = score([Path(f) for f in summary["runs"]], config, output)
    ctx.note(
        object_id=f"{object_id(output)}:{compound}",
        before={"listed": before},
        after={"action": action, "hits": len(result.hits)},
    )
    return result


@service("screen.cherrypick", object_type="screen")
def cherry_pick_hits(
    ctx: ServiceContext,
    screen: Path,
    design: Any,
    template: Any,
    *,
    library: Any = None,
    fmt: Any = None,
    register: bool = False,
    outbox: str = "",
    scheduler_url: str = "",
) -> Any:
    """Pick a screen's hits onto confirmation plates: pick list and layouts in
    ``<screen>/cherrypick/``; ``register`` saves the layouts as versions keyed by their barcodes
    (the plates are then analysed with them when imaged); ``outbox`` / ``scheduler_url`` send a
    ``cherrypick.request`` per plate. The screen summary is rewritten."""
    import json

    from hcs_studio.io.layout_store import LayoutStore, plate_key
    from hcs_studio.screening.cherrypick import event_payload, run_cherry_pick
    from hcs_studio.screening.report import write_summary

    screen = Path(screen)
    result, files = run_cherry_pick(screen, design, template, library=library, fmt=fmt)
    summary_file = screen / "cherrypick" / "cherrypick.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    name = json.loads((screen / "screen.json").read_text(encoding="utf-8")).get("name", screen.name)
    if register:
        registered = []
        for barcode, layout in result.layouts.items():
            saved = LayoutStore(plate_key(barcode=barcode)).save(
                layout, f"confirmation plate of screen {name}"
            )
            registered.append(f"{barcode} v{saved.version}")
        summary["registered"] = registered
    if outbox or scheduler_url:
        from hcs_studio.automation.intake import send

        summary["events"] = [
            send(
                "cherrypick.request",
                event_payload(str(name), barcode, result.picks, files[0]),
                outbox=outbox,
                scheduler_url=scheduler_url,
            )["event_id"]
            for barcode in result.layouts
        ]
    summary_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_summary(screen)
    ctx.note(
        object_id=object_id(screen),
        after={
            k: summary.get(k) for k in ("plates", "transfers", "compounds", "registered", "events")
        },
    )
    return result
