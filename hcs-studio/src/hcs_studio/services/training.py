"""Service functions for ground truth and models (Phase 25)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hcs_studio.services import ServiceContext, service

if TYPE_CHECKING:
    from hcs_studio.pipeline.queue import JobQueue
    from hcs_studio.training.groundtruth import AnnotationSet
    from hcs_studio.training.registry import ModelRecord
    from hcs_studio.training.train import TrainingResult, TrainingSpec


def _summary(ann: AnnotationSet) -> dict[str, Any]:
    return {
        "ref": ann.ref,
        "items": len(ann.items),
        "content_sha256": ann.content_sha256,
        "splits": ann.counts(),
    }


@service("groundtruth.import", object_type="annotation_set")
def import_ground_truth(
    ctx: ServiceContext, folder: Path, name: str, **kwargs: Any
) -> AnnotationSet:
    """A new ground-truth version from Cellpose-style training pairs."""
    from hcs_studio.training.groundtruth import import_pairs

    ann = import_pairs(
        Path(folder), name, author=kwargs.pop("author", "") or ctx.actor.name, **kwargs
    )
    ctx.note(object_id=ann.ref, after=_summary(ann))
    return ann


@service("groundtruth.create", object_type="annotation_set")
def create_ground_truth(ctx: ServiceContext, name: str, items: Any, **kwargs: Any) -> AnnotationSet:
    """A new ground-truth version from images and masks (e.g. edited segmentations)."""
    from hcs_studio.training.groundtruth import create_set

    ann = create_set(name, items, author=kwargs.pop("author", "") or ctx.actor.name, **kwargs)
    ctx.note(object_id=ann.ref, after=_summary(ann))
    return ann


@service("groundtruth.split", object_type="annotation_set")
def split_ground_truth(ctx: ServiceContext, ref: str, **kwargs: Any) -> AnnotationSet:
    """A new version with train / validation / test assigned by plate, well or cell line."""
    from hcs_studio.training.groundtruth import split_set

    ann = split_set(ref, **kwargs)
    ctx.note(object_id=ann.ref, before={"ref": ref}, after=_summary(ann))
    return ann


@service("models.register", object_type="model")
def register_model(
    ctx: ServiceContext, name: str, kind: str, source: Path, **kwargs: Any
) -> ModelRecord:
    """Copy a model file into the registry as a new candidate version."""
    from hcs_studio.training.registry import register_model as register

    record = register(
        name, kind, Path(source), author=kwargs.pop("author", "") or ctx.actor.name, **kwargs
    )
    ctx.note(object_id=record.ref, after=record.summary())
    return record


@service("models.status", object_type="model", reason_required=True)
def set_model_status(
    ctx: ServiceContext, name: str, version: int, status: str, *, approved: bool = False
) -> ModelRecord:
    """Move a version through candidate → validated → production → retired. Production needs
    ``approved`` (a confirmation; Phase 26 adds roles) and demotes the current production."""
    from hcs_studio.training.registry import get_model, set_status

    before = get_model(name, version).status
    record, demoted = set_status(
        name, version, status, approved=approved, notes=f"{status}: {ctx.reason}"
    )
    ctx.note(
        object_id=record.ref,
        before={"status": before},
        after={"status": record.status, "demoted": [d.ref for d in demoted]},
    )
    return record


@service("models.train", object_type="model")
def train_model(ctx: ServiceContext, spec: TrainingSpec, log: Any = None) -> TrainingResult:
    """Train in this process, validate and register a candidate."""
    from hcs_studio.training.train import train

    if not spec.author:
        spec = spec.model_copy(update={"author": ctx.actor.name})
    result = train(spec, log)
    ctx.note(
        object_id=result.model.ref,
        after={
            **result.model.summary(),
            "validation_ap50": result.validation.get("overall", {}).get("ap50"),
        },
    )
    return result


@service("models.train_submitted", object_type="model")
def submit_training(
    ctx: ServiceContext, queue: JobQueue, spec: TrainingSpec, priority: str = "batch"
) -> str:
    """Queue a training job (own process; GPU jobs run alone)."""
    from hcs_studio.pipeline.jobs import Job
    from hcs_studio.pipeline.queue import JobSpec, Resources

    if not spec.author:
        spec = spec.model_copy(update={"author": ctx.actor.name})
    job = Job(
        plate_source={},
        pipeline={},
        label=f"train {spec.name} ({spec.kind})",
        extra={"task": "train", "spec": spec.model_dump(mode="json")},
    )
    gpu = spec.kind == "cellpose" and spec.device != "cpu"
    job_id = queue.submit(
        JobSpec(
            job=job,
            kind="train",
            priority=priority,  # type: ignore[arg-type]
            owner=ctx.actor,
            resources=Resources(gpu=gpu),
        )
    )
    ctx.note(object_id=f"job:{job_id}", after={"spec": spec.model_dump(mode="json")})
    return job_id


@service("models.compare", object_type="model")
def compare_models(ctx: ServiceContext, candidate: str, **kwargs: Any) -> Any:
    """Score a candidate against a baseline on held-out ground truth; save the report."""
    from hcs_studio.training.compare import compare_candidate

    comparison, report = compare_candidate(candidate, **kwargs)
    ctx.note(
        object_id=candidate,
        after={
            "report": str(report),
            "set": comparison.set_ref,
            "ap50": {k: r.overall["ap50"] for k, r in comparison.results.items()},
        },
    )
    return comparison, report


@service("models.impact", object_type="model")
def model_impact(
    ctx: ServiceContext, reference_run: Path, candidate: str, out: Path, **kwargs: Any
) -> Any:
    """Re-run a reference plate with a candidate and compare counts and potencies (MSR)."""
    from hcs_studio.training.impact import downstream_impact, write_impact

    result = downstream_impact(Path(reference_run), candidate, Path(out), **kwargs)
    report = write_impact(result, Path(out))
    ctx.note(
        object_id=candidate,
        after={"report": str(report), "verdict": result.verdict, "counts": result.counts_summary},
    )
    return result, report


@service("qa.spot_check_queue", object_type="run")
def queue_spot_checks(
    ctx: ServiceContext, folder: Path, n: int = 3, seed: int | None = None
) -> list[int]:
    from hcs_studio.training.qa import queue_spot_checks as queue

    ids = queue(Path(folder), n, seed)
    ctx.note(object_id=str(folder), after={"spot_checks": ids})
    return ids


@service("qa.spot_check_review", object_type="spot_check")
def review_spot_check(ctx: ServiceContext, check_id: int, status: str) -> dict[str, Any]:
    """Accept or flag (reason required) a spot-checked field."""
    from hcs_studio.training.qa import review_spot_check as review

    record = review(check_id, status, ctx.actor.name, ctx.reason)
    ctx.note(
        object_id=f"spot_check:{check_id}",
        before={"status": record["before"]},
        after={
            "status": status,
            "run": record["run"],
            "well": record["well"],
            "field": record["field"],
        },
    )
    return record
