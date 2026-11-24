"""Service functions for ground truth and models (Phase 25)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hcs_studio.services import ServiceContext, service

if TYPE_CHECKING:
    from hcs_studio.training.groundtruth import AnnotationSet


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
