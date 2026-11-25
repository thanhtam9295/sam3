"""Validating a segmentation against a ground-truth set (Phase 25).

The model under test is given as pipeline steps (any segmentation module: Cellpose, classical,
SAM, … with their preprocessing). Each item image of the chosen split runs through the steps
that do not need a plate-level fit (measurement steps are skipped); the label set ``labels``
(default: the last one made) is compared with the item's masks (``training/metrics.py``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.training.groundtruth import AnnotationSet, Item, load_set, read_item
from hcs_studio.training.metrics import THRESHOLDS, field_metrics, segmentation_report


@dataclass
class SegmentationValidation:
    set_ref: str
    set_sha256: str
    split: str
    model: str
    fields: pd.DataFrame
    report: dict[str, Any]
    predictions: dict[str, NDArray[np.int32]] = field(default_factory=dict)

    @property
    def overall(self) -> dict[str, Any]:
        return dict(self.report["overall"])


def segmentation_steps(pipeline: Pipeline | Sequence[PipelineStep]) -> list[PipelineStep]:
    """The image steps of a pipeline up to its last segmentation, without measurements."""
    from hcs_studio.processing.base import MeasurementModule

    steps = pipeline.image_steps() if isinstance(pipeline, Pipeline) else list(pipeline)
    kept = [s for s in steps if not issubclass(type(s.create()), MeasurementModule)]
    if not kept:
        raise ValueError("The pipeline has no image steps to validate")
    return kept


def image_data(image: NDArray[Any], channels: Sequence[str], pixel_size_um: float | None) -> Any:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData

    names = list(channels) or [f"c{i + 1}" for i in range(image.shape[0])]
    if len(names) != image.shape[0]:
        names = [f"c{i + 1}" for i in range(image.shape[0])]
    metadata = ImageMetadata(
        channels=tuple(ChannelInfo(n) for n in names), pixel_size_um=pixel_size_um
    )
    return ImageData(np.asarray(image, np.float32)[:, None], metadata)


def predict(
    steps: Sequence[PipelineStep],
    image: NDArray[Any],
    *,
    channels: Sequence[str] = (),
    pixel_size_um: float | None = None,
    labels: str | None = None,
    states: dict[int, Any] | None = None,
) -> NDArray[np.int32]:
    """Labels (Y, X) of one (C, Y, X) image after the steps. Steps whose fit needs no plate
    (``fit_without_plate``, e.g. ilastik) are fitted once and kept in ``states``."""
    from hcs_studio.processing.base import ModuleError

    states = {} if states is None else states
    data = image_data(image, channels, pixel_size_um)
    for index, step in enumerate(steps):
        module = step.create()
        state = None
        if module.needs_fit:
            if not module.fit_without_plate:
                raise ModuleError(
                    f"Step '{step.module}' needs a plate-level fit; validate a pipeline without it"
                )
            module.validate(data.metadata)
            if index not in states:
                states[index] = module.fit(None)  # type: ignore[arg-type]
            state = states[index]
        module.validate(data.metadata)
        data = module.apply(data, state)
    if not data.labels:
        raise ModuleError("The steps made no segmentation")
    chosen = data.label_set(labels) if labels else data.labels[-1]
    return np.asarray(chosen.data[0], dtype=np.int32)


def validate_segmentation(
    ref: str | AnnotationSet,
    steps: Sequence[PipelineStep],
    *,
    split: str | None = "test",
    labels: str | None = None,
    model: str = "",
    pixel_size_um: float | None = None,
    keep_predictions: bool = False,
    root: Path | None = None,
    thresholds: Sequence[float] = THRESHOLDS,
) -> SegmentationValidation:
    ann = ref if isinstance(ref, AnnotationSet) else load_set(ref, root)
    items: list[Item] = ann.subset(split)
    if not items:
        raise ValueError(f"{ann.ref}: no items in split {split!r} (split the set first)")
    rows = []
    predictions = {}
    states: dict[int, Any] = {}
    for item in items:
        image, truth = read_item(ann, item, root)
        if truth is None:
            continue
        pred = predict(
            steps,
            image,
            channels=ann.channels,
            labels=labels,
            pixel_size_um=item.pixel_size_um or pixel_size_um,
            states=states,
        )
        if keep_predictions:
            predictions[item.id] = pred
        rows.append(
            {
                "item": item.id,
                "plate": item.plate,
                "well": item.well,
                "field": item.field,
                "cell_line": item.cell_line,
                **field_metrics(truth, pred, thresholds),
            }
        )
    fields = pd.DataFrame(rows)
    if fields.empty:
        raise ValueError(f"{ann.ref}: no items with masks in split {split!r}")
    return SegmentationValidation(
        set_ref=ann.ref,
        set_sha256=ann.content_sha256,
        split=split or "all",
        model=model or "+".join(s.module for s in steps),
        fields=fields,
        report=segmentation_report(fields, "cell_line", thresholds),
        predictions=predictions,
    )


def write_validation(result: SegmentationValidation, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    result.fields.to_csv(out / "fields.csv", index=False)
    record = {
        "set": result.set_ref,
        "set_sha256": result.set_sha256,
        "split": result.split,
        "model": result.model,
        **result.report,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    return ["fields.csv", "metrics.json"]
