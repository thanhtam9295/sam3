"""Pipeline presets around an ilastik project (PROMPT_ILASTIK.md §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcs_studio.pipeline.schema import Pipeline, PipelineStep


def pixel_preset(
    project: Path,
    info: Any,
    *,
    object_class: str | None = None,
    channels: list[int] | None = None,
    projection: bool = False,
) -> Pipeline:
    """``[projection →] ilastik Pixel Classification / Autocontext → probabilities → objects →
    measure objects → class areas → field QC``. The object class defaults to the last class
    (ilastik projects usually list the background first)."""
    if info.workflow == "objects":
        return objects_preset(project, channels=channels)
    names = list(info.label_names)
    if not names:
        raise ValueError("The project's class names were not found: give --class")
    chosen = object_class or names[-1]
    if chosen not in names:
        raise ValueError(f"No class {chosen!r} in the project ({', '.join(names)})")
    module = "ilastik_autocontext" if info.workflow == "autocontext" else "ilastik_pixel"
    params: dict[str, Any] = {"project": str(Path(project).absolute())}
    if channels:
        params["channels"] = channels
    from hcs_studio.segmentation.ilastik_backend.steps import class_channel

    label = chosen.strip().lower().replace(" ", "_") or "objects"
    steps = [PipelineStep(module="projection")] if projection else []
    steps += [
        PipelineStep(module=module, params=params),
        PipelineStep(
            module="probabilities_to_objects",
            params={"channel": class_channel(chosen), "name": label},
        ),
        PipelineStep(module="measure_objects", params={"labels": [label]}),
        PipelineStep(module="ilastik_class_areas"),
        PipelineStep(module="qc"),
    ]
    return Pipeline(steps=steps)


def objects_preset(
    project: Path, *, channels: list[int] | None = None, nucleus_channel: int = 1
) -> Pipeline:
    """``Cellpose nuclei → ilastik Object Classification → measure objects → field QC →
    class counts and fractions`` (a "from label image" project; for pharmacology, use the
    ``class_ilastik_<class>_fraction`` readouts)."""
    params: dict[str, Any] = {"project": str(Path(project).absolute()), "labels": "nuclei"}
    if channels:
        params["channels"] = channels
    return Pipeline(
        steps=[
            PipelineStep(module="segment_cellpose", params={"channel": nucleus_channel}),
            PipelineStep(module="ilastik_objects", params=params),
            PipelineStep(module="measure_objects", params={"labels": ["nuclei"]}),
            PipelineStep(module="qc"),
            PipelineStep(module="ilastik_object_classes", params={"object_type": "nuclei"}),
        ]
    )
