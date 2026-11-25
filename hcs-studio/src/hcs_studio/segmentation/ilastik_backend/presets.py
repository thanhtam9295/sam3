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
    if info.workflow == "tracking":
        return tracking_preset(project)
    if info.workflow == "multicut":
        raise ValueError(
            "A Multicut project needs its boundary pixel-classification project "
            "too: give it with --boundary"
        )
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


def multicut_preset(boundary_project: Path, multicut_project: Path) -> Pipeline:
    """``ilastik boundary pixel classification → Multicut → measure objects → field QC``."""
    return Pipeline(
        steps=[
            PipelineStep(
                module="ilastik_pixel", params={"project": str(Path(boundary_project).absolute())}
            ),
            PipelineStep(
                module="ilastik_multicut",
                params={"project": str(Path(multicut_project).absolute())},
            ),
            PipelineStep(module="measure_objects", params={"labels": ["cells"]}),
            PipelineStep(module="qc"),
        ]
    )


def tracking_preset(tracking_project: Path, nucleus_channel: int = 1) -> Pipeline:
    """``nuclei → ilastik Tracking → measure tracked objects → ilastik tracks`` (time-lapse)."""
    return Pipeline(
        steps=[
            PipelineStep(module="segment_cellpose", params={"channel": nucleus_channel}),
            PipelineStep(
                module="ilastik_tracking",
                params={"project": str(Path(tracking_project).absolute())},
            ),
            PipelineStep(module="measure_objects", params={"labels": ["tracked"]}),
            PipelineStep(module="ilastik_tracks"),
        ]
    )


def _pixel_step(project: Path, info: Any, channels: list[int] | None) -> PipelineStep:
    if info.workflow not in ("pixel", "autocontext"):
        raise ValueError(
            f"This preset needs a Pixel Classification or Autocontext project, not "
            f"{info.workflow_name or info.workflow}"
        )
    module = "ilastik_autocontext" if info.workflow == "autocontext" else "ilastik_pixel"
    params: dict[str, Any] = {"project": str(Path(project).absolute())}
    if channels:
        params["channels"] = channels
    return PipelineStep(module=module, params=params)


def _class(info: Any, object_class: str | None) -> str:
    from hcs_studio.segmentation.ilastik_backend.steps import class_channel

    names = list(info.label_names)
    if not names:
        raise ValueError("The project's class names were not found: give --class")
    chosen = object_class or names[-1]
    if chosen not in names:
        raise ValueError(f"No class {chosen!r} in the project ({', '.join(names)})")
    return class_channel(chosen)


def spheroid_preset(
    project: Path,
    info: Any,
    *,
    brightfield_channel: int = 1,
    object_class: str | None = None,
    channels: list[int] | None = None,
    organoids: bool = False,
    min_area_um2: float = 300.0,
) -> Pipeline:
    """Phase 21 with ilastik detection: ``ilastik pixel → detect_spheroids (probability of the
    spheroid class) → measure_spheroids → spheroid`` (single spheroid or organoid mode)."""
    from hcs_studio.assays.spheroids import preset

    base = preset({"brightfield": "Brightfield"}, organoids=organoids, min_area_um2=min_area_um2)
    steps = [_pixel_step(project, info, channels)]
    for step in base.steps:
        params = dict(step.params)
        if step.module == "detect_spheroids":
            params = {
                "channel": brightfield_channel,
                "method": "probability",
                "probability_channel": _class(info, object_class),
                "min_area_um2": min_area_um2,
            }
        elif "channel" in params:
            params["channel"] = brightfield_channel
        steps.append(PipelineStep(module=step.module, params=params))
    return Pipeline(steps=steps)


def confluence_preset(
    project: Path,
    info: Any,
    *,
    object_class: str | None = None,
    channels: list[int] | None = None,
) -> Pipeline:
    """Phase 22 with ilastik: ``ilastik pixel → confluence (probability of the cell class) →
    class areas``; add the ``kinetics`` table step for growth curves of ``confluence_pct``."""
    return Pipeline(
        steps=[
            _pixel_step(project, info, channels),
            PipelineStep(
                module="confluence",
                params={"method": "probability", "class_channel": _class(info, object_class)},
            ),
            PipelineStep(module="ilastik_class_areas"),
        ]
    )
