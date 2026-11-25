"""The "Train in ilastik" round trip (PROMPT_ILASTIK.md §6), headless.

1. ``representative_fields``: fields covering every cell line, each control role and the top
   dose of every compound (from the plate layout), else wells spread over the plate.
2. ``export_training_set``: those fields as ilastik-ready HDF5 at the **stage the pipeline feeds
   ilastik** (raw, or recomputed from a finished run up to its first ilastik step, with the
   run's fitted plate-level estimates), plus ``training_set.json`` (source, stage, channels,
   fields with the reason each was chosen, file checksums).
3. ``ilastik_gui_command``: ilastik's own window, on an existing project or a new one.
4. Registration of the saved ``.ilp`` goes through the model registry (kind ``ilastik``;
   ``services.training.register_model``) with ``project_metadata`` (workflow, classes, axes).

Files are never written into raw data. Nothing here modifies a project.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.model.plate import parse_well_name, well_name

ILASTIK_STEPS = (
    "ilastik_pixel",
    "ilastik_autocontext",
    "ilastik_objects",
    "ilastik_multicut",
    "ilastik_tracking",
)
#: ilastik's workflow names for ``--workflow`` (new projects in the GUI)
GUI_WORKFLOWS = {
    "pixel": "PixelClassificationWorkflow",
    "autocontext": "NewAutocontextWorkflowBase",
    "objects": "ObjectClassificationWorkflowPrediction",
    "multicut": "EdgeTrainingWithMulticutWorkflow",
    "tracking": "ConservationTrackingWorkflowFromBinary",
}
FORMAT = "hcs-studio-ilastik-training-set/1"


@dataclass(frozen=True)
class TrainingField:
    well: str
    field: int
    timepoint: int = 0
    reason: str = ""


def representative_fields(
    wells: list[tuple[int, int]],
    fields_per_well: dict[tuple[int, int], list[int]],
    layout: dict[tuple[int, int], Any] | None = None,
    *,
    max_fields: int = 24,
) -> list[TrainingField]:
    """Every cell line × (each control role, vehicle, the top dose of every compound), one well
    each, first field; without a layout, wells spread evenly over the plate."""
    wells = sorted(w for w in wells if fields_per_well.get(w))
    if not wells:
        return []
    chosen: dict[tuple[int, int], str] = {}
    if layout:
        groups: dict[str, tuple[int, int]] = {}
        top: dict[tuple[str, str], tuple[float, tuple[int, int]]] = {}
        for key in wells:
            annotation = layout.get(key)
            if annotation is None or annotation.role in ("empty", "excluded", "blank"):
                continue
            line = annotation.cell_line or "no cell line"
            if annotation.role != "sample" or not annotation.treatments:
                label = f"{line}: {annotation.role.replace('_', ' ')}"
                groups.setdefault(label, key)
                continue
            for treatment in annotation.treatments:
                best = top.get((line, treatment.compound))
                if best is None or treatment.concentration > best[0]:
                    top[(line, treatment.compound)] = (treatment.concentration, key)
        for (line, compound), (_, key) in sorted(top.items()):
            groups.setdefault(f"{line}: {compound} top dose", key)
        for label, key in sorted(groups.items()):
            chosen.setdefault(key, label)
    if not chosen:
        count = min(max_fields, len(wells))
        picks = np.linspace(0, len(wells) - 1, count).round().astype(int)
        chosen = {wells[i]: "spread over the plate" for i in sorted(set(picks.tolist()))}
    out = [
        TrainingField(well_name(*key), fields_per_well[key][0], 0, reason)
        for key, reason in sorted(chosen.items())
    ]
    return out[:max_fields]


def ilastik_stage(run: Path) -> tuple[str, list[int] | None]:
    """The source a run's first ilastik step sees (``raw`` or ``step:<n>``) and the channels it
    sends (from 1; None = all)."""
    from hcs_studio.pipeline.schema import Pipeline

    steps = Pipeline.load(Path(run) / "pipeline.yaml").image_steps()
    for index, step in enumerate(steps):
        if step.module in ILASTIK_STEPS:
            channels = step.params.get("channels")
            return ("raw" if index == 0 else f"step:{index}"), channels
    raise ValueError(f"{Path(run).name} has no ilastik step: choose the stage yourself")


def export_training_set(
    source: Path,
    fields: list[TrainingField],
    folder: Path,
    *,
    stage: str = "raw",
    channels: list[int] | None = None,
) -> Path:
    """Write the fields as HDF5 + ``training_set.json``; returns the JSON file.

    ``source``: a finished run folder (stages ``raw``, ``step:<n>`` or ``ilastik`` = what its
    first ilastik step sees), or a raw plate (stage ``raw`` only)."""
    from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes, to_ilastik
    from hcs_studio.segmentation.ilastik_backend.h5io import write_h5

    if not fields:
        raise ValueError("Choose at least one field")
    source = Path(source)
    is_run = (source / "run_manifest.json").exists()
    if stage == "ilastik":
        if not is_run:
            raise ValueError("The 'ilastik' stage needs a run folder (its pipeline)")
        stage, step_channels = ilastik_stage(source)
        channels = channels or step_channels
    if stage != "raw" and not is_run:
        raise ValueError(f"Stage {stage!r} needs a run folder; a raw plate gives 'raw' only")
    folder = Path(folder).absolute()
    raw_root = _raw_root(source, is_run)
    if raw_root is not None and (folder == raw_root or raw_root in folder.parents):
        raise ValueError("Write the training set outside the raw data")
    folder.mkdir(parents=True, exist_ok=True)
    plate = None if is_run else _open(source)
    records = []
    channel_names: list[str] = []
    for item in fields:
        image = _field_image(source, is_run, plate, item, stage)
        data = image.data
        names = [c.name for c in image.metadata.channels]
        if channels:
            bad = [c for c in channels if not 1 <= c <= len(names)]
            if bad:
                raise ValueError(f"Channel {bad[0]} does not exist ({len(names)} channels)")
            data = data[[c - 1 for c in channels]]
            names = [names[c - 1] for c in channels]
        channel_names = names
        stack = np.asarray(data, dtype=np.float32)[np.newaxis]  # (1, C, Z, Y, X)
        axes = ilastik_axes(stack.shape)
        name = f"{item.well}_f{item.field:03d}_t{item.timepoint}.h5"
        path = write_h5(folder / name, to_ilastik(stack, axes), axes)
        records.append({**asdict(item), "file": name, "axes": axes, "sha256": _sha256(path)})
    from hcs_studio import __version__

    manifest = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software": f"hcs-studio {__version__}",
        "source": str(source.absolute()),
        "source_kind": "run" if is_run else "plate",
        "stage": stage,
        "channels": channel_names,
        "channel_numbers": channels,
        "fields": records,
        "note": "Train with the same stage and channels the pipeline will send to ilastik.",
    }
    target = folder / "training_set.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def project_metadata(project: Path, training_set: Path | None = None) -> dict[str, Any]:
    """What the model registry stores with an ilastik project (read-only; never unpickles)."""
    from hcs_studio.segmentation.ilastik_backend.project import read_project

    info = read_project(Path(project))
    out: dict[str, Any] = {
        "workflow": info.workflow,
        "workflow_name": info.workflow_name,
        "classes": list(info.label_names),
        "axes": info.axes,
        "channels": info.channels,
        "saved_with": info.ilastik_version,
    }
    if training_set is not None:
        data = json.loads(Path(training_set).read_text(encoding="utf-8"))
        out["training_set"] = {
            "file": str(Path(training_set).absolute()),
            "stage": data.get("stage"),
            "channels": data.get("channels"),
            "fields": len(data.get("fields", [])),
            "sha256": _sha256(Path(training_set)),
        }
    return out


def ilastik_gui_command(
    executable: str | Path, project: Path, workflow: str = "pixel"
) -> list[str]:
    """ilastik's window on ``project``: opened when it exists, else created for ``workflow``
    (``--new_project`` needs an absolute path; checked on ilastik 1.4.2)."""
    from hcs_studio.segmentation.ilastik_backend.settings import ilastik_command

    project = Path(project).absolute()
    command = ilastik_command(executable)
    if project.exists():
        return [*command, f"--project={project}"]
    if workflow not in GUI_WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r} ({', '.join(GUI_WORKFLOWS)})")
    return [*command, f"--new_project={project}", f"--workflow={GUI_WORKFLOWS[workflow]}"]


# ------------------------------------------------------------------ helpers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(path: Path) -> Any:
    from hcs_studio.io.base import open_plate

    return open_plate(path)


def _raw_root(source: Path, is_run: bool) -> Path | None:
    if not is_run:
        return source.absolute()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    path = manifest.get("plate", {}).get("path")
    return Path(path).absolute() if path else None


def _field_image(source: Path, is_run: bool, plate: Any, item: TrainingField, stage: str) -> Any:
    from hcs_studio.processing.base import ImageData

    if is_run:
        from hcs_studio.compare.sources import _recompute

        return _recompute(source, item.well, item.field, stage, item.timepoint).image
    key = parse_well_name(item.well)
    if key not in plate.wells:
        raise ValueError(f"{item.well} is not on the plate")
    field = next((f for f in plate.wells[key].fields if f.index == item.field), None)
    if field is None:
        raise ValueError(f"{item.well} has no field {item.field}")
    data = np.asarray(field.load()[item.timepoint], dtype=np.float32)
    return ImageData(data, plate.metadata)


def fields_per_well(plate: Any) -> dict[tuple[int, int], list[int]]:
    return {key: [f.index for f in well.fields] for key, well in plate.wells.items()}
