"""ilastik pipeline steps (PROMPT_ILASTIK.md §7.1, step b).

- ``ilastik_pixel`` / ``ilastik_autocontext``: run a trained Pixel Classification /
  Autocontext project on every field of a well in **one** ilastik call (well batch). The
  result is added to the image as one channel per class, named ``ilastik_<class>``:
  probabilities (0–1), or 0/1 masks from ilastik's Simple Segmentation. The original channels
  are untouched (still linear); later steps measure the class channels like any other (mean
  probability per object) and use them to make objects.
- ``probabilities_to_objects``: a class channel → instance labels (threshold with optional
  hysteresis, smoothing, size filter, split touching objects by distance transform or by seeds
  from another label set).
- ``ilastik_class_areas``: % of each field per class (argmax of the class channels) →
  ``ilastik_area_pct_<class>`` in the fields table, averaged per well.

Run folder: ``<run>/ilastik/projects_used.json``, ``logs/<well>.log`` (the exact command
lines), ``inputs/`` (cached HDF5, deleted after each well unless ``keep_inputs``) and
``wells/<W>/ilastik/<field>_<export>.h5`` (ilastik's own outputs) + ``class_summary.json``.
The manifest's fitted state records the project's SHA-256, ilastik's version and the runner.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import ndimage

from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    BatchFolders,
    FitContext,
    ImageData,
    LabelSet,
    MeasureContext,
    MeasurementModule,
    Module,
    ModuleError,
    ModuleParams,
)

CHANNEL_PREFIX = "ilastik_"


def class_channel(name: str) -> str:
    return CHANNEL_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()


class PixelParams(ModuleParams):
    project: str = Field(default="", description="The trained ilastik project (.ilp).")
    export: Literal["probabilities", "segmentation"] = Field(
        default="probabilities",
        description="Class probabilities (0–1) or ilastik's Simple Segmentation (0/1 masks).",
    )
    channels: list[int] | None = Field(
        default=None, description="Channels sent to ilastik, in the trained order; empty = all."
    )
    class_names: list[str] | None = Field(
        default=None, description="Class names when the project does not hold them."
    )
    keep_inputs: bool = Field(default=False, description="Keep the HDF5 copies sent to ilastik.")
    timeout_min: float = Field(default=120.0, gt=0, description="Give up on one well after this.")


class IlastikPixel(Module):
    """Pixel Classification (one call per well)."""

    name: ClassVar[str] = "ilastik_pixel"
    title: ClassVar[str] = "ilastik Pixel Classification"
    Params = PixelParams
    needs_fit: ClassVar[bool] = True
    exclusive: ClassVar[bool] = True
    batch_scope: ClassVar[str] = "well"
    workflow: ClassVar[str] = "pixel"
    sources: ClassVar[dict[str, str]] = {
        "probabilities": "Probabilities",
        "segmentation": "Simple Segmentation",
    }

    # --- project and classes ---------------------------------------------------------------

    def project_info(self) -> Any:
        from hcs_studio.segmentation.ilastik_backend.project import read_project

        params: PixelParams = self.params
        if not params.project:
            raise ModuleError("Choose the ilastik project (.ilp)")
        path = Path(_project_path(params.project))
        try:
            return read_project(path)
        except (FileNotFoundError, ValueError) as exc:
            raise ModuleError(str(exc)) from None

    def class_names(self) -> list[str]:
        params: PixelParams = self.params
        if params.class_names:
            return list(params.class_names)
        names = self.project_info().label_names
        if not names:
            raise ModuleError(
                "The project's class names were not found: enter them in the step (class_names)"
            )
        return list(names)

    def validate(self, metadata: ImageMetadata) -> None:
        params: PixelParams = self.params
        info = self.project_info()
        if info.workflow not in ("unknown", self.workflow):
            raise ModuleError(
                f"{Path(params.project).name} is a {info.workflow_name or info.workflow} "
                f"project; this step needs {self.title}"
            )
        for channel in params.channels or []:
            if not 1 <= channel <= len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")
        names = [class_channel(n) for n in self.class_names()]
        if len(set(names)) != len(names):
            raise ModuleError("Class names must differ after cleaning: " + ", ".join(names))
        if any(c.name in names for c in metadata.channels):
            raise ModuleError("The image already has ilastik class channels (two ilastik steps?)")

    def output_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        extra = tuple(ChannelInfo(class_channel(n)) for n in self.class_names())
        return replace(metadata, channels=(*metadata.channels, *extra))

    # --- plate-level: which ilastik, which project ------------------------------------------

    def fit(self, context: FitContext) -> dict[str, Any]:
        from hcs_studio.segmentation.ilastik_backend.job import IlastikError
        from hcs_studio.segmentation.ilastik_backend.runners import runner_state

        try:
            installation = runner_state()
        except IlastikError as exc:
            raise ModuleError(str(exc)) from None
        info = self.project_info()
        return {
            "project": info.path,
            "sha256": info.sha256,
            "workflow": info.workflow_name or info.workflow,
            "saved_with": info.ilastik_version,
            "classes": self.class_names(),
            **installation,
        }

    def describe_state(self, state: Any) -> dict[str, Any]:
        return dict(state or {})

    # --- running ----------------------------------------------------------------------------

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return self.apply_batch([image], state, [(1, 0)], None)[0]

    def apply_batch(
        self,
        images: list[ImageData],
        state: Any,
        keys: list[tuple[int, int]],
        workdir: Any = None,
    ) -> list[ImageData]:
        from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes, to_ilastik
        from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, read_h5, write_h5
        from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob
        from hcs_studio.segmentation.ilastik_backend.project import check_compatible
        from hcs_studio.segmentation.ilastik_backend.runners import run_job

        params: PixelParams = self.params
        if not images:
            return []
        state = state or self.fit(None)  # type: ignore[arg-type]
        info = self.project_info()
        if info.sha256 != state["sha256"]:
            raise ModuleError("The ilastik project changed since the run started")
        temporary = None
        if isinstance(workdir, BatchFolders):
            run_folder, well_folder, well = (
                Path(workdir.run) / "ilastik",
                Path(workdir.well),
                workdir.well_name,
            )
        else:
            temporary = Path(tempfile.mkdtemp(prefix="hcs-ilastik-"))
            run_folder, well_folder, well = temporary, temporary / "well", "preview"
        inputs = run_folder / "inputs" / well
        outputs = well_folder / "ilastik"
        selected = [c - 1 for c in params.channels] if params.channels else None
        specs, names = [], []
        try:
            for image, (field_index, t) in zip(images, keys, strict=True):
                data = image.data if selected is None else image.data[selected]
                stack = data[np.newaxis]  # (T=1, C, Z, Y, X)
                axes = ilastik_axes(stack.shape)
                problems = check_compatible(info, axes, stack.shape[1])
                if problems:
                    raise ModuleError("; ".join(problems))
                name = f"f{field_index:03d}_t{t}"
                path = write_h5(
                    inputs / f"{name}.h5", to_ilastik(stack, axes), axes, compression=None
                )
                specs.append(input_spec(path))
                names.append(name)
            suffix = self.sources[params.export].replace(" ", "_")
            job = IlastikJob(
                project=state["project"],
                workflow=self.workflow,
                raw_data=specs,
                export_source=self._export_source(params.export),
                output_folder=str(outputs),
                output_suffix=suffix,
                threads=int(state.get("threads", -1)),
                ram_mb=int(state.get("ram_mb", 0)),
                timeout_s=params.timeout_min * 60,
            )
            try:
                result = run_job(state, job, log_path=run_folder / "logs" / f"{well}.log")
            except IlastikError as exc:
                raise ModuleError(str(exc)) from None
            classes = state["classes"]
            out_images, summary = [], {}
            for image, name, output in zip(images, names, result.outputs, strict=True):
                array, axes = read_h5(Path(output))
                planes = _class_planes(array, axes, len(classes), params.export, image.data.shape)
                summary[name] = {
                    c: round(float(p.mean()) * 100, 3) for c, p in zip(classes, planes, strict=True)
                }
                metadata = replace(
                    image.metadata,
                    channels=(
                        *image.metadata.channels,
                        *(ChannelInfo(class_channel(c)) for c in classes),
                    ),
                )
                out_images.append(
                    ImageData(
                        data=np.concatenate([image.data, planes], axis=0).astype(np.float32),
                        metadata=metadata,
                        linear=image.linear,
                        history=(*image.history, self.name),
                        reports=(
                            *image.reports,
                            {
                                "module": self.name,
                                "seconds_per_field": result.seconds / len(images),
                            },
                        ),
                        labels=image.labels,
                    )
                )
            if temporary is None:
                outputs.mkdir(parents=True, exist_ok=True)
                (outputs / "class_summary.json").write_text(
                    json.dumps(
                        {
                            "classes": classes,
                            "area_pct_by_field": summary,
                            "command": result.command,
                            "seconds": result.seconds,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _projects_used(run_folder, state, [result.runner])
            return out_images
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            elif not params.keep_inputs:
                shutil.rmtree(inputs, ignore_errors=True)

    def _export_source(self, export: str) -> str:
        return self.sources[export]


@MODULES.register("ilastik_autocontext")
class IlastikAutocontext(IlastikPixel):
    """Autocontext (the second stage's result)."""

    name: ClassVar[str] = "ilastik_autocontext"
    title: ClassVar[str] = "ilastik Autocontext"
    workflow: ClassVar[str] = "autocontext"
    sources: ClassVar[dict[str, str]] = {
        "probabilities": "Probabilities Stage 2",
        "segmentation": "Simple Segmentation Stage 2",
    }


MODULES.register("ilastik_pixel")(IlastikPixel)


def _project_path(value: str) -> str:
    """A path, or a model-registry reference ``registry:NAME@VERSION`` (Phase 25)."""
    if value.startswith("registry:"):
        from hcs_studio.training.registry import resolve_path

        return str(resolve_path(value))
    return value


def _class_planes(
    array: np.ndarray, axes: str, n_classes: int, export: str, shape: tuple[int, ...]
) -> np.ndarray:
    """ilastik's output → (classes, Z, Y, X) float32 on the image's grid."""
    from hcs_studio.segmentation.ilastik_backend.axes import from_ilastik

    tczyx = from_ilastik(array, axes)[0]  # (C, Z, Y, X)
    if export == "segmentation":
        values = tczyx[0]
        planes = np.stack([(values == k + 1) for k in range(n_classes)]).astype(np.float32)
    else:
        if tczyx.shape[0] != n_classes:
            raise ModuleError(
                f"ilastik returned {tczyx.shape[0]} probability channels for {n_classes} classes"
            )
        planes = tczyx.astype(np.float32)
    if planes.shape[1:] != tuple(shape[1:]):
        raise ModuleError(
            f"ilastik's output {planes.shape[1:]} does not match the field {tuple(shape[1:])}"
        )
    return planes


def _projects_used(folder: Path, state: dict[str, Any], runners: list[str]) -> None:
    """``<run>/ilastik/projects_used.json``: one entry per project (by SHA-256) with the runners
    that actually ran it (a job the Fiji runner cannot do runs headless)."""
    path = folder / "projects_used.json"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        records = []
    record = next((r for r in records if r.get("sha256") == state.get("sha256")), None)
    if record is None:
        keys = ["project", "sha256", "workflow", "saved_with", "ilastik", "ilastik_version"]
        keys += ["classes"] + [k for k in ("fiji", "ilastik4ij") if k in state]
        record = {k: state.get(k) for k in keys}
        record["runner"] = ""
        records.append(record)
    known = {r for r in str(record.get("runner") or "").split(", ") if r}
    record["runner"] = ", ".join(sorted(known | set(runners)))
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


# --------------------------------------------------------------- probabilities → objects


class ProbObjectsParams(ModuleParams):
    channel: str = Field(
        default="", description="Class channel (e.g. ilastik_nuclei, or the class name)."
    )
    threshold: float = Field(default=0.5, gt=0, lt=1, description="Probability for an object.")
    low_threshold: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Hysteresis: grow objects into pixels above this (≤ threshold).",
    )
    smoothing_px: float = Field(default=1.0, ge=0, description="Gaussian smoothing first.")
    min_size_px: int = Field(default=20, ge=0, description="Remove smaller objects.")
    max_size_px: int | None = Field(default=None, ge=1, description="Remove larger objects.")
    split: Literal["none", "distance", "seeds"] = Field(
        default="distance", description="Split touching objects."
    )
    min_distance_px: float = Field(default=5.0, gt=0, description="Distance split: centre spacing.")
    seeds: str | None = Field(default=None, description="Seeds: a label set (e.g. nuclei).")
    name: str = Field(default="objects", description="Name of the new label set.")


@MODULES.register("probabilities_to_objects")
class ProbabilitiesToObjects(Module):
    name: ClassVar[str] = "probabilities_to_objects"
    title: ClassVar[str] = "Probabilities → objects"
    Params = ProbObjectsParams

    def _index(self, metadata: ImageMetadata) -> int:
        params: ProbObjectsParams = self.params
        names = [c.name for c in metadata.channels]
        wanted = [params.channel, class_channel(params.channel)] if params.channel else []
        for name in wanted:
            if name in names:
                return names.index(name)
        classes = [n for n in names if n.startswith(CHANNEL_PREFIX)]
        raise ModuleError(
            f"No class channel {params.channel!r}; the image has: {', '.join(classes) or 'none'} "
            "(put an ilastik step before this one)"
        )

    def validate(self, metadata: ImageMetadata) -> None:
        params: ProbObjectsParams = self.params
        self._index(metadata)
        if params.low_threshold is not None and params.low_threshold > params.threshold:
            raise ModuleError("The low threshold must not be above the threshold")
        if params.split == "seeds" and not params.seeds:
            raise ModuleError("Name the label set that gives the seeds")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ProbObjectsParams = self.params
        self.validate(image.metadata)
        prob = image.data[self._index(image.metadata)]
        seeds = (
            image.label_set(params.seeds).data if params.split == "seeds" and params.seeds else None
        )
        labels = probabilities_to_labels(
            prob,
            params.threshold,
            params.low_threshold,
            params.smoothing_px,
            params.min_size_px,
            params.max_size_px,
            params.split,
            params.min_distance_px,
            seeds,
        )
        label_set = LabelSet(params.name, labels, source=self.name)
        report = {"module": self.name, "objects": int(label_set.count)}
        return image.with_labels(label_set, self.name, report)


def probabilities_to_labels(
    prob: np.ndarray,
    threshold: float = 0.5,
    low: float | None = None,
    smoothing: float = 1.0,
    min_size: int = 20,
    max_size: int | None = None,
    split: str = "distance",
    min_distance: float = 5.0,
    seeds: np.ndarray | None = None,
) -> np.ndarray:
    """Instance labels (int32, Z, Y, X) from a probability map (Z, Y, X); 2D per plane."""
    from skimage.feature import peak_local_max
    from skimage.filters import apply_hysteresis_threshold
    from skimage.segmentation import watershed

    out = np.zeros(prob.shape, dtype=np.int32)
    next_id = 0
    for z in range(prob.shape[0]):
        plane = (
            ndimage.gaussian_filter(prob[z].astype(np.float32), smoothing) if smoothing else prob[z]
        )
        if low is not None:
            mask = apply_hysteresis_threshold(plane, low, threshold)
        else:
            mask = plane >= threshold
        mask = ndimage.binary_fill_holes(mask)
        if split == "seeds" and seeds is not None:
            markers = np.where(mask, seeds[z], 0)
            labels = watershed(-plane, markers=markers, mask=mask)
        elif split == "distance":
            distance = ndimage.distance_transform_edt(mask)
            components, _ = ndimage.label(mask)
            peaks = peak_local_max(
                distance,
                min_distance=max(1, int(min_distance)),
                labels=components,
                exclude_border=False,
            )
            markers = np.zeros(mask.shape, dtype=np.int32)
            markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
            labels = watershed(-distance, markers=markers, mask=mask)
        else:
            labels, _ = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())
        drop = np.zeros(len(sizes), dtype=bool)
        if min_size:
            drop |= sizes < min_size
        if max_size is not None:
            drop |= sizes > max_size
        drop[0] = True
        keep = np.flatnonzero(~drop)
        remap = np.zeros(len(sizes), dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1) + next_id
        out[z] = remap[labels]
        next_id += len(keep)
    return out


# ---------------------------------------------------------------------- class areas


class ClassAreaParams(ModuleParams):
    pass


@MODULES.register("ilastik_class_areas")
class IlastikClassAreas(MeasurementModule):
    """% of each field per ilastik class (the class with the highest value per pixel)."""

    name: ClassVar[str] = "ilastik_class_areas"
    title: ClassVar[str] = "ilastik class areas"
    Params = ClassAreaParams

    def validate(self, metadata: ImageMetadata) -> None:
        if not any(c.name.startswith(CHANNEL_PREFIX) for c in metadata.channels):
            raise ModuleError("No ilastik class channels: put an ilastik step before this one")

    def measure(self, context: MeasureContext) -> dict[str, pd.DataFrame]:
        image = context.corrected
        index = [
            i for i, c in enumerate(image.metadata.channels) if c.name.startswith(CHANNEL_PREFIX)
        ]
        planes = image.data[index]
        winner = np.argmax(planes, axis=0)
        row = {
            f"ilastik_area_pct_{image.metadata.channels[i].name[len(CHANNEL_PREFIX) :]}": float(
                (winner == k).mean() * 100
            )
            for k, i in enumerate(index)
        }
        return {"fields": pd.DataFrame([row])}
