"""ilastik Object Classification (PROMPT_ILASTIK.md §7.2, step c).

``ilastik_objects`` (image step, one ilastik call per well) classifies objects with a trained
Object Classification project:

- ``input: labels`` (default) — an existing label set (Cellpose, SAM, classical, or objects
  from ilastik probabilities) is sent as ``--label_image``, which keeps HCS Studio's object
  IDs. Each object gets the class most of its pixels have in ilastik's "Object Predictions"
  export (majority vote; the agreement is recorded). For projects trained "from label image".
- ``input: prediction_maps`` / ``segmentation_image`` — projects trained on probabilities or a
  binary mask: ilastik makes its own objects. Two calls per well ("Object Identities", then
  "Object Predictions"); ilastik's objects become a new label set ``<name>_objects``.

The classes go to ``wells/<W>/ilastik/object_classes_<name>.parquet``; the table step
``ilastik_object_classes`` adds ``class_<name>`` (class name) and ``class_<name>_agreement`` to
the object tables and per-well ``class_<name>_<class>_count`` / ``_fraction`` — the same columns
as HCS Studio's own classifier, so plate maps, reports and pharmacology use them as they are.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from hcs_studio.analysis.table_steps import (
    TableModule,
    TableParams,
    WellContext,
    count_fractions,
)
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    BatchFolders,
    FitContext,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
)

INPUT_FLAGS = {
    "labels": "label_image",
    "prediction_maps": "prediction_maps",
    "segmentation_image": "segmentation_image",
}


class ObjectsParams(ModuleParams):
    project: str = Field(default="", description="The trained Object Classification project.")
    input: Literal["labels", "prediction_maps", "segmentation_image"] = Field(
        default="labels",
        description="What the project was trained on: a label image (your objects), "
        "probabilities or a binary mask.",
    )
    labels: str = Field(default="nuclei", description="input labels: the objects to classify.")
    source_channels: list[str] | None = Field(
        default=None,
        description="prediction_maps / segmentation_image: the class channels sent "
        "(default: every ilastik_ channel; a mask uses the first).",
    )
    channels: list[int] | None = Field(
        default=None, description="Raw channels sent, in the trained order; empty = all."
    )
    class_names: list[str] | None = Field(
        default=None, description="Class names when the project does not hold them."
    )
    name: str = Field(default="ilastik", description="Column name: class_<name>.")
    keep_inputs: bool = Field(default=False, description="Keep the HDF5 copies sent to ilastik.")
    timeout_min: float = Field(default=120.0, gt=0, description="Give up on one well after this.")


@MODULES.register("ilastik_objects")
class IlastikObjects(Module):
    """Object Classification with ilastik (one call per well; two for ilastik's own objects)."""

    name: ClassVar[str] = "ilastik_objects"
    title: ClassVar[str] = "ilastik Object Classification"
    Params = ObjectsParams
    needs_fit: ClassVar[bool] = True
    exclusive: ClassVar[bool] = True
    batch_scope: ClassVar[str] = "well"
    fit_without_plate = True

    def project_info(self) -> Any:
        from hcs_studio.segmentation.ilastik_backend.project import read_project
        from hcs_studio.segmentation.ilastik_backend.steps import _project_path

        params: ObjectsParams = self.params
        if not params.project:
            raise ModuleError("Choose the ilastik Object Classification project (.ilp)")
        try:
            return read_project(Path(_project_path(params.project)))
        except (FileNotFoundError, ValueError) as exc:
            raise ModuleError(str(exc)) from None

    def class_names(self) -> list[str]:
        params: ObjectsParams = self.params
        names = list(params.class_names or self.project_info().label_names)
        if not names:
            raise ModuleError("The project's class names were not found: enter class_names")
        return names

    def validate(self, metadata: ImageMetadata) -> None:
        params: ObjectsParams = self.params
        info = self.project_info()
        if info.workflow not in ("unknown", "objects"):
            raise ModuleError(
                f"{Path(params.project).name} is a {info.workflow_name or info.workflow} "
                "project; this step needs an Object Classification project"
            )
        for channel in params.channels or []:
            if not 1 <= channel <= len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")
        if params.input != "labels":
            available = [c.name for c in metadata.channels if c.name.startswith("ilastik_")]
            wanted = params.source_channels or available
            missing = [c for c in wanted if c not in available]
            if not wanted or missing:
                raise ModuleError(
                    f"{params.input} needs ilastik class channels (put ilastik_pixel first)"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                )
        self.class_names()

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
            "input": self.params.input,
            **installation,
        }

    def describe_state(self, state: Any) -> dict[str, Any]:
        return dict(state or {})

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return self.apply_batch([image], state, [(1, 0)], None)[0]

    def _raw(self, image: ImageData) -> np.ndarray:
        params: ObjectsParams = self.params
        keep = [
            i for i, c in enumerate(image.metadata.channels) if not c.name.startswith("ilastik_")
        ]
        if params.channels:
            keep = [c - 1 for c in params.channels]
        return image.data[keep]

    def _second_input(self, image: ImageData) -> tuple[np.ndarray, bool]:
        """(C, Z, Y, X) data of the second input role and whether it is a label image."""
        params: ObjectsParams = self.params
        if params.input == "labels":
            try:
                labels = image.label_set(params.labels).data
            except KeyError as exc:
                raise ModuleError(f"No label set {params.labels!r}") from exc
            return labels[np.newaxis].astype(np.uint32), True
        names = [c.name for c in image.metadata.channels]
        wanted = params.source_channels or [n for n in names if n.startswith("ilastik_")]
        data = image.data[[names.index(n) for n in wanted]]
        if params.input == "segmentation_image":
            return (data[:1] >= 0.5).astype(np.uint8), True
        return data.astype(np.float32), False

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
        from hcs_studio.segmentation.ilastik_backend.steps import _projects_used

        params: ObjectsParams = self.params
        if not images:
            return []
        state = state or self.fit(None)  # type: ignore[arg-type]
        info = self.project_info()
        if info.sha256 != state["sha256"]:
            raise ModuleError("The ilastik project changed since the run started")
        temporary = None
        if isinstance(workdir, BatchFolders):
            run_folder = Path(workdir.run) / "ilastik"
            well_folder, well = Path(workdir.well), workdir.well_name
        else:
            temporary = Path(tempfile.mkdtemp(prefix="hcs-ilastik-"))
            run_folder, well_folder, well = temporary, temporary / "well", "preview"
        inputs = run_folder / "inputs" / f"{well}_objects"
        outputs = well_folder / "ilastik"
        raw_specs, second_specs, names = [], [], []
        try:
            for image, (field_index, t) in zip(images, keys, strict=True):
                raw = self._raw(image)[np.newaxis]
                axes = ilastik_axes(raw.shape)
                problems = check_compatible(info, axes, raw.shape[1])
                if problems:
                    raise ModuleError("; ".join(problems))
                name = f"f{field_index:03d}_t{t}"
                second, _ = self._second_input(image)
                second_axes = ilastik_axes(second[np.newaxis].shape)
                raw_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}.h5", to_ilastik(raw, axes), axes, compression=None
                        )
                    )
                )
                second_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}_{params.input}.h5",
                            to_ilastik(second[np.newaxis], second_axes),
                            second_axes,
                            compression=1,
                        )
                    )
                )
                names.append(name)
            log_path = run_folder / "logs" / f"{well}.log"
            used: list[str] = []

            def call(source: str, suffix: str) -> list[str]:
                job = IlastikJob(
                    project=state["project"],
                    workflow="objects",
                    raw_data=raw_specs,
                    roles={INPUT_FLAGS[params.input]: second_specs},
                    export_source=source,
                    output_folder=str(outputs),
                    output_suffix=suffix,
                    threads=int(state.get("threads", -1)),
                    ram_mb=int(state.get("ram_mb", 0)),
                    timeout_s=params.timeout_min * 60,
                )
                try:
                    result = run_job(state, job, log_path)
                except IlastikError as exc:
                    raise ModuleError(str(exc)) from None
                used.append(result.runner)
                return result.outputs

            predictions = call("Object Predictions", "Object_Predictions")
            identities = (
                call("Object Identities", "Object_Identities")
                if params.input != "labels"
                else [None] * len(images)
            )
            classes = state["classes"]
            object_type = params.labels if params.input == "labels" else f"{params.name}_objects"
            rows: list[dict[str, Any]] = []
            out_images = []
            for image, (field_index, t), predicted, identity in zip(
                images, keys, predictions, identities, strict=True
            ):
                class_image = _zyx(*read_h5(Path(predicted)))
                if identity is None:
                    objects = image.label_set(params.labels).data
                    out = image
                else:
                    objects = _relabel(_zyx(*read_h5(Path(identity))))
                    out = image.with_labels(
                        LabelSet(object_type, objects, source=self.name), self.name
                    )
                for object_id, class_id, agreement in majority_classes(objects, class_image):
                    rows.append(
                        {
                            "field": field_index,
                            "timepoint": t,
                            "object_type": object_type,
                            "object_id": object_id,
                            "class_id": class_id,
                            "class_name": classes[class_id - 1]
                            if 1 <= class_id <= len(classes)
                            else "unassigned",
                            "agreement": agreement,
                        }
                    )
                out_images.append(out)
            if temporary is None:
                outputs.mkdir(parents=True, exist_ok=True)
                table = pd.DataFrame(
                    rows,
                    columns=[
                        "field",
                        "timepoint",
                        "object_type",
                        "object_id",
                        "class_id",
                        "class_name",
                        "agreement",
                    ],
                )
                table.to_parquet(outputs / f"object_classes_{params.name}.parquet", index=False)
                (outputs / f"object_classes_{params.name}.json").write_text(
                    json.dumps(
                        {
                            "classes": classes,
                            "objects": len(table),
                            "by_class": table["class_name"].value_counts().to_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _projects_used(run_folder, state, used)
            self.last_rows = rows  # previews and tests
            return out_images
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            elif not params.keep_inputs:
                shutil.rmtree(inputs, ignore_errors=True)


def _zyx(array: np.ndarray, axes: str) -> np.ndarray:
    """An ilastik single-channel image → (Z, Y, X)."""
    from hcs_studio.segmentation.ilastik_backend.axes import from_ilastik

    tczyx = from_ilastik(array, axes)
    return np.asarray(tczyx[0, 0])


def _relabel(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    values = values[values > 0]
    lookup = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int32)
    lookup[values] = np.arange(1, len(values) + 1)
    return lookup[labels.astype(np.int64)]


def majority_classes(objects: np.ndarray, class_image: np.ndarray) -> list[tuple[int, int, float]]:
    """(object ID, majority class, agreement) per object: the class most of the object's
    pixels have in ilastik's prediction image (0 = no class); agreement = that share."""
    if objects.shape != class_image.shape:
        raise ModuleError(
            f"ilastik's prediction {class_image.shape} does not match the objects {objects.shape}"
        )
    ids = objects.ravel().astype(np.int64)
    inside = ids > 0
    if not inside.any():
        return []
    ids = ids[inside]
    cls = class_image.ravel().astype(np.int64)[inside]
    n_cls = int(np.max(cls)) + 1
    flat = np.bincount(ids * n_cls + cls, minlength=(int(np.max(ids)) + 1) * n_cls)
    counts = flat.reshape(-1, n_cls)
    out = []
    for object_id in np.unique(ids):
        row = counts[object_id]
        # an object's background pixels (class 0) only win when nothing else is predicted
        best = int(np.argmax(row[1:]) + 1) if row[1:].any() else 0
        out.append((int(object_id), best, round(float(row[best] / row.sum()), 4)))
    return out


# ------------------------------------------------------------------------------ table step


class ObjectClassesParams(TableParams):
    name: str = Field(default="ilastik", description="As in the ilastik_objects step.")


@MODULES.register("ilastik_object_classes")
class IlastikObjectClasses(TableModule):
    """ilastik's object classes in the object tables; per-well class counts and fractions."""

    name: ClassVar[str] = "ilastik_object_classes"
    title: ClassVar[str] = "ilastik object classes"
    Params = ObjectClassesParams

    def _classes(self, well: WellContext) -> pd.DataFrame | None:
        path = well.folder / "ilastik" / f"object_classes_{self.params.name}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        params: ObjectClassesParams = self.params
        prefix = f"class_{params.name}"
        out = pd.DataFrame(index=objects.index)
        out[prefix] = pd.Series(None, index=objects.index, dtype="object")
        out[f"{prefix}_agreement"] = np.nan
        classes = self._classes(well)
        rows = self.selected(objects)
        if classes is None or not rows.any():
            return out
        classes = classes[classes["object_type"] == params.object_type]
        keys = ["field", "timepoint", "object_id"]
        present = [k for k in keys if k in objects.columns]
        merged = objects.loc[rows, present].merge(
            classes[[*present, "class_name", "agreement"]], on=present, how="left"
        )
        merged.index = objects.index[rows]
        out.loc[rows, prefix] = merged["class_name"].fillna("unassigned")
        out.loc[rows, f"{prefix}_agreement"] = merged["agreement"]
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prefix = f"class_{self.params.name}"
        values = objects.loc[self.selected(objects), prefix] if prefix in objects else pd.Series()
        info = well.folder / "ilastik" / f"object_classes_{self.params.name}.json"
        try:  # every class of the project, so a well without one still reports 0
            labels: list[str] | None = list(json.loads(info.read_text("utf-8"))["classes"])
        except (OSError, ValueError, KeyError):
            labels = None
        if labels is not None and (values == "unassigned").any():
            labels.append("unassigned")
        return count_fractions(prefix, values, labels), {}


__all__ = ["IlastikObjectClasses", "IlastikObjects", "majority_classes"]
