"""ilastik Multicut and Tracking (PROMPT_ILASTIK.md §7.3–7.4, step d).

- ``ilastik_multicut``: a trained Multicut (boundary-based segmentation) project turns a
  boundary probability map — the class channels of an earlier ``ilastik_pixel`` /
  ``ilastik_autocontext`` step trained with a "boundary" class — into objects with unique IDs
  (``--probabilities``, export "Multicut Segmentation"). Preset: boundary pixel classification
  → Multicut → measure objects.
- ``ilastik_tracking``: a trained Tracking project links objects over the time points of each
  field. ilastik gets each field's whole time series in one call, with a binary mask
  (``segmentation_image``: from an existing label set or a class channel) or the class
  probabilities (``prediction_maps``). Two exports: "Object-Identities" (the objects of every
  time point, kept as the label set ``tracked``) and the "CSV-Table" plugin (each object's
  track and parent track). The table step ``ilastik_tracks`` turns them into the same track
  table and per-well ``trk_`` readouts as HCS Studio's own tracker (Phase 22d): divisions,
  cycle time, speed, straightness.

The CSV columns of ilastik's tracking export are not confirmed on a real installation yet;
:func:`read_tracking_csv` accepts the usual spellings and says which column it misses.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from hcs_studio.kinetics.tracking import Tracking, track_table
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

CLASS_PREFIX = "ilastik_"


class _IlastikWellStep(Module):
    """Shared plumbing of the ilastik steps below: project, fit (which ilastik), folders."""

    needs_fit: ClassVar[bool] = True
    exclusive: ClassVar[bool] = True
    batch_scope: ClassVar[str] = "well"
    fit_without_plate = True
    workflow: ClassVar[str] = ""
    workflow_title: ClassVar[str] = ""

    def project_info(self) -> Any:
        from hcs_studio.segmentation.ilastik_backend.project import read_project
        from hcs_studio.segmentation.ilastik_backend.steps import _project_path

        if not self.params.project:
            raise ModuleError(f"Choose the ilastik {self.workflow_title} project (.ilp)")
        try:
            return read_project(Path(_project_path(self.params.project)))
        except (FileNotFoundError, ValueError) as exc:
            raise ModuleError(str(exc)) from None

    def validate(self, metadata: ImageMetadata) -> None:
        info = self.project_info()
        if info.workflow not in ("unknown", self.workflow):
            raise ModuleError(
                f"{Path(self.params.project).name} is a {info.workflow_name or info.workflow} "
                f"project; this step needs an ilastik {self.workflow_title} project"
            )
        for channel in self.params.channels or []:
            if not 1 <= channel <= len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")

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
            "classes": list(info.label_names),
            **installation,
        }

    def describe_state(self, state: Any) -> dict[str, Any]:
        return dict(state or {})

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return self.apply_batch([image], state, [(1, 0)], None)[0]

    def _raw(self, image: ImageData) -> np.ndarray:
        if self.params.channels:
            return image.data[[c - 1 for c in self.params.channels]]
        keep = [
            i for i, c in enumerate(image.metadata.channels) if not c.name.startswith(CLASS_PREFIX)
        ]
        return image.data[keep]

    def _class_channels(self, image: ImageData, wanted: list[str] | None) -> np.ndarray:
        names = [c.name for c in image.metadata.channels]
        chosen = wanted or [n for n in names if n.startswith(CLASS_PREFIX)]
        missing = [c for c in chosen if c not in names]
        if not chosen or missing:
            raise ModuleError(
                "This step needs ilastik class channels (put ilastik_pixel first)"
                + (f"; missing {', '.join(missing)}" if missing else "")
            )
        return image.data[[names.index(n) for n in chosen]]

    def _setup(
        self, state: Any, workdir: Any
    ) -> tuple[dict[str, Any], Path, Path, str, Path | None]:
        state = state or self.fit(None)  # type: ignore[arg-type]
        if self.project_info().sha256 != state["sha256"]:
            raise ModuleError("The ilastik project changed since the run started")
        if isinstance(workdir, BatchFolders):
            return state, Path(workdir.run) / "ilastik", Path(workdir.well), workdir.well_name, None
        temporary = Path(tempfile.mkdtemp(prefix="hcs-ilastik-"))
        return state, temporary, temporary / "well", "preview", temporary

    def _run(self, state: dict[str, Any], log: Path, used: list[str], **job: Any) -> list[str]:
        from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob
        from hcs_studio.segmentation.ilastik_backend.runners import run_job

        spec = IlastikJob(
            project=state["project"],
            workflow=self.workflow,
            threads=int(state.get("threads", -1)),
            ram_mb=int(state.get("ram_mb", 0)),
            timeout_s=self.params.timeout_min * 60,
            **job,
        )
        try:
            result = run_job(state, spec, log)
        except IlastikError as exc:
            raise ModuleError(str(exc)) from None
        used.append(result.runner)
        return result.outputs


# --------------------------------------------------------------------------- multicut


class MulticutParams(ModuleParams):
    project: str = Field(default="", description="The trained Multicut project (.ilp).")
    source_channels: list[str] | None = Field(
        default=None,
        description="Class channels sent as the boundary probabilities "
        "(default: every ilastik_ channel, in order).",
    )
    channels: list[int] | None = Field(
        default=None, description="Raw channels sent, in the trained order; empty = all."
    )
    name: str = Field(default="cells", description="Name of the new label set.")
    keep_inputs: bool = Field(default=False, description="Keep the HDF5 copies sent to ilastik.")
    timeout_min: float = Field(default=120.0, gt=0, description="Give up on one well after this.")


@MODULES.register("ilastik_multicut")
class IlastikMulticut(_IlastikWellStep):
    """Boundary-based segmentation with ilastik's Multicut (one call per well)."""

    name: ClassVar[str] = "ilastik_multicut"
    title: ClassVar[str] = "ilastik Multicut"
    Params = MulticutParams
    workflow: ClassVar[str] = "multicut"
    workflow_title: ClassVar[str] = "Multicut"

    def validate(self, metadata: ImageMetadata) -> None:
        super().validate(metadata)
        names = [c.name for c in metadata.channels]
        wanted = self.params.source_channels or [n for n in names if n.startswith(CLASS_PREFIX)]
        if not wanted or any(c not in names for c in wanted):
            raise ModuleError(
                "Multicut needs the boundary probabilities: put ilastik_pixel "
                "(trained with a boundary class) before it"
            )

    def apply_batch(
        self,
        images: list[ImageData],
        state: Any,
        keys: list[tuple[int, int]],
        workdir: Any = None,
    ) -> list[ImageData]:
        from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes, to_ilastik
        from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, read_h5, write_h5
        from hcs_studio.segmentation.ilastik_backend.objects import _relabel, _zyx
        from hcs_studio.segmentation.ilastik_backend.steps import _projects_used

        params: MulticutParams = self.params
        if not images:
            return []
        state, run_folder, well_folder, well, temporary = self._setup(state, workdir)
        inputs = run_folder / "inputs" / f"{well}_multicut"
        outputs = well_folder / "ilastik"
        try:
            raw_specs, prob_specs = [], []
            for image, (field_index, t) in zip(images, keys, strict=True):
                name = f"f{field_index:03d}_t{t}"
                raw = self._raw(image)[np.newaxis]
                probs = self._class_channels(image, params.source_channels)[np.newaxis]
                axes = ilastik_axes(raw.shape)
                prob_axes = ilastik_axes(probs.shape)
                raw_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}.h5", to_ilastik(raw, axes), axes, compression=None
                        )
                    )
                )
                prob_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}_boundaries.h5",
                            to_ilastik(probs, prob_axes),
                            prob_axes,
                            compression=None,
                        )
                    )
                )
            used: list[str] = []
            produced = self._run(
                state,
                run_folder / "logs" / f"{well}.log",
                used,
                raw_data=raw_specs,
                roles={"probabilities": prob_specs},
                export_source="Multicut Segmentation",
                output_folder=str(outputs),
                output_suffix="Multicut_Segmentation",
            )
            out = []
            for image, path in zip(images, produced, strict=True):
                labels = _relabel(_zyx(*read_h5(Path(path)))).astype(np.int32)
                if labels.shape != image.data.shape[1:]:
                    raise ModuleError(
                        f"Multicut returned {labels.shape} for a field of {image.data.shape[1:]}"
                    )
                label_set = LabelSet(params.name, labels, source=self.name)
                out.append(
                    image.with_labels(
                        label_set, self.name, {"module": self.name, "objects": label_set.count}
                    )
                )
            if temporary is None:
                _projects_used(run_folder, state, used)
            return out
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            elif not params.keep_inputs:
                shutil.rmtree(inputs, ignore_errors=True)


# --------------------------------------------------------------------------- tracking


class TrackingStepParams(ModuleParams):
    project: str = Field(default="", description="The trained Tracking project (.ilp).")
    input: Literal["segmentation_image", "prediction_maps"] = Field(
        default="segmentation_image",
        description="What the project was trained on: a binary mask or probabilities.",
    )
    labels: str | None = Field(
        default="nuclei",
        description="segmentation_image: the mask from this label set (or, if empty, the "
        "first of source_channels ≥ 0.5).",
    )
    source_channels: list[str] | None = Field(
        default=None, description="Class channels sent (default: every ilastik_ channel)."
    )
    channels: list[int] | None = Field(
        default=None, description="Raw channels sent, in the trained order; empty = all."
    )
    name: str = Field(default="tracked", description="Label set of ilastik's tracked objects.")
    keep_inputs: bool = Field(default=False, description="Keep the HDF5 copies sent to ilastik.")
    timeout_min: float = Field(default=240.0, gt=0, description="Give up on one well after this.")


@MODULES.register("ilastik_tracking")
class IlastikTracking(_IlastikWellStep):
    """Tracking with ilastik: each field's time series in one ilastik call per export."""

    name: ClassVar[str] = "ilastik_tracking"
    title: ClassVar[str] = "ilastik Tracking"
    Params = TrackingStepParams
    workflow: ClassVar[str] = "tracking"
    workflow_title: ClassVar[str] = "Tracking"

    def _mask_or_probs(self, image: ImageData) -> np.ndarray:
        params: TrackingStepParams = self.params
        if params.input == "prediction_maps":
            return self._class_channels(image, params.source_channels).astype(np.float32)
        if params.labels:
            try:
                return (image.label_set(params.labels).data[np.newaxis] > 0).astype(np.uint8)
            except KeyError as exc:
                raise ModuleError(f"No label set {params.labels!r}") from exc
        first = self._class_channels(image, params.source_channels)[:1]
        return (first >= 0.5).astype(np.uint8)

    def apply_batch(
        self,
        images: list[ImageData],
        state: Any,
        keys: list[tuple[int, int]],
        workdir: Any = None,
    ) -> list[ImageData]:
        from hcs_studio.segmentation.ilastik_backend.axes import (
            from_ilastik,
            ilastik_axes,
            to_ilastik,
        )
        from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, read_h5, write_h5
        from hcs_studio.segmentation.ilastik_backend.steps import _projects_used

        params: TrackingStepParams = self.params
        if not images:
            return []
        fields = sorted({f for f, _ in keys})
        for field_index in fields:
            if sum(1 for f, _ in keys if f == field_index) < 2:
                raise ModuleError("Tracking needs a time-lapse plate (several time points)")
        state, run_folder, well_folder, well, temporary = self._setup(state, workdir)
        inputs = run_folder / "inputs" / f"{well}_tracking"
        outputs = well_folder / "ilastik"
        position = {key: i for i, key in enumerate(keys)}
        try:
            raw_specs, second_specs = [], []
            for field_index in fields:
                order = sorted(t for f, t in keys if f == field_index)
                stack = np.stack([self._raw(images[position[(field_index, t)]]) for t in order])
                second = np.stack(
                    [self._mask_or_probs(images[position[(field_index, t)]]) for t in order]
                )
                axes = ilastik_axes(stack.shape)
                second_axes = ilastik_axes(second.shape)
                name = f"f{field_index:03d}"
                raw_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}.h5", to_ilastik(stack, axes), axes, compression=None
                        )
                    )
                )
                second_specs.append(
                    input_spec(
                        write_h5(
                            inputs / f"{name}_{params.input}.h5",
                            to_ilastik(second, second_axes),
                            second_axes,
                            compression=1,
                        )
                    )
                )
            log = run_folder / "logs" / f"{well}.log"
            common = {
                "raw_data": raw_specs,
                "roles": {params.input: second_specs},
                "output_folder": str(outputs),
            }
            used: list[str] = []
            identities = self._run(
                state,
                log,
                used,
                export_source="Object-Identities",
                output_suffix="Object-Identities",
                **common,
            )
            tables = self._run(
                state,
                log,
                used,
                export_source="Plugin",
                export_plugin="CSV-Table",
                output_suffix="CSV-Table",
                output_extension="csv",
                **common,
            )
            out: list[ImageData | None] = [None] * len(images)
            links = []
            for field_index, ids_path, csv_path in zip(fields, identities, tables, strict=True):
                array, axes = read_h5(Path(ids_path))
                stack = from_ilastik(array, axes)  # (T, 1, Z, Y, X)
                order = sorted(t for f, t in keys if f == field_index)
                if stack.shape[0] != len(order):
                    raise ModuleError(
                        f"ilastik returned {stack.shape[0]} time points for {len(order)}"
                    )
                for k, t in enumerate(order):
                    image = images[position[(field_index, t)]]
                    labels = np.asarray(stack[k, 0], dtype=np.int32)
                    out[position[(field_index, t)]] = image.with_labels(
                        LabelSet(params.name, labels, source=self.name), self.name
                    )
                table = read_tracking_csv(Path(csv_path))
                table["timepoint"] = [
                    order[int(f)] if 0 <= int(f) < len(order) else int(f)
                    for f in table["timepoint"]
                ]
                table.insert(0, "field", field_index)
                links.append(table)
            if temporary is None:
                outputs.mkdir(parents=True, exist_ok=True)
                pd.concat(links, ignore_index=True).to_csv(
                    outputs / f"tracks_{params.name}.csv", index=False
                )
                _projects_used(run_folder, state, used)
            self.last_links = pd.concat(links, ignore_index=True)
            return [image for image in out if image is not None]
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            elif not params.keep_inputs:
                shutil.rmtree(inputs, ignore_errors=True)


#: our name → spellings in ilastik's tracking CSV export (unconfirmed on real files)
CSV_COLUMNS = {
    "timepoint": ("frame", "timestep", "time", "t"),
    "object_id": ("labelimage_oid", "labelimage_id", "object_id", "oid"),
    "track_id": ("track_id1", "track_id", "trackid", "track"),
    "parent_track_id": ("parent_track_id", "parent_id", "parenttrackid", "parent"),
    "lineage_id": ("lineage_id", "lineageid", "lineage"),
}


def read_tracking_csv(path: Path) -> pd.DataFrame:
    """ilastik's tracking CSV → timepoint (frame index), object_id, track_id, parent_track_id
    (0 = none), lineage_id."""
    raw = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    out = pd.DataFrame()
    for ours, spellings in CSV_COLUMNS.items():
        found = next((lower[s] for s in spellings if s in lower), None)
        if found is None:
            if ours in ("parent_track_id", "lineage_id"):
                out[ours] = 0
                continue
            raise ModuleError(
                f"ilastik's tracking table {path.name} has no {ours} column (columns: "
                f"{', '.join(map(str, raw.columns))}); please send it to the developers"
            )
        out[ours] = pd.to_numeric(raw[found], errors="coerce").fillna(0).astype(int)
    out.loc[out["parent_track_id"] < 0, "parent_track_id"] = 0
    return out[out["object_id"] > 0].reset_index(drop=True)


# ------------------------------------------------------------------------ table step


class IlastikTracksParams(ModuleParams):
    object_type: str = Field(default="tracked", description="ilastik's tracked objects.")
    name: str = Field(default="tracked", description="As the ilastik_tracking step's name.")
    min_track_frames: int = Field(default=3, ge=1, description="Shorter tracks are not summarised.")


@MODULES.register("ilastik_tracks")
class IlastikTracks(Tracking):
    """ilastik's tracks in the object tables; track table and ``trk_`` readouts as the
    built-in tracker (divisions, cycle time, speed)."""

    name = "ilastik_tracks"
    title = "ilastik tracks"
    Params = IlastikTracksParams  # type: ignore[assignment]

    def transform(self, objects: pd.DataFrame, state: Any, well: Any) -> pd.DataFrame:
        params: IlastikTracksParams = self.params
        times = self.times()
        out = pd.DataFrame(
            {"trk_id": 0, "trk_parent": 0, "trk_divided": False}, index=objects.index
        )
        path = well.folder / "ilastik" / f"tracks_{params.name}.csv"
        chosen = objects[self.selected(objects)]
        if chosen.empty or not path.exists():
            return out
        links = pd.read_csv(path)
        first = links.groupby(["field", "track_id"])["timepoint"].transform("min")
        links["born"] = (links["parent_track_id"] > 0) & (links["timepoint"] == first)
        merged = (
            chosen[["field", "timepoint", "object_id"]]
            .reset_index()
            .merge(links, on=["field", "timepoint", "object_id"], how="left")
            .set_index("index")
        )
        offset = merged["field"].astype(int) * 100_000
        track = merged["track_id"].fillna(0).astype(int)
        parent = merged["parent_track_id"].fillna(0).astype(int)
        out.loc[merged.index, "trk_id"] = np.where(track > 0, track + offset, 0)
        out.loc[merged.index, "trk_parent"] = np.where(parent > 0, parent + offset, 0)
        out.loc[merged.index, "trk_divided"] = merged["born"].fillna(False).astype(bool)
        for field_index, group in chosen.groupby("field"):
            ids = out.loc[group.index]
            ids = ids[ids["trk_id"] > 0]
            if ids.empty:
                continue
            table = track_table(pd.concat([group.loc[ids.index], ids], axis=1), times)
            if not table.empty:
                table.insert(0, "field", int(str(field_index)))
                table.insert(0, "well", well.well)
                self._tracks.append(table)
        return out
