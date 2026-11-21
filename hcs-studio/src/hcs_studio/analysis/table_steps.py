"""Plate-level table steps: gates, classifiers, clustering and spatial statistics.

Table steps run after every well has been processed, on the per-object tables. A step may
learn plate-level state from a sample of objects (``fit_table``), then annotates each well's
objects (``transform``) and summarises the well (``summarize``). Annotations are written next
to the measurements as ``wells/<W>/objects_annotations.parquet`` (the measurements themselves
are never rewritten), merged into ``tables/per_object_all.parquet``, and the per-well values
join ``tables/per_well_summary.csv``. Because they only read tables, table steps can also be
applied to a finished run without reprocessing images (``hcs-studio analyze``).
"""

from __future__ import annotations

import difflib
import json
import math
import re
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.processing.base import MODULES, ImageData, Module, ModuleError, ModuleParams

#: Columns that identify an object across tables.
OBJECT_KEY = ("well", "field", "timepoint", "object_type", "object_id")
#: Columns added by table steps start with one of these.
ANNOTATION_PREFIXES = ("gate_", "class_", "cluster_", "spatial_", "qc_", "cc_")
#: Numeric columns that are positions or identifiers, not features.
NON_FEATURES = frozenset(
    {
        "row",
        "col",
        "field",
        "timepoint",
        "object_id",
        "parent_id",
        "well_x_um",
        "well_y_um",
        "in_overlap_duplicate",
    }
)
ANNOTATIONS_FILE = "objects_annotations.parquet"


def column_name(text: str) -> str:
    """A label (class name, cluster) as a safe column-name fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "x"


def feature_columns(frame: pd.DataFrame, wanted: Sequence[str] | None = None) -> list[str]:
    """Numeric feature columns: ``wanted`` (checked) or every measured feature."""
    if wanted:
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            hints = []
            for name in missing[:3]:
                close = difflib.get_close_matches(name, list(frame.columns), n=2)
                hints.append(f"{name}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
            raise ModuleError(f"Feature(s) not in the object table: {'; '.join(hints)}")
        return list(wanted)
    numeric = frame.select_dtypes("number").columns
    return [
        str(c)
        for c in numeric
        if c not in NON_FEATURES and not str(c).startswith(("centroid_", *ANNOTATION_PREFIXES))
    ]


def qc_excluded(frame: pd.DataFrame) -> pd.Series:
    """True for objects that object QC left out (``qc_excluded``)."""
    if "qc_excluded" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["qc_excluded"].fillna(False).astype(bool)


def unique_rows(frame: pd.DataFrame) -> pd.Series:
    """True for objects counted once (not the duplicate in an overlap of stitched fields)."""
    if "in_overlap_duplicate" not in frame:
        return pd.Series(True, index=frame.index)
    return ~frame["in_overlap_duplicate"].fillna(False).astype(bool)


@dataclass(frozen=True)
class WellContext:
    """What a table step knows about the well besides its objects."""

    well: str
    folder: Path
    #: Field (height, width) in µm.
    field_size_um: tuple[float, float] | None = None
    #: Stitched top-left (y, x) of each field in well µm, when the run stitched wells.
    field_offsets_um: dict[int, tuple[float, float]] | None = None

    @classmethod
    def load(cls, folder: Path) -> WellContext:
        try:
            data = json.loads((folder / "well_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        size = data.get("field_size_um")
        offsets = (data.get("stitching") or {}).get("field_offsets_um")
        return cls(
            well=str(data.get("well", folder.name)),
            folder=folder,
            field_size_um=(float(size[0]), float(size[1])) if size else None,
            field_offsets_um=(
                {int(k): (float(v[0]), float(v[1])) for k, v in offsets.items()}
                if offsets
                else None
            ),
        )


class TableParams(ModuleParams):
    object_type: str = Field(
        default="nuclei", description="Objects to work on (a segmentation name)."
    )


class TableModule(Module):
    """A step that runs on the object tables after all wells (see the module docstring)."""

    #: True when ``fit_table`` learns plate-level state from a sample of objects.
    needs_fit: ClassVar[bool] = False
    #: Objects sampled from the plate for ``fit_table``.
    max_fit_objects: ClassVar[int] = 200_000
    #: The run folder, set before ``fit_table`` (assay modules read the layout and the
    #: control wells' objects from it).
    run_folder: Path | None = None

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return image  # table steps do not touch images

    def fit_table(self, sample: pd.DataFrame) -> Any:
        """Plate-level state from objects of ``params.object_type`` sampled across wells."""
        return None

    @abstractmethod
    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        """New columns for one well's objects (same index as ``objects``). Rows of other
        object types get missing values."""

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """(per-well summary values, JSON details for ``wells/<W>/analysis.json``)."""
        return {}, {}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        """After every well: write the step's own files (assay reports); default nothing."""

    def selected(self, objects: pd.DataFrame) -> pd.Series:
        """Rows this step annotates: its object type, each object counted once, without
        objects excluded by object QC (``qc_excluded``)."""
        wanted: pd.Series = objects["object_type"] == str(self.params.object_type)
        return wanted & unique_rows(objects) & ~qc_excluded(objects)


def count_fractions(
    prefix: str, values: pd.Series, labels: Sequence[Any] | None = None
) -> dict[str, Any]:
    """``<prefix>_<label>_count`` and ``_fraction`` for each label of a categorical column."""
    values = values.dropna()
    total = len(values)
    counts = values.value_counts()
    out: dict[str, Any] = {}
    for label in labels if labels is not None else sorted(counts.index, key=str):
        n = int(counts.get(label, 0))
        out[f"{prefix}_{column_name(label)}_count"] = n
        out[f"{prefix}_{column_name(label)}_fraction"] = n / total if total else math.nan
    return out


# --------------------------------------------------------------------- running


@dataclass
class TableRunResult:
    #: Per-well summary values by well name.
    per_well: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Annotation columns added to the object tables.
    columns: list[str] = field(default_factory=list)
    #: One record per step for ``tables/analysis_summary.json``.
    steps: list[dict[str, Any]] = field(default_factory=list)


def read_well_objects(folder: Path) -> pd.DataFrame:
    """All objects of one well (``wells/<W>``), with any table-step annotations."""
    objects = pd.read_parquet(folder / "objects.parquet")
    path = folder / ANNOTATIONS_FILE
    if path.exists():
        extra = pd.read_parquet(path)
        extra = extra[[c for c in extra.columns if c not in objects.columns]]
        if len(extra) == len(objects):
            objects = pd.concat([objects, extra], axis=1)
    return objects


def _load_objects(folder: Path, annotations: pd.DataFrame | None) -> pd.DataFrame:
    objects = pd.read_parquet(folder / "objects.parquet")
    if annotations is not None:
        objects = pd.concat([objects, annotations.reset_index(drop=True)], axis=1)
    return objects


def _sample(
    folders: list[Path],
    annotations: dict[Path, pd.DataFrame],
    object_type: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Objects of one type spread evenly over the wells (at most ``max_rows``)."""
    rng = np.random.default_rng(seed)
    per_well = max(1, -(-max_rows // max(1, len(folders))))
    parts = []
    for folder in folders:
        objects = _load_objects(folder, annotations.get(folder))
        objects = objects[(objects["object_type"] == object_type) & unique_rows(objects)]
        if len(objects) > per_well:
            objects = objects.iloc[np.sort(rng.choice(len(objects), per_well, replace=False))]
        parts.append(objects)
    if not parts or not any(len(p) for p in parts):
        raise ModuleError(f"No '{object_type}' objects in this run to learn from")
    return pd.concat(parts, ignore_index=True)


def run_table_steps(
    folder: Path,
    modules: Sequence[TableModule],
    log: Callable[[str], None] | None = None,
) -> TableRunResult:
    """Apply table steps to every well of a run folder; writes the annotation files."""
    result = TableRunResult()
    well_folders = sorted(p for p in (folder / "wells").iterdir() if p.is_dir())
    for well_folder in well_folders:  # results of an earlier analysis are replaced
        (well_folder / ANNOTATIONS_FILE).unlink(missing_ok=True)
        (well_folder / "analysis.json").unlink(missing_ok=True)
    folders = [p for p in well_folders if (p / "objects.parquet").exists()]
    contexts = {p: WellContext.load(p) for p in folders}
    annotations: dict[Path, pd.DataFrame] = {}
    details: dict[Path, dict[str, Any]] = {p: {} for p in folders}
    for number, module in enumerate(modules, start=1):
        label = f"Table step {number} ({module.title})"
        if log:
            log(f"{label}: running on {len(folders)} wells")
        step_columns: list[str] = []
        try:
            state = None
            module.run_folder = Path(folder)
            if module.needs_fit:
                sample = _sample(
                    folders,
                    annotations,
                    module.params.object_type,
                    getattr(module.params, "max_fit_objects", module.max_fit_objects),
                    int(getattr(module.params, "seed", 0)),
                )
                state = module.fit_table(sample)
            for well_folder in folders:
                objects = _load_objects(well_folder, annotations.get(well_folder))
                new = module.transform(objects, state, contexts[well_folder])
                clash = [c for c in new.columns if c in objects.columns]
                if clash:
                    raise ModuleError(
                        f"column(s) {clash} already exist; give each step its own name"
                    )
                if len(new) != len(objects):
                    raise ModuleError("a table step must return one row per object")
                new = new.reset_index(drop=True)
                step_columns.extend(c for c in new.columns if c not in step_columns)
                previous = annotations.get(well_folder)
                annotations[well_folder] = (
                    new if previous is None else pd.concat([previous, new], axis=1)
                )
                summary, extra = module.summarize(
                    pd.concat([objects.reset_index(drop=True), new], axis=1),
                    state,
                    contexts[well_folder],
                )
                result.per_well.setdefault(contexts[well_folder].well, {}).update(summary)
                if extra:
                    details[well_folder][f"step{number}_{module.name}"] = extra
        except ModuleError as exc:
            raise ModuleError(f"{label}: {exc}") from None
        try:
            module.finish_table(state, Path(folder), result.per_well)
        except ModuleError as exc:
            raise ModuleError(f"{label}: {exc}") from None
        result.columns.extend(c for c in step_columns if c not in result.columns)
        result.steps.append(
            {
                "step": number,
                "module": module.name,
                "title": module.title,
                "params": module.params.model_dump(mode="json"),
                "state": module.describe_state(state) if state is not None else {},
            }
        )
    for well_folder, table in annotations.items():
        keys = pd.read_parquet(well_folder / "objects.parquet", columns=list(OBJECT_KEY))
        out = pd.concat([keys, table], axis=1)
        part = well_folder / (ANNOTATIONS_FILE + ".part")
        out.to_parquet(part, index=False)
        part.replace(well_folder / ANNOTATIONS_FILE)
        if details[well_folder]:
            (well_folder / "analysis.json").write_text(
                json.dumps(details[well_folder], indent=1, default=_json_default), encoding="utf-8"
            )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ------------------------------------------------------------------------ gates


class GateRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    min: float | None = None
    max: float | None = None


class GateParams(TableParams):
    name: str = Field(
        default="positive",
        pattern=r"^[A-Za-z0-9_]+$",
        description="Gate name; objects get the column gate_<name>.",
    )
    rules: list[GateRule] = Field(
        default_factory=list, description="Feature thresholds (min and/or max, inclusive)."
    )
    combine: Literal["all", "any"] = Field(
        default="all", description="Inside the gate when all rules hold, or any."
    )


@MODULES.register("gate_objects")
class GateObjects(TableModule):
    """Rule-based gate: objects whose features fall inside thresholds (e.g. drawn on a
    histogram or scatter plot). Missing feature values are outside the gate."""

    name = "gate_objects"
    title = "Gate objects"
    Params = GateParams

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        params: GateParams = self.params
        if not params.rules:
            raise ModuleError("add at least one rule (a feature with a min and/or max)")
        feature_columns(objects, [r.feature for r in params.rules])
        tests = []
        for rule in params.rules:
            values = pd.to_numeric(objects[rule.feature], errors="coerce")
            inside = values.notna()
            if rule.min is not None:
                inside &= values >= rule.min
            if rule.max is not None:
                inside &= values <= rule.max
            tests.append(inside)
        stacked = np.vstack([t.to_numpy(bool) for t in tests])
        combined = stacked.all(axis=0) if params.combine == "all" else stacked.any(axis=0)
        column = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        rows = self.selected(objects)
        column[rows] = pd.Series(combined, index=objects.index)[rows]
        return pd.DataFrame({f"gate_{params.name}": column})

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        values = objects[f"gate_{self.params.name}"].dropna()
        n, inside = len(values), int(values.sum())
        return {
            f"gate_{self.params.name}_count": inside,
            f"gate_{self.params.name}_fraction": inside / n if n else math.nan,
        }, {}


def is_table_module(name: str) -> bool:
    return issubclass(MODULES.get(name), TableModule)
