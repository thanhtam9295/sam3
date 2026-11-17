"""Results of a finished run: per-well readouts, normalisation, Z′, and field previews.

Used by the in-app results plate and by the HTML report. Reads only the run folder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from hcs_studio.model.layout import WellAnnotation, read_plate_layout
from hcs_studio.model.plate import parse_well_name, well_name

Normalisation = Literal["none", "robust_z_plate", "robust_z_negative"]

#: Readouts listed first when present (in this order).
_PREFERRED = (
    "count",
    "per_mm2",
    "_fraction",
    "profile_distance",
    "objects_measured",
    "_mean_mean",
    "coloc_",
    "pearson",
    "fields_",
)


@dataclass
class WellField:
    index: int
    enhanced: str | None
    corrected: str | None
    raw: str | None
    overlays: dict[str, str] = field(default_factory=dict)
    objects: dict[str, int] = field(default_factory=dict)


def robust_z(values: dict[str, float], reference: list[str] | None = None) -> dict[str, float]:
    """(x − median) / (1.4826 · MAD) of the reference wells (default: all wells)."""
    pool = [values[w] for w in (reference if reference is not None else values) if w in values]
    pool = [v for v in pool if v is not None and math.isfinite(v)]
    if len(pool) < 2:
        return {w: float("nan") for w in values}
    median = float(np.median(pool))
    mad = 1.4826 * float(np.median(np.abs(np.asarray(pool) - median)))
    if mad == 0:
        return {w: float("nan") for w in values}
    return {
        w: (v - median) / mad if v is not None and math.isfinite(v) else float("nan")
        for w, v in values.items()
    }


def z_prime(positives: list[float], negatives: list[float]) -> float | None:
    """Z′ = 1 − 3(σp + σn) / |μp − μn|; None with fewer than two finite controls of each kind."""
    p = np.asarray([v for v in positives if math.isfinite(v)], dtype=float)
    n = np.asarray([v for v in negatives if math.isfinite(v)], dtype=float)
    if len(p) < 2 or len(n) < 2 or p.mean() == n.mean():
        return None
    return float(1 - 3 * (p.std(ddof=1) + n.std(ddof=1)) / abs(p.mean() - n.mean()))


class RunResults:
    """A run folder, read for display."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        try:
            self.manifest: dict[str, Any] = json.loads(
                (self.folder / "run_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            raise ValueError(f"{self.folder} is not an HCS Studio run folder") from None
        plate = self.manifest["plate"]
        self.plate_name: str = plate["name"]
        self.rows: int = int(plate["rows"])
        self.cols: int = int(plate["cols"])
        tables = self.folder / "tables"
        summary_file = tables / "per_well_summary.csv"
        self.summary = (
            pd.read_csv(summary_file).set_index("well") if summary_file.exists() else pd.DataFrame()
        )
        self._add_profile_readouts()
        flags_file = tables / "qc_flags.csv"
        self.flags = pd.read_csv(flags_file) if flags_file.exists() else None
        layout_file = self.folder / "plate_layout.csv"
        self.layout: dict[str, WellAnnotation] = {}
        if layout_file.exists():
            self.layout = {well_name(*k): v for k, v in read_plate_layout(layout_file).items()}
        self.wells: list[str] = (
            sorted(
                p.name
                for p in (self.folder / "wells").iterdir()
                if (p / "well_summary.json").exists()
            )
            if (self.folder / "wells").exists()
            else []
        )
        self._objects_file = tables / "per_object_all.parquet"

    # ---------------------------------------------------------------- profiles

    def _profile_wells(self) -> pd.DataFrame:
        path = self.folder / "advanced" / "well_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        return frame[frame["run"] == self.folder.name] if "run" in frame else frame

    def _add_profile_readouts(self) -> None:
        """Per-well profile values (Phase 9) become readouts: distance to control, PCs."""
        wells = self._profile_wells()
        if wells.empty:
            return
        names = {
            "profile_distance_to_control": "profile_distance_to_control",
            "profile_distance_ratio": "profile_distance_ratio",
            "PC1": "profile_PC1",
            "PC2": "profile_PC2",
            "UMAP1": "profile_UMAP1",
            "UMAP2": "profile_UMAP2",
        }
        extra = wells.set_index("well")[[c for c in names if c in wells]].rename(columns=names)
        if self.summary.empty:
            self.summary = extra
        else:
            new = [c for c in extra.columns if c not in self.summary.columns]
            self.summary = self.summary.join(extra[new], how="outer")

    def profile_embedding(self, max_cells: int = 3000, seed: int = 0) -> dict[str, Any] | None:
        """PCA/UMAP coordinates of this run's wells and a sample of its objects, for plots:
        {"wells": [...], "cells": [...], "variance": [...], "spaces": [...]}, or None."""
        wells = self._profile_wells()
        if wells.empty:
            return None
        advanced = self.folder / "advanced"
        cells = pd.read_parquet(advanced / "pca_coordinates.parquet")
        if (advanced / "umap_coordinates.parquet").exists():
            umap = pd.read_parquet(advanced / "umap_coordinates.parquet")
            if len(umap) == len(cells):
                cells["UMAP1"] = umap["UMAP1"].to_numpy()
                cells["UMAP2"] = umap["UMAP2"].to_numpy()
        if "run" in cells:
            cells = cells[cells["run"] == self.folder.name]
        if len(cells) > max_cells:
            cells = cells.sample(max_cells, random_state=seed)
        spaces = [s for s in ("PC", "UMAP") if f"{s}1" in cells and f"{s}2" in cells]

        def points(frame: pd.DataFrame, extra: tuple[str, ...]) -> list[dict[str, Any]]:
            columns = [c for c in ("well", "PC1", "PC2", "UMAP1", "UMAP2", *extra) if c in frame]
            out = []
            for record in frame[columns].to_dict("records"):
                point: dict[str, Any] = {}
                for key, value in record.items():
                    if isinstance(value, (float, np.floating)):
                        if math.isfinite(value):
                            point[str(key)] = round(float(value), 4)
                    elif isinstance(value, (int, np.integer)):
                        point[str(key)] = int(value)
                    else:
                        point[str(key)] = value
                out.append(point)
            return out

        variance_file = advanced / "pca_variance.csv"
        variance = (
            pd.read_csv(variance_file)["explained_variance_ratio"].round(4).tolist()
            if variance_file.exists()
            else []
        )
        return {
            "wells": points(wells, ("profile_distance_to_control",)),
            "cells": points(cells, ("field", "object_id")),
            "variance": variance,
            "spaces": spaces,
        }

    def object_crop(
        self, well: str, field: int, y_um: float, x_um: float, size_um: float = 40.0
    ) -> Any:
        """RGB crop (uint8) of a field's enhanced preview centred on a position in the field
        (µm), or None when the preview or the field size is unknown."""
        from PIL import Image

        directory = self.folder / "wells" / well
        try:
            summary = json.loads((directory / "well_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        size = summary.get("field_size_um")
        preview = directory / "previews" / f"f{int(field):03d}.png"
        if not size or not preview.exists():
            return None
        image = np.asarray(Image.open(preview).convert("RGB"))
        scale = image.shape[0] / float(size[0])  # preview pixels per µm
        half = max(4, round(size_um * scale / 2))
        row, col = round(y_um * scale), round(x_um * scale)
        r0, c0 = max(0, row - half), max(0, col - half)
        crop = image[r0 : row + half, c0 : col + half]
        return crop if crop.size else None

    # ---------------------------------------------------------------- readouts

    def readout_columns(self) -> list[str]:
        """Numeric per-well summary columns, the most commonly used first."""
        numeric = [c for c in self.summary.select_dtypes(include=["number", "bool"]).columns]

        def rank(column: str) -> tuple[int, str]:
            for i, key in enumerate(_PREFERRED):
                if key in column:
                    return (i, column)
            return (len(_PREFERRED), column)

        return sorted(numeric, key=rank)

    def values(self, column: str) -> dict[str, float]:
        if column not in self.summary:
            raise KeyError(f"No readout '{column}' in this run")
        series = self.summary[column]
        return {str(w): float(v) if pd.notna(v) else float("nan") for w, v in series.items()}

    def object_features(self) -> dict[str, list[str]]:
        """Numeric object features per object type, for % positive."""
        if not self._objects_file.exists():
            return {}
        import pyarrow.parquet as pq

        schema = pq.read_schema(self._objects_file)
        skip = {
            "object_id",
            "parent_id",
            "row",
            "col",
            "field",
            "timepoint",
            "well_x_um",
            "well_y_um",
            "spatial_colony",
        }
        numeric = [
            f.name
            for f in schema
            if f.name not in skip
            and str(f.type).startswith(("double", "float", "int"))
            and not f.name.startswith("centroid_")
        ]
        types = (
            pq.read_table(self._objects_file, columns=["object_type"])
            .column(0)
            .unique()
            .to_pylist()
        )
        return {str(t): numeric for t in types}

    def percent_positive(
        self, object_type: str, feature: str, threshold: float
    ) -> dict[str, float]:
        """% of objects of a type with ``feature`` above ``threshold``, per well.

        Objects in fields excluded by QC (out of focus, edge, as configured) are left out.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(
            self._objects_file, columns=["well", "field", "object_type", feature]
        ).to_pandas()
        table = table[table.object_type == object_type]
        if self.flags is not None and "excluded" in self.flags:
            excluded = set(
                map(tuple, self.flags[self.flags.excluded][["well", "field"]].to_numpy().tolist())
            )
            if excluded:
                keep = [
                    (w, f) not in excluded for w, f in zip(table.well, table.field, strict=True)
                ]
                table = table[keep]
        result = {}
        for well, group in table.groupby("well"):
            valid = group[feature].dropna()
            result[str(well)] = (
                100.0 * float((valid > threshold).mean()) if len(valid) else float("nan")
            )
        return {w: result.get(w, float("nan")) for w in self.wells}

    def normalise(self, values: dict[str, float], method: Normalisation) -> dict[str, float]:
        if method == "none":
            return values
        if method == "robust_z_plate":
            return robust_z(values)
        negatives = self.control_wells("negative")
        if len(negatives) < 2:
            raise ValueError(
                "Robust Z against controls needs at least two negative-control wells in the layout"
            )
        return robust_z(values, negatives)

    def control_wells(self, kind: str) -> list[str]:
        return sorted(w for w, a in self.layout.items() if a.control_type == kind)

    def z_prime(self, values: dict[str, float]) -> float | None:
        return z_prime(
            [values.get(w, float("nan")) for w in self.control_wells("positive")],
            [values.get(w, float("nan")) for w in self.control_wells("negative")],
        )

    # ------------------------------------------------------------------ wells

    def well_summary(self, well: str) -> dict[str, Any]:
        path = self.folder / "wells" / well / "well_summary.json"
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def well_fields(self, well: str) -> list[WellField]:
        """Preview paths (relative to the run folder) of each field of a well."""
        summary = self.well_summary(well)
        result = []
        for entry in summary.get("fields", []):
            files = entry.get("files", {})

            def rel(key: str, files: dict[str, Any] = files) -> str | None:
                value = files.get(key)
                return f"wells/{well}/{value}" if value else None

            result.append(
                WellField(
                    index=int(entry["field"]),
                    enhanced=rel("preview"),
                    corrected=rel("preview_corrected"),
                    raw=rel("preview_raw"),
                    overlays={
                        k: f"wells/{well}/{v}" for k, v in (files.get("overlays") or {}).items()
                    },
                    objects={
                        k: int(v.get("count", 0)) for k, v in (entry.get("objects") or {}).items()
                    },
                )
            )
        return result

    def well_flags(self, well: str) -> list[str]:
        """QC flags of a well, as short phrases."""
        flags = []
        if self.flags is not None and not self.flags.empty:
            rows = self.flags[self.flags.well == well]
            for column, text in (
                ("out_of_focus", "out-of-focus field"),
                ("saturated", "saturated field"),
                ("edge_field", "edge field"),
            ):
                count = int(rows[column].sum()) if column in rows else 0
                if count:
                    flags.append(f"{count} {text}{'s' if count > 1 else ''}")
            if "low_count_well" in rows and bool(rows["low_count_well"].any()):
                flags.append("low count")
        error = self.well_summary(well).get("error") if well in self.wells else None
        if error:
            flags.append(f"failed: {error}")
        return flags

    def position(self, well: str) -> tuple[int, int]:
        return parse_well_name(well)
