"""Proliferation markers, micronuclei and multinucleation (Phase 20e).

Two steps:

- **``mitotic_errors``** (image step, right after the nuclear segmentation): **micronuclei** —
  small DNA objects (area between ``min_area_um2`` and ``max_area_fraction`` × the median
  nuclear area) within ``search_um`` of a nucleus edge but not touching it; objects that small
  are taken out of the nuclei (a threshold segmentation keeps the larger micronuclei as
  "nuclei"), and dim ones below the segmentation's size limit are found with their own
  threshold (``threshold_fraction`` × the median nuclear DNA intensity). Each micronucleus
  belongs to its nearest nucleus (label set ``micronuclei``, so measurement gives
  ``micronuclei_count`` per nucleus). **Multinucleation** — with a cell segmentation made on a
  cell stain (``cells``; not grown from the nuclei, which would hold one nucleus each) each
  nucleus is linked to the cell it overlaps most, so measurement gives ``nuclei_count`` per cell.
- **``proliferation``** (assay table step): % positive nuclei for Ki67, EdU/BrdU and pH3 by a
  nuclear intensity threshold set from the controls, intensity distributions per well, and the
  micronucleus and multinucleation frequencies.

**Thresholds** per marker (``threshold_method``):

- ``secondary_only``: median + ``threshold_sd`` (5) robust SD of log intensity of the nuclei of
  secondary-antibody-only (no primary) wells — layout role ``unstained`` (with a DNA stain), or
  ``secondary_only_wells``. Also used for EdU/BrdU wells without the click reaction / antibody.
- ``vehicle_mixture``: two log-normal populations fitted to the vehicle nuclei; the threshold
  is where both are equally likely, so it does not depend on the positive share (which differs
  between treated wells) — for markers with a large positive share in growing cells (Ki67,
  EdU); warns when the populations are not separated (Ashman's D < 2).
- ``vehicle_otsu``: Otsu's threshold of log intensity over the vehicle nuclei (biased towards
  the smaller population: with 70 % Ki67-positive vehicle cells it cuts into the negatives).
- ``vehicle_robust``: median + ``threshold_sd`` robust SD over the vehicle nuclei — for rare
  positives (pH3).
- ``auto`` (default): ``secondary_only`` when such wells exist, else ``vehicle_robust`` for pH3
  and ``vehicle_mixture`` for the others. ``thresholds`` fixes a marker's value.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.table_steps import TableParams, WellContext
from hcs_studio.assays.base import (
    POSITIVE_ROLES,
    VEHICLE_ROLES,
    AssayModule,
    ChannelRole,
    channel_column,
)
from hcs_studio.assays.cell_cycle import otsu_log, robust_upper_log
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)

MARKERS = ("ki67", "edu", "ph3")
SECONDARY_ROLES = ("unstained",)
TITLES = {"ki67": "Ki67", "edu": "EdU/BrdU", "ph3": "pH3"}


# ------------------------------------------------------------ thresholds


def mixture_threshold(values: NDArray[Any], iterations: int = 200) -> tuple[float, float]:
    """Two log-normal populations fitted to ``values`` (EM on log10, started from Otsu's
    split); returns (threshold where both populations are equally likely — independent of
    their proportions —, Ashman's D)."""
    x = np.log10(np.asarray(values, dtype=np.float64))
    x = x[np.isfinite(x)]
    if len(x) < 20:
        raise ModuleError("too few values to set a threshold")
    split = math.log10(otsu_log(10**x))
    low, high = x[x <= split], x[x > split]
    if len(low) < 3 or len(high) < 3:
        raise ModuleError("the values do not form two populations")
    m = np.array([low.mean(), high.mean()])
    s = np.array([max(low.std(), 1e-3), max(high.std(), 1e-3)])
    w = np.array([len(low), len(high)], dtype=float) / len(x)
    for _ in range(iterations):
        density = w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2)
        resp = density / np.maximum(density.sum(axis=1, keepdims=True), 1e-300)
        n = resp.sum(axis=0)
        if np.any(n < 2):
            break
        new_m = (resp * x[:, None]).sum(axis=0) / n
        s = np.sqrt((resp * (x[:, None] - new_m) ** 2).sum(axis=0) / n).clip(1e-3)
        w, converged, m = n / len(x), np.allclose(new_m, m, atol=1e-7), new_m
        if converged:
            break
    order = np.argsort(m)
    m, s = m[order], s[order]
    # equal densities: a x² + b x + c = 0, root between the means
    a = 1 / s[1] ** 2 - 1 / s[0] ** 2
    b = 2 * (m[0] / s[0] ** 2 - m[1] / s[1] ** 2)
    c = m[1] ** 2 / s[1] ** 2 - m[0] ** 2 / s[0] ** 2 + 2 * math.log(s[1] / s[0])
    if abs(a) < 1e-12:
        root = -c / b
    else:
        roots = np.roots([a, b, c]).real
        inside = [r for r in roots if m[0] <= r <= m[1]]
        root = inside[0] if inside else float(np.mean(m))
    d = math.sqrt(2) * abs(m[1] - m[0]) / math.sqrt(s[0] ** 2 + s[1] ** 2)
    return float(10**root), float(d)


# --------------------------------------------------------- micronuclei step


def find_micronuclei(
    nuclei: NDArray[Any],
    dna: NDArray[Any],
    *,
    max_area_fraction: float,
    min_area_px: float,
    search_px: float,
    threshold_fraction: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], int]:
    """2D: (nuclei without the small objects, micronuclei labels, parent nucleus of each
    micronucleus, number of small objects taken out of the nuclei)."""
    from scipy import ndimage

    nuclei = np.asarray(nuclei, dtype=np.int32)
    areas = np.bincount(nuclei.ravel())
    ids = np.nonzero(areas)[0]
    ids = ids[ids > 0]
    if ids.size == 0:
        return nuclei, np.zeros_like(nuclei), np.zeros(1, dtype=np.int32), 0
    median_area = float(np.median(areas[ids]))
    max_px = max_area_fraction * median_area
    small = ids[areas[ids] < max_px]
    big = np.where(np.isin(nuclei, small), 0, nuclei).astype(np.int32)
    smooth = ndimage.gaussian_filter(np.asarray(dna, dtype=np.float64), 0.7)
    means = ndimage.mean(smooth, big, ids[~np.isin(ids, small)])
    level = threshold_fraction * float(np.median(means)) if len(means) else math.inf
    inside = big > 0
    candidates, _ = ndimage.label((smooth > level) & ~inside)
    touching = np.unique(candidates[ndimage.binary_dilation(inside, iterations=1)])
    sizes = np.bincount(candidates.ravel())
    keep = np.zeros(len(sizes), dtype=bool)
    keep[1:] = (sizes[1:] >= min_area_px) & (sizes[1:] <= max_px)
    keep[touching] = False
    distance, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    labels = np.zeros_like(nuclei)
    parents = [0]
    for index in np.nonzero(keep)[0]:
        ys, xs = np.nonzero(candidates == index)
        closest = int(np.argmin(distance[ys, xs]))
        if distance[ys[closest], xs[closest]] > search_px:
            continue
        labels[ys, xs] = len(parents)
        parents.append(int(big[iy[ys[closest], xs[closest]], ix[ys[closest], xs[closest]]]))
    return big, labels, np.asarray(parents, dtype=np.int32), int(small.size)


class MitoticErrorsParams(ModuleParams):
    channel: int = Field(ge=1, description="DNA channel (starting at 1).")
    nuclei: str = Field(default="nuclei", description="Label set of the nuclei.")
    micronuclei: bool = Field(default=True, description="Detect micronuclei.")
    output: str = Field(default="micronuclei", description="Name of the micronucleus label set.")
    max_area_fraction: float = Field(
        default=0.2, gt=0, lt=1, description="Largest micronucleus, × median nuclear area."
    )
    min_area_um2: float = Field(default=0.5, ge=0, description="Smallest micronucleus.")
    search_um: float = Field(default=5.0, gt=0, description="Largest distance from a nucleus edge.")
    threshold_fraction: float = Field(
        default=0.4, gt=0, lt=1, description="DNA threshold, × median nuclear DNA intensity."
    )
    cells: str = Field(
        default="",
        description="Cells segmented on a cell stain (not grown from the nuclei): each nucleus "
        "is linked to its cell, for nuclei per cell. Empty = no multinucleation.",
    )


@MODULES.register("mitotic_errors")
class MitoticErrors(Module):
    """Micronuclei next to nuclei, and nuclei per cell (see module docs)."""

    name = "mitotic_errors"
    title = "Micronuclei and multinucleation"
    Params = MitoticErrorsParams

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.search_um, metadata, "The micronucleus search distance")
        if self.params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        from hcs_studio.segmentation.objects import overlap_parents

        params: MitoticErrorsParams = self.params
        nuclei = image.label_set(params.nuclei)
        children = [ls.name for ls in image.labels if ls.parent == nuclei.name]
        if children:
            raise ModuleError(
                f"{', '.join(children)} already belong to the {nuclei.name}: put mitotic_errors "
                "right after the nuclear segmentation"
            )
        if nuclei.data.shape[0] > 1:
            raise ModuleError("micronuclei are found in 2D: add a Z projection first")
        report: dict[str, Any] = {"module": self.name}
        data = nuclei.data
        labels: list[LabelSet] = []
        if params.micronuclei:
            pixel_area = (image.metadata.pixel_size_um or 1.0) ** 2
            big, mn, parents, moved = find_micronuclei(
                nuclei.data[0],
                image.data[params.channel - 1, 0],
                max_area_fraction=params.max_area_fraction,
                min_area_px=params.min_area_um2 / pixel_area,
                search_px=um_to_px(params.search_um, image.metadata, "The search distance"),
                threshold_fraction=params.threshold_fraction,
            )
            data = big[None]
            labels.append(LabelSet(params.output, mn[None], parent=nuclei.name,
                                   parent_ids=parents, source="micronuclei"))  # fmt: skip
            report.update({"micronuclei": int(len(parents) - 1), "moved_from_nuclei": moved})
        parent, parent_ids = nuclei.parent, nuclei.parent_ids
        if params.cells:
            cells = image.label_set(params.cells)
            if cells.parent == nuclei.name:
                raise ModuleError(
                    f"the {cells.name} were made from the {nuclei.name} and hold one nucleus "
                    "each: segment cells on a cell stain without the nuclei as parent"
                )
            parent, parent_ids = cells.name, overlap_parents(data, cells.data)
            report["nuclei_in_cells"] = int((parent_ids[1:] > 0).sum())
        updated = LabelSet(nuclei.name, np.asarray(data, dtype=np.int32), parent, parent_ids,
                           nuclei.source)  # fmt: skip
        names = {nuclei.name} | {ls.name for ls in labels}
        kept = tuple(updated if ls.name == nuclei.name else ls for ls in image.labels
                     if ls.name not in names - {nuclei.name})  # fmt: skip
        return ImageData(data=image.data, metadata=image.metadata, linear=image.linear,
                         history=(*image.history, self.name), reports=(*image.reports, report),
                         labels=(*kept, *labels))  # fmt: skip


# -------------------------------------------------------------- assay step


class ProliferationParams(TableParams):
    dna_channel: str = Field(default="DAPI", description="Nuclear stain.")
    ki67_channel: str = Field(default="", description="Ki67 (or another nuclear marker).")
    edu_channel: str = Field(default="", description="EdU or BrdU.")
    ph3_channel: str = Field(default="", description="Phospho-histone H3 (mitoses).")
    cells_channel: str = Field(
        default="", description="Cell stain for multinucleation (the preset segments cells on it)."
    )
    feature: str = Field(
        default="mean", description="Nuclear intensity feature (mean, integrated)."
    )
    threshold_method: Literal[
        "auto", "secondary_only", "vehicle_mixture", "vehicle_otsu", "vehicle_robust"
    ] = Field(
        default="auto",
        description="auto: secondary-only wells if any, else vehicle_robust for pH3 and "
        "vehicle_mixture for the others.",
    )
    threshold_sd: float = Field(default=5.0, gt=0)
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Fixed thresholds by marker (ki67, edu, ph3)."
    )
    secondary_only_wells: list[str] = Field(
        default_factory=list,
        description="Secondary-only wells; empty = wells with role 'unstained' in the layout.",
    )
    micronuclei: str = Field(default="micronuclei", description="Micronucleus label set.")
    cells: str = Field(default="cells", description="Cell label set (multinucleation).")


@MODULES.register("proliferation")
class Proliferation(AssayModule):
    """% marker-positive nuclei, micronuclei and multinucleated cells (module docs)."""

    name = "proliferation"
    title = "Assay: proliferation markers"
    Params = ProliferationParams
    prefix: ClassVar[str] = "pf_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole("dna", "DNA", "DAPI or Hoechst: nuclei and micronuclei"),
        ChannelRole("ki67", "Ki67", "Ki67 or another nuclear marker", required=False),
        ChannelRole("edu", "EdU/BrdU", "EdU or BrdU incorporation", required=False),
        ChannelRole("ph3", "pH3", "phospho-histone H3 (mitoses)", required=False),
        ChannelRole(
            "cells", "Cell stain", "for multinucleation (CellMask, actin …)", required=False
        ),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}
        self.distributions: dict[str, dict[str, Any]] = {}
        self.controls: dict[str, dict[str, NDArray[Any]]] = {}

    @property
    def p(self) -> ProliferationParams:
        params: ProliferationParams = self.params
        return params

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    def markers(self) -> dict[str, str]:
        """Marker key → measurement column."""
        return {
            key: channel_column(getattr(self.p, f"{key}_channel"), self.p.feature)
            for key in MARKERS
            if getattr(self.p, f"{key}_channel")
        }

    def well_roles(self) -> dict[str, str]:
        roles = super().well_roles()
        for well in self.p.secondary_only_wells:
            roles[well.upper()] = "unstained"
        return roles

    # ------------------------------------------------------------ calibration

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        markers = self.markers()
        kind = str(self.params.object_type)
        vehicle = self.control_objects(VEHICLE_ROLES)
        source = "vehicle wells"
        if vehicle.empty:
            vehicle = sample[sample["object_type"] == kind]
            source = "all wells (no vehicle wells in the plate layout)"
        secondary = self.control_objects(SECONDARY_ROLES)
        positive = self.control_objects(POSITIVE_ROLES)
        state: dict[str, Any] = {"calibrated_on": source, "vehicle_nuclei": len(vehicle),
                                 "secondary_only_nuclei": len(secondary), "markers": {},
                                 "warnings": []}  # fmt: skip
        if not markers:
            state["warnings"].append("no marker channel: micronuclei and multinucleation only")
        for key, column in markers.items():
            self.require(vehicle, [column])

            def values(frame: pd.DataFrame, column: str = column) -> NDArray[Any]:
                if frame.empty or column not in frame:
                    return np.array([])
                x = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                return x[np.isfinite(x) & (x > 0)]

            v, s, pos = values(vehicle), values(secondary), values(positive)
            self.controls[key] = {"vehicle": v, "secondary only": s, "positive control": pos}
            method: str = self.p.threshold_method
            if method == "auto":
                method = ("secondary_only" if s.size >= 20
                          else "vehicle_robust" if key == "ph3" else "vehicle_mixture")  # fmt: skip
            entry: dict[str, Any] = {"column": column}
            if key in self.p.thresholds:
                method, threshold = "fixed", float(self.p.thresholds[key])
            elif method == "secondary_only":
                if s.size < 20:
                    raise ModuleError(
                        f"{TITLES[key]}: no secondary-only nuclei (mark the wells 'unstained' in "
                        "the layout or list them in secondary_only_wells)"
                    )
                threshold = robust_upper_log(s, self.p.threshold_sd)
            elif method == "vehicle_robust":
                threshold = robust_upper_log(v, self.p.threshold_sd)
            elif method == "vehicle_mixture":
                threshold, d = mixture_threshold(v)
                entry["ashman_d"] = d
                if d < 2:
                    state["warnings"].append(
                        f"{TITLES[key]}: the vehicle nuclei do not form two separate "
                        f"populations (Ashman's D = {d:.1f} < 2); use secondary-only wells"
                    )
            else:
                threshold = otsu_log(v)
                low, high = np.log10(v[v <= threshold]), np.log10(v[v > threshold])
                if len(low) > 2 and len(high) > 2:
                    d = (
                        math.sqrt(2)
                        * abs(high.mean() - low.mean())
                        / math.sqrt(low.var() + high.var())
                    )
                    entry["ashman_d"] = float(d)
                    if d < 2:
                        state["warnings"].append(
                            f"{TITLES[key]}: the vehicle nuclei do not form two separate "
                            f"populations (Ashman's D = {d:.1f} < 2); use secondary-only wells"
                        )
            entry.update({"method": method, "threshold": float(threshold),
                          "vehicle_positive_pct": 100 * float((v > threshold).mean()) if v.size else math.nan})  # fmt: skip
            state["markers"][key] = entry
        return state

    def preview_values(self, state: Any) -> tuple[Any, str]:
        if not self.controls:
            return np.array([]), ""
        key = next(iter(self.controls))
        return np.log10(self.controls[key]["vehicle"]), f"log10 {TITLES[key]} intensity, vehicle"

    # --------------------------------------------------------------- per well

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        rows = self.selected(objects)
        out = pd.DataFrame(index=objects.index)
        for key, entry in state["markers"].items():
            flag = pd.Series(pd.NA, index=objects.index, dtype="boolean")
            values = pd.to_numeric(objects.loc[rows, entry["column"]], errors="coerce")
            flag.loc[values.index] = (values > entry["threshold"]).to_numpy(bool)
            flag.loc[values.index[values.isna()]] = pd.NA
            out[f"pf_{key}_positive"] = flag
        count = f"{self.p.micronuclei}_count"
        if count in objects:
            mn = pd.Series(np.nan, index=objects.index)
            mn.loc[rows] = pd.to_numeric(objects.loc[rows, count], errors="coerce").fillna(0)
            out["pf_micronuclei"] = mn
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = self.selected(objects)
        n = int(rows.sum())
        values: dict[str, Any] = {"pf_nuclei": n}
        details: dict[str, Any] = {}
        for key, entry in state["markers"].items():
            flags = objects.loc[rows, f"pf_{key}_positive"].dropna().astype(bool)
            values[f"pf_{key}_pos_pct"] = 100 * float(flags.mean()) if len(flags) else math.nan
            intensity = pd.to_numeric(objects.loc[rows, entry["column"]], errors="coerce").dropna()
            values[f"pf_{key}_median"] = float(intensity.median()) if len(intensity) else math.nan
            positive = intensity[intensity > 0]
            if len(positive):
                q = np.percentile(np.log10(positive), [5, 25, 50, 75, 95])
                self.distributions.setdefault(key, {})[well.well] = q.tolist()
        if "pf_micronuclei" in objects:
            mn = objects.loc[rows, "pf_micronuclei"].fillna(0)
            values["pf_mn_pct"] = 100 * float((mn > 0).mean()) if n else math.nan
            values["pf_mn_per_100"] = 100 * float(mn.sum()) / n if n else math.nan
        cells = objects[objects["object_type"] == self.p.cells]
        count = f"{self.params.object_type}_count"
        if not cells.empty and count in cells:
            k = pd.to_numeric(cells[count], errors="coerce").fillna(0)
            k = k[k > 0]
            if len(k):
                values["pf_cells"] = len(k)
                values["pf_multinucleated_pct"] = 100 * float((k >= 2).mean())
                values["pf_binucleated_pct"] = 100 * float((k == 2).mean())
                values["pf_nuclei_per_cell"] = float(k.mean())
        self.wells[well.well] = values
        return values, details

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        roles = self.well_roles()
        vehicle = [w for w, r in roles.items() if r in VEHICLE_ROLES and w in self.wells]
        positive = [w for w, r in roles.items() if r in POSITIVE_ROLES and w in self.wells]
        readouts = [f"pf_{k}_pos_pct" for k in state["markers"]]
        readouts += [r for r in ("pf_mn_per_100", "pf_multinucleated_pct")
                     if any(r in v for v in self.wells.values())]  # fmt: skip
        z_prime: dict[str, float] = {}
        for readout in readouts:
            ref = [self.wells[w][readout] for w in vehicle if readout in self.wells[w]]
            level = float(np.median(ref)) if ref else math.nan
            for well, values in self.wells.items():
                if readout in values:
                    fold = values[readout] / level if level > 0 else math.nan
                    values[readout.replace("_pct", "") + "_vs_vehicle"] = fold
                    per_well.setdefault(well, {})[readout.replace("_pct", "") + "_vs_vehicle"] = (
                        fold
                    )
            if len(vehicle) >= 2 and len(positive) >= 2:
                a = np.array([self.wells[w].get(readout, math.nan) for w in vehicle])
                b = np.array([self.wells[w].get(readout, math.nan) for w in positive])
                spread = abs(float(b.mean() - a.mean()))
                z_prime[readout] = (
                    float(1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / spread) if spread else math.nan
                )
        state["z_prime"] = z_prime
        out = self.output_folder(folder)
        calibration = {"module": self.name, "params": self.p.model_dump(mode="json"), **state}
        (out / "calibration.json").write_text(
            json.dumps(calibration, indent=1, default=str), "utf-8"
        )
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "proliferation_wells.csv", index=False)
        write_report(out, calibration, table, self.controls, self.distributions)


def write_report(
    folder: Path,
    calibration: dict[str, Any],
    table: pd.DataFrame,
    controls: dict[str, dict[str, NDArray[Any]]],
    distributions: dict[str, dict[str, Any]],
) -> Path:
    """Per marker: control histograms with the threshold, and per-well intensity boxes."""
    import base64
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figures = []
    for key, entry in calibration["markers"].items():
        fig = Figure(figsize=(9, 3.0), dpi=100)
        FigureCanvasAgg(fig)
        left, right = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, (2, 3))
        groups = {name: np.log10(v) for name, v in controls.get(key, {}).items() if len(v)}
        if groups:
            bins: Any = np.linspace(
                min(map(np.min, groups.values())), max(map(np.max, groups.values())), 60
            )
            for (name, values), colour in zip(
                groups.items(), ("#4477aa", "#999999", "#ee6677"), strict=False
            ):
                left.hist(values, bins=bins, alpha=0.6, density=True, color=colour, label=name)
            left.legend(fontsize=6)
        threshold = math.log10(entry["threshold"])
        left.axvline(threshold, color="black", linestyle="--")
        left.set_xlabel(f"log10 {TITLES[key]}")
        wells = distributions.get(key, {})
        stats = [{"label": w, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4]}
                 for w, q in sorted(wells.items())]  # fmt: skip
        if stats:
            right.bxp(stats, showfliers=False)
            right.tick_params(axis="x", labelrotation=90, labelsize=6)
        right.axhline(threshold, color="black", linestyle="--")
        right.set_ylabel(f"log10 {TITLES[key]} (5–95 %)")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", metadata={"Software": None})
        figures.append(
            f"<h2>{TITLES[key]}</h2><p>{html.escape(entry['method'])}: threshold {entry['threshold']:.4g}"
            f" ({entry['vehicle_positive_pct']:.1f} % of vehicle nuclei positive).</p>"
            f'<img alt="{TITLES[key]}" src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode("ascii")}">'
        )
    shown = [c for c in table.columns if c == "well" or c.startswith("pf_")]

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"<td>{value:.3g}</td>" if math.isfinite(value) else "<td></td>"
        return f"<td>{html.escape(str(value))}</td>"

    rows = "".join(
        "<tr>" + "".join(cell(r.get(c, math.nan)) for c in shown) + "</tr>"
        for r in table.to_dict("records")
    )
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown)
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in calibration.get("warnings", []))
    z = ", ".join(
        f"{k} {v:.2f}" for k, v in calibration.get("z_prime", {}).items() if math.isfinite(v)
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Proliferation</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}th{{background:#f3f3f3}}.warn{{color:#b00020}}</style></head><body>
<h1>Proliferation markers</h1>
<p>Calibrated on {html.escape(str(calibration.get("calibrated_on", "")))} ({calibration.get("vehicle_nuclei", 0)} vehicle nuclei,
{calibration.get("secondary_only_nuclei", 0)} secondary-only nuclei). {("Z′ (positive control vs vehicle): " + html.escape(z) + ".") if z else ""}</p>
{f'<ul class="warn">{warnings}</ul>' if warnings else ""}
{"".join(figures)}
<h2>Per well</h2><table><tr>{head}</tr>{rows}</table></body></html>"""
    target = folder / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    nucleus_diameter_um: float = 10.0,
    cell_diameter_um: float = 25.0,
    z_planes: int = 1,
    micronuclei: bool = True,
) -> Any:
    """Background, nuclei (Otsu), cells on the cell stain if mapped, micronuclei /
    multinucleation, measurement and the proliferation step."""
    from hcs_studio.pipeline.schema import Pipeline

    def number(name: str) -> int:
        return channels.index(name) + 1 if channels and name in channels else 1

    dna = mapping.get("dna", "DAPI")
    cells = mapping.get("cells", "")
    steps: list[dict[str, Any]] = []
    if z_planes > 1:
        steps.append({"module": "projection", "params": {"method": "max"}})
    steps += [
        {
            "module": "background",
            "params": {"method": "tophat", "radius_um": 2.5 * cell_diameter_um},
        },
        {
            "module": "segment_classical",
            "params": {
                "channel": number(dna),
                "method": "otsu",
                "min_distance_um": nucleus_diameter_um / 4,
            },
        },
    ]
    if cells:
        steps.append({"module": "segment_classical", "params": {
            "channel": number(cells), "method": "otsu", "output": "cells",
            "min_distance_um": cell_diameter_um / 2}})  # fmt: skip
    if micronuclei or cells:
        steps.append({"module": "mitotic_errors", "params": {
            "channel": number(dna), "micronuclei": micronuclei, "cells": "cells" if cells else ""}})  # fmt: skip
    params = {
        f"{k}_channel": mapping[k] for k in ("dna", "ki67", "edu", "ph3", "cells") if mapping.get(k)
    }
    steps += [
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {"module": "proliferation", "params": params},
    ]
    return Pipeline.model_validate({"name": "proliferation", "steps": steps})
