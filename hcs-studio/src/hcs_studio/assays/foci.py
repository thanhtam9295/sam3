"""DNA-damage foci: γH2AX, 53BP1, RAD51 and others (Phase 20b).

Two pipeline steps:

- ``detect_foci`` (image step): spots inside each nucleus. The foci channel is band-pass
  filtered (Laplacian of Gaussian scaled by σ², or a white top-hat), candidate foci are pixels
  above a threshold, split at local maxima (seeded watershed of the filter response) and kept
  when their size is plausible. The threshold is **calibrated on the vehicle wells**: median +
  ``threshold_sd`` × robust SD (1.4826 × MAD) of the filter response over the vehicle nuclei's
  pixels (robust to the few foci vehicle nuclei have), or a fixed value. With a Z stack the
  detection is 3D (anisotropic filter, foci counted once through Z). Foci become the label set
  ``foci`` whose parents are the nuclei, so ``measure_objects`` reports per nucleus the count,
  mean area and total intensity of its foci.
- ``foci`` (assay table step): per well the mean foci per nucleus and % of nuclei with ≥ N foci
  (``positive_min_foci``), mean foci area and intensity, and **pan-nuclear** γH2AX nuclei
  (mean nuclear intensity of the foci channel ≥ ``pan_factor`` × the vehicle nuclei's median:
  bright, diffuse nuclei — apoptosis or replication stress), reported separately and left out
  of the focal statistics. When the cell-cycle step ran first, foci are broken down by phase
  (RAD51 foci are expected in S/G2).

A maximum projection merges foci at different depths into one: the step warns when the run
used one (count in 3D, or use a sum projection and accept merged foci).
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
from scipy import ndimage

from hcs_studio.analysis.table_steps import TableParams, WellContext
from hcs_studio.assays.base import (
    POSITIVE_ROLES,
    VEHICLE_ROLES,
    AssayModule,
    ChannelRole,
    channel_column,
)
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    FitContext,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)

ROBUST = 1.4826


# -------------------------------------------------------------- detection


def spot_response(
    data: NDArray[Any], sigma_xy: float, z_ratio: float = 1.0, method: str = "log"
) -> NDArray[np.float64]:
    """Band-pass response of a (Z, Y, X) stack: 2D per plane when Z = 1, else 3D with the
    Z sigma scaled by ``z_ratio`` (pixel size / Z step)."""
    stack = np.asarray(data, dtype=np.float64)
    three_d = stack.shape[0] > 1
    if method == "log":
        sigma: Any = (max(sigma_xy * z_ratio, 0.5), sigma_xy, sigma_xy) if three_d else sigma_xy
        if three_d:
            return np.asarray(-ndimage.gaussian_laplace(stack, sigma) * sigma_xy**2)
        return np.stack([-ndimage.gaussian_laplace(p, sigma) * sigma**2 for p in stack])
    radius = max(1, round(2.5 * sigma_xy))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disc = (yy**2 + xx**2) <= radius**2
    smooth = ndimage.gaussian_filter(stack, (0, 0.7, 0.7))
    return np.stack([ndimage.white_tophat(p, footprint=disc) for p in smooth])


def nuclear_mask(labels: NDArray[Any], depth: int) -> NDArray[Any]:
    """Nuclear labels as (Z, Y, X) of the image's depth (2D masks are repeated through Z)."""
    if labels.shape[0] == depth:
        return labels
    if labels.shape[0] == 1:
        return np.repeat(labels, depth, axis=0)
    raise ModuleError(f"Nuclear masks have {labels.shape[0]} planes, the image {depth}")


def find_foci(
    response: NDArray[np.float64],
    nuclei: NDArray[Any],
    threshold: float,
    *,
    min_size_px: int,
    max_size_px: int,
    separation_px: float,
) -> NDArray[np.int32]:
    """Foci labels: response above ``threshold`` inside nuclei, split at local maxima."""
    from skimage.segmentation import watershed

    inside = nuclei > 0
    mask = (response > threshold) & inside
    if not mask.any():
        return np.zeros(response.shape, dtype=np.int32)
    size = max(3, round(separation_px) | 1)
    footprint = (1 if response.shape[0] == 1 else 3, size, size)
    peaks = (response == ndimage.maximum_filter(response, size=footprint)) & mask
    markers, _ = ndimage.label(peaks)
    labels = watershed(-response, markers, mask=mask)
    sizes = np.bincount(labels.ravel())
    keep = (sizes >= min_size_px) & (sizes <= max_size_px)
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    from hcs_studio.segmentation.objects import relabel_sequential

    return np.asarray(relabel_sequential(labels), dtype=np.int32)


class DetectFociParams(ModuleParams):
    channel: int = Field(ge=1, description="Foci channel (γH2AX, 53BP1, RAD51 …).")
    nuclei: str = Field(default="nuclei", description="Nuclear label set (the foci's parents).")
    output: str = Field(default="foci", description="Name of the foci label set.")
    diameter_um: float = Field(default=0.7, gt=0, description="Typical focus diameter in µm.")
    method: Literal["log", "tophat"] = Field(
        default="log", description="Laplacian of Gaussian, or white top-hat."
    )
    threshold: float | None = Field(
        default=None, description="Fixed filter-response threshold; empty = from vehicle wells."
    )
    threshold_sd: float = Field(
        default=5.0, gt=0, description="Calibration: median + this × robust SD of vehicle nuclei."
    )
    min_area_um2: float = Field(default=0.05, ge=0, description="Smallest focus (µm²; per plane).")
    max_area_um2: float | None = Field(
        default=None, gt=0,
        description="Largest focus (µm²; per plane); empty = a disc twice the focus diameter.",
    )  # fmt: skip
    calibration_fields: int = Field(default=24, ge=1, description="Vehicle fields sampled.")


@MODULES.register("detect_foci")
class DetectFoci(Module):
    """Foci inside nuclei with a threshold calibrated on the vehicle wells (see module docs)."""

    name = "detect_foci"
    title = "Assay: detect DNA-damage foci"
    Params = DetectFociParams

    @property
    def needs_fit(self) -> bool:  # type: ignore[override]
        return self.params.threshold is None

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.diameter_um, metadata, "The focus diameter")
        if self.params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.channel} does not exist")

    def _geometry(self, metadata: ImageMetadata) -> tuple[float, float]:
        diameter = um_to_px(self.params.diameter_um, metadata, "The focus diameter")
        sigma = max(diameter / (2 * np.sqrt(2)), 0.6)
        ratio = 1.0
        if metadata.z_step_um and metadata.pixel_size_um:
            ratio = metadata.pixel_size_um / metadata.z_step_um
        return sigma, ratio

    def fit(self, context: FitContext) -> dict[str, Any]:
        params: DetectFociParams = self.params
        vehicle = [
            key for key, a in context.layout.items()
            if a.role == "negative_control" or (a.role == "untreated" and not a.active_treatments)
        ]  # fmt: skip
        present = [w for w in vehicle if w in context.plate.wells]
        source = "vehicle wells" if present else "all wells (no vehicle wells in the layout)"
        values: list[NDArray[Any]] = []
        fields = 0
        for image in context.sample(params.calibration_fields, present or None):
            sigma, ratio = self._geometry(image.metadata)
            data = image.data[params.channel - 1]
            nuclei = nuclear_mask(image.label_set(params.nuclei).data, data.shape[0])
            response = spot_response(data, sigma, ratio, params.method)
            values.append(response[nuclei > 0])
            fields += 1
        pooled = np.concatenate(values) if values else np.array([])
        if pooled.size < 100:
            raise ModuleError("No nuclei in the calibration fields to set the foci threshold")
        median = float(np.median(pooled))
        sd = ROBUST * float(np.median(np.abs(pooled - median)))
        return {
            "threshold": median + params.threshold_sd * sd,
            "median": median,
            "robust_sd": sd,
            "calibrated_on": source,
            "fields": fields,
            "pixels": int(pooled.size),
        }

    def describe_state(self, state: Any) -> dict[str, Any]:
        return dict(state) if state else {"threshold": self.params.threshold}

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: DetectFociParams = self.params
        threshold = params.threshold if params.threshold is not None else state["threshold"]
        sigma, ratio = self._geometry(image.metadata)
        data = image.data[params.channel - 1]
        nuclei_set = image.label_set(params.nuclei)
        nuclei = nuclear_mask(nuclei_set.data, data.shape[0])
        response = spot_response(data, sigma, ratio, params.method)
        pixel = image.metadata.pixel_size_um or 1.0
        min_px = max(1, int(params.min_area_um2 / pixel**2))
        largest = params.max_area_um2 or float(np.pi * params.diameter_um**2)
        max_px = int(largest / pixel**2) * max(1, data.shape[0])
        foci = find_foci(
            response, nuclei, threshold, min_size_px=min_px, max_size_px=max_px,
            separation_px=2 * sigma * np.sqrt(2),
        )  # fmt: skip
        from hcs_studio.segmentation.objects import point_parents

        parent_ids = point_parents(foci, nuclei)
        result = LabelSet(params.output, foci, parent=params.nuclei, parent_ids=parent_ids,
                          source=f"{params.method} foci")  # fmt: skip
        report = {
            "module": self.name,
            "foci": result.count,
            "threshold": float(threshold),
            "three_d": bool(data.shape[0] > 1),
        }
        return image.with_labels(result, self.name, report)


# -------------------------------------------------------------- assay step


class FociParams(TableParams):
    foci_channel: str = Field(default="gH2AX", description="Foci channel name (as on the plate).")
    foci: str = Field(default="foci", description="Label set of the foci (detect_foci output).")
    positive_min_foci: int = Field(
        default=5, ge=1, description="A nucleus is foci-positive with at least this many foci."
    )
    pan_factor: float = Field(
        default=4.0,
        gt=1,
        description="Pan-nuclear: mean nuclear intensity ≥ this × vehicle median.",
    )


@MODULES.register("foci")
class Foci(AssayModule):
    """DNA-damage foci per nucleus and well, pan-nuclear nuclei apart (see module docs)."""

    name = "foci"
    title = "Assay: DNA-damage foci (γH2AX, 53BP1, RAD51)"
    Params = FociParams
    prefix: ClassVar[str] = "dd_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole("dna", "DNA", "nuclear stain used for the nuclei (DAPI, Hoechst)"),
        ChannelRole("foci", "Foci", "γH2AX, 53BP1, RAD51 or another foci marker"),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}

    @property
    def p(self) -> FociParams:
        params: FociParams = self.params
        return params

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    def columns(self) -> dict[str, str]:
        name = self.p.foci
        return {
            "count": f"{name}_count",
            "area": f"{name}_mean_area_um2",
            "intensity": f"{name}_{channel_column(self.p.foci_channel, 'total_intensity')}",
            "nuclear_mean": channel_column(self.p.foci_channel, "mean"),
        }

    def run_warnings(self) -> list[str]:
        warnings = []
        manifest = self.manifest()
        for step in (manifest.get("pipeline") or {}).get("steps", []):
            if step.get("module") == "projection" and step.get("enabled", True):
                method = (step.get("params") or {}).get("method", "max")
                warnings.append(
                    f"the images were Z-projected ({method}): foci at different depths of a "
                    "nucleus merge in the projection; count in 3D (no projection) for exact counts"
                )
        return warnings

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        columns = self.columns()
        vehicle = self.control_objects(VEHICLE_ROLES)
        source = "vehicle wells"
        if vehicle.empty:
            vehicle = sample[sample["object_type"] == str(self.params.object_type)]
            source = "all wells (no vehicle wells in the plate layout)"
        self.require(vehicle, [columns["nuclear_mean"]])
        if columns["count"] not in vehicle and columns["count"] not in sample:
            raise ModuleError(
                f"no foci counts ({columns['count']}): add detect_foci before measure_objects"
            )
        median = float(np.nanmedian(vehicle[columns["nuclear_mean"]].to_numpy(float)))
        counts = vehicle.get(columns["count"], pd.Series(0.0, index=vehicle.index)).fillna(0.0)
        return {
            "calibrated_on": source,
            "vehicle_nuclei": len(vehicle),
            "vehicle_nuclear_mean_median": median,
            "pan_nuclear_above": self.p.pan_factor * median,
            "vehicle_foci_per_nucleus": float(counts.mean()),
            "foci_threshold": self._detection_threshold(),
            "warnings": self.run_warnings(),
        }

    def preview_values(self, state: Any) -> tuple[Any, str]:
        column = self.columns()["count"]
        vehicle = self.control_objects(VEHICLE_ROLES, [column])
        values = vehicle[column].fillna(0).to_numpy(float) if column in vehicle else np.array([])
        return values, "foci per vehicle nucleus"

    def _detection_threshold(self) -> Any:
        manifest = self.manifest()
        for entry in manifest.get("fitted_state") or []:
            if entry.get("module") == "detect_foci":
                return entry.get("threshold")
        return None

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        columns = self.columns()
        rows = self.selected(objects)
        # nuclei without foci have no child row: no value means 0 foci
        if columns["count"] in objects:
            count = pd.to_numeric(objects[columns["count"]], errors="coerce").fillna(0)
        else:
            count = pd.Series(0.0, index=objects.index)
        mean = pd.to_numeric(objects[columns["nuclear_mean"]], errors="coerce")
        pan = mean >= state["pan_nuclear_above"]
        positive = (count >= self.p.positive_min_foci) & ~pan
        return pd.DataFrame(
            {
                "dd_foci": count.where(rows, np.nan),
                "dd_pan_nuclear": pd.Series(pan, index=objects.index, dtype="boolean").where(
                    rows, pd.NA
                ),
                "dd_positive": pd.Series(positive, index=objects.index, dtype="boolean").where(
                    rows, pd.NA
                ),
            }
        )

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        columns = self.columns()
        rows = objects[objects["dd_foci"].notna()]
        pan = rows["dd_pan_nuclear"].astype(bool)
        focal = rows[~pan]
        values: dict[str, Any] = {
            "dd_nuclei": len(rows),
            "dd_pan_nuclear_pct": 100 * float(pan.mean()) if len(rows) else math.nan,
            "dd_foci_per_nucleus": float(focal["dd_foci"].mean()) if len(focal) else math.nan,
            "dd_foci_per_nucleus_median": float(focal["dd_foci"].median()) if len(focal) else math.nan,
            "dd_positive_pct": 100 * float(focal["dd_positive"].astype(bool).mean()) if len(focal) else math.nan,
        }  # fmt: skip
        with_foci = focal[focal["dd_foci"] > 0]
        for key, column in (("area", "dd_focus_area_um2"), ("intensity", "dd_foci_intensity")):
            source = columns[key]
            values[column] = (
                float(pd.to_numeric(with_foci[source], errors="coerce").mean())
                if source in with_foci and len(with_foci)
                else math.nan
            )
        reference = state.get("vehicle_foci_per_nucleus") or math.nan
        values["dd_foci_fold_vs_vehicle"] = (
            values["dd_foci_per_nucleus"] / reference if reference and reference > 0 else math.nan
        )
        if "cc_phase" in focal:
            for phase, label in (("g1", "g1"), ("s", "s"), ("g2m", "g2m")):
                part = focal[focal["cc_phase"] == phase]
                values[f"dd_foci_per_nucleus_{label}"] = (
                    float(part["dd_foci"].mean()) if len(part) else math.nan
                )
                values[f"dd_positive_{label}_pct"] = (
                    100 * float(part["dd_positive"].astype(bool).mean()) if len(part) else math.nan
                )
        self.wells[well.well] = values
        histogram = np.bincount(focal["dd_foci"].astype(int).clip(0, 40), minlength=41)
        return values, {"foci_histogram": histogram.tolist()}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        out = self.output_folder(folder)
        calibration = {"module": self.name, "params": self.p.model_dump(mode="json"), **state}
        (out / "calibration.json").write_text(
            json.dumps(calibration, indent=1, default=str), "utf-8"
        )
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "foci_wells.csv", index=False)
        write_report(out, calibration, table, self.well_roles())


# ----------------------------------------------------------------- report


def write_report(
    folder: Path, calibration: dict[str, Any], table: pd.DataFrame, roles: dict[str, str]
) -> Path:
    import base64
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 3), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if not table.empty:
        colours = ["#4477aa" if roles.get(w) in VEHICLE_ROLES else "#ee6677" if roles.get(w) in POSITIVE_ROLES else "#999999" for w in table["well"]]  # fmt: skip
        ax.bar(range(len(table)), table["dd_foci_per_nucleus"], color=colours)
        ax.set_xticks(range(len(table)), table["well"], rotation=90, fontsize=6)
    ax.set_ylabel("foci per nucleus (focal nuclei)")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    shown = [c for c in table.columns if c == "well" or c.startswith("dd_")]

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"<td>{value:.2f}</td>" if math.isfinite(value) else "<td></td>"
        return f"<td>{html.escape(str(value))}</td>"

    rows = "".join(
        "<tr>" + "".join(cell(r[c]) for c in shown) + "</tr>" for r in table.to_dict("records")
    )
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown)
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in calibration.get("warnings", []))
    threshold = calibration.get("foci_threshold")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DNA-damage foci</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}th{{background:#f3f3f3}}.warn{{color:#b00020}}</style></head><body>
<h1>DNA-damage foci</h1>
<p>Calibrated on {html.escape(str(calibration.get("calibrated_on", "")))} ({calibration.get("vehicle_nuclei", 0)} nuclei):
{calibration.get("vehicle_foci_per_nucleus", 0):.2f} foci per vehicle nucleus;
pan-nuclear above {calibration.get("pan_nuclear_above", 0):.4g} mean intensity
({calibration["params"]["pan_factor"]:g} × the vehicle median);
foci-positive with ≥ {calibration["params"]["positive_min_foci"]} foci.
{f"Detection threshold {threshold['threshold']:.4g} ({html.escape(str(threshold.get('calibrated_on', '')))})." if isinstance(threshold, dict) and threshold.get("threshold") is not None else ""}</p>
{f'<ul class="warn">{warnings}</ul>' if warnings else ""}
<img alt="foci per nucleus" src="data:image/png;base64,{image}">
<h2>Per well</h2><table><tr>{head}</tr>{rows}</table></body></html>"""
    target = folder / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


# ----------------------------------------------------------------- preset


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    nucleus_diameter_um: float = 10.0,
    z_planes: int = 1,
    focus_diameter_um: float = 0.7,
) -> Any:
    """Background removal, nuclei on the DNA channel, foci (3D when the plate has Z planes),
    measurement, and the foci step."""
    from hcs_studio.pipeline.schema import Pipeline

    dna, marker = mapping.get("dna", "DAPI"), mapping.get("foci", "gH2AX")

    def number(name: str, default: int) -> int:
        return channels.index(name) + 1 if channels and name in channels else default

    steps: list[dict[str, Any]] = [
        {
            "module": "background",
            "params": {"method": "tophat", "radius_um": 2.5 * nucleus_diameter_um},
        },
        {
            "module": "segment_classical",
            "params": {
                "channel": number(dna, 1),
                "method": "triangle",
                "min_distance_um": nucleus_diameter_um / 3,
            },
        },
        {
            "module": "detect_foci",
            "params": {"channel": number(marker, 2), "diameter_um": focus_diameter_um},
        },
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {"module": "foci", "params": {"foci_channel": marker}},
    ]
    return Pipeline.model_validate({"name": "dna-damage-foci", "steps": steps})
