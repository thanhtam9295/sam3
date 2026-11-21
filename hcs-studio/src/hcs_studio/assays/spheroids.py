"""Spheroids and organoids (Phase 21a): detection, size, shape, volume, QC flags, gallery.

Three steps:

- **``detect_spheroids``** (image step, 2D): ``brightfield`` — textured objects are found by
  the local standard deviation of a difference of Gaussians at the ``texture_um`` scale (Otsu
  on log SD), in which smooth dark regions such as the well edge and meniscus shadows cancel
  out (objects that are not textured inside are dropped); each object's outline is then
  refined at half height between its interior and its local background (the edge of a blurred
  step lies at half height, so blur does not change the size). ``dark`` uses the
  half-height refinement on the darkness alone; ``bright`` (fluorescence) uses Otsu's
  threshold. Objects smaller than ``min_area_um2`` are dropped; holes are filled. Label set
  ``spheroids`` (largest first).
- **``measure_spheroids``** (measurement step, instead of ``measure_objects`` for this label
  set): area, equivalent diameter, perimeter, circularity (4πA/P²), solidity, compactness (mean
  squared distance from the centroid / that of a disc of the same area: 1 for a disc), major and
  minor axes, volume estimates (sphere from the equivalent diameter: πd³/6; ellipsoid π/6·L·W²),
  brightfield darkness (1 − mean inside / local background), edge sharpness (``edge_sigma_um``:
  the Gaussian blur that explains the contrast / gradient ratio at the outline), position
  (distance of the centroid from the field centre, touching the image border).
- **``spheroid``** (assay table step, one spheroid per well): picks each well's spheroid,
  reports its readouts (``sp_``), QC flags, volume vs vehicle, and writes ``spheroids/``:
  per-spheroid table, per-well table, crops with outlines and a gallery sortable by any readout.
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
from hcs_studio.assays.base import POSITIVE_ROLES, VEHICLE_ROLES, AssayModule, ChannelRole
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    MeasureContext,
    MeasurementModule,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)

FLAGS = ("no_spheroid", "multiple", "touching_edge", "off_centre", "out_of_focus", "debris",
         "irregular")  # fmt: skip
SPHEROID_COLUMNS = ("area_um2", "equivalent_diameter_um", "volume_sphere_um3",
                    "volume_ellipsoid_um3", "circularity", "solidity", "compactness",
                    "darkness", "edge_sigma_um", "offcentre")  # fmt: skip


# ------------------------------------------------------------- detection


def local_sd(image: NDArray[Any], sigma: float) -> NDArray[np.float64]:
    """Local standard deviation in a Gaussian window."""
    from scipy import ndimage

    x = np.asarray(image, dtype=np.float64)
    mean = ndimage.gaussian_filter(x, sigma)
    square = ndimage.gaussian_filter(x * x, sigma)
    return np.asarray(np.sqrt(np.clip(square - mean * mean, 0, None)), dtype=np.float64)


def _ring(mask: NDArray[np.bool_], inner: float, outer: float) -> NDArray[np.bool_]:
    from scipy import ndimage

    distance = np.asarray(ndimage.distance_transform_edt(~mask), dtype=np.float64)
    return np.asarray((distance > inner) & (distance <= outer), dtype=bool)


def texture_map(image: NDArray[Any], texture_px: float) -> NDArray[np.float64]:
    """Local SD of a difference of Gaussians (σ texture/3 − σ texture): cell-scale texture and
    sharp outlines; smooth shading (well edge, meniscus) cancels out."""
    from scipy import ndimage

    pad = math.ceil(4 * texture_px)
    # odd (point) reflection continues gradients across the image border: no kink to respond to
    x = np.pad(np.asarray(image, dtype=np.float64), pad, mode="reflect", reflect_type="odd")
    dog = ndimage.gaussian_filter(x, texture_px / 3) - ndimage.gaussian_filter(x, texture_px)
    return local_sd(dog, texture_px)[pad:-pad, pad:-pad]


def detect(
    image: NDArray[Any],
    *,
    method: str = "brightfield",
    texture_px: float = 4.0,
    min_area_px: float = 100.0,
) -> NDArray[np.int32]:
    """2D spheroid labels, largest first (see the module docstring)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), 1.0)
    textured = np.ones(smooth.shape, dtype=bool)
    if method == "bright":
        coarse = smooth > threshold_otsu(smooth)
    elif method == "brightfield":
        sd = texture_map(image, texture_px)
        band = math.ceil(2 * texture_px)
        inner = np.zeros(sd.shape, dtype=bool)
        inner[band:-band, band:-band] = True  # padding at the image border distorts the filters
        level = float(np.exp(threshold_otsu(np.log(sd[inner] + 1e-6))))
        # a field without a spheroid: Otsu would split the noise; texture must stand out from it
        level = max(level, 4 * float(np.percentile(sd[inner], 10)))
        textured = sd > level
        coarse = ndimage.binary_opening(textured & inner, iterations=2)
    else:  # dark
        coarse = ndimage.binary_opening(smooth < threshold_otsu(smooth), iterations=2)
    coarse = ndimage.binary_fill_holes(coarse)
    labels, n = ndimage.label(coarse)
    out = np.zeros(smooth.shape, dtype=np.int32)
    pieces: list[NDArray[np.bool_]] = []
    for index in range(1, n + 1):
        component = labels == index
        if component.sum() < min_area_px / 4:
            continue
        if method == "bright":
            pieces.append(component)
            continue
        # outline at half height between the interior and the local background
        core = ndimage.binary_erosion(component, iterations=max(1, int(texture_px)))
        inside = float(np.median(smooth[core if core.any() else component]))
        ring = _ring(component, 2 * texture_px, 5 * texture_px) & ~coarse
        if not ring.any():
            continue
        level = (inside + float(np.median(smooth[ring]))) / 2
        roi = _ring(component, -1, 2 * texture_px) | component
        refined = ndimage.binary_fill_holes((smooth < level) & roi)
        refined = ndimage.binary_opening(refined, iterations=1)
        parts, k = ndimage.label(refined)
        if k == 0:
            continue
        overlap = np.bincount(parts[component], minlength=k + 1)
        for part in range(1, k + 1):  # touching objects share a coarse component
            if overlap[part] < min_area_px / 4:
                continue
            piece = parts == part
            if method == "brightfield" and textured[piece].mean() < 0.5:
                continue  # smooth (well edge, meniscus), not cells
            pieces.append(piece)
    sizes = [int(p.sum()) for p in pieces]
    for rank, i in enumerate(sorted(range(len(pieces)), key=lambda i: -sizes[i]), start=1):
        if sizes[i] >= min_area_px:
            out[pieces[i] & (out == 0)] = rank
    from hcs_studio.segmentation.objects import relabel_sequential

    return relabel_sequential(out)


class DetectSpheroidsParams(ModuleParams):
    channel: int = Field(ge=1, description="Brightfield (or fluorescence) channel.")
    method: Literal["brightfield", "dark", "bright"] = Field(
        default="brightfield",
        description="brightfield: texture, then half-height outline; dark: darker than the "
        "background; bright: fluorescence (Otsu).",
    )
    texture_um: float = Field(default=6.0, gt=0, description="Texture window (Gaussian σ).")
    min_area_um2: float = Field(default=300.0, ge=0, description="Smaller objects are dropped.")
    output: str = Field(default="spheroids", description="Name of the label set.")


@MODULES.register("detect_spheroids")
class DetectSpheroids(Module):
    """Spheroids or organoids on brightfield (texture) or fluorescence (see module docs)."""

    name = "detect_spheroids"
    title = "Detect spheroids"
    Params = DetectSpheroidsParams

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.texture_um, metadata, "The texture window")
        if self.params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: DetectSpheroidsParams = self.params
        if image.data.shape[1] > 1:
            raise ModuleError("spheroid detection is 2D: add a Z projection first")
        pixel_area = (image.metadata.pixel_size_um or 1.0) ** 2
        labels = detect(
            image.data[params.channel - 1, 0],
            method=params.method,
            texture_px=um_to_px(params.texture_um, image.metadata, "The texture window"),
            min_area_px=params.min_area_um2 / pixel_area,
        )
        result = LabelSet(params.output, labels[None], source=f"spheroids ({params.method})")
        report = {"module": self.name, "objects": result.count}
        return image.with_labels(result, self.name, report)


# ----------------------------------------------------------- measurement


def spheroid_table(
    labels: NDArray[Any], image: NDArray[Any] | None, pixel_um: float, edge_px: float = 4.0
) -> pd.DataFrame:
    """Per object: size, shape, volume, darkness, edge sharpness and position (2D)."""
    from scipy import ndimage
    from skimage.measure import regionprops

    height, width = labels.shape
    smooth = (
        ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), 1.0)
        if image is not None
        else None
    )
    gradient = (
        (np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1)) / 8)
        if smooth is not None
        else None
    )
    everything = labels > 0
    rows = []
    for region in regionprops(labels):
        mask = labels == region.label
        area = region.num_pixels * pixel_um**2
        perimeter = float(region.perimeter) * pixel_um
        diameter = 2 * math.sqrt(area / math.pi)
        major, minor = region.axis_major_length * pixel_um, region.axis_minor_length * pixel_um
        ys, xs = np.nonzero(mask)
        cy, cx = region.centroid
        second = float(np.mean((ys - cy) ** 2 + (xs - cx) ** 2)) * pixel_um**2
        row: dict[str, Any] = {
            "object_id": int(region.label),
            "centroid_y_um": cy * pixel_um,
            "centroid_x_um": cx * pixel_um,
            "area_um2": area,
            "perimeter_um": perimeter,
            "equivalent_diameter_um": diameter,
            "major_axis_um": major,
            "minor_axis_um": minor,
            "volume_sphere_um3": math.pi / 6 * diameter**3,
            "volume_ellipsoid_um3": math.pi / 6 * major * minor**2,
            "circularity": 4 * math.pi * area / perimeter**2 if perimeter > 0 else math.nan,
            "solidity": float(region.solidity),
            "compactness": second / (area / (2 * math.pi)),
            "touching_edge": bool(
                ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1
            ),
            "offcentre": math.hypot(cy - (height - 1) / 2, cx - (width - 1) / 2)
            / (min(height, width) / 2),
            "pixel_um": pixel_um,
            "field_height_px": height,
            "field_width_px": width,
        }
        if smooth is not None and gradient is not None:
            radius = diameter / 2 / pixel_um
            ring = _ring(everything, edge_px, edge_px + max(3.0, radius / 2))
            core = ndimage.binary_erosion(mask, iterations=max(1, int(edge_px)))
            inside = float(np.median(smooth[core if core.any() else mask]))
            background = float(np.median(smooth[ring])) if ring.any() else math.nan
            boundary = mask & ~ndimage.binary_erosion(mask)
            steepest = float(np.percentile(gradient[ndimage.binary_dilation(boundary)], 75))
            contrast = abs(background - inside)
            row.update(
                mean_intensity=float(smooth[mask].mean()),
                background=background,
                darkness=1 - inside / background if background > 0 else math.nan,
                # the 1-px smoothing above is removed in quadrature
                edge_sigma_um=math.sqrt(max((contrast / (steepest * math.sqrt(2 * math.pi))) ** 2 - 1, 0))
                * pixel_um if steepest > 0 else math.nan,
            )  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


class MeasureSpheroidsParams(ModuleParams):
    labels: str = Field(default="spheroids", description="Label set of the spheroids.")
    channel: int | None = Field(
        default=None, ge=1, description="Brightfield/fluorescence channel for darkness and focus."
    )


@MODULES.register("measure_spheroids")
class MeasureSpheroids(MeasurementModule):
    """Size, shape, volume, darkness, focus and position of each spheroid (module docs)."""

    name = "measure_spheroids"
    title = "Measure spheroids"
    Params = MeasureSpheroidsParams

    def measure(self, context: MeasureContext) -> dict[str, Any]:
        params: MeasureSpheroidsParams = self.params
        labels = next((ls for ls in context.labels if ls.name == params.labels), None)
        if labels is None:
            raise ModuleError(f"No label set '{params.labels}': run detect_spheroids first")
        if labels.data.shape[0] > 1:
            raise ModuleError("spheroid measurement is 2D: add a Z projection first")
        pixel = context.corrected.metadata.pixel_size_um or 1.0
        image = context.corrected.data[params.channel - 1, 0] if params.channel else None
        table = spheroid_table(labels.data[0], image, pixel)
        if table.empty:  # an empty well still gets a (0-row) table: it is reported, as no_spheroid
            table = pd.DataFrame({"object_id": pd.Series(dtype=int),
                                  "area_um2": pd.Series(dtype=float)})  # fmt: skip
        table.insert(0, "object_type", params.labels)
        return {"objects": table}


# -------------------------------------------------------------- assay step


class SpheroidParams(TableParams):
    object_type: str = Field(default="spheroids", description="Label set of the spheroids.")
    brightfield_channel: str = Field(default="", description="Brightfield channel (preset).")
    fluorescence_channel: str = Field(default="", description="Fluorescence channel (preset).")
    volume: Literal["sphere", "ellipsoid"] = Field(
        default="sphere", description="Volume estimate reported as sp_volume_um3."
    )
    min_spheroid_um2: float = Field(
        default=2000.0, ge=0, description="Smaller objects are debris, never the spheroid."
    )
    multiple_fraction: float = Field(
        default=0.25, gt=0, le=1, description="Another object ≥ this × the spheroid: 'multiple'."
    )
    max_offcentre: float = Field(
        default=0.5, gt=0, description="Centroid distance from the field centre / half field."
    )
    max_edge_sigma_um: float = Field(default=4.0, gt=0, description="Blurrier: 'out_of_focus'.")
    debris_fraction: float = Field(
        default=0.02, ge=0, description="Other objects' area above this × spheroid: 'debris'."
    )
    min_solidity: float = Field(default=0.95, gt=0, le=1, description="Below: 'irregular'.")
    exclude_flags: list[str] = Field(
        default_factory=lambda: ["no_spheroid", "multiple", "touching_edge", "out_of_focus"],
        description="Flags that fail a well (sp_qc_ok = 0).",
    )


@MODULES.register("spheroid")
class Spheroid(AssayModule):
    """One spheroid per well: readouts, QC flags, gallery (module docs)."""

    name = "spheroid"
    title = "Assay: spheroids (one per well)"
    Params = SpheroidParams
    prefix: ClassVar[str] = "sp_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole(
            "brightfield", "Brightfield", "brightfield or digital phase contrast", required=False
        ),
        ChannelRole(
            "fluorescence",
            "Fluorescence",
            "a whole-spheroid stain (if no brightfield)",
            required=False,
        ),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}
        self.objects: list[pd.DataFrame] = []

    @property
    def p(self) -> SpheroidParams:
        params: SpheroidParams = self.params
        return params

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    def output_folder(self, folder: Path) -> Path:
        out = Path(folder) / "spheroids"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        return {"warnings": self.run_warnings()}

    def run_warnings(self) -> list[str]:
        steps = (self.manifest().get("pipeline") or {}).get("steps", [])
        if any(s.get("module") == "measure_objects" and not (s.get("params") or {}).get("labels")
               for s in steps):  # fmt: skip
            return ["measure_objects also measured the spheroids: their rows appear twice "
                    "(limit measure_objects to other label sets)"]  # fmt: skip
        return []

    def _main(self, rows: pd.DataFrame) -> Any:
        """Index of the well's spheroid: the largest object not touching the image border."""
        big = rows[rows["area_um2"] >= self.p.min_spheroid_um2]
        if big.empty:
            return None
        inside = big[~big["touching_edge"].astype(bool)]
        return (inside if not inside.empty else big)["area_um2"].idxmax()

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        rows = objects[self.selected(objects)]
        out = pd.DataFrame(index=objects.index)
        main = self._main(rows) if not rows.empty else None
        flag = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        flag.loc[rows.index] = rows.index == main
        out["sp_main"] = flag
        volume = f"volume_{self.p.volume}_um3"
        out["sp_volume_um3"] = np.nan
        if volume in rows:
            out.loc[rows.index, "sp_volume_um3"] = rows[volume].to_numpy(float)
        return out

    def flags(self, rows: pd.DataFrame, main: Any) -> list[str]:
        p = self.p
        if main is None:
            return ["no_spheroid"]
        spheroid = rows.loc[main]
        others = rows.drop(index=main)
        flags = []
        if (others["area_um2"] >= p.multiple_fraction * spheroid["area_um2"]).any():
            flags.append("multiple")
        if bool(spheroid["touching_edge"]):
            flags.append("touching_edge")
        if spheroid["offcentre"] > p.max_offcentre:
            flags.append("off_centre")
        if float(spheroid.get("edge_sigma_um", 0) or 0) > p.max_edge_sigma_um:
            flags.append("out_of_focus")
        small = others[others["area_um2"] < p.multiple_fraction * spheroid["area_um2"]]
        if small["area_um2"].sum() > p.debris_fraction * spheroid["area_um2"]:
            flags.append("debris")
        if spheroid["solidity"] < p.min_solidity:
            flags.append("irregular")
        return flags

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = objects[self.selected(objects)]
        main = self._main(rows) if not rows.empty else None
        flags = self.flags(rows, main)
        values: dict[str, Any] = {"sp_objects": len(rows), "sp_found": int(main is not None)}
        if main is not None:
            spheroid = rows.loc[main]
            for column in SPHEROID_COLUMNS:
                if column in spheroid:
                    values[f"sp_{column.replace('equivalent_', '')}"] = float(spheroid[column])
            values["sp_volume_um3"] = float(spheroid[f"volume_{self.p.volume}_um3"])
            values["sp_field"] = int(spheroid["field"])
            self.objects.append(rows.assign(sp_main=rows.index == main))
        values["sp_flags"] = ",".join(flags)
        values["sp_qc_ok"] = int(not set(flags) & set(self.p.exclude_flags))
        self.wells[well.well] = values
        return values, {"flags": flags}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        roles = self.well_roles()
        vehicle = [
            w
            for w, r in roles.items()
            if r in VEHICLE_ROLES and self.wells.get(w, {}).get("sp_qc_ok")
        ]
        positive = [
            w
            for w, r in roles.items()
            if r in POSITIVE_ROLES and self.wells.get(w, {}).get("sp_qc_ok")
        ]
        volumes = [
            self.wells[w]["sp_volume_um3"] for w in vehicle if "sp_volume_um3" in self.wells[w]
        ]
        reference = float(np.median(volumes)) if volumes else math.nan
        for well, values in self.wells.items():
            if "sp_volume_um3" in values and reference > 0:
                values["sp_volume_vs_vehicle_pct"] = 100 * values["sp_volume_um3"] / reference
                per_well.setdefault(well, {})["sp_volume_vs_vehicle_pct"] = values[
                    "sp_volume_vs_vehicle_pct"
                ]
        if len(vehicle) >= 2 and len(positive) >= 2:
            a = np.array([self.wells[w]["sp_volume_um3"] for w in vehicle])
            b = np.array([self.wells[w]["sp_volume_um3"] for w in positive])
            spread = abs(float(a.mean() - b.mean()))
            state["z_prime_volume"] = (
                float(1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / spread) if spread else math.nan
            )
        state["flag_counts"] = {
            f: sum(f in v["sp_flags"].split(",") for v in self.wells.values()) for f in FLAGS
        }
        out = self.output_folder(folder)
        (out / "calibration.json").write_text(
            json.dumps({"module": self.name, "params": self.p.model_dump(mode="json"), **state},
                       indent=1, default=str), "utf-8")  # fmt: skip
        if self.objects:
            pd.concat(self.objects, ignore_index=True).to_csv(out / "spheroids.csv", index=False)
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "wells.csv", index=False)
        crops = write_crops(folder, out, self.objects, self.p.object_type)
        write_gallery(out, table, crops, roles, state)


def write_crops(run: Path, out: Path, objects: list[pd.DataFrame], labels: str) -> dict[str, str]:
    """A crop of each well's spheroid from the run's preview, with its outline."""
    from PIL import Image

    (out / "crops").mkdir(exist_ok=True)
    crops = {}
    for rows in objects:
        spheroid = rows[rows["sp_main"]].iloc[0]
        well, field = str(spheroid["well"]), int(spheroid["field"])
        previews = run / "wells" / well / "previews"
        preview_path = previews / f"f{field:03d}.png"
        if not preview_path.exists():
            continue
        with Image.open(preview_path) as image:
            preview = image.convert("RGB")
        overlay_path = previews / f"f{field:03d}_outline_{labels}.png"
        if overlay_path.exists():
            with Image.open(overlay_path) as overlay:
                preview.paste(overlay.convert("RGBA"), (0, 0), overlay.convert("RGBA"))
        scale = preview.width / float(spheroid["field_width_px"]) / float(spheroid["pixel_um"])
        cy = float(spheroid["centroid_y_um"]) * scale
        cx = float(spheroid["centroid_x_um"]) * scale
        half = max(float(spheroid["major_axis_um"]) / 2 * 1.4 * scale, 8.0)
        box = (max(0, int(cx - half)), max(0, int(cy - half)),
               min(preview.width, int(cx + half)), min(preview.height, int(cy + half)))  # fmt: skip
        crop = preview.crop(box)
        crop.thumbnail((240, 240))
        name = f"crops/{well}.png"
        crop.save(out / name, format="PNG")
        crops[well] = name
    return crops


def write_gallery(
    out: Path,
    table: pd.DataFrame,
    crops: dict[str, str],
    roles: dict[str, str],
    state: dict[str, Any],
) -> Path:
    """Cards with the crop and readouts; sortable by any readout (inline JavaScript)."""
    readouts = [c for c in table.columns if c.startswith("sp_") and c not in ("sp_flags", "sp_field")
                and pd.api.types.is_numeric_dtype(table[c])]  # fmt: skip
    cards = []
    for record in table.to_dict("records"):
        well = str(record["well"])
        data = " ".join(
            f'data-{c.replace("_", "-")}="{record[c]:.6g}"' for c in readouts
            if isinstance(record.get(c), int | float) and math.isfinite(float(record[c]))
        )  # fmt: skip
        flags = [f for f in str(record.get("sp_flags") or "").split(",") if f]
        image = (
            f'<img src="{html.escape(crops[well])}" alt="{well}">'
            if well in crops
            else '<div class="none">no image</div>'
        )
        volume = record.get("sp_volume_um3")
        caption = (
            f"{volume / 1e6:.3g} nL" if isinstance(volume, float) and math.isfinite(volume) else "—"
        )
        badges = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in flags)
        cards.append(f'<div class="card" data-well="{well}" {data}>{image}<div><b>{well}</b> '
                     f'<small>{html.escape(roles.get(well, ""))}</small><br>{caption} {badges}</div></div>')  # fmt: skip
    options = "".join(
        f'<option value="{c.replace("_", "-")}">{html.escape(c)}</option>' for c in readouts
    )
    z = state.get("z_prime_volume")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Spheroids</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}.grid{{display:flex;flex-wrap:wrap;gap:8px}}
.card{{width:170px;border:1px solid #ccc;border-radius:6px;padding:4px;font-size:12px}}.card img{{width:160px;height:160px;object-fit:contain;background:#111}}
.none{{width:160px;height:160px;background:#eee}}.flag{{background:#b00020;color:#fff;border-radius:3px;padding:0 3px;margin-right:2px;font-size:10px}}</style></head><body>
<h1>Spheroids</h1>
<p>{len(table)} wells; flags: {html.escape(", ".join(f"{k} {v}" for k, v in state.get("flag_counts", {}).items() if v) or "none")}.
{f"Z′ (volume, vehicle vs positive control) = {z:.2f}." if isinstance(z, float) and math.isfinite(z) else ""}</p>
<p>Sort by <select id="key"><option value="well">well</option>{options}</select>
<select id="order"><option value="1">ascending</option><option value="-1">descending</option></select></p>
<div class="grid" id="grid">{"".join(cards)}</div>
<script>
function sortCards(){{const key=document.getElementById('key').value,order=+document.getElementById('order').value;
const grid=document.getElementById('grid');const cards=[...grid.children];
cards.sort((a,b)=>{{const va=a.dataset[key.replace(/-([a-z0-9])/g,(m,c)=>c.toUpperCase())],vb=b.dataset[key.replace(/-([a-z0-9])/g,(m,c)=>c.toUpperCase())];
if(key==='well')return order*a.dataset.well.localeCompare(b.dataset.well);
const x=va===undefined?Infinity:+va,y=vb===undefined?Infinity:+vb;return order*(x-y);}});cards.forEach(c=>grid.appendChild(c));}}
document.getElementById('key').onchange=sortCards;document.getElementById('order').onchange=sortCards;
</script></body></html>"""
    target = out / "gallery.html"
    target.write_text(page, encoding="utf-8")
    return target


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    texture_um: float = 6.0,
    min_area_um2: float = 300.0,
) -> Any:
    """Detection on brightfield (texture) or fluorescence (Otsu), measurement, the spheroid step."""
    from hcs_studio.pipeline.schema import Pipeline

    brightfield = mapping.get("brightfield", "")
    name = brightfield or mapping.get("fluorescence", "") or "Brightfield"
    number = channels.index(name) + 1 if channels and name in channels else 1
    method = "brightfield" if brightfield or not mapping.get("fluorescence") else "bright"
    steps = [
        {
            "module": "detect_spheroids",
            "params": {
                "channel": number,
                "method": method,
                "texture_um": texture_um,
                "min_area_um2": min_area_um2,
            },
        },
        {"module": "measure_spheroids", "params": {"channel": number}},
        {
            "module": "spheroid",
            "params": {
                "brightfield_channel": brightfield,
                "fluorescence_channel": mapping.get("fluorescence", ""),
            },
        },
    ]
    return Pipeline.model_validate({"name": "spheroids", "steps": steps})
