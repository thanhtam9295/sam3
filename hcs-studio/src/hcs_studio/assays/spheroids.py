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
from dataclasses import dataclass
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
SPHEROID_COLUMNS = (
    "area_um2",
    "equivalent_diameter_um",
    "volume_sphere_um3",
    "volume_ellipsoid_um3",
    "circularity",
    "solidity",
    "compactness",
    "darkness",
    "edge_sigma_um",
    "offcentre",
)
EXTRA_COLUMNS = ("core_area_um2", "core_fraction", "core_diameter_um", "rim_um", "core_contrast",
                 "live_area_um2", "dead_area_um2", "dead_fraction", "live_fraction",
                 "invasion_core_area_um2", "invasion_area_um2", "invasion_fraction", "protrusions",
                 "protrusion_length_um", "max_invasion_um", "skeleton_length_um",
                 "skeleton_branches")  # fmt: skip  # fmt: skip


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
    reach_px: float = 0.0,
) -> NDArray[np.int32]:
    """2D spheroid labels, largest first (see the module docstring). ``reach_px``: how far
    from the textured body dark protrusions (invading cells) are followed."""
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
        # outline at half height between the interior and the local background; the interior
        # is the dark class of the region (the texture region of a small object is mostly
        # background around it)
        ring = _ring(component, 2 * texture_px, 5 * texture_px) & ~coarse
        if not ring.any():
            continue
        background = float(np.median(smooth[ring]))
        values = smooth[component]
        split = threshold_otsu(values) if np.ptp(values) > 0 else float(values[0])
        dark = values[values <= split]
        inside = float(np.median(dark)) if dark.size else float(np.median(values))
        if inside >= background:
            continue
        level = (inside + background) / 2
        roi = _ring(component, -1, 2 * texture_px + reach_px) | component
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
    protrusion_reach_um: float = Field(
        default=0.0, ge=0,
        description="Invasion assays: follow dark protrusions this far from the spheroid body.",
    )  # fmt: skip
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
            reach_px=params.protrusion_reach_um / math.sqrt(pixel_area),
        )
        result = LabelSet(params.output, labels[None], source=f"spheroids ({params.method})")
        report = {"module": self.name, "objects": result.count}
        return image.with_labels(result, self.name, report)


# ----------------------------------------------------------- measurement


def spheroid_table(
    labels: NDArray[Any],
    image: NDArray[Any] | None,
    pixel_um: float,
    edge_px: float = 4.0,
    planes: dict[str, NDArray[Any]] | None = None,
    options: ExtraOptions | None = None,
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
        if options is not None and (planes or options.invasion):
            row.update(spheroid_extras(mask, everything, planes or {}, pixel_um, options))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ExtraOptions:
    """What :func:`spheroid_extras` measures (lengths in pixels)."""

    shells: int = 10
    core_polarity: str = "dark"
    core_smooth_px: float = 3.0
    min_core_contrast: float = 0.15
    threshold_sd: float = 5.0
    live_threshold: float | None = None
    dead_threshold: float | None = None
    invasion: bool = False
    invasion_core_px: float = 8.0
    min_protrusion_px: float = 5.0


def radial_shells(mask: NDArray[np.bool_], shells: int) -> NDArray[np.int32]:
    """Shell index of each pixel of ``mask`` (0 = outer edge … shells − 1 = centre; −1 outside),
    from the distance to the edge relative to the deepest point."""
    from scipy import ndimage

    depth = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)
    top = float(depth.max()) if mask.any() else 1.0
    index = np.minimum((depth / max(top, 1e-9) * shells).astype(np.int32), shells - 1)
    return np.where(mask, index, -1).astype(np.int32)


def necrotic_core(
    plane: NDArray[Any], mask: NDArray[np.bool_], options: ExtraOptions
) -> tuple[NDArray[np.bool_], float]:
    """Core mask (at half height between the rim and the centre of ``plane``) and the
    rim–centre contrast; an empty core when the contrast is below ``min_core_contrast``."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(np.asarray(plane, dtype=np.float64), options.core_smooth_px)
    depth = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)
    relative = depth / max(float(depth.max()), 1e-9)
    rim_pixels = smooth[mask & (relative > 0.1) & (relative < 0.3)]
    centre_pixels = smooth[mask & (relative > 0.85)]
    if rim_pixels.size == 0 or centre_pixels.size == 0:
        return np.zeros_like(mask), math.nan
    rim, centre = float(np.median(rim_pixels)), float(np.median(centre_pixels))
    dark = options.core_polarity == "dark"
    contrast = (rim - centre if dark else centre - rim) / max(abs(rim), abs(centre), 1e-9)
    if contrast < options.min_core_contrast:
        return np.zeros_like(mask), contrast
    level = (rim + centre) / 2
    candidate = mask & ((smooth < level) if dark else (smooth > level))
    candidate = ndimage.binary_opening(ndimage.binary_fill_holes(candidate), iterations=1)
    parts, n = ndimage.label(candidate)
    if n == 0:
        return np.zeros_like(mask), contrast
    deepest = np.unravel_index(int(np.argmax(np.where(mask, depth, -1))), mask.shape)
    keep = parts[deepest] or int(np.argmax(np.bincount(parts.ravel())[1:])) + 1
    return np.asarray(ndimage.binary_fill_holes(parts == keep), dtype=bool), contrast


def stain_threshold(plane: NDArray[Any], everything: NDArray[np.bool_], k: float) -> float:
    """Background (outside all objects) median + k robust SD."""
    outside = np.asarray(plane, dtype=np.float64)[~everything]
    if outside.size == 0:
        return math.inf
    median = float(np.median(outside))
    return median + k * 1.4826 * float(np.median(np.abs(outside - median)))


def invasion_measures(mask: NDArray[np.bool_], options: ExtraOptions) -> dict[str, Any]:
    """Body (morphological opening by ``invasion_core_px``, done with distance transforms) vs
    protrusions: area, count and reach of protrusions, skeleton length and branches (pixels)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    r = options.invasion_core_px
    eroded = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64) > r
    body = (np.asarray(ndimage.distance_transform_edt(~eroded), dtype=np.float64) <= r) & mask
    parts, n = ndimage.label(body)
    if n > 1:
        body = parts == int(np.argmax(np.bincount(parts.ravel())[1:])) + 1
    if not body.any():
        body = mask.copy()
    reach = np.asarray(ndimage.distance_transform_edt(~body), dtype=np.float64)
    pieces, k = ndimage.label(mask & ~body)
    kept = np.zeros_like(mask)
    lengths = []
    for index in range(1, k + 1):
        piece = pieces == index
        length = float(reach[piece].max())
        if length >= options.min_protrusion_px:
            kept |= piece
            lengths.append(length)
    skeleton = skeletonize(kept) if kept.any() else kept
    neighbours = ndimage.convolve(skeleton.astype(int), np.ones((3, 3), int), mode="constant")
    ends = int((skeleton & (neighbours == 2)).sum())
    return {
        "invasion_core_px2": float(body.sum()),
        "invasion_px2": float(kept.sum()),
        "protrusions": len(lengths),
        "protrusion_length_px": float(np.mean(lengths)) if lengths else 0.0,
        "max_invasion_px": float(max(lengths, default=0.0)),
        "skeleton_length_px": float(skeleton.sum()),
        "skeleton_branches": ends,
    }


def spheroid_extras(
    mask: NDArray[np.bool_],
    everything: NDArray[np.bool_],
    planes: dict[str, NDArray[Any]],
    pixel_um: float,
    options: ExtraOptions,
) -> dict[str, Any]:
    """Radial profiles, necrotic core, live/dead areas and invasion of one spheroid."""
    row: dict[str, Any] = {}
    area_px = float(mask.sum())
    shells = radial_shells(mask, options.shells)
    for name, plane in planes.items():
        values = np.asarray(plane, dtype=np.float64)
        means = np.bincount(shells[mask], weights=values[mask], minlength=options.shells)
        counts = np.bincount(shells[mask], minlength=options.shells)
        for s in range(options.shells):
            row[f"radial_{name}_s{s + 1:02d}"] = (
                float(means[s] / counts[s]) if counts[s] else math.nan
            )
    if "core" in planes:
        core, contrast = necrotic_core(planes["core"], mask, options)
        core_px = float(core.sum())
        row["core_contrast"] = contrast
        row["core_area_um2"] = core_px * pixel_um**2
        row["core_fraction"] = core_px / area_px if area_px else math.nan
        row["core_diameter_um"] = 2 * math.sqrt(core_px / math.pi) * pixel_um
        row["rim_um"] = (math.sqrt(area_px / math.pi) - math.sqrt(core_px / math.pi)) * pixel_um
    stained = {}
    for name in ("live", "dead"):
        if name in planes:
            fixed = options.live_threshold if name == "live" else options.dead_threshold
            level = (
                fixed
                if fixed is not None
                else stain_threshold(planes[name], everything, options.threshold_sd)
            )
            stained[name] = mask & (np.asarray(planes[name]) > level)
            row[f"{name}_area_um2"] = float(stained[name].sum()) * pixel_um**2
            row[f"{name}_threshold"] = float(level)
    if stained:
        union = np.zeros_like(mask)
        for positive in stained.values():
            union |= positive
        total = float(union.sum())
        if "dead" in stained:
            row["dead_fraction"] = float(stained["dead"].sum()) / total if total else math.nan
        if "live" in stained:
            live_only = stained["live"] & ~stained.get("dead", np.zeros_like(mask))
            row["live_fraction"] = float(live_only.sum()) / total if total else math.nan
    if options.invasion:
        measured = invasion_measures(mask, options)
        row.update({
            "invasion_core_area_um2": measured["invasion_core_px2"] * pixel_um**2,
            "invasion_area_um2": measured["invasion_px2"] * pixel_um**2,
            "invasion_fraction": measured["invasion_px2"] / area_px if area_px else math.nan,
            "protrusions": measured["protrusions"],
            "protrusion_length_um": measured["protrusion_length_px"] * pixel_um,
            "max_invasion_um": measured["max_invasion_px"] * pixel_um,
            "skeleton_length_um": measured["skeleton_length_px"] * pixel_um,
            "skeleton_branches": measured["skeleton_branches"],
        })  # fmt: skip
    return row


class MeasureSpheroidsParams(ModuleParams):
    labels: str = Field(default="spheroids", description="Label set of the spheroids.")
    channel: int | None = Field(
        default=None, ge=1, description="Brightfield/fluorescence channel for darkness and focus."
    )
    core: bool = Field(default=True, description="Measure the necrotic core.")
    core_channel: int | None = Field(
        default=None, ge=1,
        description="Channel for the core: empty = the dead stain if any, else `channel`.",
    )  # fmt: skip
    core_polarity: Literal["auto", "dark", "bright"] = Field(
        default="auto", description="auto: bright for a dead stain, dark otherwise (brightfield)."
    )
    core_smooth_um: float = Field(
        default=6.0, gt=0, description="Smoothing before the core outline."
    )
    min_core_contrast: float = Field(
        default=0.15, ge=0, description="Rim–centre contrast below this: no core."
    )
    shells: int = Field(default=10, ge=2, le=50, description="Radial profile shells.")
    live_channel: int | None = Field(default=None, ge=1, description="Live stain (calcein).")
    dead_channel: int | None = Field(default=None, ge=1, description="Dead stain (EthD-1, PI).")
    threshold_sd: float = Field(
        default=5.0, gt=0, description="Stain positive above background median + this × robust SD."
    )
    live_threshold: float | None = Field(default=None, description="Fixed live threshold.")
    dead_threshold: float | None = Field(default=None, description="Fixed dead threshold.")
    invasion: bool = Field(default=False, description="Measure invasion (protrusions).")
    invasion_core_um: float = Field(
        default=20.0, gt=0, description="Opening radius: thinner structures are protrusions."
    )
    min_protrusion_um: float = Field(default=10.0, ge=0, description="Shorter: not a protrusion.")


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
        data = context.corrected.data
        for number in (
            params.channel,
            params.core_channel,
            params.live_channel,
            params.dead_channel,
        ):
            if number is not None and number > data.shape[0]:
                raise ModuleError(f"Channel {number} does not exist")
        image = data[params.channel - 1, 0] if params.channel else None
        planes: dict[str, NDArray[Any]] = {}
        if params.live_channel:
            planes["live"] = data[params.live_channel - 1, 0]
        if params.dead_channel:
            planes["dead"] = data[params.dead_channel - 1, 0]
        polarity = params.core_polarity
        if params.core:
            core = params.core_channel or params.dead_channel or params.channel
            if core is not None:
                planes["core"] = data[core - 1, 0]
                if polarity == "auto":
                    polarity = "bright" if core == params.dead_channel else "dark"
        options = ExtraOptions(
            shells=params.shells,
            core_polarity=polarity,
            core_smooth_px=params.core_smooth_um / pixel,
            min_core_contrast=params.min_core_contrast,
            threshold_sd=params.threshold_sd,
            live_threshold=params.live_threshold,
            dead_threshold=params.dead_threshold,
            invasion=params.invasion,
            invasion_core_px=params.invasion_core_um / pixel,
            min_protrusion_px=params.min_protrusion_um / pixel,
        )
        table = spheroid_table(labels.data[0], image, pixel, planes=planes, options=options)
        if table.empty:  # an empty well still gets a (0-row) table: it is reported, as no_spheroid
            table = pd.DataFrame({"object_id": pd.Series(dtype=int),
                                  "area_um2": pd.Series(dtype=float)})  # fmt: skip
        table.insert(0, "object_type", params.labels)
        return {"objects": table}


# -------------------------------------------------------------- assay step


class SpheroidParams(TableParams):
    object_type: str = Field(default="spheroids", description="Label set of the spheroids.")
    mode: Literal["single", "organoids"] = Field(
        default="single",
        description="single: one spheroid per well (ULA plates); organoids: many per well.",
    )
    min_organoid_um2: float = Field(default=200.0, ge=0, description="Organoid mode: smallest.")
    well_area_mm2: float | None = Field(
        default=None, gt=0,
        description="Organoid mode: well area, to extrapolate counts from the imaged fields "
        "(empty = the fields cover the whole well).",
    )  # fmt: skip
    brightfield_channel: str = Field(default="", description="Brightfield channel (preset).")
    fluorescence_channel: str = Field(default="", description="Fluorescence channel (preset).")
    live_channel: str = Field(default="", description="Live stain channel (preset).")
    dead_channel: str = Field(default="", description="Dead stain channel (preset).")
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
    title = "Assay: spheroids and organoids"
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
        ChannelRole("live", "Live", "calcein-AM or another live stain", required=False),
        ChannelRole("dead", "Dead", "EthD-1, PI or another dead stain", required=False),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}
        self.objects: list[pd.DataFrame] = []
        self.diameters: dict[str, list[float]] = {}

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
        flag = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        if self.p.mode == "organoids":
            flag.loc[rows.index] = (rows["area_um2"] >= self.p.min_organoid_um2).to_numpy(bool)
        else:
            main = self._main(rows) if not rows.empty else None
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
        if self.p.mode == "organoids":
            return self.organoids(rows, well)
        main = self._main(rows) if not rows.empty else None
        flags = self.flags(rows, main)
        values: dict[str, Any] = {"sp_objects": len(rows), "sp_found": int(main is not None)}
        if main is not None:
            spheroid = rows.loc[main]
            for column in SPHEROID_COLUMNS + EXTRA_COLUMNS:
                if column in spheroid and pd.notna(spheroid[column]):
                    values[f"sp_{column.replace('equivalent_', '')}"] = float(spheroid[column])
            values["sp_volume_um3"] = float(spheroid[f"volume_{self.p.volume}_um3"])
            values["sp_field"] = int(spheroid["field"])
            self.objects.append(rows.assign(sp_main=rows.index == main))
        values["sp_flags"] = ",".join(flags)
        values["sp_qc_ok"] = int(not set(flags) & set(self.p.exclude_flags))
        self.wells[well.well] = values
        return values, {"flags": flags}

    def organoids(
        self, rows: pd.DataFrame, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Organoid mode: count, size distribution, density, forming efficiency, viability."""
        valid = rows[rows["area_um2"] >= self.p.min_organoid_um2] if not rows.empty else rows
        complete = valid[~valid["touching_edge"].astype(bool)] if not valid.empty else valid
        values: dict[str, Any] = {"sp_objects": len(rows), "sp_org_count": len(valid),
                                  "sp_org_count_complete": len(complete)}  # fmt: skip
        diameters = (
            complete["equivalent_diameter_um"].to_numpy(float)
            if not complete.empty
            else np.array([])
        )
        self.diameters[well.well] = diameters.tolist()
        if diameters.size:
            values.update({
                "sp_org_diameter_mean_um": float(diameters.mean()),
                "sp_org_diameter_median_um": float(np.median(diameters)),
                "sp_org_diameter_p10_um": float(np.percentile(diameters, 10)),
                "sp_org_diameter_p90_um": float(np.percentile(diameters, 90)),
            })  # fmt: skip
        if not valid.empty:
            values["sp_org_area_total_um2"] = float(valid["area_um2"].sum())
            values["sp_org_volume_total_um3"] = float(valid[f"volume_{self.p.volume}_um3"].sum())
            for column in ("dead_fraction", "core_fraction"):
                if column in valid and valid[column].notna().any():
                    known = valid[valid[column].notna()]
                    values[f"sp_{column}"] = float(
                        (known[column] * known["area_um2"]).sum() / known["area_um2"].sum()
                    )
            self.objects.append(rows.assign(sp_main=rows.index.isin(valid.index)))
        imaged = self._imaged_mm2(well, rows)
        if imaged:
            values["sp_org_per_mm2"] = len(valid) / imaged
            seeded = self._seeding(well.well)
            if seeded:
                estimate = len(valid) * (
                    self.p.well_area_mm2 / imaged if self.p.well_area_mm2 else 1.0
                )
                values["sp_ofe_pct"] = 100 * estimate / seeded
        flags = [] if len(valid) else ["no_organoids"]
        values["sp_flags"] = ",".join(flags)
        values["sp_qc_ok"] = int(not flags)
        self.wells[well.well] = values
        return values, {"flags": flags}

    def _imaged_mm2(self, well: WellContext, rows: pd.DataFrame) -> float:
        try:
            summary = json.loads((well.folder / "well_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
        fields = len(summary.get("fields") or []) or (
            rows["field"].nunique() if not rows.empty else 0
        )
        if rows.empty or not fields:
            if well.field_size_um and fields:
                return fields * well.field_size_um[0] * well.field_size_um[1] / 1e6
            return 0.0
        first = rows.iloc[0]
        area = (
            float(first["field_height_px"])
            * float(first["field_width_px"])
            * float(first["pixel_um"]) ** 2
        )
        return fields * area / 1e6

    def _seeding(self, well: str) -> float:
        layout = self.layout()
        if layout is None:
            return 0.0
        from hcs_studio.model.plate import parse_well_name

        annotation = layout.wells.get(parse_well_name(well))
        return float(annotation.seeding_density or 0) if annotation is not None else 0.0

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
        key = "sp_volume_um3" if self.p.mode == "single" else "sp_org_volume_total_um3"
        volumes = [self.wells[w][key] for w in vehicle if key in self.wells[w]]
        reference = float(np.median(volumes)) if volumes else math.nan
        for well, values in self.wells.items():
            if key in values and reference > 0:
                values["sp_volume_vs_vehicle_pct"] = 100 * values[key] / reference
                per_well.setdefault(well, {})["sp_volume_vs_vehicle_pct"] = values[
                    "sp_volume_vs_vehicle_pct"
                ]
        if len(vehicle) >= 2 and len(positive) >= 2:
            a = np.array([self.wells[w].get(key, math.nan) for w in vehicle])
            b = np.array([self.wells[w].get(key, math.nan) for w in positive])
            spread = abs(float(a.mean() - b.mean()))
            state["z_prime_volume"] = (
                float(1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / spread) if spread else math.nan
            )
        state["flag_counts"] = {
            f: sum(f in v["sp_flags"].split(",") for v in self.wells.values())
            for f in (*FLAGS, "no_organoids")
        }
        out = self.output_folder(folder)
        (out / "calibration.json").write_text(
            json.dumps({"module": self.name, "params": self.p.model_dump(mode="json"), **state},
                       indent=1, default=str), "utf-8")  # fmt: skip
        figures = []
        if self.objects:
            everything = pd.concat(self.objects, ignore_index=True)
            everything.to_csv(out / "spheroids.csv", index=False)
            figures += write_profiles(out, everything[everything["sp_main"].astype(bool)], roles)
        if self.p.mode == "organoids" and self.diameters:
            pd.DataFrame([{"well": w, "diameter_um": d} for w, ds in sorted(self.diameters.items())
                          for d in ds]).to_csv(out / "organoid_sizes.csv", index=False)  # fmt: skip
            figures += write_size_distribution(out, self.diameters)
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "wells.csv", index=False)
        crops = write_crops(
            folder, out, self.objects, self.p.object_type, whole_field=self.p.mode == "organoids"
        )
        write_gallery(out, table, crops, roles, state, figures)


def write_crops(
    run: Path, out: Path, objects: list[pd.DataFrame], labels: str, whole_field: bool = False
) -> dict[str, str]:
    """A crop of each well's spheroid from the run's preview, with its outline (organoids: the
    field with the largest organoid, whole)."""
    from PIL import Image

    (out / "crops").mkdir(exist_ok=True)
    crops = {}
    for rows in objects:
        main = rows[rows["sp_main"].astype(bool)]
        if main.empty:
            continue
        spheroid = main.iloc[int(np.argmax(main["area_um2"].to_numpy(float)))]
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
        if whole_field:
            cy, cx, half = preview.height / 2, preview.width / 2, max(preview.size) / 2
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
    figures: list[str] | None = None,
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
        count = record.get("sp_org_count")
        if isinstance(count, int | float) and math.isfinite(float(count)):
            caption = f"{int(count)} organoids"
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
{"".join(f'<p><img src="{html.escape(f)}" alt="{html.escape(f)}" style="max-width:100%"></p>' for f in figures or [])}
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


def write_profiles(out: Path, spheroids: pd.DataFrame, roles: dict[str, str]) -> list[str]:
    """Radial profiles (long table) and a figure: mean profile per role, per channel."""
    columns = [c for c in spheroids.columns if c.startswith("radial_")]
    if not columns or spheroids.empty:
        return []
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    long = spheroids.melt(id_vars=["well", "field", "object_id"], value_vars=columns,
                          var_name="column", value_name="shell_mean")  # fmt: skip
    parts = long["column"].str.extract(r"radial_(?P<channel>.+)_s(?P<shell>\d+)$")
    long = pd.concat([long.drop(columns="column"), parts], axis=1)
    long["shell"] = long["shell"].astype(int)
    shells = int(long["shell"].max())
    long["depth"] = (long["shell"] - 0.5) / shells  # 0 = edge, 1 = centre
    long.to_csv(out / "radial_profiles.csv", index=False)
    channels = sorted(long["channel"].unique())
    fig = Figure(figsize=(3.2 * len(channels), 2.8), dpi=100)
    FigureCanvasAgg(fig)
    long["role"] = [roles.get(str(w), "sample") for w in long["well"]]
    for i, channel in enumerate(channels, start=1):
        ax = fig.add_subplot(1, len(channels), i)
        for role, group in long[long["channel"] == channel].groupby("role"):
            mean = group.groupby("depth")["shell_mean"].mean()
            ax.plot(mean.index, mean.to_numpy(), marker=".", label=str(role))
        ax.set_title(channel, fontsize=9)
        ax.set_xlabel("edge → centre", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "radial_profiles.png", metadata={"Software": None})
    return ["radial_profiles.png"]


def write_size_distribution(out: Path, diameters: dict[str, list[float]]) -> list[str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    wells = [w for w in sorted(diameters) if diameters[w]]
    if not wells:
        return []
    fig = Figure(figsize=(max(4.0, 0.25 * len(wells)), 3.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot([diameters[w] for w in wells], showfliers=False)
    ax.set_xticks(range(1, len(wells) + 1), wells, rotation=90, fontsize=6)
    ax.set_ylabel("organoid diameter (µm)")
    fig.tight_layout()
    fig.savefig(out / "size_distribution.png", metadata={"Software": None})
    return ["size_distribution.png"]


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    texture_um: float = 6.0,
    min_area_um2: float = 300.0,
    invasion: bool = False,
    protrusion_reach_um: float = 300.0,
    organoids: bool = False,
) -> Any:
    """Detection on brightfield (texture) or fluorescence (Otsu), measurement (with live/dead
    stains, core and invasion when asked), the spheroid step (single or organoid mode)."""
    from hcs_studio.pipeline.schema import Pipeline

    def number(name: str) -> int | None:
        if not name:
            return None
        return channels.index(name) + 1 if channels and name in channels else 1

    brightfield = mapping.get("brightfield", "")
    name = brightfield or mapping.get("fluorescence", "") or "Brightfield"
    detection = number(name) or 1
    method = "brightfield" if brightfield or not mapping.get("fluorescence") else "bright"
    measure: dict[str, Any] = {"channel": detection}
    for role in ("live", "dead"):
        if mapping.get(role):
            measure[f"{role}_channel"] = number(mapping[role])
    if invasion:
        measure["invasion"] = True
    detect: dict[str, Any] = {"channel": detection, "method": method, "texture_um": texture_um,
                              "min_area_um2": min_area_um2}  # fmt: skip
    if invasion:
        detect["protrusion_reach_um"] = protrusion_reach_um
    step: dict[str, Any] = {f"{role}_channel": mapping.get(role, "")
                            for role in ("brightfield", "fluorescence", "live", "dead")}  # fmt: skip
    if organoids:
        step["mode"] = "organoids"
    steps = [
        {"module": "detect_spheroids", "params": detect},
        {"module": "measure_spheroids", "params": measure},
        {"module": "spheroid", "params": step},
    ]
    return Pipeline.model_validate({"name": "spheroids", "steps": steps})
