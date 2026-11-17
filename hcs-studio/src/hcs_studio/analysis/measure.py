"""Measurement steps: object features, colocalization, field QC; and channel registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy import ndimage

from hcs_studio.analysis import coloc, features
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

# ------------------------------------------------------------------ objects


class MeasureObjectsParams(ModuleParams):
    labels: list[str] | None = Field(
        default=None, description="Label sets to measure; empty = all."
    )
    channels: list[int] | None = Field(
        default=None, description="Channels to measure; empty = all."
    )
    percentiles: list[float] = Field(default=[5, 25, 75, 95], description="Intensity percentiles.")
    radial_bins: int = Field(default=3, ge=1, le=10, description="Rings from edge to centre.")
    texture: bool = Field(default=True, description="Haralick texture features.")
    texture_scales_px: list[int] = Field(default=[1, 3], description="Haralick pixel distances.")
    granularity: bool = Field(default=True, description="Granularity spectrum.")
    granularity_max_radius_px: int = Field(
        default=6, ge=1, le=30, description="Largest opening radius."
    )
    zernike: bool = Field(default=False, description="Zernike shape moments (2D).")
    neighbor_radius_um: float = Field(
        default=20.0, gt=0, description="Radius for neighbour counts."
    )
    nucleus_labels: str | None = Field(
        default="nuclei",
        description="Label set used for nuclear/cytoplasmic ratios of its children.",
    )


@MODULES.register("measure_objects")
class MeasureObjects(MeasurementModule):
    """Morphology, intensity, texture, children and neighbours of every object."""

    name = "measure_objects"
    title = "Measure: objects"
    Params = MeasureObjectsParams

    def validate(self, metadata: ImageMetadata) -> None:
        if metadata.pixel_size_um is None:
            raise ModuleError(
                "Object measurements are in µm; set the pixel size in the Metadata panel"
            )
        bad = [c for c in self.params.channels or [] if not 1 <= c <= len(metadata.channels)]
        if bad:
            raise ModuleError(f"Channel(s) {bad} do not exist")

    def measure(self, context: MeasureContext) -> dict[str, Any]:
        params: MeasureObjectsParams = self.params
        if not context.labels:
            raise ModuleError("No segmentation to measure: add a segmentation step before this one")
        options = features.FeatureOptions(
            channels=[c - 1 for c in params.channels] if params.channels else None,
            percentiles=tuple(params.percentiles),
            radial_bins=params.radial_bins,
            texture=params.texture,
            texture_scales_px=tuple(params.texture_scales_px),
            granularity=params.granularity,
            granularity_max_radius_px=params.granularity_max_radius_px,
            zernike=params.zernike,
            neighbor_radius_um=params.neighbor_radius_um,
            nucleus_labels=params.nucleus_labels,
        )
        wanted = params.labels or [ls.name for ls in context.labels]
        sets = {ls.name: ls for ls in context.labels}
        missing = [name for name in wanted if name not in sets]
        if missing:
            raise ModuleError(f"No segmentation named {missing}; available: {', '.join(sets)}")
        image = context.corrected.data
        # Children first (spots before cells, cells before nuclei) so parents can summarise them.
        order = sorted(wanted, key=lambda n: -_depth(sets[n], sets))
        tables: dict[str, pd.DataFrame] = {}
        for name in order:
            tables[name] = features.measure_label_set(
                sets[name], image, context.corrected.metadata, options, context.labels, tables
            )
        frames = []
        for name in wanted:
            frame = tables[name].copy()
            frame.insert(0, "object_type", name)
            frames.append(frame)
        return {"objects": pd.concat(frames, ignore_index=True)}


def _depth(label_set: LabelSet, sets: dict[str, LabelSet]) -> int:
    depth, current = 0, label_set
    while current.parent is not None and current.parent in sets and depth < 10:
        depth += 1
        current = sets[current.parent]
    return depth


# ---------------------------------------------------------- colocalization


class ColocParams(ModuleParams):
    channel_a: int = Field(ge=1, description="First channel.")
    channel_b: int = Field(ge=1, description="Second channel.")
    scope: Literal["objects", "field"] = Field(
        default="objects",
        description="objects: each object of 'labels'; field: all of them together.",
    )
    labels: str | None = Field(
        default="cells",
        description="Label set giving the objects or the mask; empty = whole field.",
    )
    background: Literal["outside_objects", "percentile", "none"] = Field(
        default="outside_objects",
        description="Subtracted before Manders: median outside objects, 1st percentile, or none.",
    )
    thresholds: list[float] | None = Field(
        default=None,
        description="Manual thresholds [A, B] after background subtraction; empty = Costes.",
    )
    costes_below: Literal["and", "or"] = Field(
        default="and", description="Costes: pixels below both thresholds ('and') or either ('or')."
    )
    randomizations: int = Field(
        default=0, ge=0, le=1000, description="Costes randomization test; 0 = off."
    )
    block_size_um: float | None = Field(
        default=None,
        gt=0,
        description="Scramble block size; empty = lateral PSF FWHM from the metadata.",
    )
    saturation_warning_fraction: float = Field(
        default=0.001, ge=0, le=1, description="Warn when more raw pixels than this are saturated."
    )
    object_labels_a: str | None = Field(
        default=None, description="Object-based: fraction of these objects (e.g. spots) near ..."
    )
    object_labels_b: str | None = Field(default=None, description="... these objects.")
    object_distance_um: float = Field(
        default=0.0, ge=0, description="Object-based: 0 = overlapping."
    )

    @model_validator(mode="after")
    def _check(self) -> ColocParams:
        if self.channel_a == self.channel_b:
            raise ValueError("Choose two different channels")
        if self.thresholds is not None and len(self.thresholds) != 2:
            raise ValueError("thresholds needs two values [A, B]")
        if (self.object_labels_a is None) != (self.object_labels_b is None):
            raise ValueError(
                "Object-based colocalization needs both object_labels_a and object_labels_b"
            )
        return self


@MODULES.register("colocalization")
class Colocalization(MeasurementModule):
    name = "colocalization"
    title = "Measure: colocalization"
    Params = ColocParams

    def validate(self, metadata: ImageMetadata) -> None:
        params: ColocParams = self.params
        for channel in (params.channel_a, params.channel_b):
            if channel > len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")

    def _block(self, metadata: ImageMetadata, three_d: bool) -> tuple[int, ...]:
        params: ColocParams = self.params
        if params.block_size_um is not None:
            size_um = params.block_size_um
        else:
            emission = metadata.channels[params.channel_a - 1].emission_nm
            if emission is None or metadata.objective_na is None:
                raise ModuleError(
                    "The randomization test needs block_size_um, or the NA and emission "
                    "wavelength to derive it from the PSF"
                )
            size_um = 0.51 * emission / 1000 / metadata.objective_na
        side = max(2, round(um_to_px(size_um, metadata, "The block size")))
        if three_d:
            z_step = metadata.z_step_um or metadata.pixel_size_um or 1.0
            return (max(1, round(size_um * 2.5 / z_step)), side, side)
        return (side, side)

    def measure(self, context: MeasureContext) -> dict[str, Any]:
        params: ColocParams = self.params
        image = context.corrected
        metadata = image.metadata
        a_all = image.data[params.channel_a - 1].astype(np.float64)
        b_all = image.data[params.channel_b - 1].astype(np.float64)
        labels = _label_data(context, params.labels, a_all.shape)
        objects = labels > 0 if labels is not None else np.ones(a_all.shape, dtype=bool)

        common: list[str] = []
        if not image.linear:
            common.append("input is not linear (display-only steps applied)")
        for channel in (params.channel_a, params.channel_b):
            fraction = _saturated(context.raw, channel - 1)
            if fraction > params.saturation_warning_fraction:
                common.append(f"channel {channel}: {100 * fraction:.2f}% of pixels saturated")
        shift = coloc.channel_shift(a_all, b_all)
        if max(abs(s) for s in shift[-2:]) > 1.0:
            common.append(
                f"possible chromatic shift of {shift[-2]:+.1f}, {shift[-1]:+.1f} px (Y, X): "
                "check with a bead image and register the channels"
            )

        a = a_all - _background(a_all, objects, params.background, labels is not None)
        b = b_all - _background(b_all, objects, params.background, labels is not None)
        a, b = np.clip(a, 0, None), np.clip(b, 0, None)
        thresholds = (params.thresholds[0], params.thresholds[1]) if params.thresholds else None
        three_d = a.shape[0] > 1
        block = self._block(metadata, three_d) if params.randomizations else None
        data_a, data_b = (a, b) if three_d else (a[0], b[0])

        rows = []
        if params.scope == "field" or labels is None:
            mask = objects if three_d else objects[0]
            result = coloc.colocalize(
                data_a,
                data_b,
                mask,
                thresholds=thresholds,
                below=params.costes_below,
                randomizations=params.randomizations,
                block=block,
            )
            rows.append({"scope": "field", "object_id": 0, **result.row()})
        else:
            for region in ndimage.find_objects(labels):
                if region is None:
                    continue
                crop = labels[region]
                for object_id in np.unique(crop):
                    if object_id == 0:
                        continue
                    mask = crop == object_id
                    sa, sb = a[region], b[region]
                    if not three_d:
                        mask, sa, sb = mask[0], sa[0], sb[0]
                    result = coloc.colocalize(
                        sa,
                        sb,
                        mask,
                        thresholds=thresholds,
                        below=params.costes_below,
                        randomizations=params.randomizations,
                        block=block,
                    )
                    rows.append(
                        {"scope": params.labels, "object_id": int(object_id), **result.row()}
                    )
        table = pd.DataFrame(rows)
        table.insert(0, "channel_b", metadata.channels[params.channel_b - 1].name)
        table.insert(0, "channel_a", metadata.channels[params.channel_a - 1].name)
        if common:
            joined = "; ".join(common)
            table["warnings"] = [f"{joined}; {w}" if w else joined for w in table["warnings"]]
        if params.object_labels_a and params.object_labels_b:
            fraction, count = object_based(
                context, params.object_labels_a, params.object_labels_b, params.object_distance_um
            )
            table[f"fraction_{params.object_labels_a}_near_{params.object_labels_b}"] = fraction
            table[f"n_{params.object_labels_a}"] = count
        return {"coloc": table}


def _label_data(context: MeasureContext, name: str | None, shape: tuple[int, ...]) -> Any:
    if name is None:
        return None
    for label_set in context.labels:
        if label_set.name == name:
            return label_set.data
    available = ", ".join(ls.name for ls in context.labels) or "none"
    raise ModuleError(f"No segmentation named '{name}' (available: {available})")


def _background(channel: Any, objects: Any, method: str, has_objects: bool) -> float:
    if method == "none":
        return 0.0
    if method == "outside_objects" and has_objects and (~objects).any():
        return float(np.median(channel[~objects]))
    return float(np.percentile(channel, 1))


def _saturated(raw: ImageData, channel: int) -> float:
    bits = raw.metadata.bit_depth
    data = raw.data[channel]
    maximum = float(2**bits - 1) if bits else float(data.max())
    return float((data >= maximum).mean()) if maximum > 0 else 0.0


def object_based(
    context: MeasureContext, name_a: str, name_b: str, distance_um: float
) -> tuple[float, int]:
    """Fraction of objects in A whose centre lies within ``distance_um`` of an object in B."""
    a = _label_data(context, name_a, ())
    b = _label_data(context, name_b, ())
    metadata = context.corrected.metadata
    pixel = metadata.pixel_size_um or 1.0
    sampling = (
        (metadata.z_step_um or pixel, pixel, pixel) if a.shape[0] > 1 else (1.0, pixel, pixel)
    )
    distance = ndimage.distance_transform_edt(b == 0, sampling=sampling)
    ids = [i for i in np.unique(a) if i > 0]
    if not ids:
        return float("nan"), 0
    centres = ndimage.center_of_mass(np.ones_like(a), a, ids)
    near = 0
    for centre in centres:
        index = tuple(min(max(round(c), 0), n - 1) for c, n in zip(centre, a.shape, strict=True))
        near += distance[index] <= distance_um
    return near / len(ids), len(ids)


# ----------------------------------------------------------------------- QC


class QcParams(ModuleParams):
    focus_channel: int = Field(default=1, ge=1, description="Channel used for the focus score.")
    focus_fraction: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Out of focus when the score is below this × plate median.",
    )
    saturation_fraction: float = Field(
        default=0.001,
        ge=0,
        le=1,
        description="Saturated when more pixels than this are at maximum.",
    )
    low_count_fraction: float = Field(
        default=0.3, ge=0, le=1, description="Low-count well when below this × plate median count."
    )
    count_labels: str | None = Field(
        default=None, description="Label set for counts; empty = first."
    )
    exclude_out_of_focus: bool = Field(
        default=True, description="Leave out-of-focus fields out of counts."
    )
    exclude_edge_fields: bool = Field(
        default=False,
        description="Leave fields at the edge of the well out of counts (needs ≥ 9 fields).",
    )


def focus_score(plane: Any) -> float:
    """Normalised variance of the Laplacian: high for sharp images, independent of brightness."""
    plane = np.asarray(plane, dtype=np.float64)
    mean = plane.mean()
    if mean <= 0:
        return 0.0
    return float(ndimage.laplace(ndimage.gaussian_filter(plane, 1.0)).var() / mean**2)


@MODULES.register("qc")
class FieldQc(MeasurementModule):
    """Per-field focus, saturation and object counts; plate-level flags are set after the run."""

    name = "qc"
    title = "Measure: field QC and counts"
    Params = QcParams

    def validate(self, metadata: ImageMetadata) -> None:
        if self.params.focus_channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.focus_channel} does not exist")

    def measure(self, context: MeasureContext) -> dict[str, Any]:
        params: QcParams = self.params
        raw = context.raw
        planes = raw.data[params.focus_channel - 1]
        row: dict[str, Any] = {
            "focus_score": max(focus_score(p) for p in planes),
            "saturated_fraction": max(_saturated(raw, c) for c in range(raw.data.shape[0])),
        }
        pixel = raw.metadata.pixel_size_um
        if pixel is not None:
            row["area_mm2"] = raw.data.shape[-1] * raw.data.shape[-2] * pixel**2 / 1e6
        for label_set in context.labels:
            row[f"n_{label_set.name}"] = label_set.count
        row["count_labels"] = params.count_labels or (
            context.labels[0].name if context.labels else ""
        )
        return {"fields": pd.DataFrame([row])}


# ------------------------------------------------------------ registration


class RegisterParams(ModuleParams):
    shifts_um: dict[int, list[float]] = Field(
        default_factory=dict,
        description="Per channel number: [dy, dx] (or [dz, dy, dx]) in µm to apply.",
    )
    reference_channel: int = Field(
        default=1, ge=1, description="Channel the others are aligned to."
    )
    bead_image: str | None = Field(
        default=None,
        description="Image of multicolour beads to estimate the shifts from (overrides shifts_um).",
    )


@MODULES.register("register_channels")
class RegisterChannels(Module):
    """Correct chromatic shift by translating channels (linear; sub-pixel interpolation)."""

    name = "register_channels"
    title = "Register channels (chromatic shift)"
    Params = RegisterParams
    needs_fit = True

    def fit(self, context: Any) -> dict[int, tuple[float, ...]]:
        params: RegisterParams = self.params
        metadata = context.plate.metadata
        pixel = metadata.pixel_size_um or 1.0
        if params.bead_image:
            from hcs_studio.io.base import open_plate

            plate = open_plate(Path(params.bead_image))
            beads = np.asarray(
                next(iter(plate.wells.values())).fields[0].load()[0], dtype=np.float64
            )
            reference = beads[params.reference_channel - 1]
            return {
                c + 1: coloc.channel_shift(reference, beads[c])
                for c in range(beads.shape[0])
                if c != params.reference_channel - 1
            }
        shifts = {}
        for channel, values in params.shifts_um.items():
            if not 1 <= channel <= len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")
            scale = [pixel] * len(values)
            if len(values) == 3:
                scale[0] = metadata.z_step_um or pixel
            shifts[channel] = tuple(v / s for v, s in zip(values, scale, strict=True))
        return shifts

    def apply(self, image: ImageData, state: Any) -> ImageData:
        out = image.data.copy()
        for channel, shift in (state or {}).items():
            vector = tuple(shift) if len(shift) == 3 else (0.0, *shift)
            if image.data.shape[1] == 1:
                vector = (0.0, *vector[-2:])
            # The measured shift is where the channel is; move it back onto the reference.
            out[channel - 1] = ndimage.shift(
                image.data[channel - 1], [-v for v in vector], order=1, mode="nearest"
            )
        return image.derive(out, self.name)

    def describe_state(self, state: Any) -> dict[str, Any]:
        return {"shifts_px": {str(k): list(v) for k, v in (state or {}).items()}}


def coloc_inputs(
    image: ImageData,
    labels: tuple[LabelSet, ...],
    channel_a: int,
    channel_b: int,
    label_name: str | None,
    object_id: int,
    z: int,
    background: str = "outside_objects",
) -> tuple[Any, Any, Any]:
    """Background-subtracted 2D planes of two channels (0-based) and the mask, as the step uses.

    ``object_id`` 0 means all objects of ``label_name`` together (or the whole plane when
    ``label_name`` is None).
    """
    a = image.data[channel_a, z].astype(np.float64)
    b = image.data[channel_b, z].astype(np.float64)
    if label_name is None:
        objects = np.ones(a.shape, dtype=bool)
        mask = objects
        has_objects = False
    else:
        data = next((ls.data for ls in labels if ls.name == label_name), None)
        if data is None:
            raise ModuleError(f"No segmentation named '{label_name}'")
        objects = data[z] > 0
        mask = objects if object_id == 0 else data[z] == object_id
        has_objects = True
        if not mask.any():
            raise ModuleError(f"Object {object_id} is not on plane {z}")
    a = np.clip(a - _background(a, objects, background, has_objects), 0, None)
    b = np.clip(b - _background(b, objects, background, has_objects), 0, None)
    return a, b, mask


def colocalization_map(a: Any, b: Any, mask: Any) -> Any:
    """Normalised mean deviation product (nMDP): > 0 where both channels are above their means."""
    ma, mb = a[mask].mean(), b[mask].mean()
    product = (a - ma) * (b - mb)
    scale = (a[mask].max() - ma) * (b[mask].max() - mb)
    out = np.where(mask, product / scale if scale > 0 else 0.0, 0.0)
    return np.clip(out, -1, 1).astype(np.float32)
