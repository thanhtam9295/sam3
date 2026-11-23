"""Confluence: % of the field covered by cells (Phase 22a).

``confluence`` is a measurement step that writes ``confluence_pct`` (and ``confluence_area_um2``)
to the per-field table:

- ``texture`` (default; brightfield or phase contrast): cells are textured, the background is
  not. The texture is the local SD (Gaussian window of σ ``texture_um``) of the image minus its
  median filter (size ≈ 3σ): the median keeps steps, so cell *outlines* — far stronger than the
  texture inside cells — leave no wide halo, while fine texture remains. The log texture is
  thresholded at Otsu's level of the **pooled** values of fields sampled across the plate
  (``fit``; a single field may be all background or all cells, which its own histogram cannot
  split). The window still spreads the texture past the cell edge: where the window holds a
  fraction p of cell, the SD is √(n² + p·T²) (n: background noise, T: cell texture, the medians
  of the two Otsu classes), which reaches the level L at p = (L² − n²)/T², i.e. a distance
  σ·Φ⁻¹(1 − p) outside the edge; the mask is eroded by that distance. A plate whose
  calibration fields show no texture above the noise (T < 2n) is reported, and its level is
  set to 3n so that noise is not counted as cells;
- ``threshold`` (fluorescence: a cytoplasmic stain or a nuclear stain at low magnification):
  Otsu's level of the pooled intensities;
- ``labels``: the area fraction of a label set (e.g. Cellpose cells).

Specks smaller than ``min_area_um2`` and holes smaller than it are cleaned up. A fixed
``threshold`` skips the calibration (no halo correction then).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    FitContext,
    MeasureContext,
    MeasurementModule,
    ModuleError,
    ModuleParams,
    um_to_px,
)


def texture(plane: NDArray[Any], sigma_px: float) -> NDArray[np.float64]:
    """Local SD (Gaussian window σ) of the plane minus its median filter (size ≈ 3σ)."""
    from scipy import ndimage

    from hcs_studio.assays.spheroids import local_sd

    x = np.asarray(plane, dtype=np.float64)
    size = 2 * max(round(1.5 * sigma_px), 1) + 1
    residual = x - ndimage.median_filter(x, size=size, mode="reflect")
    return local_sd(residual, sigma_px)


def signal(plane: NDArray[Any], method: str, texture_px: float) -> NDArray[np.float64]:
    """The value thresholded for confluence: log texture or (smoothed) intensity."""
    if method == "texture":
        return np.log(texture(plane, texture_px) + 1e-6)
    from scipy import ndimage

    return np.asarray(ndimage.gaussian_filter(np.asarray(plane, dtype=np.float64), 1.0))


def texture_level(values: NDArray[Any]) -> dict[str, Any]:
    """Level, noise, texture and halo (in units of σ) from pooled log texture values."""
    from scipy.stats import norm
    from skimage.filters import threshold_otsu

    level = float(threshold_otsu(values))
    noise = float(np.exp(np.median(values[values <= level])))
    cells = float(np.exp(np.median(values[values > level])))
    state: dict[str, Any] = {"level": level, "noise": noise, "texture": cells}
    if cells < 2 * noise:
        state["level"] = float(np.log(3 * noise))
        state["warning"] = "the calibration fields show no texture above the noise (no cells?)"
        state["halo_sigma"] = 0.0
        return state
    fraction = (math.exp(2 * level) - noise**2) / cells**2
    state["halo_sigma"] = float(norm.isf(fraction)) if 0 < fraction < 0.5 else 0.0
    return state


def cover(mask: NDArray[np.bool_], min_px: float) -> NDArray[np.bool_]:
    """The mask without specks and holes smaller than ``min_px``."""
    from skimage.morphology import remove_small_holes, remove_small_objects

    size = max(int(min_px), 1)
    mask = remove_small_objects(mask, max_size=size)
    return np.asarray(remove_small_holes(mask, max_size=size), dtype=bool)


class ConfluenceParams(ModuleParams):
    channel: int = Field(default=1, ge=1, description="Brightfield/phase or stain channel.")
    method: Literal["texture", "threshold", "labels"] = Field(
        default="texture",
        description="texture: brightfield/phase; threshold: a fluorescent stain; labels: area "
        "of a label set.",
    )
    labels: str = Field(default="cells", description="labels: the label set.")
    texture_um: float = Field(default=1.0, gt=0, description="texture: σ of the window.")
    threshold: float | None = Field(default=None, description="Fixed level (no calibration).")
    min_area_um2: float = Field(default=20.0, ge=0, description="Smaller specks/holes removed.")
    calibration_fields: int = Field(default=24, ge=2, description="Fields sampled for the level.")


@MODULES.register("confluence")
class Confluence(MeasurementModule):
    """% of the field covered by cells, per field and time point (see module docs)."""

    name = "confluence"
    title = "Confluence"
    Params = ConfluenceParams

    @property
    def needs_fit(self) -> bool:  # type: ignore[override]
        return self.params.method != "labels" and self.params.threshold is None

    def validate(self, metadata: ImageMetadata) -> None:
        params: ConfluenceParams = self.params
        if params.method != "labels" and params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {params.channel} does not exist")
        if params.method == "texture":
            um_to_px(params.texture_um, metadata, "The texture window")

    def _plane(self, image: Any) -> NDArray[Any]:
        data: NDArray[Any] = image.data[self.params.channel - 1].max(axis=0)
        return data

    def describe_state(self, state: Any) -> dict[str, Any]:
        keys = ("level", "noise", "texture", "halo_sigma", "fields", "warning")
        return {k: state[k] for k in keys if k in state}

    def _texture_px(self, metadata: ImageMetadata) -> float:
        if self.params.method != "texture":
            return 1.0
        return um_to_px(self.params.texture_um, metadata, "The texture window")

    def fit(self, context: FitContext) -> dict[str, Any]:
        from skimage.filters import threshold_otsu

        params: ConfluenceParams = self.params
        pooled = []
        rng = np.random.default_rng(0)
        for image in context.sample(params.calibration_fields):
            values = signal(self._plane(image), params.method, self._texture_px(image.metadata))
            values = values.ravel()
            pooled.append(rng.choice(values, size=min(values.size, 20000), replace=False))
        if not pooled:
            raise ModuleError("No fields to calibrate the confluence threshold")
        values = np.concatenate(pooled)
        if params.method == "texture":
            state = texture_level(values)
        else:
            state = {"level": float(threshold_otsu(values))}
        return {**state, "fields": len(pooled), "method": params.method}

    def measure(self, context: MeasureContext) -> dict[str, Any]:
        params: ConfluenceParams = self.params
        metadata = context.corrected.metadata
        pixel_area = (metadata.pixel_size_um or 1.0) ** 2
        if params.method == "labels":
            labels = next((ls for ls in context.labels if ls.name == params.labels), None)
            if labels is None:
                raise ModuleError(f"No label set '{params.labels}' for confluence")
            mask = labels.data.max(axis=0) > 0
        else:
            texture_px = self._texture_px(metadata)
            state = context.state or {}
            level = params.threshold if params.threshold is not None else state["level"]
            values = signal(self._plane(context.corrected), params.method, texture_px)
            mask = values > level
            halo = float(state.get("halo_sigma", 0.0)) * texture_px
            if params.threshold is None and halo > 0:
                from scipy import ndimage

                mask = ndimage.distance_transform_edt(mask) > halo
            mask = cover(mask, params.min_area_um2 / pixel_area)
        return {"fields": pd.DataFrame([{
            "confluence_pct": 100 * float(mask.mean()),
            "confluence_area_um2": float(mask.sum()) * pixel_area,
        }])}  # fmt: skip
