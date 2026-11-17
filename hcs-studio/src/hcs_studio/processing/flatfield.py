"""Retrospective flat-field (shading) and dark-offset correction.

The illumination profile is estimated from many fields of the plate: each field is
divided by its own mean, the per-pixel median across fields removes the cells, and a
wide Gaussian keeps only the smooth shading. This is a median-based estimate in the
spirit of BaSiC, not the BaSiC low-rank algorithm itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field
from scipy import ndimage

from hcs_studio.processing.base import (
    MODULES,
    FitContext,
    ImageData,
    Module,
    ModuleError,
    ModuleParams,
    block_reduce_mean,
    resize_to,
)

_ESTIMATE_SIZE = 256


class FlatFieldParams(ModuleParams):
    dark_offset: float = Field(
        default=0.0, ge=0, description="Camera offset (counts) subtracted before correction."
    )
    max_fields: int = Field(
        default=200, ge=2, description="Maximum number of fields sampled across the plate."
    )
    min_fields: int = Field(
        default=8,
        ge=2,
        description="Fewer fields than this cannot separate shading from cells; the step fails.",
    )
    smoothing_fraction: float = Field(
        default=0.05,
        gt=0,
        le=0.5,
        description="Gaussian smoothing of the profile, as a fraction of the image width.",
    )


@MODULES.register("flatfield")
class FlatField(Module):
    name = "flatfield"
    title = "Flat-field correction"
    Params = FlatFieldParams
    needs_fit = True

    def fit(self, context: FitContext) -> dict[str, Any]:
        params: FlatFieldParams = self.params
        stack: list[np.ndarray[Any, Any]] = []
        for image in context.sample(params.max_fields):
            projected = image.data.mean(axis=1) - params.dark_offset  # (C, Y, X)
            small = np.stack([block_reduce_mean(p, _ESTIMATE_SIZE) for p in projected])
            means = small.mean(axis=(1, 2), keepdims=True)
            if np.all(means > 0):
                stack.append(small / means)
        if len(stack) < params.min_fields:
            raise ModuleError(
                f"Flat-field estimation needs at least {params.min_fields} usable fields; "
                f"found {len(stack)}"
            )
        median = np.median(np.stack(stack), axis=0)
        sigma = params.smoothing_fraction * max(median.shape[1:])
        flat = np.stack([ndimage.gaussian_filter(m, sigma, mode="nearest") for m in median])
        flat /= flat.mean(axis=(1, 2), keepdims=True)
        if not np.all(np.isfinite(flat)) or flat.min() <= 0:
            raise ModuleError("Flat-field estimation produced a non-positive profile")
        return {"flat": flat.astype(np.float32), "n_fields": len(stack)}

    def apply(self, image: ImageData, state: Any) -> ImageData:
        if state is None:
            raise ModuleError("Flat-field correction was not fitted")
        flat = state["flat"]
        if flat.shape[0] != image.data.shape[0]:
            raise ModuleError("Flat-field profile and image have different channel counts")
        out = image.data - np.float32(self.params.dark_offset)
        for c in range(out.shape[0]):
            out[c] /= resize_to(flat[c], out.shape[2:])
        return image.derive(out, self.name)

    def describe_state(self, state: Any) -> dict[str, Any]:
        flat = state["flat"]
        return {
            "n_fields": state["n_fields"],
            "profile_min": [float(f.min()) for f in flat],
            "profile_max": [float(f.max()) for f in flat],
        }
