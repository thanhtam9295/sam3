"""Depth-dependent intensity loss in confocal stacks (Phase 21c).

Light is scattered and absorbed on its way into thick samples (spheroids, organoids, tissue), so
deeper planes are dimmer. The step measures the signal of each plane — the ``percentile`` (90th)
of the foreground pixels (above Otsu's threshold of the stack), which follows the brightest,
uniformly stained structures rather than the plane's object area — minus the plane's
background (median of the pixels below the threshold; an offset would flatten the decay) — over
the planes where the
foreground covers at least ``min_area_fraction`` of the largest foreground area (the caps of a
spheroid, mostly blur, are left out), then:

- ``exponential`` (default): fits I(z) = I₀ · exp(−(z − z₀)/λ) (robust Theil–Sen line through
  log I) and multiplies plane z by exp((z − z₀)/λ); a smooth correction that does not follow
  plane-to-plane noise or real biological differences between planes;
- ``per_plane``: divides each plane by its own signal relative to the reference plane z₀ (the
  first plane used), interpolated between the planes used.

Only the signal above each plane's background is scaled. Gains are capped at ``max_gain``
(noise is amplified by the same factor); a field whose correction needs more is reported with a
warning, as is a stack that gets *brighter* with depth (no correction then). The correction is
a known factor per plane: intensities stay linear.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ChannelParams,
    ImageData,
    Module,
    ModuleError,
)


def plane_signal(
    stack: NDArray[Any], percentile: float, min_area_fraction: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Per plane: the foreground ``percentile`` above the plane's background, whether the plane
    is used, and the background (median of the pixels below the threshold)."""
    from skimage.filters import threshold_otsu

    data = np.asarray(stack, dtype=np.float64)
    level = float(threshold_otsu(data)) if np.ptp(data) > 0 else math.inf
    foreground = data > level
    areas = foreground.reshape(len(data), -1).sum(axis=1)
    used = areas >= max(min_area_fraction * float(np.max(areas)), 1)
    signal = np.full(len(data), np.nan)
    background = np.array([float(np.median(plane[~mask])) if (~mask).any() else 0.0
                           for plane, mask in zip(data, foreground, strict=True)])  # fmt: skip
    for z in np.nonzero(used)[0]:
        signal[z] = float(np.percentile(data[z][foreground[z]], percentile)) - background[z]
    return signal, used, background


def correction_factors(
    signal: NDArray[Any], used: NDArray[np.bool_], method: str, z_step: float
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Multiplicative factor per plane and a description (λ in µm for ``exponential``)."""
    from scipy.stats import theilslopes

    n = len(signal)
    planes = np.nonzero(used & np.isfinite(signal) & (signal > 0))[0]
    factors = np.ones(n)
    if len(planes) < 3:
        return factors, {"planes_used": len(planes), "note": "too few planes with signal"}
    reference = int(planes[0])
    if method == "per_plane":
        relative = signal[planes] / signal[reference]
        profile = np.interp(np.arange(n), planes, relative)
        factors = 1 / np.clip(profile, 1e-6, None)
        factors[:reference] = 1.0
        return factors, {"planes_used": len(planes), "reference_plane": reference + 1}
    slope, _intercept, *_ = theilslopes(np.log(signal[planes]), planes * z_step)
    info: dict[str, Any] = {"planes_used": len(planes), "reference_plane": reference + 1}
    if slope >= 0:
        info["attenuation_length_um"] = math.inf
        info["note"] = "the signal does not fall with depth: not corrected"
        return factors, info
    length = -1 / slope
    depth = (np.arange(n) - reference) * z_step
    factors = np.where(depth > 0, np.exp(depth / length), 1.0)
    info["attenuation_length_um"] = float(length)
    return factors, info


class AttenuationParams(ChannelParams):
    method: Literal["exponential", "per_plane"] = Field(
        default="exponential",
        description="exponential: fit I₀·exp(−z/λ) and invert it; per_plane: normalise each "
        "plane to the first plane with signal.",
    )
    percentile: float = Field(
        default=90.0, gt=0, lt=100, description="Foreground percentile measured per plane."
    )
    min_area_fraction: float = Field(
        default=0.5, gt=0, le=1,
        description="Planes whose foreground is smaller than this × the largest are not used.",
    )  # fmt: skip
    max_gain: float = Field(default=10.0, gt=1, description="Largest correction factor.")


@MODULES.register("attenuation")
class Attenuation(Module):
    """Depth-dependent intensity loss in Z stacks, corrected per plane (see module docs)."""

    name = "attenuation"
    title = "Depth attenuation correction"
    Params = AttenuationParams

    def validate(self, metadata: ImageMetadata) -> None:
        params: AttenuationParams = self.params
        params.selected(len(metadata.channels))
        if params.method == "exponential" and metadata.z_step_um is None:
            raise ModuleError("The exponential correction needs the Z step (Metadata panel)")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: AttenuationParams = self.params
        data = image.data.copy()
        if data.shape[1] < 3:
            raise ModuleError("Attenuation correction needs a Z stack (3 planes or more)")
        z_step = image.metadata.z_step_um or 1.0
        channels: dict[str, Any] = {}
        warnings: list[str] = []
        for c in params.selected(data.shape[0]):
            signal, used, background = plane_signal(
                data[c], params.percentile, params.min_area_fraction
            )
            factors, info = correction_factors(signal, used, params.method, z_step)
            if float(np.max(factors)) > params.max_gain:
                warnings.append(
                    f"channel {c + 1}: correction capped at {params.max_gain:g}× (needed "
                    f"{float(np.max(factors)):.3g}×): the deepest planes are mostly noise"
                )
                factors = np.minimum(factors, params.max_gain)
            if info.get("note"):
                warnings.append(f"channel {c + 1}: {info['note']}")
            # only the signal above each plane's background is scaled (noise is not amplified)
            offset = background[:, None, None].astype(np.float32)
            data[c] = offset + (data[c] - offset) * factors[:, None, None].astype(np.float32)
            info["max_gain"] = float(np.max(factors))
            info["factors"] = [round(float(f), 4) for f in factors]
            channels[str(c + 1)] = info
        report = {"module": self.name, "method": params.method, "channels": channels,
                  "warnings": warnings}  # fmt: skip
        return image.derive(data, self.name, report=report)
