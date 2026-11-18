"""Autofluorescence and background removal.

Two approaches, each selectable per channel by stacking steps: reference wells
(``af_reference``) and a dedicated AF channel (``af_channel``). ``unmix`` performs full
linear spectral unmixing from a reference spectra matrix. Background removal is in
``background.py``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy import ndimage

from hcs_studio.model.plate import ChannelInfo, ImageMetadata, parse_well_name
from hcs_studio.processing.base import (
    MODULES,
    ChannelParams,
    FitContext,
    ImageData,
    Module,
    ModuleError,
    ModuleParams,
    block_reduce_mean,
    replace,
    resize_to,
)

# ------------------------------------------------------------ reference wells


class AfReferenceParams(ChannelParams):
    reference_wells: list[str] = Field(
        default_factory=list,
        description="Unstained / vehicle-only wells, e.g. [A01, A02]. "
        "Empty = wells marked 'unstained' in the plate layout.",
    )
    mode: Literal["scalar", "image"] = Field(
        default="scalar",
        description="scalar: one value per channel; image: smooth per-pixel AF profile.",
    )
    max_fields: int = Field(default=50, ge=1, description="Maximum reference fields sampled.")
    smoothing_fraction: float = Field(
        default=0.02, gt=0, le=0.5, description="Smoothing of the AF image (fraction of width)."
    )


@MODULES.register("af_reference")
class AfReference(Module):
    name = "af_reference"
    title = "Autofluorescence: reference wells"
    Params = AfReferenceParams
    needs_fit = True

    def _wells(self, context: FitContext) -> list[tuple[int, int]]:
        params: AfReferenceParams = self.params
        if params.reference_wells:
            try:
                wells = [parse_well_name(w) for w in params.reference_wells]
            except ValueError as exc:
                raise ModuleError(str(exc)) from None
        else:
            wells = [k for k, v in context.layout.items() if v.control_type == "unstained"]
        if not wells:
            raise ModuleError(
                "No reference wells: list them in 'reference_wells' or mark wells as "
                "'unstained' in the plate layout"
            )
        present = [w for w in wells if w in context.plate.wells]
        if not present:
            raise ModuleError("None of the reference wells contain images on this plate")
        return present

    def fit(self, context: FitContext) -> dict[str, Any]:
        params: AfReferenceParams = self.params
        medians: list[NDArray[Any]] = []
        images: list[NDArray[Any]] = []
        for image in context.sample(params.max_fields, self._wells(context)):
            projected = image.data.mean(axis=1)
            medians.append(np.median(projected, axis=(1, 2)))
            if params.mode == "image":
                images.append(np.stack([block_reduce_mean(p, 256) for p in projected]))
        if not medians:
            raise ModuleError("The reference wells contain no readable fields")
        state: dict[str, Any] = {
            "scalar": np.median(np.stack(medians), axis=0).astype(np.float32),
            "n_fields": len(medians),
        }
        if params.mode == "image":
            median = np.median(np.stack(images), axis=0)
            sigma = params.smoothing_fraction * max(median.shape[1:])
            state["image"] = np.stack(
                [ndimage.gaussian_filter(m, sigma, mode="nearest") for m in median]
            ).astype(np.float32)
        return state

    def apply(self, image: ImageData, state: Any) -> ImageData:
        if state is None:
            raise ModuleError("Reference-well autofluorescence was not fitted")
        out = image.data.copy()
        for c in self.params.selected(out.shape[0]):
            if "image" in state:
                out[c] -= resize_to(state["image"][c], out.shape[2:])
            else:
                out[c] -= state["scalar"][c]
        np.clip(out, 0, None, out=out)
        return image.derive(out, self.name)

    def describe_state(self, state: Any) -> dict[str, Any]:
        return {
            "n_fields": state["n_fields"],
            "af_per_channel": [float(v) for v in state["scalar"]],
        }


# ----------------------------------------------------------------- AF channel


def robust_slope(x: NDArray[Any], y: NDArray[Any], iterations: int = 30) -> tuple[float, float]:
    """Slope and intercept of ``y ≈ k·x + b`` with Tukey-biweight IRLS.

    Pixels carrying real signal lie above the bleed-through line and are down-weighted
    to zero, provided background pixels are the majority.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 10 or np.ptp(x) == 0:
        raise ModuleError("Not enough variation in the AF channel to estimate bleed-through")
    # Start from the lower half of y so bright signal does not anchor the first fit.
    low = y <= np.median(y)
    weights = low.astype(np.float64)
    slope = intercept = 0.0
    for _ in range(iterations):
        total = weights.sum()
        if total <= 0:
            break
        mean_x = (weights * x).sum() / total
        mean_y = (weights * y).sum() / total
        var_x = (weights * (x - mean_x) ** 2).sum()
        if var_x <= 0:
            break
        new_slope = (weights * (x - mean_x) * (y - mean_y)).sum() / var_x
        new_intercept = mean_y - new_slope * mean_x
        residual = y - (new_slope * x + new_intercept)
        core = residual[weights > 0]
        scale = 1.4826 * np.median(np.abs(core - np.median(core)))
        converged = abs(new_slope - slope) < 1e-6 * max(1.0, abs(new_slope))
        slope, intercept = float(new_slope), float(new_intercept)
        if scale <= 0 or converged:
            break
        u = residual / (4.685 * scale)
        weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
    return slope, intercept


class AfChannelParams(ChannelParams):
    af_channel: int = Field(ge=1, description="Number of the autofluorescence / empty channel.")
    coefficients: dict[int, float] | None = Field(
        default=None,
        description="Manual bleed-through coefficient per channel number; empty = estimate.",
    )
    max_fields: int = Field(default=20, ge=1, description="Fields sampled for the estimate.")
    pixels_per_field: int = Field(default=20000, ge=100, description="Pixels sampled per field.")


@MODULES.register("af_channel")
class AfChannel(Module):
    name = "af_channel"
    title = "Autofluorescence: AF channel subtraction"
    Params = AfChannelParams
    needs_fit = True

    def _targets(self, n_channels: int) -> tuple[int, list[int]]:
        params: AfChannelParams = self.params
        if params.af_channel > n_channels:
            raise ModuleError(f"AF channel {params.af_channel} does not exist")
        af = params.af_channel - 1
        return af, [c for c in params.selected(n_channels) if c != af]

    def validate(self, metadata: ImageMetadata) -> None:
        self._targets(len(metadata.channels))

    def fit(self, context: FitContext) -> dict[str, Any]:
        params: AfChannelParams = self.params
        n_channels = len(context.plate.metadata.channels)
        af, targets = self._targets(n_channels)
        if params.coefficients is not None:
            missing = [t + 1 for t in targets if t + 1 not in params.coefficients]
            if missing:
                raise ModuleError(f"No manual coefficient given for channel(s) {missing}")
            return {"k": {t: float(params.coefficients[t + 1]) for t in targets}, "n_fields": 0}
        rng = np.random.default_rng(0)
        samples: list[NDArray[Any]] = []
        for image in context.sample(params.max_fields):
            flat = image.data.reshape(image.data.shape[0], -1)
            count = min(params.pixels_per_field, flat.shape[1])
            samples.append(flat[:, rng.choice(flat.shape[1], count, replace=False)])
        if not samples:
            raise ModuleError("No fields available to estimate bleed-through")
        pixels = np.concatenate(samples, axis=1)
        k = {t: max(0.0, robust_slope(pixels[af], pixels[t])[0]) for t in targets}
        return {"k": k, "n_fields": len(samples)}

    def apply(self, image: ImageData, state: Any) -> ImageData:
        if state is None:
            raise ModuleError("AF-channel subtraction was not fitted")
        af, _ = self._targets(image.data.shape[0])
        out = image.data.copy()
        for target, k in state["k"].items():
            out[target] -= np.float32(k) * image.data[af]
        np.clip(out, 0, None, out=out)
        return image.derive(out, self.name)

    def describe_state(self, state: Any) -> dict[str, Any]:
        return {
            "n_fields": state["n_fields"],
            "coefficients": {str(t + 1): k for t, k in state["k"].items()},
        }


# ------------------------------------------------------------------- unmixing


class UnmixParams(ModuleParams):
    mixing_matrix: list[list[float]] = Field(
        description="Reference spectra: one row per detection channel, one column per "
        "fluorophore (signal of that fluorophore in each channel)."
    )
    fluorophores: list[str] = Field(description="Name of each fluorophore (matrix column).")

    @model_validator(mode="after")
    def _check(self) -> UnmixParams:
        matrix = np.asarray(self.mixing_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("mixing_matrix must be a non-empty rectangular matrix")
        if matrix.shape[1] != len(self.fluorophores):
            raise ValueError("One fluorophore name per matrix column is required")
        if matrix.shape[1] > matrix.shape[0]:
            raise ValueError("More fluorophores than detection channels cannot be unmixed")
        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            raise ValueError("The reference spectra are linearly dependent")
        return self


@MODULES.register("unmix")
class Unmix(Module):
    name = "unmix"
    title = "Linear spectral unmixing"
    Params = UnmixParams

    def validate(self, metadata: ImageMetadata) -> None:
        rows = len(self.params.mixing_matrix)
        if rows != len(metadata.channels):
            raise ModuleError(
                f"The mixing matrix has {rows} rows but the image has "
                f"{len(metadata.channels)} channels"
            )

    def output_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return replace(
            metadata, channels=tuple(ChannelInfo(name=n) for n in self.params.fluorophores)
        )

    def apply(self, image: ImageData, state: Any) -> ImageData:
        self.validate(image.metadata)
        inverse = np.linalg.pinv(np.asarray(self.params.mixing_matrix, dtype=np.float64))
        out = np.tensordot(inverse.astype(np.float32), image.data, axes=([1], [0]))
        np.clip(out, 0, None, out=out)
        return image.derive(out, self.name, metadata=self.output_metadata(image.metadata))
