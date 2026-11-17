"""Enhancement steps: denoise, contrast, CLAHE, sharpening, clarity preset, Z projection.

Contrast, CLAHE, sharpening and clarity change intensities non-linearly. Their output
is for viewing and as segmentation input only; it is flagged ``linear=False`` so that
measurement steps can refuse it.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy import ndimage

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ChannelParams,
    FloatImage,
    ImageData,
    Module,
    ModuleError,
    replace,
    um_to_px,
)


def _per_plane(image: ImageData, channels: list[int], function: Any) -> FloatImage:
    out = image.data.copy()
    for c in channels:
        for z in range(out.shape[1]):
            out[c, z] = function(out[c, z])
    return out


def _unit_range(plane: NDArray[Any], low_pct: float, high_pct: float) -> FloatImage:
    low, high = np.percentile(plane, (low_pct, high_pct))
    if high <= low:
        return np.zeros_like(plane, dtype=np.float32)
    return np.asarray(np.clip((plane - low) / (high - low), 0, 1), dtype=np.float32)


def _noise_sigma(residual: NDArray[Any]) -> float:
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise(plane: NDArray[Any]) -> float:
    """Noise SD of an image by Immerkær's method (insensitive to smooth structure)."""
    if min(plane.shape) < 3:
        return 0.0
    response = ndimage.convolve(plane.astype(np.float32), _IMMERKAER, mode="nearest")
    return float(np.sqrt(np.pi / 2) / 6 * np.abs(response).mean())


def gated_unsharp(
    plane: NDArray[Any], denoise_sigma: float, detail_sigma: float, amount: float, noise_gate: float
) -> FloatImage:
    """Mild denoise, then unsharp masking that leaves noise-level detail alone."""
    smooth = ndimage.gaussian_filter(plane, denoise_sigma, mode="nearest")
    detail = smooth - ndimage.gaussian_filter(smooth, detail_sigma, mode="nearest")
    # Noise SD that survives into the detail band: raw noise SD times the L2 norm of
    # the band-pass kernel, measured from its impulse response.
    size = 2 * int(4 * (denoise_sigma + detail_sigma) + 1) + 1
    impulse = np.zeros((size, size), dtype=np.float32)
    impulse[size // 2, size // 2] = 1.0
    kernel = ndimage.gaussian_filter(impulse, denoise_sigma, mode="constant")
    kernel = kernel - ndimage.gaussian_filter(kernel, detail_sigma, mode="constant")
    threshold = noise_gate * estimate_noise(plane) * float(np.sqrt((kernel**2).sum()))
    # Soft gate: ~0 for detail at the noise level, ~1 for clear structure.
    gate = detail**2 / (detail**2 + threshold**2) if threshold > 0 else 1.0
    return np.asarray(np.clip(smooth + amount * gate * detail, 0, None), dtype=np.float32)


# -------------------------------------------------------------------- denoise


class DenoiseParams(ChannelParams):
    method: Literal["gaussian", "median", "nlm"] = Field(
        default="gaussian", description="nlm = non-local means (slow, preserves edges)."
    )
    sigma_px: float = Field(default=1.0, gt=0, description="Gaussian sigma in pixels.")
    median_size_px: int = Field(default=3, ge=3, description="Median window in pixels (odd).")
    nlm_strength: float = Field(
        default=1.0, gt=0, description="Non-local means strength, in units of the noise level."
    )


@MODULES.register("denoise")
class Denoise(Module):
    name = "denoise"
    title = "Denoise"
    Params = DenoiseParams

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: DenoiseParams = self.params

        def run(plane: FloatImage) -> NDArray[Any]:
            if params.method == "gaussian":
                return np.asarray(ndimage.gaussian_filter(plane, params.sigma_px, mode="nearest"))
            if params.method == "median":
                return np.asarray(ndimage.median_filter(plane, size=params.median_size_px))
            from skimage.restoration import denoise_nl_means

            sigma = estimate_noise(plane)
            if sigma <= 0:
                return plane
            return np.asarray(
                denoise_nl_means(
                    plane,
                    h=params.nlm_strength * sigma,
                    sigma=sigma,
                    patch_size=5,
                    patch_distance=6,
                    fast_mode=True,
                    preserve_range=True,
                )
            )

        return image.derive(_per_plane(image, params.selected(image.data.shape[0]), run), self.name)


# ------------------------------------------------------------------- contrast


class ContrastParams(ChannelParams):
    low_percentile: float = Field(default=0.1, ge=0, lt=100, description="Maps to black.")
    high_percentile: float = Field(default=99.9, gt=0, le=100, description="Maps to white.")
    gamma: float = Field(default=1.0, gt=0, description="Gamma applied after rescaling to 0-1.")

    @model_validator(mode="after")
    def _check(self) -> ContrastParams:
        if self.high_percentile <= self.low_percentile:
            raise ValueError("high_percentile must be greater than low_percentile")
        return self


@MODULES.register("contrast")
class Contrast(Module):
    name = "contrast"
    title = "Contrast (percentile rescale, gamma)"
    Params = ContrastParams
    preserves_linearity = False

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ContrastParams = self.params
        out = image.data.copy()
        for c in params.selected(out.shape[0]):
            # One range per channel (not per plane) keeps Z planes comparable.
            scaled = _unit_range(out[c], params.low_percentile, params.high_percentile)
            out[c] = scaled**params.gamma if params.gamma != 1.0 else scaled
        return image.derive(out, self.name, linear=False)


# ---------------------------------------------------------------------- CLAHE


def _clahe(plane: NDArray[Any], tile_px: int, clip_limit: float) -> FloatImage:
    from skimage.exposure import equalize_adapthist

    scaled = _unit_range(plane, 0.01, 99.99)
    if not scaled.any():
        return scaled
    tile = max(2, min(tile_px, min(plane.shape)))
    return np.asarray(
        equalize_adapthist(scaled, kernel_size=tile, clip_limit=clip_limit), dtype=np.float32
    )


class ClaheParams(ChannelParams):
    tile_um: float = Field(default=50.0, gt=0, description="Tile size in µm.")
    clip_limit: float = Field(
        default=0.01, gt=0, le=1, description="Contrast limit; higher = stronger."
    )


@MODULES.register("clahe")
class Clahe(Module):
    name = "clahe"
    title = "CLAHE (local contrast)"
    Params = ClaheParams
    preserves_linearity = False

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.tile_um, metadata, "The CLAHE tile size")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ClaheParams = self.params
        tile = round(um_to_px(params.tile_um, image.metadata, "The CLAHE tile size"))
        out = _per_plane(
            image,
            params.selected(image.data.shape[0]),
            lambda plane: _clahe(plane, tile, params.clip_limit),
        )
        return image.derive(out, self.name, linear=False)


# -------------------------------------------------------------------- sharpen


class SharpenParams(ChannelParams):
    method: Literal["unsharp", "dog"] = Field(
        default="unsharp", description="unsharp mask, or difference-of-Gaussians band boost."
    )
    sigma_px: float = Field(default=2.0, gt=0, description="Blur sigma (unsharp) in pixels.")
    sigma_large_px: float = Field(default=6.0, gt=0, description="Large sigma (dog) in pixels.")
    amount: float = Field(default=1.0, gt=0, description="Strength of the added detail.")

    @model_validator(mode="after")
    def _check(self) -> SharpenParams:
        if self.method == "dog" and self.sigma_large_px <= self.sigma_px:
            raise ValueError("sigma_large_px must be greater than sigma_px")
        return self


@MODULES.register("sharpen")
class Sharpen(Module):
    name = "sharpen"
    title = "Sharpen (unsharp mask / DoG)"
    Params = SharpenParams
    preserves_linearity = False

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: SharpenParams = self.params

        def run(plane: FloatImage) -> NDArray[Any]:
            small = ndimage.gaussian_filter(plane, params.sigma_px, mode="nearest")
            if params.method == "unsharp":
                detail = plane - small
            else:
                detail = small - ndimage.gaussian_filter(
                    plane, params.sigma_large_px, mode="nearest"
                )
            return np.clip(plane + params.amount * detail, 0, None)

        out = _per_plane(image, params.selected(image.data.shape[0]), run)
        return image.derive(out, self.name, linear=False)


# -------------------------------------------------------------------- clarity


class ClarityParams(ChannelParams):
    amount: float = Field(default=0.8, gt=0, le=3, description="Strength of detail enhancement.")
    denoise_sigma_px: float = Field(default=0.7, gt=0, description="Mild pre-smoothing (pixels).")
    detail_sigma_px: float = Field(default=3.0, gt=0, description="Scale of enhanced detail.")
    noise_gate: float = Field(
        default=3.0,
        ge=0,
        description="Detail weaker than this many noise SDs is not amplified (limits ringing).",
    )
    tile_um: float = Field(default=60.0, gt=0, description="CLAHE tile size in µm.")
    clip_limit: float = Field(default=0.005, gt=0, le=1, description="CLAHE contrast limit.")


@MODULES.register("clarity")
class Clarity(Module):
    """Preset: mild denoise → noise-gated unsharp mask → CLAHE."""

    name = "clarity"
    title = "Clarity preset"
    Params = ClarityParams
    preserves_linearity = False

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.tile_um, metadata, "The clarity tile size")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ClarityParams = self.params
        tile = round(um_to_px(params.tile_um, image.metadata, "The clarity tile size"))

        def run(plane: FloatImage) -> NDArray[Any]:
            sharpened = gated_unsharp(
                plane,
                params.denoise_sigma_px,
                params.detail_sigma_px,
                params.amount,
                params.noise_gate,
            )
            return _clahe(sharpened, tile, params.clip_limit)

        out = _per_plane(image, params.selected(image.data.shape[0]), run)
        return image.derive(out, self.name, linear=False)


# ----------------------------------------------------------------- projection


class ProjectionParams(ChannelParams):
    method: Literal["max", "mean", "edf"] = Field(
        default="max", description="edf = extended depth of field (sharpest plane per pixel)."
    )
    channels: list[int] | None = Field(default=None, exclude=True, description="Not used.")


def _edf(stack: FloatImage) -> FloatImage:
    focus = np.stack([ndimage.gaussian_filter(ndimage.laplace(plane) ** 2, 2.0) for plane in stack])
    best = ndimage.median_filter(np.argmax(focus, axis=0), size=5)
    return np.asarray(np.take_along_axis(stack, best[None], axis=0)[0], dtype=np.float32)


@MODULES.register("projection")
class Projection(Module):
    name = "projection"
    title = "Z projection"
    Params = ProjectionParams

    def output_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return replace(metadata, z_step_um=None)

    def apply(self, image: ImageData, state: Any) -> ImageData:
        method = self.params.method
        if image.data.shape[1] == 1:
            return image.derive(image.data, self.name)
        if method == "max":
            out = image.data.max(axis=1, keepdims=True)
        elif method == "mean":
            out = image.data.mean(axis=1, keepdims=True)
        elif method == "edf":
            out = np.stack([_edf(channel) for channel in image.data])[:, None]
        else:  # pragma: no cover - guarded by pydantic
            raise ModuleError(f"Unknown projection method {method!r}")
        return image.derive(out, self.name, metadata=self.output_metadata(image.metadata))
