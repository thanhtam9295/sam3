"""Deconvolution: Richardson-Lucy with total-variation regularisation, and Wiener.

Richardson-Lucy follows Dey et al. (2006) for the TV term and Biggs & Andrews (1997)
for acceleration. The forward model is ``data ≈ PSF ⊛ estimate + background``; the
returned estimate excludes the constant background and is non-negative.

This is classic RL/TV deconvolution. Validate it against your reference software
(e.g. Huygens) on bead and cell images before using it for quantitative claims.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy import fft as sfft

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing import psf as psf_tools
from hcs_studio.processing.base import (
    MODULES,
    ChannelParams,
    FloatImage,
    ImageData,
    Module,
    ModuleError,
)

_EPS = 1e-6

# ------------------------------------------------------------------- backends


class NumpyBackend:
    """CPU backend: scipy FFT on all cores, float32 throughout."""

    name = "cpu (numpy)"

    def array(self, data: NDArray[Any]) -> Any:
        return np.ascontiguousarray(data, dtype=np.float32)

    def numpy(self, data: Any) -> FloatImage:
        return np.asarray(data, dtype=np.float32)

    def rfftn(self, data: Any) -> Any:
        return sfft.rfftn(data, workers=-1)

    def irfftn(self, data: Any, shape: tuple[int, ...]) -> Any:
        return sfft.irfftn(data, s=shape, workers=-1)

    def conj(self, data: Any) -> Any:
        return np.conj(data)

    def clip_min(self, data: Any, minimum: float) -> Any:
        return np.maximum(data, np.float32(minimum))

    def roll(self, data: Any, shift: int, axis: int) -> Any:
        return np.roll(data, shift, axis)

    def sqrt(self, data: Any) -> Any:
        return np.sqrt(data)

    def abs(self, data: Any) -> Any:
        return np.abs(data)

    def total(self, data: Any) -> float:
        return float(np.sum(data, dtype=np.float64))


class TorchBackend:
    """PyTorch backend (CUDA when available)."""

    def __init__(self, device: str) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.name = f"{device} (torch {torch.__version__})"

    def array(self, data: NDArray[Any]) -> Any:
        return self._torch.as_tensor(
            np.ascontiguousarray(data, dtype=np.float32), device=self.device
        )

    def numpy(self, data: Any) -> FloatImage:
        return np.asarray(data.detach().cpu().numpy(), dtype=np.float32)

    def rfftn(self, data: Any) -> Any:
        return self._torch.fft.rfftn(data)

    def irfftn(self, data: Any, shape: tuple[int, ...]) -> Any:
        return self._torch.fft.irfftn(data, s=shape)

    def conj(self, data: Any) -> Any:
        return self._torch.conj(data)

    def clip_min(self, data: Any, minimum: float) -> Any:
        return self._torch.clamp(data, min=minimum)

    def roll(self, data: Any, shift: int, axis: int) -> Any:
        return self._torch.roll(data, shift, axis)

    def sqrt(self, data: Any) -> Any:
        return self._torch.sqrt(data)

    def abs(self, data: Any) -> Any:
        return self._torch.abs(data)

    def total(self, data: Any) -> float:
        return float(data.sum(dtype=self._torch.float64))


Backend = NumpyBackend | TorchBackend
Device = Literal["auto", "cpu", "gpu"]


def get_backend(device: Device = "auto") -> Backend:
    """GPU via PyTorch when available (or required), otherwise numpy on the CPU."""
    if device == "cpu":
        return NumpyBackend()
    try:
        import torch

        if torch.cuda.is_available():
            return TorchBackend("cuda")
    except Exception:
        pass
    if device == "gpu":
        raise ModuleError("No CUDA GPU is available to PyTorch; choose device 'auto' or 'cpu'")
    return NumpyBackend()


# ------------------------------------------------------------------ algorithm


@dataclass(frozen=True)
class DeconvolutionResult:
    image: FloatImage
    iterations: int
    #: Mean absolute change of the estimate in the last iteration, relative to its mean.
    relative_change: float | None
    #: Mean |data - model| relative to the mean of the data, after the last iteration.
    residual: float
    background: float
    seconds: float
    backend: str
    tiles: int = 1


def estimate_background(data: NDArray[Any]) -> float:
    """Constant background (camera offset, diffuse signal).

    A low percentile of the raw data is biased low by noise (by about 2.5 noise SDs for
    the 0.5th percentile). Smoothing first shrinks the noise spread, so the 1st percentile
    of the smoothed image lies within a fraction of a noise SD of the true background.
    """
    from scipy import ndimage

    sigma = [0.0 if n == 1 else 2.0 for n in data.shape]
    smooth = ndimage.gaussian_filter(np.asarray(data, dtype=np.float32), sigma, mode="reflect")
    return float(np.percentile(smooth, 1.0))


def _padding(shape: tuple[int, ...], psf_shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Mirror padding per axis: the PSF half-size, rounded up to an FFT-friendly size."""
    pads = []
    for size, kernel in zip(shape, psf_shape, strict=True):
        if size == 1:
            pads.append((0, 0))
            continue
        half = min(kernel // 2, size - 1)
        target = sfft.next_fast_len(size + 2 * half, real=True)
        pads.append((half, target - size - half))
    return pads


#: Peak working memory of one Richardson-Lucy + TV run, per padded voxel (measured: ~80).
BYTES_PER_PADDED_VOXEL = 96
_MIN_TILE = 128


def estimate_memory(shape: tuple[int, ...], psf_shape: tuple[int, ...]) -> int:
    """Bytes needed to deconvolve an image of ``shape`` in one piece."""
    pads = _padding(shape, psf_shape)
    voxels = int(np.prod([n + a + b for n, (a, b) in zip(shape, pads, strict=True)]))
    return voxels * BYTES_PER_PADDED_VOXEL


def auto_tile_size(shape: tuple[int, int, int], psf_shape: tuple[int, ...], budget: float) -> int:
    """0 when the whole image fits in ``budget`` bytes, else the largest Y/X tile that does."""
    if estimate_memory(shape, psf_shape) <= budget:
        return 0
    tile = (max(shape[1], shape[2]) // 64) * 64
    while tile >= _MIN_TILE:
        if (
            estimate_memory((shape[0], min(tile, shape[1]), min(tile, shape[2])), psf_shape)
            <= budget
        ):
            return tile
        tile -= 64
    needed = estimate_memory((shape[0], _MIN_TILE, _MIN_TILE), psf_shape) / 1024**3
    raise ModuleError(
        f"Not enough memory: even {_MIN_TILE}×{_MIN_TILE} tiles need {needed:.1f} GB; "
        "raise max_memory_gb or deconvolve each plane separately (force_2d)"
    )


def _available_ram() -> float | None:
    import os
    import sys

    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return float(status.avail_phys)
        return None
    try:
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def memory_budget(backend: Backend, max_memory_gb: float | None) -> float:
    """Bytes one deconvolution may use: the user limit, else a share of free (V)RAM."""
    if max_memory_gb is not None:
        return max_memory_gb * 1024**3
    if isinstance(backend, TorchBackend) and backend.device.type == "cuda":
        free, _ = backend._torch.cuda.mem_get_info(backend.device)
        return 0.8 * float(free)
    available = _available_ram()
    return 0.5 * available if available else 4 * 1024**3


def _prepare(
    image: NDArray[Any], psf: NDArray[Any]
) -> tuple[FloatImage, FloatImage, tuple[slice, ...]]:
    """Mirror-pad the image by the PSF half-size (to FFT-friendly sizes) and build the kernel."""
    if psf.ndim != image.ndim:
        raise ModuleError(f"The PSF is {psf.ndim}D but the image is {image.ndim}D")
    pads = _padding(image.shape, psf.shape)
    crop = [slice(a, a + n) for n, (a, _) in zip(image.shape, pads, strict=True)]
    # 'reflect' needs pads smaller than the axis; fall back to 'symmetric' for tiny axes.
    fits = all(max(p) < n for p, n in zip(pads, image.shape, strict=True))
    padded = np.pad(
        image.astype(np.float32, copy=False), pads, mode="reflect" if fits else "symmetric"
    )
    kernel = psf_tools.centre_crop_or_pad(psf_tools.normalise(psf), padded.shape)
    kernel = np.fft.ifftshift(kernel / kernel.sum())
    return padded, np.asarray(kernel, dtype=np.float32), tuple(crop)


def _tv_denominator(backend: Backend, x: Any, tv_lambda: float, weights: tuple[float, ...]) -> Any:
    """``1 - λ·div(∇x / |∇x|)`` with periodic differences (the image is padded)."""
    grads: list[Any] = []
    for axis in range(x.ndim):
        if x.shape[axis] == 1:
            grads.append(None)
            continue
        grads.append((backend.roll(x, -1, axis) - x) * weights[axis])
    norm = None
    for grad in grads:
        if grad is not None:
            norm = grad * grad if norm is None else norm + grad * grad
    if norm is None:
        return 1.0
    norm = backend.sqrt(norm + _EPS)
    divergence = None
    for axis, grad in enumerate(grads):
        if grad is None:
            continue
        unit = grad / norm
        term = (unit - backend.roll(unit, 1, axis)) * weights[axis]
        divergence = term if divergence is None else divergence + term
    if divergence is None:
        return 1.0
    return backend.clip_min(1.0 - tv_lambda * divergence, 0.1)


def _richardson_lucy(
    backend: Backend,
    data: FloatImage,
    kernel: FloatImage,
    *,
    iterations: int,
    tv_lambda: float,
    accelerate: bool,
    stop_rel_change: float,
    background: float,
    weights: tuple[float, ...],
) -> tuple[Any, int, float | None, float]:
    shape = tuple(data.shape)
    d = backend.array(data)
    otf = backend.rfftn(backend.array(kernel))
    otf_conj = backend.conj(otf)

    def forward(x: Any) -> Any:
        return backend.irfftn(backend.rfftn(x) * otf, shape) + background

    # Strictly positive start: RL is multiplicative, so zeros would stay zero.
    mean = float(np.mean(np.clip(data - background, 0, None)))
    x = backend.clip_min(d - background, 0.0) * 0.5 + 0.5 * max(mean, _EPS)
    x_old = x
    g1 = g2 = None
    done = 0
    change: float | None = None
    for _ in range(iterations):
        if accelerate and g1 is not None and g2 is not None:
            alpha = backend.total(g1 * g2) / (backend.total(g2 * g2) + 1e-30)
            alpha = min(max(alpha, 0.0), 1.0)
            y = backend.clip_min(x + alpha * (x - x_old), 0.0)
        else:
            y = x
        ratio = d / backend.clip_min(forward(y), _EPS)
        x_new = y * backend.irfftn(backend.rfftn(ratio) * otf_conj, shape)
        if tv_lambda > 0:
            x_new = x_new / _tv_denominator(backend, y, tv_lambda, weights)
        x_new = backend.clip_min(x_new, 0.0)
        g2, g1 = g1, x_new - y
        change = backend.total(backend.abs(x_new - x)) / max(backend.total(x), 1e-30)
        x_old, x = x, x_new
        done += 1
        if stop_rel_change > 0 and change < stop_rel_change:
            break
    residual = backend.total(backend.abs(d - forward(x))) / max(backend.total(d), 1e-30)
    return x, done, change, residual


def _wiener(
    backend: Backend, data: FloatImage, kernel: FloatImage, nsr: float, background: float
) -> tuple[Any, float]:
    shape = tuple(data.shape)
    d = backend.array(data)
    otf = backend.rfftn(backend.array(kernel))
    power = backend.abs(otf) ** 2
    spectrum = backend.rfftn(backend.clip_min(d - background, 0.0))
    x = backend.clip_min(backend.irfftn(spectrum * backend.conj(otf) / (power + nsr), shape), 0.0)
    model = backend.irfftn(backend.rfftn(x) * otf, shape) + background
    residual = backend.total(backend.abs(d - model)) / max(backend.total(d), 1e-30)
    return x, residual


def tile_slices(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Start/stop of overlapping tiles that cover ``length``."""
    if tile <= 0 or length <= tile:
        return [(0, length)]
    if overlap >= tile:
        raise ModuleError("The tile size must be larger than the tile overlap")
    step = tile - overlap
    starts = list(range(0, max(1, length - overlap), step))
    return [(min(s, length - tile), min(s, length - tile) + tile) for s in starts]


def _ramp(length: int, start: int, stop: int, overlap: int) -> NDArray[Any]:
    """Blend weight of a tile along one axis: linear ramps over the overlaps, 1 elsewhere."""
    weight = np.ones(stop - start, dtype=np.float32)
    ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
    if start > 0 and overlap:
        weight[:overlap] = ramp
    if stop < length and overlap:
        weight[-overlap:] = ramp[::-1]
    return weight


def deconvolve(
    image: NDArray[Any],
    psf: NDArray[Any],
    *,
    method: Literal["rl", "wiener"] = "rl",
    iterations: int = 40,
    tv_lambda: float = 0.002,
    accelerate: bool = True,
    stop_rel_change: float = 0.0,
    background: float | None = None,
    wiener_nsr: float = 0.01,
    z_to_xy_ratio: float = 1.0,
    tile_size: int = 0,
    max_memory_gb: float | None = None,
    device: Device = "auto",
    backend: Backend | None = None,
) -> DeconvolutionResult:
    """Deconvolve a (Z, Y, X) image (``Z == 1`` for 2D) with a PSF of the same dimensionality.

    ``z_to_xy_ratio`` is pixel size / Z step, used to weight the Z gradient of the TV term.
    ``tile_size`` > 0 processes Y/X tiles of that size with blended overlaps; 0 chooses
    tiles automatically when the whole image would exceed the memory budget.
    """
    if image.ndim != 3:
        raise ModuleError(f"Expected a (Z, Y, X) image; got shape {image.shape}")
    if psf.ndim == 2:
        psf = psf[None]
    if image.shape[0] == 1 and psf.shape[0] > 1:
        psf = psf[psf.shape[0] // 2][None]  # 2D image: in-focus plane of the PSF
    backend = backend or get_backend(device)
    started = time.perf_counter()
    offset = estimate_background(image) if background is None else float(background)
    weights = (z_to_xy_ratio, 1.0, 1.0)

    def run(block: NDArray[Any]) -> tuple[FloatImage, int, float | None, float]:
        padded, kernel, crop = _prepare(block, psf)
        if method == "rl":
            estimate, done, change, residual = _richardson_lucy(
                backend,
                padded,
                kernel,
                iterations=iterations,
                tv_lambda=tv_lambda,
                accelerate=accelerate,
                stop_rel_change=stop_rel_change,
                background=offset,
                weights=weights,
            )
        elif method == "wiener":
            estimate, residual = _wiener(backend, padded, kernel, wiener_nsr, offset)
            done, change = 1, None
        else:
            raise ModuleError(f"Unknown deconvolution method {method!r}")
        return backend.numpy(estimate)[crop], done, change, residual

    if tile_size == 0:
        tile_size = auto_tile_size(image.shape, psf.shape, memory_budget(backend, max_memory_gb))
    overlap = max(16, 2 * (psf.shape[-1] // 2))
    ys = tile_slices(image.shape[1], tile_size, overlap)
    xs = tile_slices(image.shape[2], tile_size, overlap)
    if len(ys) == 1 and len(xs) == 1:
        result, done, change, residual = run(image)
        tiles = 1
    else:
        result = np.zeros(image.shape, dtype=np.float32)
        total = np.zeros(image.shape[1:], dtype=np.float32)
        done, change, residuals = 0, None, []
        for y0, y1 in ys:
            for x0, x1 in xs:
                tile, n, c, r = run(image[:, y0:y1, x0:x1])
                weight = np.outer(
                    _ramp(image.shape[1], y0, y1, overlap), _ramp(image.shape[2], x0, x1, overlap)
                )
                result[:, y0:y1, x0:x1] += tile * weight
                total[y0:y1, x0:x1] += weight
                done = max(done, n)
                change = c if change is None or (c is not None and c > change) else change
                residuals.append(r)
        result /= np.maximum(total, 1e-6)
        residual = float(np.mean(residuals))
        tiles = len(ys) * len(xs)
    return DeconvolutionResult(
        image=result,
        iterations=done,
        relative_change=change,
        residual=residual,
        background=offset,
        seconds=time.perf_counter() - started,
        backend=backend.name,
        tiles=tiles,
    )


# --------------------------------------------------------------------- module


class DeconvolutionParams(ChannelParams):
    method: Literal["rl", "wiener"] = Field(
        default="rl", description="rl: Richardson-Lucy with TV; wiener: fast, for quick previews."
    )
    iterations: int = Field(default=40, ge=1, le=1000, description="Maximum RL iterations.")
    tv_lambda: float = Field(
        default=0.002, ge=0, le=0.1, description="Total-variation regularisation; 0 = off."
    )
    accelerate: bool = Field(default=True, description="Biggs-Andrews acceleration.")
    stop_rel_change: float = Field(
        default=0.0, ge=0, description="Stop when the relative change falls below this; 0 = off."
    )
    background: float | None = Field(
        default=None, ge=0, description="Constant background in counts; empty = estimate."
    )
    wiener_nsr: float = Field(default=0.01, gt=0, description="Wiener noise-to-signal ratio.")
    psf_file: str | None = Field(
        default=None, description="Measured PSF file (from the PSF tool); empty = theoretical."
    )
    psf_model: Literal["vectorial", "scalar", "gaussian"] = Field(
        default="vectorial", description="Theoretical PSF model."
    )
    sample_ri: float = Field(default=1.33, gt=1, lt=2, description="Sample refractive index.")
    pinhole_au: float | None = Field(
        default=None, gt=0, description="Confocal pinhole in Airy units; empty = from metadata."
    )
    force_2d: bool = Field(default=False, description="Deconvolve each Z plane separately.")
    tile_size_px: int = Field(
        default=0,
        ge=0,
        description="Process Y/X tiles of this size; 0 = automatic (tiles only when needed).",
    )
    max_memory_gb: float | None = Field(
        default=None,
        gt=0,
        description="Memory one deconvolution may use; empty = half the free RAM "
        "(80% of free GPU memory).",
    )
    device: Literal["auto", "cpu", "gpu"] = Field(default="auto", description="Where to compute.")


@MODULES.register("deconvolution")
class Deconvolution(Module):
    name = "deconvolution"
    title = "Deconvolution"
    Params = DeconvolutionParams

    def __init__(self, params: Any = None) -> None:
        super().__init__(params)
        self._psfs: dict[tuple[int, bool], tuple[FloatImage, dict[str, Any]]] = {}

    def validate(self, metadata: ImageMetadata) -> None:
        params: DeconvolutionParams = self.params
        if metadata.pixel_size_um is None:
            raise ModuleError("Deconvolution needs the pixel size; set it in the Metadata panel")
        if params.psf_file is None:
            for channel in params.selected(len(metadata.channels)):
                psf_tools.spec_from_metadata(metadata, channel)

    def psf_for(
        self, metadata: ImageMetadata, channel: int, three_d: bool
    ) -> tuple[FloatImage, dict[str, Any]]:
        """PSF for one channel, matched to the image voxel size, with a description."""
        key = (channel, three_d)
        if key in self._psfs:
            return self._psfs[key]
        params: DeconvolutionParams = self.params
        assert metadata.pixel_size_um is not None
        if three_d and metadata.z_step_um is None:
            raise ModuleError("3D deconvolution needs the Z step; set it in the Metadata panel")
        voxel = (
            metadata.z_step_um if three_d else None,
            metadata.pixel_size_um,
            metadata.pixel_size_um,
        )
        info: dict[str, Any]
        if params.psf_file is not None:
            from pathlib import Path

            measured, pixel, z_step = psf_tools.load_psf(Path(params.psf_file))
            if three_d and measured.shape[0] == 1:
                raise ModuleError("The PSF file is 2D but the image is a Z stack; use force_2d")
            source = (z_step, pixel or metadata.pixel_size_um, pixel or metadata.pixel_size_um)
            psf = psf_tools.resample_psf(measured, source, voxel)
            info = {"source": "measured", "file": params.psf_file}
            if pixel is None:
                info["note"] = "PSF file has no voxel size; assumed equal to the image"
        else:
            spec, notes = psf_tools.spec_from_metadata(
                metadata,
                channel,
                sample_ri=params.sample_ri,
                pinhole_au=params.pinhole_au,
                model=params.psf_model,
            )
            psf = psf_tools.theoretical_psf(spec, three_d)
            info = {
                "source": "theoretical",
                "model": params.psf_model,
                "confocal": spec.confocal,
                "assumptions": notes,
            }
        info["shape"] = list(psf.shape)
        info["fwhm_um"] = psf_tools.measure_fwhm(psf, voxel)
        self._psfs[key] = (psf, info)
        return psf, info

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: DeconvolutionParams = self.params
        self.validate(image.metadata)
        metadata = image.metadata
        three_d = image.data.shape[1] > 1 and not params.force_2d
        assert metadata.pixel_size_um is not None
        ratio = (
            metadata.pixel_size_um / metadata.z_step_um if three_d and metadata.z_step_um else 1.0
        )
        backend = get_backend(params.device)
        out = image.data.copy()
        channels = []
        for c in params.selected(out.shape[0]):
            psf, psf_info = self.psf_for(metadata, c, three_d)
            blocks = [out[c]] if three_d else [out[c, z][None] for z in range(out.shape[1])]
            results = [
                deconvolve(
                    block,
                    psf,
                    method=params.method,
                    iterations=params.iterations,
                    tv_lambda=params.tv_lambda,
                    accelerate=params.accelerate,
                    stop_rel_change=params.stop_rel_change,
                    background=params.background,
                    wiener_nsr=params.wiener_nsr,
                    z_to_xy_ratio=ratio,
                    tile_size=params.tile_size_px,
                    max_memory_gb=params.max_memory_gb,
                    backend=backend,
                )
                for block in blocks
            ]
            out[c] = results[0].image if three_d else np.concatenate([r.image for r in results])
            channels.append(
                {
                    "channel": metadata.channels[c].name,
                    "mode": "3D" if three_d else "2D",
                    "iterations": max(r.iterations for r in results),
                    "relative_change": max((r.relative_change or 0.0) for r in results),
                    "residual": float(np.mean([r.residual for r in results])),
                    "background": float(np.mean([r.background for r in results])),
                    "seconds": round(sum(r.seconds for r in results), 3),
                    "tiles": results[0].tiles,
                    "psf": psf_info,
                }
            )
        report = {
            "module": self.name,
            "method": params.method,
            "backend": backend.name,
            "channels": channels,
        }
        return image.derive(out, self.name, report=report)
