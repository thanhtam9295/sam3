"""Deconvolution benchmark on a synthetic stack, to report runtime on this machine."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcs_studio.processing.deconvolution import Device, deconvolve, get_backend
from hcs_studio.processing.psf import PsfSpec, theoretical_psf


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, int, int]
    iterations: int
    backend: str
    tiles: int
    seconds: float
    seconds_per_iteration: float
    peak_memory_gb: float | None
    psf_shape: tuple[int, ...]

    def summary(self) -> str:
        z, y, x = self.shape
        memory = (
            f"{self.peak_memory_gb:.1f} GB peak process memory"
            if self.peak_memory_gb is not None
            else "peak memory not available on this platform"
        )
        return (
            f"Deconvolution benchmark: {x}×{y}×{z} voxels, Richardson-Lucy + TV, "
            f"{self.iterations} iterations, PSF {self.psf_shape}\n"
            f"Backend: {self.backend}; tiles: {self.tiles}\n"
            f"Runtime: {self.seconds:.1f} s ({self.seconds_per_iteration:.2f} s per iteration)\n"
            f"Memory: {memory}\n"
            f"Machine: {platform.processor() or platform.machine()}, {platform.system()}"
        )


def _peak_memory_gb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return float(peak) / (1024**3 if platform.system() == "Darwin" else 1024**2)


def synthetic_stack(shape: tuple[int, int, int], seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    data = rng.poisson(100.0, size=shape).astype(np.float32)
    count = max(10, shape[1] * shape[2] // 4000)
    zs = rng.integers(0, shape[0], count)
    ys = rng.integers(0, shape[1], count)
    xs = rng.integers(0, shape[2], count)
    data[zs, ys, xs] += 20000
    return data


def run_benchmark(
    shape: tuple[int, int, int] = (30, 2048, 2048),
    iterations: int = 40,
    device: Device = "auto",
    tile_size: int = 0,
    max_memory_gb: float | None = None,
) -> BenchmarkResult:
    """Deconvolve a synthetic stack with a 60x/1.4 oil widefield PSF and time it."""
    spec = PsfSpec(na=1.4, emission_nm=520, pixel_size_um=0.1, immersion_ri=1.515, z_step_um=0.3)
    psf = theoretical_psf(spec, three_d=shape[0] > 1)
    data = synthetic_stack(shape)
    backend = get_backend(device)
    started = time.perf_counter()
    result = deconvolve(
        data,
        psf,
        iterations=iterations,
        backend=backend,
        tile_size=tile_size,
        max_memory_gb=max_memory_gb,
        z_to_xy_ratio=1 / 3,
    )
    seconds = time.perf_counter() - started
    return BenchmarkResult(
        shape=shape,
        iterations=result.iterations,
        backend=result.backend,
        tiles=result.tiles,
        seconds=seconds,
        seconds_per_iteration=seconds / max(1, result.iterations),
        peak_memory_gb=_peak_memory_gb(),
        psf_shape=tuple(psf.shape),
    )
