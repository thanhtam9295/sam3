"""Synthetic fluorescence images with a known ground truth, for tests and demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticField:
    """A (C, Y, X) uint16 image plus the nucleus centres used to draw it."""

    image: NDArray[np.uint16]
    channel_names: tuple[str, ...]
    centers_yx: NDArray[np.float64]
    pixel_size_um: float


def make_cell_field(
    shape: tuple[int, int] = (512, 512),
    n_cells: int = 40,
    pixel_size_um: float = 0.325,
    seed: int = 0,
) -> SyntheticField:
    """Draw ``n_cells`` cells: Gaussian nuclei (channel 0) inside wider cytoplasm (channel 1).

    Deterministic for a given seed. Includes a constant camera offset and Poisson noise.
    """
    if n_cells < 0:
        raise ValueError("n_cells must be non-negative")
    rng = np.random.default_rng(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    margin = 12.0
    centers = np.column_stack(
        [
            rng.uniform(margin, height - margin, n_cells),
            rng.uniform(margin, width - margin, n_cells),
        ]
    )
    nuclei = np.zeros(shape, dtype=np.float64)
    cytoplasm = np.zeros(shape, dtype=np.float64)
    for cy, cx in centers:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        nuclei += rng.uniform(1500, 4000) * np.exp(-r2 / (2 * rng.uniform(5.0, 8.0) ** 2))
        cytoplasm += rng.uniform(500, 1500) * np.exp(-r2 / (2 * rng.uniform(12.0, 18.0) ** 2))
    offset = 100.0
    stack = np.stack([nuclei, cytoplasm]) + offset
    noisy = rng.poisson(stack).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    return SyntheticField(
        image=noisy,
        channel_names=("Nuclei (synthetic)", "Cytoplasm (synthetic)"),
        centers_yx=centers,
        pixel_size_um=pixel_size_um,
    )
