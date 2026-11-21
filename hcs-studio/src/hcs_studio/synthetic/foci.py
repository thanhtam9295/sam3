"""Nuclei with a known number of foci (Phase 20b).

:class:`FociNuclei` draws nuclei (discs, DNA channel 0) on a jittered grid; in the foci
channel (1) each nucleus has a dim uniform background and ``count(row, col)`` Gaussian foci
(σ ``focus_sigma_px``) at least ``separation_px`` apart and inside the nucleus, or — for a
share ``pan_fraction(row, col)`` of nuclei — bright, diffuse pan-nuclear staining without foci.
The truth records every nucleus: centre (px), number of foci, pan-nuclear.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FociNuclei:
    count: Callable[[int, int, np.random.Generator], int]
    pan_fraction: Callable[[int, int], float] = lambda r, c: 0.0
    name: str = "foci"
    radius_px: float = 10.0
    grid: int = 6
    dna_intensity: float = 800.0
    nuclear_background: float = 60.0
    focus_amplitude: float = 500.0
    focus_sigma_px: float = 1.0
    separation_px: float = 5.0
    pan_intensity: float = 900.0

    def _positions(self, k: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        placed: list[tuple[float, float]] = []
        limit = self.radius_px - 2.5
        tries = 0
        while len(placed) < k and tries < 5000:
            tries += 1
            y, x = rng.uniform(-limit, limit, 2)
            if y * y + x * x > limit * limit:
                continue
            if all((y - a) ** 2 + (x - b) ** 2 >= self.separation_px**2 for a, b in placed):
                placed.append((float(y), float(x)))
        return placed

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        pitch = min(h, w) / (self.grid + 0.4)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        nuclei = []
        for i in range(self.grid**2):
            cy = pitch * (0.7 + i // self.grid) + ctx.rng.uniform(-1.5, 1.5)
            cx = pitch * (0.7 + i % self.grid) + ctx.rng.uniform(-1.5, 1.5)
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.radius_px**2
            image[0, :, disc] += self.dna_intensity
            pan = bool(ctx.rng.random() < self.pan_fraction(ctx.row, ctx.col))
            if pan:
                image[1, :, disc] += self.pan_intensity
                nuclei.append({"centre_px": [round(cy, 1), round(cx, 1)], "foci": 0, "pan": True})
                continue
            image[1, :, disc] += self.nuclear_background
            wanted = int(self.count(ctx.row, ctx.col, ctx.rng))
            spots = self._positions(wanted, ctx.rng)
            for dy, dx in spots:
                spot = self.focus_amplitude * np.exp(
                    -((yy - cy - dy) ** 2 + (xx - cx - dx) ** 2) / (2 * self.focus_sigma_px**2)
                )
                image[1] += spot[None]
            nuclei.append(
                {"centre_px": [round(cy, 1), round(cx, 1)], "foci": len(spots), "pan": False}
            )
        return {"count": len(nuclei), "nuclei": nuclei}
