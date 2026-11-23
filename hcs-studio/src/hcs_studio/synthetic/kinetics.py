"""Growing cells on a time-lapse plate with known counts, confluence and drift (Phase 22).

:class:`GrowingCells` draws, at each time point, ``count(row, col, hours)`` cells of a fixed
pool of positions (the same cells stay where they are; new ones appear): a nucleus (disc of
radius ``nucleus_px`` in ``nucleus_channel``) inside a textured, darker cell body (disc of
``body_px`` in ``body_channel`` — a brightfield/phase-like image on a bright background). The
whole field moves by ``drift(row, col, hours)`` (µm, stage drift). The truth per time point:
cells, confluence (% of the field covered by bodies) and the drift in pixels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


def exponential(n0: float, doubling_h: float, cap: int | None = None) -> Callable[..., int]:
    """count(row, col, hours) for exponential growth from ``n0`` (capped at ``cap``)."""

    def count(row: int, col: int, hours: float) -> int:
        n = round(n0 * 2 ** (hours / doubling_h))
        return min(n, cap) if cap is not None else n

    return count


@dataclass
class GrowingCells:
    count: Callable[[int, int, float], int]
    name: str = "cells"
    nucleus_channel: int = 0
    body_channel: int | None = 1
    nucleus_px: float = 5.0
    body_px: float = 16.0
    #: Cells stay this far from the field's edge (room for the drift).
    margin_px: float = 0.0
    drift: Callable[[int, int, float], tuple[float, float]] | None = None
    background: float = 1000.0
    body: float = 650.0
    texture: float = 60.0
    nucleus: float = 900.0

    def _pool(self, ctx: Any) -> np.ndarray:
        """Cell positions of this field, in the order cells appear (same at every time)."""
        h, w = ctx.image.shape[-2:]
        pitch = 2 * self.body_px + 4
        edge = self.body_px + 6 + self.margin_px
        ys = np.arange(edge, h - edge, pitch)
        xs = np.arange(edge, w - edge, pitch)
        grid = np.array([(y, x) for y in ys for x in xs], dtype=float)
        rng = np.random.default_rng([ctx.row, ctx.col, ctx.field, 7])
        grid += rng.uniform(-1.0, 1.0, grid.shape)
        return grid[rng.permutation(len(grid))]

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        hours = ctx.time_s / 3600
        pool = self._pool(ctx)
        n = min(int(self.count(ctx.row, ctx.col, hours)), len(pool))
        dy_um, dx_um = self.drift(ctx.row, ctx.col, hours) if self.drift else (0.0, 0.0)
        dy, dx = dy_um / ctx.pixel_size_um, dx_um / ctx.pixel_size_um
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        bodies = np.zeros((h, w), dtype=bool)
        nuclei = np.zeros((h, w), dtype=bool)
        for y, x in pool[:n]:
            distance = (yy - (y + dy)) ** 2 + (xx - (x + dx)) ** 2
            bodies |= distance <= self.body_px**2
            nuclei |= distance <= self.nucleus_px**2
        image[self.nucleus_channel, :, nuclei] += self.nucleus
        if self.body_channel is not None:
            from scipy import ndimage

            noise = np.random.default_rng([ctx.row, ctx.col, ctx.field, 3]).normal(0, 1, (h, w))
            texture = ndimage.shift(ndimage.gaussian_filter(noise, 1.0), (dy, dx), order=1)
            texture *= self.texture / max(float(texture.std()), 1e-9)
            # the optics blur outlines (σ 1 px): no pixel-sharp steps
            plane = ndimage.gaussian_filter(np.where(bodies, self.body, self.background), 1.0)
            plane = plane + np.where(bodies, texture, 0.0)
            image[self.body_channel] += plane[None]
        return {"count": n, "confluence_pct": 100 * float(bodies.mean()),
                "drift_px": [round(dy, 4), round(dx, 4)]}  # fmt: skip
