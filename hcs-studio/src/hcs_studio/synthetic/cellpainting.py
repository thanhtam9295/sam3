"""Synthetic Cell Painting fields with known phenotypes (Phase 24).

``PaintedCells`` draws cells on a jittered grid in five channels — DNA (0, nucleus), ER (1,
perinuclear cytoplasm, fading outward), RNA (2, nucleoli in the nucleus + faint cytoplasm), AGP
(3, cell edge + faint cytoplasm) and Mito (4, punctate cytoplasm) — so that nuclei, cells and
cytoplasm can be segmented and measured. ``spec(row, col)`` gives the well's phenotype as
multipliers of the vehicle cell (all 1.0 by default):

- ``nucleus_size``: nuclear radius; ``er``: ER intensity; ``agp``: edge (actin) intensity;
- ``mito_spots``: number of mitochondrial puncta (fewer, larger spots below 1: fragmentation
  read as texture and granularity); ``rna``: nucleolar RNA intensity.

The truth per field: the phenotype and the number of cells.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PHENOTYPE_KEYS = ("nucleus_size", "er", "agp", "mito_spots", "rna")


@dataclass
class PaintedCells:
    spec: Callable[[int, int], Mapping[str, float]]
    name: str = "painted"
    cells_per_side: int = 5
    nucleus_px: float = 7.0
    cell_px: float = 15.0
    seed: int = 0

    def draw(self, ctx: Any) -> dict[str, Any]:
        from scipy import ndimage

        phenotype = {k: float(self.spec(ctx.row, ctx.col).get(k, 1.0)) for k in PHENOTYPE_KEYS}
        h, w = ctx.image.shape[-2:]
        rng = np.random.default_rng([self.seed, ctx.row, ctx.col, ctx.field])
        pitch = min(h, w) / self.cells_per_side
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        dna = np.zeros((h, w))
        er = np.zeros((h, w))
        rna = np.zeros((h, w))
        agp = np.zeros((h, w))
        mito = np.zeros((h, w))
        nucleus_r = self.nucleus_px * phenotype["nucleus_size"]
        count = 0
        for i in range(self.cells_per_side):
            for j in range(self.cells_per_side):
                cy = (i + 0.5) * pitch + rng.uniform(-1.5, 1.5)
                cx = (j + 0.5) * pitch + rng.uniform(-1.5, 1.5)
                d = np.hypot(yy - cy, xx - cx)
                nucleus = d <= nucleus_r
                cell = d <= self.cell_px
                cyto = cell & ~nucleus
                dna[nucleus] += 1000
                er[cyto] += 600 * phenotype["er"] * np.exp(-(d[cyto] - nucleus_r) / 6)
                for _ in range(2):  # nucleoli
                    oy, ox = rng.normal(0, nucleus_r / 3, 2)
                    rna[(np.hypot(yy - cy - oy, xx - cx - ox) <= 1.8) & nucleus] += (
                        700 * phenotype["rna"]
                    )
                rna[cyto] += 120
                ring = cell & (d > self.cell_px - 2.5)
                agp[ring] += 800 * phenotype["agp"]
                agp[cyto & ~ring] += 150
                spots = max(round(30 * phenotype["mito_spots"]), 1)
                radius = 1.2 / np.sqrt(min(phenotype["mito_spots"], 1.0))
                for _ in range(spots):
                    angle, r = (
                        rng.uniform(0, 2 * np.pi),
                        rng.uniform(nucleus_r + 1, self.cell_px - 2),
                    )
                    sy, sx = cy + r * np.sin(angle), cx + r * np.cos(angle)
                    mito[(np.hypot(yy - sy, xx - sx) <= radius) & cyto] += 900
                count += 1
        for c, plane in enumerate((dna, er, rna, agp, mito)):
            ctx.image[c, :] += 100 + ndimage.gaussian_filter(plane, 0.8)[None]
        return {"count": count, "phenotype": phenotype}
