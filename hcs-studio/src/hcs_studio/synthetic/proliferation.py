"""Proliferation markers, micronuclei and multinucleated cells with known answers (Phase 20e).

Marker intensities per nucleus (log-normal, σ 0.35): positive ≈ 10× negative; nuclei of
secondary-only wells sit a little below the negatives (no primary antibody).

:func:`write_proliferation_run` writes a finished-run folder (nuclei with marker means and
``micronuclei_count``, cells with ``nuclei_count``); :class:`ProliferationCells` draws round
cells (cell stain), one or two nuclei each, Ki67 per cell and micronuclei next to the nuclei.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEVELS = {"ki67": (80.0, 800.0), "edu": (60.0, 1000.0), "ph3": (50.0, 1500.0)}
SECONDARY = 0.75  # secondary-only level, × the negative level
SIGMA = 0.35


def marker_values(
    fraction: float, n: int, rng: np.random.Generator, key: str, *, secondary: bool = False
) -> tuple[Any, Any]:
    """(intensities, positive flags) with exactly round(fraction · n) positives."""
    negative, positive = LEVELS[key]
    flags = np.zeros(n, dtype=bool)
    if not secondary:
        flags[rng.permutation(n)[: round(fraction * n)]] = True
    centre = np.where(flags, positive, negative * (SECONDARY if secondary else 1.0))
    return rng.lognormal(np.log(centre), SIGMA), flags


def write_proliferation_run(
    folder: Path,
    wells: Mapping[str, tuple[str, Mapping[str, float], int]],
    *,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """``wells``: well → (role, fractions, nuclei). Fractions: ``ki67``/``edu``/``ph3``
    (positive), ``mn`` (nuclei with one micronucleus), ``multi`` (binucleated cells). Role
    ``unstained`` = secondary-only (all markers at the secondary level). Returns the truth."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.model.plate import parse_well_name

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    layout = PlateLayout.for_plate_type(96)
    truth: dict[str, dict[str, float]] = {}
    for well, (role, fractions, n) in wells.items():
        ids = np.arange(1, n + 1)
        nuclei = pd.DataFrame({"well": well, "field": 1, "timepoint": 0, "object_type": "nuclei",
                               "object_id": ids, "area_um2": rng.normal(90, 8, n),
                               "dapi_mean": rng.normal(500, 30, n)})  # fmt: skip
        expected: dict[str, float] = {}
        for key in LEVELS:
            values, flags = marker_values(fractions.get(key, 0.0), n, rng, key,
                                          secondary=role == "unstained")  # fmt: skip
            nuclei[f"{key}_mean"] = values
            expected[f"{key}_pos_pct"] = 100 * float(flags.mean())
        mn = np.zeros(n, dtype=int)
        mn[rng.permutation(n)[: round(fractions.get("mn", 0.0) * n)]] = 1
        nuclei["micronuclei_count"] = np.where(mn > 0, mn, np.nan)  # absent = none
        expected["mn_pct"] = 100 * float((mn > 0).mean())
        # cells: binucleated ones take two consecutive nuclei
        binucleated = round(fractions.get("multi", 0.0) * n / 2)
        counts = [2] * binucleated + [1] * (n - 2 * binucleated)
        cells = pd.DataFrame({"well": well, "field": 1, "timepoint": 0, "object_type": "cells",
                              "object_id": np.arange(1, len(counts) + 1),
                              "area_um2": 300.0, "nuclei_count": counts})  # fmt: skip
        expected["multinucleated_pct"] = 100 * binucleated / len(counts)
        objects = pd.concat([nuclei, cells], ignore_index=True)
        (folder / "wells" / well).mkdir(parents=True, exist_ok=True)
        objects.to_parquet(folder / "wells" / well / "objects.parquet", index=False)
        (folder / "wells" / well / "well_summary.json").write_text(json.dumps({"well": well}))
        layout = set_role(layout, [parse_well_name(well)], role, cell_line="HeLa")
        truth[well] = expected
    save_layout(layout, folder / "plate_layout.json")
    manifest = {"plate": {"name": "ProlifSynth", "barcode": "PF-SYNTH", "rows": 8, "cols": 12,
                          "metadata": {}},
                "pipeline": {"name": "synthetic", "steps": []}, "status": "completed"}  # fmt: skip
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return truth


@dataclass
class ProliferationCells:
    """Round cells on a grid (channel 2: cell stain), each with one nucleus (radius 7 px) or —
    for a share ``fractions['multi']`` — two (radius 6 px, 16 px apart); Ki67 (channel 1) per
    cell; a share ``fractions['mn']`` of cells has one micronucleus (radius 1.5–2.5 px, ≥ 3 px
    from the nearest nucleus edge, inside the cell)."""

    fractions: Callable[[int, int], Mapping[str, float]]
    name: str = "proliferation"
    grid: int = 6
    cell_radius_px: float = 16.0
    dna_intensity: float = 600.0
    cell_intensity: float = 300.0

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        fractions = self.fractions(ctx.row, ctx.col)
        n = self.grid**2
        rng = ctx.rng
        ki67, positive = marker_values(fractions.get("ki67", 0.0), n, rng, "ki67")
        multi = rng.random(n) < fractions.get("multi", 0.0)
        micronucleus = rng.random(n) < fractions.get("mn", 0.0)
        pitch = min(h, w) / (self.grid + 0.4)
        yy, xx = np.mgrid[0:h, 0:w]
        cells = []
        for i in range(n):
            cy = pitch * (0.7 + i // self.grid) + rng.uniform(-1, 1)
            cx = pitch * (0.7 + i % self.grid) + rng.uniform(-1, 1)
            cell = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.cell_radius_px**2
            image[2, :, cell] += self.cell_intensity
            centres = [(cy, cx - 8), (cy, cx + 8)] if multi[i] else [(cy, cx)]
            radius = 6.0 if multi[i] else 7.0
            for y, x in centres:
                disc = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
                image[0, :, disc] += self.dna_intensity
                image[1, :, disc] += ki67[i]
            mn_radius = 0.0
            if micronucleus[i]:
                mn_radius = float(rng.uniform(1.5, 2.5))
                if multi[i]:
                    y, x = cy - 10, cx
                else:
                    angle = rng.uniform(0, 2 * np.pi)
                    distance = radius + 3 + mn_radius
                    y, x = cy + distance * np.sin(angle), cx + distance * np.cos(angle)
                disc = (yy - y) ** 2 + (xx - x) ** 2 <= mn_radius**2
                image[0, :, disc] += self.dna_intensity
            cells.append({"centre_px": [round(cy, 1), round(cx, 1)], "nuclei": len(centres),
                          "ki67": bool(positive[i]), "micronuclei": int(micronucleus[i]),
                          "micronucleus_radius_px": round(mn_radius, 2)})  # fmt: skip
        return {"count": n, "cells": cells}
