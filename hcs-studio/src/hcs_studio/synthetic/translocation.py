"""Cells with a known nuclear/cytoplasmic ratio of a reporter (Phase 20d).

Resting cells keep the reporter in the cytoplasm (log2 N/C ≈ ``resting_log2``), responding
cells in the nucleus (log2 N/C ≈ ``responding_log2``), each with a cell-to-cell spread.

:func:`write_translocation_run` writes a finished-run folder (nuclei and cytoplasm rows per
well, layout with roles and treatments); :class:`TranslocationCells` draws cells into images:
nucleus (DNA channel 0 and reporter channel 1 at N/C × the cytoplasm level) inside a round cell
of uniform cytoplasm, with the per-cell N/C ratio in the truth.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESTING_LOG2 = -1.0
RESPONDING_LOG2 = 1.0
SPREAD = 0.25


def hill(concentration: float, ec50: float, hill_slope: float = 1.0, top: float = 1.0) -> float:
    """Fraction responding at a concentration (0 at 0)."""
    if concentration <= 0:
        return 0.0
    return float(top / (1 + (ec50 / concentration) ** hill_slope))


def sample_ratios(fraction: float, n: int, rng: np.random.Generator) -> tuple[Any, Any]:
    """(log2 N/C per cell, responding flags) with exactly round(fraction · n) responders."""
    k = round(fraction * n)
    responding = np.zeros(n, dtype=bool)
    responding[rng.permutation(n)[:k]] = True
    centre = np.where(responding, RESPONDING_LOG2, RESTING_LOG2)
    return rng.normal(centre, SPREAD), responding


def write_translocation_run(
    folder: Path,
    wells: Mapping[str, tuple[str, float, int, tuple[str, float] | None]],
    *,
    seed: int = 0,
    reporter: str = "p65",
    cytoplasm_level: float = 400.0,
    cells: bool = False,
) -> dict[str, dict[str, float]]:
    """A finished-run folder: ``wells`` maps a well to (role, responding fraction, cells,
    (compound, concentration in M) or None). With ``cells`` the cytoplasm is on 'cells' rows
    as measure_objects writes it for cells grown from nuclei (``<reporter>_cytoplasm_mean``,
    cell = nucleus + cytoplasm). Returns per well the true % responding, median
    N/C and mean log2 N/C."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.model.plate import parse_well_name

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    layout = PlateLayout.for_plate_type(96)
    truth = {}
    column = f"{reporter.lower()}_mean"
    for well, (role, fraction, n, treatment) in wells.items():
        log2, responding = sample_ratios(fraction, n, rng)
        cyto = rng.normal(cytoplasm_level, 0.05 * cytoplasm_level, n).clip(1)
        nuc = cyto * 2.0**log2
        ids = np.arange(1, n + 1)
        base = {"well": well, "field": 1, "timepoint": 0, "object_id": ids}
        nuclei = pd.DataFrame({**base, "object_type": "nuclei", "area_um2": rng.normal(90, 8, n),
                               "dapi_mean": rng.normal(500, 30, n), column: nuc})  # fmt: skip
        if cells:
            cytoplasm = pd.DataFrame({**base, "object_type": "cells", "area_um2": 150.0,
                                      column: (90 * nuc + 60 * cyto) / 150,
                                      f"{reporter.lower()}_cytoplasm_mean": cyto})  # fmt: skip
        else:
            cytoplasm = pd.DataFrame({**base, "object_type": "cytoplasm",
                                      "area_um2": rng.normal(60, 6, n), column: cyto})  # fmt: skip
        objects = pd.concat([nuclei, cytoplasm], ignore_index=True)
        (folder / "wells" / well).mkdir(parents=True, exist_ok=True)
        objects.to_parquet(folder / "wells" / well / "objects.parquet", index=False)
        (folder / "wells" / well / "well_summary.json").write_text(
            json.dumps({"well": well}), encoding="utf-8"
        )
        key = parse_well_name(well)
        layout = set_role(layout, [key], role, cell_line="HeLa")
        if treatment is not None:
            compound, concentration = treatment
            layout = apply_fields(
                layout, [key], treatments=(Treatment(compound, concentration, "µM"),)
            )
        ratio = nuc / cyto
        truth[well] = {
            "responding_pct": 100 * float(responding.mean()),
            "nc_ratio": float(np.median(ratio)),
            "log2_nc": float(np.mean(np.log2(ratio))),
        }
    save_layout(layout, folder / "plate_layout.json")
    manifest = {"plate": {"name": "TranslocationSynth", "barcode": "TL-SYNTH", "rows": 8,
                          "cols": 12, "metadata": {}},
                "pipeline": {"name": "synthetic", "steps": []}, "status": "completed"}  # fmt: skip
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return truth


@dataclass
class TranslocationCells:
    """Round cells on a grid: share ``fraction(row, col)`` of them responding."""

    fraction: Callable[[int, int], float]
    name: str = "translocation"
    nucleus_radius_px: float = 8.0
    cell_radius_px: float = 15.0
    grid: int = 7
    dna_intensity: float = 600.0
    cytoplasm_level: float = 400.0

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        n = self.grid**2
        log2, responding = sample_ratios(self.fraction(ctx.row, ctx.col), n, ctx.rng)
        pitch = min(h, w) / (self.grid + 0.4)
        yy, xx = np.mgrid[0:h, 0:w]
        cells = []
        for i in range(n):
            cy = pitch * (0.7 + i // self.grid) + ctx.rng.uniform(-1, 1)
            cx = pitch * (0.7 + i % self.grid) + ctx.rng.uniform(-1, 1)
            distance = (yy - cy) ** 2 + (xx - cx) ** 2
            nucleus = distance <= self.nucleus_radius_px**2
            cell = (distance <= self.cell_radius_px**2) & ~nucleus
            image[0, :, nucleus] += self.dna_intensity
            image[1, :, cell] += self.cytoplasm_level
            image[1, :, nucleus] += self.cytoplasm_level * 2.0 ** log2[i]
            ratio = float(2.0 ** log2[i])
            cells.append({"centre_px": [round(cy, 1), round(cx, 1)], "nc_ratio": ratio,
                          "responding": bool(responding[i])})  # fmt: skip
        return {"count": n, "cells": cells}
