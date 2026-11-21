"""Apoptosis data with known phenotypes (Phase 20c).

Phenotypes (relative to a healthy nucleus: area 1, DNA brightness 1):

==============  ========  ===========  =========  =====  =======================
phenotype       area      DNA mean     caspase    PI     expected category
==============  ========  ===========  =========  =====  =======================
viable          1.0       1.0          low        low    viable
early           0.9       1.1          high       low    early_apoptotic
condensed       0.45      1.9          low        low    early_apoptotic (morphology)
late            0.45      1.9          high       high   late_apoptotic
necrotic        1.2       0.9          low        high   necrotic
==============  ========  ===========  =========  =====  =======================

Fragmented nuclei add ``fragments`` small bright objects each (area 0.12), which are not cells.
:func:`write_apoptosis_run` writes a finished-run folder (objects per well, layout with roles and
time points); :class:`ApoptosisNuclei` draws the same phenotypes into images.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHENOTYPES: dict[str, dict[str, float]] = {
    "viable": {"area": 1.0, "dna": 1.0, "casp": 0, "pi": 0},
    "early": {"area": 0.9, "dna": 1.1, "casp": 1, "pi": 0},
    "condensed": {"area": 0.45, "dna": 1.9, "casp": 0, "pi": 0},
    "late": {"area": 0.45, "dna": 1.9, "casp": 1, "pi": 1},
    "necrotic": {"area": 1.2, "dna": 0.9, "casp": 0, "pi": 1},
}
CATEGORY = {"viable": "viable", "early": "early_apoptotic", "condensed": "early_apoptotic",
            "late": "late_apoptotic", "necrotic": "necrotic"}  # fmt: skip


def sample_cells(
    fractions: Mapping[str, float], n: int, rng: np.random.Generator, *, fragmented: float = 0.0
) -> pd.DataFrame:
    """``n`` nuclei with exact phenotype counts (fractions in %), plus 3 fragments for each of
    ``fragmented`` (share of the condensed + late nuclei)."""
    total = sum(fractions.values())
    counts = {k: round(n * fractions.get(k, 0.0) / total) for k in PHENOTYPES}
    counts["viable"] += n - sum(counts.values())
    rows = []
    for phenotype, k in counts.items():
        spec = PHENOTYPES[phenotype]
        rows.append(
            pd.DataFrame(
                {
                    "phenotype": phenotype,
                    "area_um2": rng.normal(90 * spec["area"], 7 * spec["area"], k),
                    "dapi_mean": rng.normal(500 * spec["dna"], 35, k),
                    "casp_mean": rng.lognormal(np.log(900 if spec["casp"] else 60), 0.3, k),
                    "pi_mean": rng.lognormal(np.log(1200 if spec["pi"] else 40), 0.3, k),
                }
            )
        )
    cells = pd.concat(rows, ignore_index=True)
    pieces = round(fragmented * (counts["condensed"] + counts["late"]))
    if pieces:
        fragments = pd.DataFrame(
            {
                "phenotype": "fragment",
                "area_um2": rng.normal(11, 2, 3 * pieces).clip(4, 18),
                "dapi_mean": rng.normal(1100, 80, 3 * pieces),
                "casp_mean": rng.lognormal(np.log(900), 0.3, 3 * pieces),
                "pi_mean": rng.lognormal(np.log(40), 0.3, 3 * pieces),
            }
        )
        cells = pd.concat([cells, fragments], ignore_index=True)
    return cells.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def expected(cells: pd.DataFrame) -> dict[str, float]:
    """The categories (%) the assay should report for these cells (fragments apart)."""
    real = cells[cells["phenotype"] != "fragment"]
    category = real["phenotype"].map(CATEGORY)
    names = ("viable", "early_apoptotic", "late_apoptotic", "necrotic")
    out = {c: 100 * float((category == c).mean()) for c in names}
    out["viable_count"] = float((category == "viable").sum())
    out["fragments"] = float((cells["phenotype"] == "fragment").sum())
    return out


def write_apoptosis_run(
    folder: Path,
    wells: Mapping[str, tuple[str, Mapping[str, float], int, float | None]],
    *,
    seed: int = 0,
    fragmented: float = 0.0,
    annexin_on_cells: bool = False,
) -> dict[str, dict[str, float]]:
    """A finished-run folder: ``wells`` maps a well to (role, phenotype %, nuclei, time point h).
    With ``annexin_on_cells`` the apoptosis marker is measured on 'cells' rows (same IDs) and
    called ``annexin_mean``. Returns the expected categories per well."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.model.plate import parse_well_name

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    layout = PlateLayout.for_plate_type(96)
    truth = {}
    for well, (role, fractions, n, time_point) in wells.items():
        cells = sample_cells(fractions, n, rng, fragmented=fragmented)
        ids = np.arange(1, len(cells) + 1)
        nuclei = pd.DataFrame({"well": well, "field": 1, "timepoint": 0, "object_type": "nuclei",
                               "object_id": ids, "area_um2": cells["area_um2"],
                               "dapi_mean": cells["dapi_mean"],
                               "dapi_p95": cells["dapi_mean"] * 1.1,
                               "pi_mean": cells["pi_mean"]})  # fmt: skip
        parts = [nuclei]
        if annexin_on_cells:
            parts.append(pd.DataFrame({"well": well, "field": 1, "timepoint": 0,
                                       "object_type": "cells", "object_id": ids,
                                       "area_um2": cells["area_um2"] * 3,
                                       "annexin_mean": cells["casp_mean"]}))  # fmt: skip
        else:
            nuclei["casp_mean"] = cells["casp_mean"]
        objects = pd.concat(parts, ignore_index=True)
        (folder / "wells" / well).mkdir(parents=True, exist_ok=True)
        objects.to_parquet(folder / "wells" / well / "objects.parquet", index=False)
        (folder / "wells" / well / "well_summary.json").write_text(json.dumps({"well": well}))
        key = parse_well_name(well)
        layout = set_role(layout, [key], role, cell_line="HeLa")
        if time_point is not None:
            layout = apply_fields(layout, [key], time_point_h=time_point)
        truth[well] = expected(cells)
    save_layout(layout, folder / "plate_layout.json")
    manifest = {"plate": {"name": "ApoSynth", "metadata": {}},
                "pipeline": {"name": "synthetic", "steps": []}, "status": "completed"}  # fmt: skip
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return truth


@dataclass
class ApoptosisNuclei:
    """Nuclei on a grid with phenotypes drawn from ``fractions(row, col)`` (%): DNA (channel 0)
    as a disc of the phenotype's size and brightness; caspase (1) and PI (2) inside the nucleus."""

    fractions: Callable[[int, int], Mapping[str, float]]
    name: str = "apoptosis"
    radius_px: float = 9.0
    grid: int = 7

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        cells = sample_cells(self.fractions(ctx.row, ctx.col), self.grid**2, ctx.rng)
        pitch = min(h, w) / (self.grid + 0.4)
        yy, xx = np.mgrid[0:h, 0:w]
        drawn = []
        for i, row in enumerate(cells.to_dict("records")):
            spec = PHENOTYPES[str(row["phenotype"])]
            cy = pitch * (0.7 + i // self.grid) + ctx.rng.uniform(-1, 1)
            cx = pitch * (0.7 + i % self.grid) + ctx.rng.uniform(-1, 1)
            radius = self.radius_px * np.sqrt(spec["area"])
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            image[0, :, disc] += 500 * spec["dna"]
            image[1, :, disc] += float(row["casp_mean"])
            if image.shape[0] > 2:
                image[2, :, disc] += float(row["pi_mean"])
            drawn.append(str(row["phenotype"]))
        return {"count": len(drawn), "phenotypes": {p: drawn.count(p) for p in PHENOTYPES}}
