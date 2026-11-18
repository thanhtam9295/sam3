"""A synthetic 384-well pharmacology plate with known answers (tests, demos, timing).

Layout (rows A–P, columns 1–24):

- rows A–L: MCF7; rows M–N: HeLa; rows O–P: MCF7 dose–response
- column 1: vehicle (DMSO 0.1 %), column 24: positive control (full kill)
- rows A–F: Drug A × Drug B, 6 × 6 (0 included), three replicate matrices (columns 2–7,
  9–14, 16–21): synergistic by construction
- rows G–L: Drug A × Drug C, 6 × 6, three replicates: exactly Bliss-additive
- rows M–N (HeLa) and O–P (MCF7): Drug D, 10 points (1:3 from 10 µM) in duplicate,
  columns 2–11; HeLa is 10× less sensitive than MCF7
- rows M–N, columns 12–21: Drug A on HeLa, 10 points in duplicate

The readout ``nuclei_count`` = 50 + 950 · (1 − I/100) (+ noise), so % inhibition recovers
the true inhibition I; ``TRUTH`` holds the single-agent parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.layout_tools import combination_matrix, dilution_series, set_role
from hcs_studio.model.plate import well_name

#: (IC50 in M, Hill) of each drug as % inhibition 0 → 100, per cell line.
TRUTH: dict[tuple[str, str], tuple[float, float]] = {
    ("MCF7", "Drug A"): (1e-6, 1.2),
    ("MCF7", "Drug B"): (3e-7, 1.5),
    ("MCF7", "Drug C"): (2e-6, 1.0),
    ("MCF7", "Drug D"): (1e-7, 1.0),
    ("HeLa", "Drug D"): (1e-6, 1.0),
    ("HeLa", "Drug A"): (2e-6, 1.3),
}
SYNERGY = 90.0  # extra inhibition of A × B where both drugs are half active (% points)


def inhibition(cell_line: str, drug: str, conc: float) -> float:
    ic50, hill = TRUTH[(cell_line, drug)]
    if conc <= 0:
        return 0.0
    return float(100.0 * conc**hill / (conc**hill + ic50**hill))


def true_response(cell_line: str, treatments: list[tuple[str, float]]) -> float:
    """True % inhibition of a well's active treatments."""
    effects = [inhibition(cell_line, d, c) / 100 for d, c in treatments]
    combined = 1.0
    for e in effects:
        combined *= 1 - e
    bliss = 100 * (1 - combined)
    names = sorted(d for d, _ in treatments)
    if names == ["Drug A", "Drug B"]:
        ea, eb = effects if treatments[0][0] == "Drug A" else effects[::-1]
        # a bump that is largest where both drugs are half active
        bliss += SYNERGY * 16 * ea * (1 - ea) * eb * (1 - eb)
    return float(min(100.0, bliss))


def layout() -> PlateLayout:
    plate = PlateLayout.for_plate_type(384).updated({}, assay="synthetic combination plate")
    lines = {r: ("HeLa" if r in (12, 13) else "MCF7") for r in range(16)}
    for r in range(16):
        plate = set_role(
            plate,
            [(r, 0)],
            "negative_control",
            vehicle="DMSO",
            vehicle_percent=0.1,
            cell_line=lines[r],
        )
        plate = set_role(plate, [(r, 23)], "positive_control", cell_line=lines[r])
    for origin_row, drug_b, top_b in ((0, "Drug B", 3.0), (6, "Drug C", 30.0)):
        plate = combination_matrix(
            plate,
            (origin_row, 1),
            drug_a="Drug A",
            top_a=10.0,
            unit_a="µM",
            drug_b=drug_b,
            top_b=top_b,
            unit_b="µM",
            factor_a=3,
            factor_b=3,
            replicates=3,
            replicate_direction="right",
            gap=1,
            cell_line="MCF7",
        )
    for row, line in ((12, "HeLa"), (14, "MCF7")):
        plate = dilution_series(
            plate, (row, 1), "Drug D", 10.0, "µM", factor=3, points=10, replicates=2, cell_line=line
        )
    return dilution_series(
        plate, (12, 11), "Drug A", 30.0, "µM", factor=3, points=10, replicates=2, cell_line="HeLa"
    )


def summary(plate: PlateLayout, noise: float = 15.0, seed: int = 0) -> pd.DataFrame:
    """``per_well_summary``-like readouts: nuclei_count and a rising % positive readout."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, a in sorted(plate.wells.items()):
        if a.role == "positive_control":
            value = 50.0
        elif a.role == "negative_control":
            value = 1000.0
        else:
            active = [(t.compound, t.concentration) for t in a.active_treatments]
            value = 50 + 950 * (1 - true_response(a.cell_line, active) / 100)
        count = max(0.0, value + rng.normal(0, noise))
        rows.append(
            {
                "well": well_name(*key),
                "nuclei_count": count,
                "pct_caspase_positive": 2 + 0.6 * (1000 - count) / 9.5,
            }
        )
    return pd.DataFrame(rows).set_index("well")


def write_run(folder: Path, noise: float = 15.0, seed: int = 0, name: str = "SynthPlate") -> Path:
    """A minimal finished-run folder (manifest, per-well summary, layout) for the
    pharmacology workflow."""
    import json

    from hcs_studio.io.layout_io import save_layout

    folder.mkdir(parents=True, exist_ok=True)
    plate = layout()
    (folder / "tables").mkdir(exist_ok=True)
    summary(plate, noise, seed).to_csv(folder / "tables" / "per_well_summary.csv")
    save_layout(plate, folder / "plate_layout.json")
    manifest: dict[str, Any] = {
        "plate": {"name": name, "rows": 16, "cols": 24},
        "pipeline": {"name": "synthetic"},
        "status": "completed",
    }
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


# ------------------------------------------------------------- image plate


N_VEHICLE = 48  # nuclei per field in vehicle wells
N_POSITIVE = 2  # nuclei per field in positive-control wells


def nuclei_count(cell_line: str, role: str, treatments: list[tuple[str, float]]) -> int:
    """True nuclei count of a well: vehicle 48, positive control 2, treated wells
    50 − 48·I/100 … (counts follow the true % inhibition I between the controls)."""
    if role == "positive_control":
        return N_POSITIVE
    if role == "negative_control" or not treatments:
        return N_VEHICLE
    inhibition = true_response(cell_line, treatments)
    return round(N_VEHICLE - (N_VEHICLE - N_POSITIVE) * inhibition / 100)


def write_image_plate(root: Path, size: int = 96, seed: int = 0) -> tuple[Path, Path]:
    """An OME-Zarr 384-well plate (one 2-channel field per well, 0.65 µm pixels) whose
    nuclei counts follow the known dose–response and combination truth of ``layout()``,
    and the layout as JSON. Nuclei are discs of radius 3.5 px on a jittered 7 × 7 grid
    (at least 3 px apart, so counts are exact)."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.utils.synthetic import write_ome_zarr_plate

    plate = layout()
    yy, xx = np.mgrid[0:size, 0:size]
    pitch = size / 7.4
    sites = [(pitch * (0.7 + i), pitch * (0.7 + j)) for i in range(7) for j in range(7)]

    def field(row: int, col: int, f: int) -> np.ndarray:
        rng = np.random.default_rng(seed + row * 24 + col)
        a = plate.get((row, col))
        drugs = [(t.compound, t.concentration) for t in a.active_treatments if t.kind != "percent"]
        n = nuclei_count(a.cell_line, a.role, drugs if a.role == "sample" else [])
        data = np.full((2, 1, size, size), 100.0)
        for k in rng.permutation(len(sites))[:n]:
            y, x = sites[k] + rng.uniform(-1.0, 1.0, 2)
            disc = (yy - y) ** 2 + (xx - x) ** 2 <= 12.25  # radius 3.5 px: gaps ≥ 3 px
            data[0, 0][disc] += 1200
            data[1, 0][disc] += 300 + 600 * (1 - n / N_VEHICLE)  # a stress marker
        data += rng.normal(0, 8, data.shape)
        return np.asarray(np.clip(data, 0, 65535), dtype=np.uint16)

    wells = [(r, c) for r in range(16) for c in range(24) if (r, c) in plate.wells]
    path = write_ome_zarr_plate(
        root / "plate.zarr",
        wells,
        field,
        channels=("DAPI", "Stress"),
        pixel_size_um=0.65,
        z_step_um=1.0,
        name="RegressionPlate",
    )
    layout_file = root / "layout.json"
    save_layout(plate, layout_file)
    return path, layout_file
