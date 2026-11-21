"""Cell-cycle data with known phase fractions (Phase 20a).

:func:`sample_nuclei` draws DNA contents (in N; G1 = 2N) for given phase fractions: sub-G1
uniform 0.6–1.35N, G1 normal at 2N, S uniform between the peaks broadened by the same CV,
G2/M normal at 4N, > 4N normal at 8N (truncated above 5.5N). EdU is high in S phase only;
pH3 is high in a given share of G2/M nuclei (mitotic). :func:`write_cycle_run` writes a
finished-run folder with per-well object tables and a layout (vehicle, positive control,
samples) for table-step tests; :class:`CycleNuclei` draws nuclei whose integrated intensity is
proportional to DNA content into a :class:`~hcs_studio.synthetic.SyntheticPlate`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHASES = ("sub_g1", "g1", "s", "g2m", "polyploid")
#: Typical proliferating vehicle and a G2/M-arrested well (% of nuclei).
VEHICLE = {"sub_g1": 2.0, "g1": 55.0, "s": 25.0, "g2m": 16.0, "polyploid": 2.0}
ARRESTED = {"sub_g1": 4.0, "g1": 20.0, "s": 14.0, "g2m": 54.0, "polyploid": 8.0}


def sample_nuclei(
    fractions: Mapping[str, float],
    n: int,
    rng: np.random.Generator,
    *,
    cv: float = 0.04,
    mitotic_share_of_g2m: float = 0.15,
) -> pd.DataFrame:
    """``n`` nuclei with exactly round(fraction × n) per phase: columns ``dna_n``, ``phase``,
    ``edu`` (a.u.), ``ph3`` (a.u.)."""
    total = sum(fractions.values())
    counts = {p: round(n * fractions.get(p, 0.0) / total) for p in PHASES}
    counts["g1"] += n - sum(counts.values())
    parts = []
    for phase, k in counts.items():
        if phase == "sub_g1":
            dna = rng.uniform(0.6, 1.35, k)
        elif phase == "g1":
            dna = rng.normal(2.0, cv * 2.0, k)
        elif phase == "s":
            t = rng.uniform(2.0, 4.0, k)
            dna = rng.normal(t, cv * t)
        elif phase == "g2m":
            dna = rng.normal(4.0, cv * 4.0, k)
        else:
            dna = np.clip(rng.normal(8.0, cv * 8.0, k), 5.5, None)
        edu = rng.lognormal(np.log(800 if phase == "s" else 60), 0.35, k)
        mitotic = (rng.random(k) < mitotic_share_of_g2m) if phase == "g2m" else np.zeros(k, bool)
        ph3 = (
            rng.lognormal(np.log(900), 0.3, k) * mitotic
            + rng.lognormal(np.log(50), 0.25, k) * ~mitotic
        )
        parts.append(
            pd.DataFrame({"dna_n": dna, "phase": phase, "edu": edu, "ph3": ph3, "mitotic": mitotic})
        )
    return pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=int(rng.integers(1 << 31))
    )


def write_cycle_run(
    folder: Path,
    wells: Mapping[str, tuple[str, Mapping[str, float]]],
    *,
    n: int = 4000,
    raw_per_n: float = 25_000.0,
    seed: int = 0,
    staining: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """A finished-run folder: ``wells`` maps a well to (role, phase fractions in %). Raw DNA
    intensity = DNA content × ``raw_per_n`` × the well's ``staining`` factor (default 1).
    Returns the truth (fractions, mitotic %, EdU+ % per well)."""
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.model.plate import parse_well_name

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    layout = PlateLayout.for_plate_type(96)
    truth: dict[str, Any] = {}
    for well, (role, fractions) in wells.items():
        nuclei = sample_nuclei(fractions, n, rng)
        factor = (staining or {}).get(well, 1.0)
        objects = pd.DataFrame(
            {
                "well": well,
                "field": 1,
                "timepoint": 0,
                "object_type": "nuclei",
                "object_id": np.arange(1, len(nuclei) + 1),
                "area_um2": rng.normal(90, 8, len(nuclei)),
                "dapi_integrated": nuclei["dna_n"].to_numpy() * raw_per_n * factor,
                "edu_mean": nuclei["edu"].to_numpy(),
                "ph3_mean": nuclei["ph3"].to_numpy(),
            }
        )
        (folder / "wells" / well).mkdir(parents=True, exist_ok=True)
        objects.to_parquet(folder / "wells" / well / "objects.parquet", index=False)
        layout = set_role(layout, [parse_well_name(well)], role, cell_line="HeLa")
        truth[well] = {
            "role": role,
            "fractions": {p: 100 * float((nuclei["phase"] == p).mean()) for p in PHASES},
            "mitotic_pct": 100 * float(nuclei["mitotic"].mean()),
            "edu_pct": 100 * float((nuclei["phase"] == "s").mean()),
        }
    save_layout(layout, folder / "plate_layout.json")
    manifest = {"plate": {"name": "CycleSynth", "metadata": {"modality": "widefield"}},
                "pipeline": {"name": "synthetic", "steps": []}, "status": "completed"}  # fmt: skip
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return truth


@dataclass
class CycleNuclei:
    """Nuclei on a jittered grid whose summed intensity is ``per_n`` × DNA content, in the DNA
    channel (0), with EdU (channel 1) bright in S phase. ``fractions(row, col)`` gives the
    well's phase fractions (%)."""

    fractions: Callable[[int, int], Mapping[str, float]]
    name: str = "cycle"
    per_n: float = 20_000.0
    radius_px: float = 6.0
    grid: int = 8
    cv: float = 0.04

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        k = self.grid**2
        nuclei = sample_nuclei(self.fractions(ctx.row, ctx.col), k, ctx.rng, cv=self.cv)
        pitch = min(h, w) / (self.grid + 0.4)
        yy, xx = np.mgrid[0:h, 0:w]
        phases = []
        records = nuclei.to_dict("records")
        for i, row in enumerate(records):
            cy = pitch * (0.7 + i // self.grid) + ctx.rng.uniform(-1, 1)
            cx = pitch * (0.7 + i % self.grid) + ctx.rng.uniform(-1, 1)
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.radius_px**2
            image[0, :, disc] += self.per_n * float(row["dna_n"]) / disc.sum()
            if image.shape[0] > 1:
                image[1, :, disc] += float(row["edu"])
            phases.append(str(row["phase"]))
        return {"count": k, "phases": {p: phases.count(p) for p in PHASES}}
