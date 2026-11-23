"""A synthetic primary screen with known answers (Phase 23).

``write_screen(root, ...)``: ``libraries`` library plates screened in ``replicates`` replicate
assay plates each (one run folder per assay plate: per-well summary, layout, manifest), 384
wells: vehicle in columns 1–2, positive control in columns 23–24, one library compound per
other well (single concentration). Two readouts:

- ``reporter_mean``: baseline 1000, noise ``noise_pct`` %, a positional artifact that differs
  per plate (a row gradient of ``row_artifact_pct`` % and an edge offset of ``edge_artifact_pct``
  %, on the samples as on the controls), seeded hits lowering it by ``hit_effect_pct`` %; the
  positive control lowers it by 90 %;
- ``nuclei_count``: baseline 800, noise 6 %; cytotoxic compounds lower both readouts (count by
  70 %).

Some compounds are autofluorescent: image QC flags their wells (``qc_flags``
"autofluorescence"), and their reporter reads high or low. Truth (``truth.json``): the seeded
hits, the cytotoxic and the autofluorescent compounds (all as compound names).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_screen(
    root: Path,
    *,
    libraries: int = 4,
    replicates: int = 2,
    hits: int = 10,
    cytotoxic: int = 4,
    autofluorescent: int = 3,
    hit_effect_pct: float = 35.0,
    noise_pct: float = 6.0,
    row_artifact_pct: float = 30.0,
    edge_artifact_pct: float = -25.0,
    seed: int = 0,
) -> tuple[list[Path], dict[str, Any]]:
    from hcs_studio.model.plate import well_name
    from hcs_studio.normalization.plate import WellInfo
    from hcs_studio.synthetic import write_values_run

    rng = np.random.default_rng(seed)
    rows, cols = 16, 24
    root = Path(root)
    folders: list[Path] = []
    truth: dict[str, Any] = {"hits": [], "cytotoxic": [], "autofluorescent": []}
    start = datetime(2026, 2, 2, 8, 0, tzinfo=UTC)
    sample_cells = [(r, c) for r in range(rows) for c in range(2, cols - 2)]
    for lib in range(libraries):
        names = {cell: f"L{lib + 1:02d}-{well_name(*cell)}" for cell in sample_cells}
        order = rng.permutation(len(sample_cells))
        kinds: dict[tuple[int, int], str] = {}
        for k, n in (("hit", hits), ("cytotoxic", cytotoxic), ("autofluorescent", autofluorescent)):
            for i in order[:n]:
                kinds[sample_cells[int(i)]] = k
            order = order[n:]
        for cell, kind in kinds.items():
            truth[{"hit": "hits"}.get(kind, kind)].append(names[cell])
        bright = {
            cell: rng.choice([-1.0, 1.0])
            for cell, kind in kinds.items()
            if kind == "autofluorescent"
        }
        for rep in range(replicates):
            barcode = f"SCR{lib + 1:02d}-R{rep + 1}"
            r_idx, c_idx = np.mgrid[0:rows, 0:cols]
            tilt = rng.uniform(0.6, 1.0) * (1 if rng.random() < 0.5 else -1)
            artifact = tilt * row_artifact_pct / 100 * (r_idx / (rows - 1) - 0.5)
            edge = (r_idx == 0) | (r_idx == rows - 1) | (c_idx == 0) | (c_idx == cols - 1)
            artifact = artifact + edge_artifact_pct / 100 * edge
            reporter, count, flags, wells = {}, {}, {}, []
            for r in range(rows):
                for c in range(cols):
                    w = well_name(r, c)
                    effect, toxic, flag = 0.0, 0.0, ""
                    compounds: tuple[tuple[str, float], ...]
                    if c < 2:
                        role, compounds = "negative_control", ()
                    elif c >= cols - 2:
                        role, compounds, effect = "positive_control", (), -0.9
                    else:
                        role, compounds = "sample", ((names[(r, c)], 1e-5),)
                        kind = kinds.get((r, c), "")
                        if kind == "hit":
                            effect = -hit_effect_pct / 100
                        elif kind == "cytotoxic":
                            effect, toxic = -0.6, -0.7
                        elif kind == "autofluorescent":
                            effect, flag = 0.6 * bright[(r, c)], "autofluorescence"
                    noise = rng.normal(0, noise_pct / 100)
                    reporter[w] = 1000 * (1 + artifact[r, c] + effect + noise)
                    count[w] = 800 * (1 + toxic + rng.normal(0, 0.06))
                    flags[w] = flag
                    wells.append(WellInfo(w, role, compounds))
            folder = write_values_run(
                root / f"{barcode}__screen",
                reporter,
                wells,
                barcode=barcode,
                readout="reporter_mean",
                started=start + timedelta(hours=lib * replicates + rep),
            )
            table = pd.DataFrame(
                {
                    "well": list(reporter),
                    "reporter_mean": list(reporter.values()),
                    "nuclei_count": [count[w] for w in reporter],
                    "qc_failed": [bool(flags[w]) for w in reporter],
                    "qc_flags": [flags[w] for w in reporter],
                    "qc_reasons": [flags[w] and "background raised" for w in reporter],
                    "low_count": [count[w] < 0.2 * 800 for w in reporter],
                }
            )
            table.to_csv(folder / "tables" / "per_well_summary.csv", index=False)
            folders.append(folder)
    for key in truth:
        truth[key] = sorted(truth[key])
    (root / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return folders, truth
