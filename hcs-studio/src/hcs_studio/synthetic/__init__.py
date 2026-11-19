"""Synthetic data with known answers for every phase (PROMPT_PLATFORM.md §2.9).

One entry point for test data:

- :class:`SyntheticPlate` builds an OME-Zarr plate from composable :class:`Feature` objects
  (nuclei with a known count per well, background, noise …). Each feature records what it
  drew, and ``truth.json`` next to the plate holds it, so a test compares results with the
  truth instead of hard-coded numbers. Later phases add features (image artifacts for QC,
  time courses, spheroids, screens with seeded hits, profiling phenotypes, cell-cycle DNA
  content, foci, N/C ratios).
- :func:`registry_runs` writes several finished runs of the known-answer dose–response plate
  (``pharmacology.synthetic``) with pharmacology results, for the experiment registry.
- The generators of earlier phases are re-exported unchanged (the regression baseline
  depends on their exact pixels): :func:`write_image_plate`, :func:`write_run`,
  :func:`write_ome_zarr_plate`, :func:`write_imagexpress_plate`, :func:`make_cell_field`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from hcs_studio.model.plate import well_name
from hcs_studio.pharmacology.synthetic import TRUTH, write_image_plate, write_run
from hcs_studio.utils.synthetic import (
    make_cell_field,
    write_imagexpress_plate,
    write_ome_zarr_plate,
)

__all__ = [
    "TRUTH",
    "Background",
    "Feature",
    "Nuclei",
    "ShotNoise",
    "SyntheticPlate",
    "make_cell_field",
    "read_truth",
    "registry_runs",
    "write_image_plate",
    "write_imagexpress_plate",
    "write_ome_zarr_plate",
    "write_run",
    "write_truth",
]

TRUTH_FILE = "truth.json"


def write_truth(folder: Path, truth: dict[str, Any]) -> Path:
    path = Path(folder) / TRUTH_FILE
    path.write_text(json.dumps(truth, indent=1, default=str), encoding="utf-8")
    return path


def read_truth(folder: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((Path(folder) / TRUTH_FILE).read_text(encoding="utf-8"))
    return data


@dataclass
class FieldContext:
    """What a feature draws into: one field of one well."""

    row: int
    col: int
    field: int
    image: NDArray[np.float64]  # (C, Z, Y, X), float while drawing
    rng: np.random.Generator
    pixel_size_um: float

    @property
    def well(self) -> str:
        return well_name(self.row, self.col)


class Feature(Protocol):
    """Draws into a field and reports its truth per well (and, if useful, per field)."""

    name: str

    def draw(self, ctx: FieldContext) -> dict[str, Any]: ...


@dataclass
class Background:
    name: str = "background"
    level: float = 100.0  # camera offset + background, all channels

    def draw(self, ctx: FieldContext) -> dict[str, Any]:
        ctx.image += self.level
        return {"level": self.level}


@dataclass
class Nuclei:
    """Discs (radius in px) on a jittered grid, at least 3 px apart: segmentation counts them
    exactly. ``count(row, col, field)`` gives the number per field (≤ ``grid``²)."""

    count: Callable[[int, int, int], int]
    name: str = "nuclei"
    channel: int = 0
    radius_px: float = 3.5
    intensity: float = 1500.0
    grid: int = 7

    def draw(self, ctx: FieldContext) -> dict[str, Any]:
        n = int(self.count(ctx.row, ctx.col, ctx.field))
        if not 0 <= n <= self.grid**2:
            raise ValueError(f"Nuclei per field must be 0–{self.grid**2}, got {n}")
        height, width = ctx.image.shape[-2:]
        pitch = min(height, width) / (self.grid + 0.4)
        sites = [
            (pitch * (0.7 + i), pitch * (0.7 + j))
            for i in range(self.grid)
            for j in range(self.grid)
        ]
        chosen: list[int] = (
            [int(k) for k in ctx.rng.choice(len(sites), size=n, replace=False)] if n else []
        )
        yy, xx = np.mgrid[0:height, 0:width]
        centres = []
        for k in chosen:
            cy, cx = sites[int(k)]
            cy += ctx.rng.uniform(-1, 1)
            cx += ctx.rng.uniform(-1, 1)
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.radius_px**2
            ctx.image[self.channel, :, disc] += self.intensity
            centres.append((round(float(cy), 2), round(float(cx), 2)))
        return {"count": n, "centres_px": centres}


@dataclass
class ShotNoise:
    name: str = "noise"
    sd: float = 5.0

    def draw(self, ctx: FieldContext) -> dict[str, Any]:
        ctx.image += ctx.rng.normal(0, self.sd, ctx.image.shape)
        return {"sd": self.sd}


@dataclass
class SyntheticPlate:
    wells: Sequence[tuple[int, int]]
    features: list[Any] = field(default_factory=list)
    channels: tuple[str, ...] = ("DAPI", "GFP")
    size: int = 96
    planes: int = 1
    n_fields: int = 1
    pixel_size_um: float = 0.65
    plate_size: tuple[int, int] = (16, 24)
    name: str = "SyntheticPlate"
    seed: int = 0

    def write(self, root: Path) -> tuple[Path, dict[str, Any]]:
        """Write ``<root>/plate.zarr`` and ``<root>/truth.json``; returns (plate, truth)."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        truth: dict[str, Any] = {
            "plate": self.name,
            "channels": list(self.channels),
            "pixel_size_um": self.pixel_size_um,
            "wells": {},
        }

        def image(row: int, col: int, f: int) -> NDArray[np.uint16]:
            rng = np.random.default_rng([self.seed, row, col, f])
            data = np.zeros((len(self.channels), self.planes, self.size, self.size))
            ctx = FieldContext(row, col, f, data, rng, self.pixel_size_um)
            per_field = {feature.name: feature.draw(ctx) for feature in self.features}
            well = truth["wells"].setdefault(ctx.well, {"fields": {}})
            well["fields"][str(f + 1)] = per_field
            return np.asarray(np.clip(data, 0, 65535), dtype=np.uint16)

        plate = write_ome_zarr_plate(
            root / "plate.zarr",
            list(self.wells),
            image,
            channels=self.channels,
            n_fields=self.n_fields,
            pixel_size_um=self.pixel_size_um,
            plate_size=self.plate_size,
            name=self.name,
        )
        for well in truth["wells"].values():  # per-well totals of countable features
            totals: dict[str, int] = {}
            for per_field in well["fields"].values():
                for name, values in per_field.items():
                    if isinstance(values, dict) and "count" in values:
                        totals[name] = totals.get(name, 0) + int(values["count"])
            well["counts"] = totals
        write_truth(root, truth)
        return plate, read_truth(root)  # as stored (JSON types)


# --------------------------------------------------------------- registry runs


def registry_runs(
    root: Path,
    n_runs: int = 3,
    *,
    noise: float = 8.0,
    start: datetime | None = None,
    pharmacology: bool = True,
) -> list[Path]:
    """``n_runs`` finished runs of the known-answer dose–response plate (one per day, plate
    barcodes ``REG-001`` …), each with a full manifest and, by default, pharmacology results
    (``pharmacology/dose_response_fits.csv`` …). ``truth.json`` in ``root`` lists the true
    EC50s (``pharmacology.synthetic.TRUTH``) and the runs."""
    root = Path(root)
    start = start or datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    runs = []
    for i in range(n_runs):
        barcode = f"REG-{i + 1:03d}"
        folder = write_run(root / f"{barcode}__regression__run", noise=noise, seed=i, name=barcode)
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        started = start + timedelta(days=i)
        manifest.update(
            {
                "software": {"name": "hcs-studio", "version": "synthetic"},
                "started": started.isoformat(),
                "finished": (started + timedelta(minutes=20)).isoformat(),
                "pipeline_hash": f"synthetic{i:07d}",
                "models": [],
            }
        )
        manifest["plate"].update(
            {
                "barcode": barcode,
                "vendor": "Synthetic",
                "path": str(root / "raw" / barcode),
                "rooted_path": f"ABSOLUTE:{root / 'raw' / barcode}",
            }
        )
        (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if pharmacology:
            from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology

            run_pharmacology(
                [folder],
                PharmOptions(readouts=["nuclei_count"], figures=False, pdf=False),
                refresh_reports=False,
            )
        runs.append(folder)
    write_truth(
        root,
        {
            "ec50_m": {f"{line}|{drug}": ec50 for (line, drug), (ec50, _) in TRUTH.items()},
            "hill": {f"{line}|{drug}": hill for (line, drug), (_, hill) in TRUTH.items()},
            "runs": [str(r) for r in runs],
            "barcodes": [f"REG-{i + 1:03d}" for i in range(n_runs)],
        },
    )
    return runs
