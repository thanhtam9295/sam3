"""A small finished run for comparison tests: 2 × 6 wells (A01–B06), 2 channels, 5 Z
planes, 2 fields per well, 64 × 64 px at 0.5 µm, with a plate layout (MCF7 / HeLa ×
6 concentrations of Drug X)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcs_studio.io.base import open_plate
from hcs_studio.io.layout_io import save_layout
from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.layout_tools import dilution_series
from hcs_studio.pipeline.executor import run_pipeline
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.utils.synthetic import write_ome_zarr_plate

PIXEL_UM = 0.5
WELLS = [(r, c) for r in range(2) for c in range(6)]
PIPELINE = """
name: compare
steps:
  - {module: segment_classical, params: {channel: 1}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
"""


def field(row: int, col: int, f: int) -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    data = np.full((2, 5, 64, 64), 100.0)
    rng = np.random.default_rng(row * 100 + col * 10 + f)
    n = 3 + f  # field 2 has one more nucleus
    for k in range(n):
        y, x = 12 + 13 * (k % 4), 14 + 18 * (k // 4) + 3 * f
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= 25
        for z in range(5):
            weight = 1.0 - 0.3 * abs(z - 2)
            data[0, z][disc] += 1500 * weight
            data[1, z][disc] += (200 + 300 * col) * weight
    data += rng.normal(0, 5, data.shape)
    return np.asarray(np.clip(data, 0, 65535), dtype=np.uint16)


def make_run(root: Path) -> Path:
    plate_dir = write_ome_zarr_plate(
        root / "plate.zarr",
        WELLS,
        field,
        channels=("DAPI", "GFP"),
        n_fields=2,
        pixel_size_um=PIXEL_UM,
        z_step_um=1.0,
        plate_size=(8, 12),
        name="CmpPlate",
    )
    layout = PlateLayout.for_plate_type(96)
    for row, line in ((0, "MCF7"), (1, "HeLa")):
        layout = dilution_series(
            layout, (row, 0), "Drug X", 10.0, "µM", factor=3, points=6, cell_line=line, reverse=True
        )
    layout_file = root / "layout.json"
    save_layout(layout, layout_file)
    result = run_pipeline(
        open_plate(plate_dir),
        Pipeline.from_text(PIPELINE),
        output_root=root / "out",
        layout=layout,
        layout_path=layout_file,
        checksums=False,
    )
    assert not result.failed_wells, result.failed_wells
    return result.folder
