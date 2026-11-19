"""The synthetic-data framework (PROMPT_PLATFORM.md §2.9): what it draws is what it reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hcs_studio.synthetic import Background, Nuclei, ShotNoise, SyntheticPlate, read_truth


def test_synthetic_plate_counts_are_recovered_by_segmentation(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline

    wells = [(0, c) for c in range(4)]
    plate = SyntheticPlate(
        wells,
        [Background(level=100), Nuclei(count=lambda r, c, f: 5 + 10 * c + f), ShotNoise(sd=4)],
        n_fields=2,
        plate_size=(8, 12),
        name="CountsPlate",
    )
    path, truth = plate.write(tmp_path)
    assert read_truth(tmp_path) == truth
    assert truth["wells"]["A03"]["counts"]["nuclei"] == (5 + 20) + (5 + 20 + 1)
    pipeline = Pipeline.from_text(
        "name: c\nsteps:\n  - {module: segment_classical, params: {channel: 1}}\n"
        "  - {module: measure_objects, params: {texture: false, granularity: false}}\n"
        "  - {module: qc}\n"
    )
    run = run_pipeline(open_plate(path), pipeline, output_root=tmp_path / "out", checksums=False)
    summary = pd.read_csv(run.folder / "tables" / "per_well_summary.csv").set_index("well")
    for well, info in truth["wells"].items():
        assert summary.loc[well, "nuclei_count"] == info["counts"]["nuclei"], well
    _, truth_again = plate.write(tmp_path / "again")
    assert truth_again == truth  # deterministic
