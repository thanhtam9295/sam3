"""Readout registry (PROMPT_PLATFORM.md §2.6): patterns, resolution, and the retrofit of the
readouts of Phases 5–15 (every numeric per-well column of a real run is registered)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hcs_studio.readouts import READOUTS, ReadoutRegistry, ReadoutSpec, label, resolve

FULL = """
name: every-readout
steps:
  - {module: projection, params: {method: max}}
  - {module: segment_classical, params: {channel: 1}}
  - {module: expand_cells, params: {distance_um: 3}}
  - {module: detect_spots, params: {channel: 2, parent: cells}}
  - {module: measure_objects}
  - {module: colocalization, params: {channel_a: 1, channel_b: 2, labels: cells}}
  - {module: qc}
  - {module: gate_objects, params: {name: big, rules: [{feature: area_um2, min: 20}]}}
  - {module: phenotypic_clusters, params: {n_clusters: 2}}
  - {module: spatial_clusters, params: {scope: field, simulations: 0, radii_um: [5.0]}}
"""


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.utils.synthetic import write_ome_zarr_plate
    from tests.compare_builders import PIXEL_UM, WELLS, field

    root = tmp_path_factory.mktemp("readouts")
    plate = write_ome_zarr_plate(
        root / "plate.zarr",
        WELLS[:4],
        field,
        channels=("DAPI", "Alexa 488"),
        n_fields=2,
        pixel_size_um=PIXEL_UM,
        plate_size=(8, 12),
    )
    result = run_pipeline(
        open_plate(plate), Pipeline.from_text(FULL), output_root=root / "out", checksums=False
    )
    assert not result.failed_wells, result.failed_wells
    return result.folder


def test_every_numeric_per_well_column_is_registered(full_run: Path) -> None:
    from hcs_studio.io.layout_io import is_annotation_column

    table = pd.read_csv(full_run / "tables" / "per_well_summary.csv")
    numeric = [
        str(c)
        for c in table.select_dtypes(include=["number", "bool"]).columns
        if not is_annotation_column(str(c), None)
    ]
    assert len(numeric) > 100
    assert READOUTS.unknown(numeric, "well") == []
    families = {resolve(c, "well").spec.module for c in numeric}  # type: ignore[union-attr]
    assert {"measure_objects", "colocalization", "qc", "gate_objects"} <= families
    assert {"phenotypic_clusters", "spatial_clusters"} <= families


def test_every_object_column_is_registered(full_run: Path) -> None:
    objects = pd.read_parquet(full_run / "tables" / "per_object_all.parquet")
    ids = {"well", "row", "col", "field", "timepoint", "object_type", "object_id", "parent_id"}
    numeric = [
        str(c)
        for c in objects.select_dtypes(include=["number"]).columns
        if c not in ids and not str(c).startswith(("gate_", "class_", "cluster_", "spatial_"))
    ]
    assert READOUTS.unknown(numeric, "object") == []


def test_resolution_names_units_and_specificity() -> None:
    count = resolve("nuclei_count")
    assert count is not None and count.spec.level == "well" and count.values == {"labels": "nuclei"}
    assert count.label == "nuclei count (objects)"
    assert count.spec.default_normalization == "percent_of_control"
    cyto = resolve("cells_alexa_488_cytoplasm_mean_median")
    assert cyto is not None and cyto.values["channel"] == "alexa_488"  # not channel "…_cytoplasm"
    assert cyto.label == "cells alexa_488 cytoplasmic mean (median) (a.u.)"
    sd = resolve("nuclei_area_um2_std")
    assert sd is not None and sd.display == "nuclei area (SD)" and sd.unit == "µm²"
    child = resolve("cells_spots_count_mean")
    assert child is not None and child.values == {
        "labels": "cells",
        "child": "spots",
        "stat": "mean",
    }
    assert resolve("gate_big_fraction").spec.module == "gate_objects"  # type: ignore[union-attr]
    assert resolve("ec50", "fit").unit == "M"  # type: ignore[union-attr]
    assert resolve("zip_mean", "fit").spec.direction == "up"  # type: ignore[union-attr]
    assert resolve("conc_1") is None and label("conc_1") == "conc_1"  # annotation, not a readout


def test_registering_new_readouts() -> None:
    registry = ReadoutRegistry()
    registry.register(
        ReadoutSpec(
            "{labels}_foci_per_nucleus_{stat:mean|median}",
            "{labels} foci per nucleus ({stat})",
            "foci",
            "well",
            "up",
            "robust_z",
            "DNA-damage foci per nucleus",
            "foci",
        ),
    )
    found = registry.resolve("nuclei_foci_per_nucleus_mean")
    assert found is not None and found.spec.module == "foci" and found.spec.direction == "up"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ReadoutSpec("{labels}_foci_per_nucleus_{stat:mean|median}", "x"))
