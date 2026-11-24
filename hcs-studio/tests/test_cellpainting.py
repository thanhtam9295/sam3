"""Phase 24a: Cell Painting features, CellProfiler-style single-cell tables and CellProfiler
import."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.analysis.features import correlations
from hcs_studio.cellpainting.cellprofiler import read_cellprofiler
from hcs_studio.cellpainting.naming import rename_objects

DATA = Path(__file__).parent / "data" / "cellpainting"
CHANNELS = ("Hoechst", "ConA", "SYTO14", "Phalloidin", "MitoTracker")
MAPPING = {
    "dna": "Hoechst",
    "er": "ConA",
    "rna": "SYTO14",
    "agp": "Phalloidin",
    "mito": "MitoTracker",
}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_cellprofiler_names() -> None:
    names = rename_objects(
        ["object_id", "area_um2", "dna_mean", "dna_p25", "er_haralick_contrast_s3",
         "mito_granularity_r2", "dna_er_correlation", "dna_radial_inner", "dna_radial_outer",
         "nearest_neighbour_um", "foci_count", "centroid_x_um", "zernike_2_0", "dna_weird"],
        "nuclei", ["dna", "er", "mito"], {"dna": "DNA", "er": "ER", "mito": "Mito"},
    )  # fmt: skip
    assert names == {
        "area_um2": "Nuclei_AreaShape_Area",
        "dna_mean": "Nuclei_Intensity_MeanIntensity_DNA",
        "dna_p25": "Nuclei_Intensity_LowerQuartileIntensity_DNA",
        "er_haralick_contrast_s3": "Nuclei_Texture_Contrast_ER_3",
        "mito_granularity_r2": "Nuclei_Granularity_2_Mito",
        "dna_er_correlation": "Nuclei_Correlation_Correlation_DNA_ER",
        "dna_radial_inner": "Nuclei_RadialDistribution_FracAtD_DNA_1of3",
        "dna_radial_outer": "Nuclei_RadialDistribution_FracAtD_DNA_3of3",
        "nearest_neighbour_um": "Nuclei_Neighbors_FirstClosestDistance",
        "foci_count": "Nuclei_Children_Foci_Count",
        "centroid_x_um": "Nuclei_Location_Center_X",
        "zernike_2_0": "Nuclei_AreaShape_Zernike_2_0",
        "dna_weird": "Nuclei_Intensity_weird_DNA",
    }
    assert rename_objects(["alexa_488_mean"], "cells", ["alexa_488"], {})["alexa_488_mean"] == (
        "Cells_Intensity_MeanIntensity_Alexa_488"
    )


def test_correlation_known_answers() -> None:
    rng = np.random.default_rng(0)
    labels = np.zeros((1, 40, 40), int)
    labels[0, 5:20, 5:20] = 1
    labels[0, 22:38, 22:38] = 2
    a = rng.normal(0, 1, (1, 40, 40))
    image = np.stack([a, 2 * a + 1, rng.normal(0, 1, (1, 40, 40)), -a])
    table = correlations(labels, image, ["dna", "er", "mito", "neg"], [0, 1, 2, 3]).set_index(
        "object_id"
    )
    assert table["dna_er_correlation"].tolist() == pytest.approx([1.0, 1.0])
    assert table["dna_neg_correlation"].tolist() == pytest.approx([-1.0, -1.0])
    assert table["dna_mito_correlation"].abs().max() < 0.2


@pytest.fixture(scope="module")
def painted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Three wells (vehicle, larger nuclei, fragmented mitochondria), 2 fields of 25 cells."""
    import os

    from hcs_studio.cellpainting.preset import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.cellpainting import PaintedCells

    root = tmp_path_factory.mktemp("painted")
    os.environ.setdefault("HCS_STUDIO_HOME", str(root / "home"))
    specs: dict[tuple[int, int], dict[str, float]] = {
        (0, 0): {}, (0, 1): {"nucleus_size": 1.3}, (0, 2): {"mito_spots": 0.3},
    }  # fmt: skip
    cells = PaintedCells(spec=lambda r, c: specs[(r, c)])
    plate = SyntheticPlate(wells=list(specs), features=[cells, ShotNoise(sd=8)],
                           channels=CHANNELS, size=192, n_fields=2, name="CP1")  # fmt: skip
    path, truth = plate.write(root / "raw")
    layout = set_role(
        PlateLayout.for_plate_type(96), [(0, 0)], "negative_control", cell_line="U2OS"
    )
    for col, name in ((1, "Bigger"), (2, "Fragmenter")):
        layout = apply_fields(layout, [(0, col)], treatments=(Treatment(name, 1e-6, "µM"),),
                              cell_line="U2OS")  # fmt: skip
    folder = run_pipeline(open_plate(path), preset(MAPPING, channels=list(CHANNELS)),
                          output_root=root / "runs", layout=replace(layout, barcode="CP1"),
                          save_images=False).folder  # fmt: skip
    return folder, truth


def test_preset_single_cell_table(painted: tuple[Path, dict[str, Any]]) -> None:
    from pycytominer.cyto_utils import infer_cp_features

    from hcs_studio.cellpainting.preset import channel_map
    from hcs_studio.cellpainting.single_cell import feature_columns, write_single_cells

    folder, truth = painted
    path, cells = write_single_cells(folder, channel_map=channel_map(MAPPING))
    table = pd.read_parquet(path)
    assert (
        cells == len(table) == sum(t["counts"]["painted"] for t in truth["wells"].values()) == 150
    )
    assert table.groupby("Metadata_Well").size().to_dict() == {"A01": 50, "A02": 50, "A03": 50}
    meta = table.drop_duplicates("Metadata_Well").set_index("Metadata_Well")
    assert (
        meta.loc["A01", "Metadata_Role"] == "negative_control"
        and meta.loc["A01", "Metadata_Compound"] == "DMSO"
    )
    assert meta.loc["A02", "Metadata_Compound"] == "Bigger"
    assert meta.loc["A02", "Metadata_Concentration"] == pytest.approx(1e-6)
    assert set(table["Metadata_Plate"]) == {"CP1"} and set(table["Metadata_Site"]) == {1, 2}
    features = feature_columns(table)
    for compartment in ("Nuclei_", "Cells_", "Cytoplasm_"):
        assert sum(f.startswith(compartment) for f in features) > 250
    assert (
        "Cells_Correlation_Correlation_ER_RNA" in table and "Cytoplasm_Granularity_2_Mito" in table
    )
    assert "Nuclei_Texture_Contrast_DNA_5" in table
    assert not [f for f in features if f.startswith("Nuclei_Other_cells_")]  # no duplicates
    assert set(infer_cp_features(table)) == set(features)  # pycytominer finds them all
    columns = ["Nuclei_AreaShape_Area", "Cytoplasm_Granularity_2_Mito"]
    median = table.groupby("Metadata_Well")[columns].median().to_dict("index")
    area = {w: float(v["Nuclei_AreaShape_Area"]) for w, v in median.items()}
    granularity = {w: float(v["Cytoplasm_Granularity_2_Mito"]) for w, v in median.items()}
    assert area["A02"] / area["A01"] == pytest.approx(1.69, rel=0.12)
    assert granularity["A03"] != pytest.approx(granularity["A01"], rel=0.1)


def _reference() -> pd.DataFrame:
    return (
        pd.read_csv(DATA / "test_SQ00014613.csv.gz")
        .sort_values("Metadata_ImageNumber")
        .reset_index(drop=True)
    )


def _same_features(ours: pd.DataFrame) -> None:
    reference = _reference()
    ours = ours.sort_values("Metadata_ImageNumber").reset_index(drop=True)
    features = [c for c in reference.columns if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))]
    assert len(features) > 1700
    assert not [c for c in features if c not in ours]
    a = ours[features].astype(float).to_numpy()
    b = reference[features].astype(float).to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) == 0.0
    assert ours["Metadata_Plate"].tolist() == ["SQ00014613", "SQ00014613"]
    assert ours["Metadata_Well"].tolist() == ["A01", "A01"]


def test_cellprofiler_sqlite_matches_pycytominer() -> None:
    """pycytominer's example SQLite imported here = pycytominer's own merged single cells."""
    _same_features(read_cellprofiler(DATA / "test_SQ00014613.sqlite"))


def test_cellprofiler_csv_folder_and_parquet(tmp_path: Path) -> None:
    """The same data as ExportToSpreadsheet CSVs (unprefixed columns, Parent_ links) and as a
    merged Parquet file."""
    with sqlite3.connect(DATA / "test_SQ00014613.sqlite") as connection:
        image = pd.read_sql_query("select * from Image", connection)
        tables = {
            n: pd.read_sql_query(f"select * from {n}", connection)
            for n in ("Cells", "Cytoplasm", "Nuclei")
        }
    folder = tmp_path / "csv"
    folder.mkdir()
    image.rename(
        columns={
            c: c.replace("Image_", "", 1) for c in image.columns if c.startswith("Image_Metadata_")
        }
    ).to_csv(folder / "Image.csv", index=False)
    for name, frame in tables.items():
        frame.rename(
            columns={c: c[len(name) + 1 :] for c in frame.columns if c.startswith(f"{name}_")}
        ).to_csv(folder / f"{name}.csv", index=False)
    _same_features(read_cellprofiler(folder))
    merged = tmp_path / "merged.parquet"
    _reference().to_parquet(merged)
    assert read_cellprofiler(merged)["Metadata_Plate"].tolist() == ["SQ00014613", "SQ00014613"]
    with pytest.raises(ValueError, match="Cannot read"):
        read_cellprofiler(tmp_path / "x.txt")


def test_cellpainting_commands(painted: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.pipeline.schema import Pipeline

    pipeline = tmp_path / "cp.yaml"
    args = ["cellpainting", "preset", "--channels", ",".join(CHANNELS), "--output", str(pipeline)]
    args += [f"--map={role}={channel}" for role, channel in MAPPING.items()]
    assert main(args) == 0
    steps = [s.module for s in Pipeline.load(pipeline).steps]
    assert steps == ["segment_classical", "expand_cells", "cytoplasm", "measure_objects", "qc"]
    folder, _ = painted
    out = tmp_path / "sc.parquet"
    assert (
        main(
            ["cellpainting", "single-cells", str(folder), "--output", str(out), "--map=dna=Hoechst"]
        )
        == 0
    )
    assert "Nuclei_Intensity_MeanIntensity_DNA" in pd.read_parquet(out)
    imported = tmp_path / "imported.parquet"
    assert (
        main(
            [
                "cellpainting",
                "import",
                str(DATA / "test_SQ00014613.sqlite"),
                "--output",
                str(imported),
            ]
        )
        == 0
    )
    assert len(pd.read_parquet(imported)) == 2
    assert (
        main(
            ["cellpainting", "preset", "--channels", "A,B", "--map=er=A", "--output", str(pipeline)]
        )
        == 2
    )
