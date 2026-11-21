"""Phase 20b: DNA-damage foci (detect_foci + the foci assay) on data with known answers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.foci import find_foci, spot_response


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _objects(folder: Path, well: str) -> pd.DataFrame:
    objects = pd.read_parquet(folder / "wells" / well / "objects.parquet")
    extra = pd.read_parquet(folder / "wells" / well / "objects_annotations.parquet")
    objects = pd.concat([objects, extra[[c for c in extra.columns if c not in objects]]], axis=1)
    return objects[objects["object_type"] == "nuclei"]


def test_foci_counts_per_nucleus_from_images(home: Path) -> None:
    """Done-when (20b): known foci counts are recovered within ±1 per nucleus for
    well-separated foci; pan-nuclear nuclei are recognised and left out."""
    from hcs_studio.assays.foci import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import Background, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.foci import FociNuclei

    rate = {0: 0.5, 1: 8.0}  # vehicle / irradiated (Poisson means)
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (1, 0), (1, 1)],
        features=[
            Background(level=100.0),
            FociNuclei(
                count=lambda r, c, rng: min(int(rng.poisson(rate[r])), 12),
                pan_fraction=lambda r, c: 0.0 if r == 0 else 0.15,
            ),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "gH2AX"),
        size=256,
        n_fields=3,
        name="FOCI",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    layout = set_role(layout, [(1, 0), (1, 1)], "positive_control")
    pipeline = preset({"dna": "DAPI", "foci": "gH2AX"}, channels=["DAPI", "gH2AX"],
                      nucleus_diameter_um=13.0, focus_diameter_um=1.5)  # fmt: skip
    result = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False)  # fmt: skip
    folder = result.folder
    errors, pan_found, pan_false = [], [], 0
    for well, data in truth["wells"].items():
        objects = _objects(folder, well)
        for field, info in data["fields"].items():
            found = objects[objects["field"] == int(field)]
            for nucleus in info["foci"]["nuclei"]:
                cy, cx = nucleus["centre_px"]
                distance = (found["centroid_y_um"] / 0.65 - cy) ** 2 + (
                    found["centroid_x_um"] / 0.65 - cx
                ) ** 2
                row = found.loc[distance.idxmin()].to_dict()
                if nucleus["pan"]:
                    pan_found.append(bool(row["dd_pan_nuclear"]))
                    continue
                pan_false += bool(row["dd_pan_nuclear"])
                errors.append(row["dd_foci"] - nucleus["foci"])
    errors_array = np.asarray(errors)
    assert len(errors_array) > 350
    assert np.all(np.abs(errors_array) <= 1), np.unique(errors_array, return_counts=True)
    assert (errors_array == 0).mean() > 0.95
    assert all(pan_found) and len(pan_found) > 20 and pan_false == 0
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    vehicle = summary.loc[["A01", "A02"], "dd_foci_per_nucleus"]
    treated = summary.loc[["B01", "B02"], "dd_foci_per_nucleus"]
    assert vehicle.between(0.3, 0.7).all() and treated.between(5.5, 8.0).all()
    assert (summary.loc[["B01", "B02"], "dd_positive_pct"] > 75).all()
    assert (summary.loc[["A01", "A02"], "dd_pan_nuclear_pct"] == 0).all()
    manifest = json.loads((folder / "run_manifest.json").read_text())
    fitted = next(e for e in manifest["fitted_state"] if e["module"] == "detect_foci")
    assert fitted["calibrated_on"] == "vehicle wells" and fitted["threshold"] > fitted["median"]
    assert {p.name for p in (folder / "assays" / "foci").iterdir()} == {
        "calibration.json", "foci_wells.csv", "report.html"}  # fmt: skip
    assert "assays/foci/report.html" in (folder / "index.html").read_text()


def test_3d_detection_counts_foci_a_max_projection_merges() -> None:
    depth, size = 15, 48
    zz, yy, xx = np.mgrid[0:depth, 0:size, 0:size].astype(float)
    stack = np.full((depth, size, size), 100.0)
    for z in (3, 11):  # two foci at the same x, y, eight planes apart
        stack += 800 * np.exp(-((zz - z) ** 2 / 2 + (yy - 24) ** 2 / 2 + (xx - 24) ** 2 / 2))
    nuclei = np.zeros((1, size, size), dtype=np.int32)
    nuclei[0, 10:38, 10:38] = 1
    response = spot_response(stack, 1.0, z_ratio=1.0)
    threshold = float(np.percentile(response, 50)) + 30
    three_d = find_foci(response, np.repeat(nuclei, depth, 0), threshold, min_size_px=1,
                        max_size_px=500, separation_px=3)  # fmt: skip
    flat = stack.max(axis=0, keepdims=True)
    projected = find_foci(spot_response(flat, 1.0), nuclei, threshold, min_size_px=1,
                          max_size_px=500, separation_px=3)  # fmt: skip
    assert int(np.max(three_d)) == 2 and int(np.max(projected)) == 1


def test_foci_step_breaks_down_by_cell_cycle_and_warns_on_projections(tmp_path: Path) -> None:
    from hcs_studio.analysis.table_steps import run_table_steps
    from hcs_studio.assays.cell_cycle import CellCycle, CellCycleParams
    from hcs_studio.assays.foci import Foci, FociParams
    from hcs_studio.synthetic.cell_cycle import ARRESTED, VEHICLE, write_cycle_run

    run = tmp_path / "run"
    write_cycle_run(
        run, {"A01": ("negative_control", VEHICLE), "A02": ("sample", ARRESTED)}, n=3000
    )
    rng = np.random.default_rng(5)
    for well in ("A01", "A02"):
        path = run / "wells" / well / "objects.parquet"
        objects = pd.read_parquet(path)
        dna = objects["dapi_integrated"] / 25_000  # N
        s_or_g2 = dna > 2.6  # RAD51-like: foci in S/G2 only
        objects["foci_count"] = np.where(
            s_or_g2, rng.poisson(6, len(objects)), rng.poisson(0.3, len(objects))
        )
        objects["gh2ax_mean"] = rng.normal(100, 8, len(objects))
        objects.loc[objects.index[:30], "gh2ax_mean"] = 900  # pan-nuclear
        objects.to_parquet(path, index=False)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pipeline"]["steps"] = [{"module": "projection", "params": {"method": "max"}}]
    manifest_path.write_text(json.dumps(manifest))
    result = run_table_steps(run, [CellCycle(CellCycleParams(dna_channel="DAPI")),
                                   Foci(FociParams(foci_channel="gH2AX"))])  # fmt: skip
    values = result.per_well["A02"]
    assert values["dd_foci_per_nucleus_g1"] < 0.6 < 5 < values["dd_foci_per_nucleus_g2m"]
    # foci above 2.6N: all of G2/M, the later part of S, almost none of G1
    # positive = ≥ 5 foci: P(Poisson(6) ≥ 5) ≈ 0.71
    assert values["dd_positive_g2m_pct"] == pytest.approx(71.5, abs=8)
    assert 20 < values["dd_positive_s_pct"] < values["dd_positive_g2m_pct"]
    assert values["dd_positive_g1_pct"] < 5
    assert values["dd_pan_nuclear_pct"] == pytest.approx(100 * 30 / 3000, abs=0.01)
    state = result.steps[1]["state"]
    assert any("foci at different depths" in w for w in state["warnings"])
    with pytest.raises(Exception, match="detect_foci"):
        run_table_steps(run, [Foci(FociParams(foci_channel="gH2AX", foci="spots"))])


def test_foci_preset_and_readouts() -> None:
    from hcs_studio.assays.foci import preset
    from hcs_studio.readouts import resolve

    pipeline = preset({"dna": "Hoechst", "foci": "53BP1"}, channels=["53BP1", "Hoechst"])
    modules = [s.module for s in pipeline.steps]
    assert modules == [
        "background",
        "segment_classical",
        "detect_foci",
        "measure_objects",
        "qc",
        "foci",
    ]
    assert pipeline.steps[1].params["channel"] == 2 and pipeline.steps[2].params["channel"] == 1
    for column in ("dd_foci_per_nucleus", "dd_positive_pct", "dd_pan_nuclear_pct",
                   "dd_foci_per_nucleus_s", "dd_positive_g2m_pct"):  # fmt: skip
        assert resolve(column, "well") is not None, column
    params: dict[str, Any] = pipeline.steps[-1].params
    assert params["foci_channel"] == "53BP1"
