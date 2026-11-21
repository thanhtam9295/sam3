"""Phase 20c: apoptosis and cytotoxicity on data with known answers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.apoptosis import Apoptosis, ApoptosisParams, growth_ratio
from hcs_studio.synthetic.apoptosis import CATEGORY, write_apoptosis_run

VEHICLE: dict[str, float] = {"viable": 96, "early": 2, "condensed": 0.5, "late": 0.5, "necrotic": 1}
STAUROSPORINE: dict[str, float] = {
    "viable": 40,
    "early": 25,
    "condensed": 10,
    "late": 20,
    "necrotic": 5,
}
NECROTIC: dict[str, float] = {"viable": 50, "early": 5, "late": 5, "necrotic": 40}
CATEGORIES = ("viable", "early_apoptotic", "late_apoptotic", "necrotic")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _plate(folder: Path, **options: Any) -> dict[str, dict[str, float]]:
    """Vehicle and staurosporine controls (72 h), a cytotoxic, a cytostatic and a necrotic
    compound, and two time-zero wells (0 h)."""
    wells: dict[str, tuple[str, Mapping[str, float], int, float | None]] = {
        "A01": ("negative_control", VEHICLE, 2000, 72),
        "A02": ("negative_control", VEHICLE, 2000, 72),
        "B01": ("positive_control", STAUROSPORINE, 1500, 72),
        "B02": ("positive_control", STAUROSPORINE, 1500, 72),
        "C01": ("sample", STAUROSPORINE, 800, 72),  # fewer viable cells than at time zero
        "C02": ("sample", VEHICLE, 1300, 72),  # healthy, growth slowed
        "C03": ("sample", NECROTIC, 1600, 72),
        "D01": ("negative_control", VEHICLE, 1000, 0),
        "D02": ("negative_control", VEHICLE, 1000, 0),
    }
    return write_apoptosis_run(folder, wells, seed=3, **options)


def _run(folder: Path, **params: Any) -> Any:
    from hcs_studio.analysis.table_steps import run_table_steps

    defaults = {"dna_channel": "DAPI", "apoptosis_channel": "Casp", "dead_channel": "PI"}
    return run_table_steps(folder, [Apoptosis(ApoptosisParams(**{**defaults, **params}))])


def test_categories_viable_counts_and_gr_match_truth(tmp_path: Path) -> None:
    """Done-when (20c): known phenotype mixtures are reported per category (fragments of
    fragmented nuclei left out), and GR separates cytotoxic from cytostatic compounds."""
    run = tmp_path / "run"
    truth = _plate(run, fragmented=1.0)
    result = _run(run)
    for well, expected in truth.items():
        values = result.per_well[well]
        for name in CATEGORIES:
            assert values[f"ap_{name}_pct"] == pytest.approx(expected[name], abs=0.5), (well, name)
        assert (
            abs(values["ap_viable_count"] - expected["viable_count"])
            <= 0.005 * expected["viable_count"] + 2
        )
        assert values["ap_fragments"] == pytest.approx(expected["fragments"], rel=0.02)
        assert values["ap_apoptotic_pct"] == pytest.approx(
            values["ap_early_apoptotic_pct"] + values["ap_late_apoptotic_pct"]
        )
    state = result.steps[0]["state"]
    assert state["calibrated_on"] == "vehicle wells" and state["time_zero_source"] == "D01,D02"
    assert 60 * 1.5 < state["apoptosis_threshold"] < 900 / 1.5  # between the populations
    assert 40 * 1.5 < state["dead_threshold"] < 1200 / 1.5
    assert state["time_zero_count"] == pytest.approx(
        np.median([truth[w]["viable_count"] for w in ("D01", "D02")])
    )
    assert state["z_prime_apoptotic"] > 0.5
    per_well = pd.DataFrame(result.per_well).T
    gr = per_well["ap_gr"].astype(float).to_dict()
    response = per_well["ap_response"].to_dict()
    assert response["C01"] == "cytotoxic" and gr["C01"] < 0
    assert response["C02"] == "cytostatic" and 0 < gr["C02"] < 0.8
    assert response["A01"] == "no effect"
    assert np.isnan(gr["D01"])  # the reference itself
    assert per_well["ap_necrotic_pct"].astype(float).to_dict()["C03"] > 35
    assert per_well.loc["C03", "ap_dead_pct"] == pytest.approx(truth["C03"]["necrotic"] + 5, abs=1)
    x0, x_vehicle = (
        state["time_zero_count"],
        np.median([truth[w]["viable_count"] for w in ("A01", "A02")]),
    )
    assert per_well.loc["C02", "ap_gr"] == pytest.approx(
        growth_ratio(truth["C02"]["viable_count"], x0, x_vehicle), abs=0.02
    )
    objects = pd.read_parquet(run / "wells" / "B01" / "objects_annotations.parquet")
    assert set(objects["ap_category"].dropna()) == {"fragment", *CATEGORIES}
    out = run / "assays" / "apoptosis"
    assert {p.name for p in out.iterdir()} == {"calibration.json", "apoptosis_wells.csv",
                                              "report.html"}  # fmt: skip
    calibration = json.loads((out / "calibration.json").read_text())
    assert calibration["params"]["threshold_method"] == "vehicle_robust"
    assert "Z′" in (out / "report.html").read_text()


def test_annexin_on_cells_and_otsu_on_controls(tmp_path: Path) -> None:
    run = tmp_path / "run"
    truth = _plate(run, annexin_on_cells=True)
    result = _run(run, apoptosis_channel="Annexin", marker_object_type="cells",
                  threshold_method="controls")  # fmt: skip
    for well in ("A01", "B01", "C03"):
        for name in CATEGORIES:
            assert result.per_well[well][f"ap_{name}_pct"] == pytest.approx(
                truth[well][name], abs=1.0
            ), (well, name)
    assert result.steps[0]["state"]["threshold_method"] == "controls"
    with pytest.raises(Exception, match="not measured on the cells"):
        _run(run, apoptosis_channel="Casp", marker_object_type="cells")


def test_vehicle_percentile_is_less_robust_than_robust_sd(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_apoptosis_run(
        run,
        {
            "A01": ("negative_control", {"viable": 94, "early": 6}, 3000, None),
            "A02": ("sample", {"viable": 94, "early": 6}, 3000, None),
        },
        seed=1,
    )
    robust = _run(run, dead_channel="").per_well["A02"]["ap_early_apoptotic_pct"]
    percentile = _run(run, dead_channel="", threshold_method="vehicle_percentile")
    assert robust == pytest.approx(6, abs=0.5)
    # 6 % positives in the vehicle: its 99th percentile lies inside them
    assert percentile.per_well["A02"]["ap_early_apoptotic_pct"] < 4


def test_morphology_only_classifier_and_fixed_time_zero(tmp_path: Path) -> None:
    run = tmp_path / "run"
    truth = write_apoptosis_run(
        run,
        {
            "A01": ("negative_control", VEHICLE, 1500, None),
            "B01": ("sample", {"viable": 70, "condensed": 30}, 1500, None),
        },
        seed=2,
    )
    result = _run(run, apoptosis_channel="", dead_channel="", time_zero_count=500.0)
    values = result.per_well["B01"]
    assert values["ap_condensed_pct"] == pytest.approx(30 + 0, abs=1.0)
    assert values["ap_early_apoptotic_pct"] == pytest.approx(30, abs=1.0)
    state = result.steps[0]["state"]
    assert any("nuclear morphology only" in w for w in state["warnings"])
    assert state["time_zero_source"] == "time_zero_count parameter"
    assert values["ap_gr"] > 0 and values["ap_response"]
    # a classifier's class column instead of the size/brightness rules
    for well in ("A01", "B01"):
        path = run / "wells" / well / "objects.parquet"
        objects = pd.read_parquet(path)
        objects["class_morphology"] = np.where(objects["area_um2"] < 60, "apoptotic", "normal")
        objects.to_parquet(path, index=False)
    by_class = _run(run, apoptosis_channel="", dead_channel="",
                    morphology_column="class_morphology")  # fmt: skip
    expected = truth["B01"]["early_apoptotic"] + truth["B01"]["late_apoptotic"]
    assert by_class.per_well["B01"]["ap_apoptotic_pct"] == pytest.approx(expected, abs=1.0)
    with pytest.raises(Exception, match="classifier step"):
        _run(run, morphology_column="class_missing")


def test_growth_ratio() -> None:
    assert growth_ratio(200, 100, 200) == pytest.approx(1.0)
    assert growth_ratio(100, 100, 200) == pytest.approx(0.0)
    assert growth_ratio(50, 100, 200) == pytest.approx(-0.5)
    assert growth_ratio(0, 100, 200) == -1.0
    assert np.isnan(growth_ratio(50, 100, 100))


def test_apoptosis_from_images(home: Path) -> None:
    """Done-when (20c): the preset pipeline recovers the phenotype mix from images."""
    from hcs_studio.assays.apoptosis import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import Background, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.apoptosis import ApoptosisNuclei

    vehicle = {"viable": 97, "early": 2, "necrotic": 1}
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (1, 0), (1, 1)],
        features=[
            Background(level=100.0),
            ApoptosisNuclei(fractions=lambda r, c: vehicle if r == 0 else STAUROSPORINE),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "Casp", "PI"),
        size=256,
        n_fields=6,
        name="APO",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    layout = set_role(layout, [(1, 0), (1, 1)], "positive_control")
    pipeline = preset({"dna": "DAPI", "apoptosis": "Casp", "dead": "PI"},
                      channels=["DAPI", "Casp", "PI"], nucleus_diameter_um=18 * 0.65)  # fmt: skip
    result = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False)  # fmt: skip
    summary = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    for well, data in truth["wells"].items():
        counts: dict[str, int] = {}
        for field in data["fields"].values():
            for phenotype, k in field["apoptosis"]["phenotypes"].items():
                counts[CATEGORY[phenotype]] = counts.get(CATEGORY[phenotype], 0) + k
        n = sum(counts.values())
        assert summary.loc[well, "ap_nuclei"] == pytest.approx(n, rel=0.03)
        for name in CATEGORIES:
            assert summary.loc[well, f"ap_{name}_pct"] == pytest.approx(
                100 * counts.get(name, 0) / n, abs=2.5
            ), (well, name)
    assert (result.folder / "assays" / "apoptosis" / "report.html").exists()


def test_apoptosis_preset_cli_and_readouts(tmp_path: Path, home: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.assays.apoptosis import preset
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.readouts import resolve

    pipeline = preset({"dna": "Hoechst", "dead": "DRAQ7"}, channels=["DRAQ7", "Hoechst"],
                      z_planes=4)  # fmt: skip
    modules = [s.module for s in pipeline.steps]
    assert modules == ["projection", "background", "segment_classical", "measure_objects", "qc",
                       "apoptosis"]  # fmt: skip
    assert pipeline.steps[2].params["channel"] == 2
    assert pipeline.steps[-1].params["dead_channel"] == "DRAQ7"
    out = tmp_path / "apo.yaml"
    command = ["assay", "preset", "apoptosis", "--map", "dna=DAPI", "--map", "apoptosis=Casp"]
    assert main([*command, "--output", str(out)]) == 0
    assert Pipeline.load(out).steps[-1].params["apoptosis_channel"] == "Casp"
    run = tmp_path / "run"
    _plate(run)
    command = ["assay", "run", str(run), "apoptosis", "--map", "dna=DAPI", "--map",
               "apoptosis=Casp", "--map", "dead=PI", "--reason", "test"]  # fmt: skip
    assert main(command) == 0
    summary = pd.read_csv(run / "tables" / "per_well_summary.csv").set_index("well")
    apoptotic = summary["ap_apoptotic_pct"].astype(float).to_dict()
    assert apoptotic["B01"] > 40 > apoptotic["A01"]
    for column in ("ap_viable_pct", "ap_early_apoptotic_pct", "ap_necrotic_pct",
                   "ap_apoptotic_pct", "ap_dead_pct", "ap_viable_count", "ap_gr",
                   "ap_viable_vs_vehicle", "ap_fragments"):  # fmt: skip
        assert resolve(column, "well") is not None, column


def test_wizard_preview_values(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _plate(run)
    module = Apoptosis(ApoptosisParams(dna_channel="DAPI", apoptosis_channel="Casp"))
    module.run_folder = run
    values, label = module.preview_values({})
    assert len(values) == 4000 and "Casp" in label  # vehicle only, not the 0 h wells
    assert np.median(values) == pytest.approx(np.log10(60), abs=0.1)
