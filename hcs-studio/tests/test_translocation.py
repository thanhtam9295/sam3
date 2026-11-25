"""Phase 20d: nuclear translocation (cytoplasm ring + N/C assay) on data with known answers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.translocation import (
    Translocation,
    TranslocationParams,
    cytoplasm_labels,
)
from hcs_studio.synthetic.translocation import hill, write_translocation_run

Wells = dict[str, tuple[str, float, int, tuple[str, float] | None]]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _run(folder: Path, **params: Any) -> Any:
    from hcs_studio.analysis.table_steps import run_table_steps

    return run_table_steps(
        folder, [Translocation(TranslocationParams(reporter_channel="p65", **params))]
    )


def _dose_plate(fractions: Mapping[str, float], roles: tuple[str, str]) -> Wells:
    """Controls in A/B (two wells each), a 9-point series of 'TNF' in C."""
    wells: Wells = {
        "A01": (roles[0], fractions["neg"], 600, None),
        "A02": (roles[0], fractions["neg"], 600, None),
        "B01": (roles[1], fractions["pos"], 600, None),
        "B02": (roles[1], fractions["pos"], 600, None),
    }
    return wells


def test_cytoplasm_ring_stays_out_of_nuclei_and_neighbours() -> None:
    nuclei = np.zeros((1, 60, 80), dtype=np.int32)
    yy, xx = np.mgrid[0:60, 0:80]
    nuclei[0][(yy - 30) ** 2 + (xx - 20) ** 2 <= 64] = 1
    nuclei[0][(yy - 30) ** 2 + (xx - 50) ** 2 <= 64] = 2  # edges at x = 28 and 42
    ring = cytoplasm_labels(nuclei, gap_px=1.0, width_px=4.0)[0]
    assert not np.any(ring[nuclei[0] > 0])
    assert ring[30, 29] == 0 and ring[30, 30] == 1 and ring[30, 33] == 1 and ring[30, 34] == 0
    wide = cytoplasm_labels(nuclei, gap_px=1.0, width_px=10.0)[0]
    for label, cx in ((1, 20), (2, 50)):
        y, x = np.nonzero(wide == label)
        own = np.hypot(y - 30, x - cx)
        other = np.hypot(y - 30, x - (70 - cx))
        assert np.all(own <= other + 0.5)  # never in the neighbour's territory
        assert own.min() > 8.5 and own.max() <= 8 + 1 + 10 + 1
    # wide rings meet half-way between the nuclei
    assert wide[30, 34] == 1 and wide[30, 36] == 2
    # a cell segmentation keeps each ring inside its own cell
    cells = np.zeros_like(nuclei)
    cells[0][(yy - 30) ** 2 + (xx - 25) ** 2 <= 100] = 1
    inside = cytoplasm_labels(nuclei, 1.0, 4.0, cells=cells)[0]
    assert set(np.unique(inside)) == {0, 1}
    assert np.all(cells[0][inside == 1] == 1)
    whole = cytoplasm_labels(nuclei, 0.0, 1.0, cells=cells, mode="cells")[0]
    assert np.array_equal(whole > 0, (cells[0] > 0) & (nuclei[0] == 0))


def test_nc_ratio_per_cell_from_images(home: Path) -> None:
    """Done-when (20d): a known N/C ratio is recovered within 5 % — per cell from images, ring
    cytoplasm and cell-minus-nucleus alike."""
    from hcs_studio.assays.translocation import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import Background, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.translocation import TranslocationCells

    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (1, 0), (1, 1)],
        features=[
            Background(level=100.0),
            TranslocationCells(fraction=lambda r, c: 0.03 if r == 0 else 0.85),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "p65"),
        size=256,
        n_fields=3,
        name="TL",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    layout = set_role(layout, [(1, 0), (1, 1)], "positive_control")
    ring = preset({"dna": "DAPI", "reporter": "p65"}, channels=["DAPI", "p65"],
                  nucleus_diameter_um=16 * 0.65)  # fmt: skip
    steps = ring.to_dict()["steps"]
    index = [s["module"] for s in steps].index("cytoplasm")
    steps[index] = {"module": "cytoplasm", "params": {"cells": "cells", "mode": "cells"}}
    steps.insert(index, {"module": "expand_cells", "params": {"distance_um": 7 * 0.65}})
    cells = Pipeline.model_validate({"name": "translocation-cells", "steps": steps})
    for pipeline in (ring, cells):
        result = run_pipeline(open_plate(path), pipeline, output_root=home / pipeline.name,
                              layout=layout, save_images=False)  # fmt: skip
        folder = result.folder
        errors, per_well = [], {}
        for well, data in truth["wells"].items():
            objects = pd.read_parquet(folder / "wells" / well / "objects.parquet")
            extra = pd.read_parquet(folder / "wells" / well / "objects_annotations.parquet")
            objects = pd.concat([objects, extra[[c for c in extra if c not in objects]]], axis=1)
            nuclei = objects[objects["object_type"] == "nuclei"]
            true_ratios = []
            for field, info in data["fields"].items():
                found = nuclei[nuclei["field"] == int(field)]
                for cell in info["translocation"]["cells"]:
                    cy, cx = cell["centre_px"]
                    distance = (found["centroid_y_um"] / 0.65 - cy) ** 2 + (
                        found["centroid_x_um"] / 0.65 - cx
                    ) ** 2
                    measured = float(found.loc[distance.idxmin(), "tl_nc_ratio"])
                    errors.append(measured / cell["nc_ratio"] - 1)
                    true_ratios.append(cell["nc_ratio"])
            per_well[well] = float(np.median(true_ratios))
        error = np.abs(np.asarray(errors))
        assert len(error) == 4 * 3 * 49, pipeline.name
        assert np.median(error) < 0.03 and (error < 0.05).mean() > 0.9, pipeline.name
        summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
        values = summary.to_dict("index")
        for well, ratio in per_well.items():
            assert values[well]["tl_nc_ratio"] == pytest.approx(ratio, rel=0.05), well
        # 1 of 49 resting cells per field responds in A, 42 of 49 in B
        assert values["A01"]["tl_responding_pct"] == pytest.approx(100 * 3 / 147, abs=1.5)
        assert values["B01"]["tl_responding_pct"] == pytest.approx(100 * 126 / 147, abs=1.5)
    calibration = json.loads(
        (folder / "assays" / "translocation" / "calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["calibrated_on"] == "unstimulated wells (agonist mode)"
    assert calibration["z_prime_responding_pct"] > 0.5 and calibration["warnings"] == []
    assert "assays/translocation/report.html" in (folder / "index.html").read_text(encoding="utf-8")


def test_agonist_ec50_through_pharmacology(tmp_path: Path, home: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology

    run = tmp_path / "run"
    wells = _dose_plate({"neg": 0.02, "pos": 0.9}, ("negative_control", "positive_control"))
    for i in range(9):
        concentration = 1e-9 * 3**i
        fraction = 0.02 + 0.88 * hill(concentration, 1e-7)
        wells[f"C{i + 1:02d}"] = ("sample", fraction, 600, ("TNF", concentration))
    truth = write_translocation_run(run, wells, seed=1)
    command = ["assay", "run", str(run), "translocation", "--map", "dna=DAPI", "--map",
               "reporter=p65", "--reason", "test"]  # fmt: skip
    assert main(command) == 0
    summary = pd.read_csv(run / "tables" / "per_well_summary.csv").set_index("well")
    responding = summary["tl_responding_pct"].astype(float).to_dict()
    for well, expected in truth.items():
        assert responding[well] == pytest.approx(expected["responding_pct"], abs=1.0), well
    activation = summary["tl_activation_pct"].astype(float).to_dict()
    assert activation["B01"] == pytest.approx(100, abs=2) and abs(activation["A01"]) < 2
    readouts = ["tl_responding_pct", "tl_log2_nc"]
    options = PharmOptions(readouts=readouts, normalization="percent_inhibition", figures=False,
                           pdf=False)  # fmt: skip
    result = run_pharmacology([run], options, write=False)
    for readout in ("tl_responding_pct", "tl_log2_nc"):
        (curve,) = result.readouts[readout].curves
        assert curve.fit.log_ec50 == pytest.approx(-7.0, abs=0.1), readout
        assert curve.fit.top == pytest.approx(100, abs=5)


def test_inhibitor_mode_calibrates_on_positive_controls(tmp_path: Path) -> None:
    from hcs_studio.pharmacology.analysis import PharmOptions, analyse
    from hcs_studio.pharmacology.normalize import load_wells

    run = tmp_path / "run"
    # all wells stimulated; vehicle = stimulus alone, positive control = unstimulated
    wells = _dose_plate({"neg": 0.9, "pos": 0.03}, ("negative_control", "positive_control"))
    for i in range(9):
        concentration = 1e-9 * 3**i
        fraction = 0.03 + 0.87 * (1 - hill(concentration, 3e-8))
        wells[f"C{i + 1:02d}"] = ("sample", fraction, 600, ("IKKi", concentration))
    write_translocation_run(run, wells, seed=2)
    result = _run(run, mode="inhibitor")
    state = result.steps[0]["state"]
    assert state["unstimulated_wells"] == ["B01", "B02"]
    assert state["unstimulated_median_ratio"] == pytest.approx(0.5, rel=0.05)
    assert state["stimulated_median_ratio"] == pytest.approx(2.0, rel=0.05)
    values = pd.DataFrame(result.per_well).T
    assert result.per_well["A01"]["tl_activation_pct"] == pytest.approx(100, abs=2)
    # IC50 against the stimulated controls (pharmacology's % inhibition)
    table = values.reset_index(names="well")
    (run / "tables").mkdir(exist_ok=True)
    table.to_csv(run / "tables" / "per_well_summary.csv", index=False)
    pharm = analyse(load_wells([run]), PharmOptions(readouts=["tl_responding_pct"],
                                                    figures=False, pdf=False))  # fmt: skip
    (curve,) = pharm.readouts["tl_responding_pct"].curves
    assert curve.fit.log_ec50 == pytest.approx(math.log10(3e-8), abs=0.1)


def test_threshold_methods_and_errors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    wells: Wells = {
        "A01": ("negative_control", 0.05, 2000, None),
        "B01": ("positive_control", 0.9, 2000, None),
        "C01": ("sample", 0.5, 2000, None),
    }
    truth = write_translocation_run(run, wells, seed=3)
    robust = _run(run)
    percentile = _run(run, threshold_method="unstimulated_percentile", percentile=95)
    controls = _run(run, threshold_method="controls")
    fixed = _run(run, ratio_threshold=1.0)
    for result in (robust, percentile, controls, fixed):
        assert result.per_well["C01"]["tl_responding_pct"] == pytest.approx(50, abs=1.5)
    for result in (robust, percentile, controls):
        assert 0.7 < result.steps[0]["state"]["ratio_threshold"] < 1.4  # between 0.5 and 2
    assert fixed.steps[0]["state"]["threshold_method"] == "fixed"
    override = _run(run, unstimulated_wells=["C01"])
    assert override.steps[0]["state"]["unstimulated_wells"] == ["C01"]
    assert truth["C01"]["responding_pct"] == 50
    with pytest.raises(Exception, match="cytoplasm step"):
        _run(run, cytoplasm="rings")


def test_cells_as_cytoplasm(tmp_path: Path) -> None:
    run = tmp_path / "run"
    wells: Wells = {"A01": ("negative_control", 0.02, 800, None),
                    "B01": ("positive_control", 0.9, 800, None)}  # fmt: skip
    truth = write_translocation_run(run, wells, seed=4, cells=True)
    result = _run(run, cytoplasm="cells")
    for well in wells:
        assert result.per_well[well]["tl_nc_ratio"] == pytest.approx(
            truth[well]["nc_ratio"], rel=1e-6
        )
    with pytest.raises(Exception, match="only the mean"):
        _run(run, cytoplasm="cells", statistic="median")


def test_warnings_about_background(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_translocation_run(run, {"A01": ("negative_control", 0.02, 300, None)})
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plate"]["metadata"]["channels"] = [{"name": "DAPI"}, {"name": "p65"}]

    def warnings(steps: list[dict[str, Any]]) -> list[str]:
        manifest["pipeline"]["steps"] = steps
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        state: dict[str, Any] = _run(run).steps[0]["state"]
        return list(state["warnings"])

    seg = {"module": "segment_classical", "params": {"channel": 1}}
    assert any("no background subtraction" in w for w in warnings([seg]))
    tophat = {"module": "background", "params": {"method": "tophat", "radius_um": 15}}
    assert any("flattens the cytoplasm" in w for w in warnings([tophat, seg]))
    dna_only = {"module": "background", "params": {"method": "tophat", "channels": [1]}}
    constant = {"module": "background", "params": {"method": "constant"}}
    assert warnings([dna_only, constant, seg]) == []
    surface = {"module": "background", "params": {"method": "polynomial"}}
    assert any("inflated" in w for w in warnings([surface, seg]))


def test_translocation_preset_and_readouts() -> None:
    from hcs_studio.assays.translocation import preset
    from hcs_studio.readouts import resolve

    pipeline = preset({"dna": "Hoechst", "reporter": "YAP"}, channels=["YAP", "Hoechst"],
                      mode="inhibitor", blank_wells=True)  # fmt: skip
    modules = [s.module for s in pipeline.steps]
    assert modules == ["background", "segment_classical", "cytoplasm", "measure_objects", "qc",
                       "translocation"]  # fmt: skip
    assert pipeline.steps[0].params["constant_source"] == "blank_wells"
    segment = pipeline.steps[1].params
    assert (segment["channel"], segment["method"], segment["min_distance_um"]) == (2, "otsu", 2.5)
    assert pipeline.steps[-1].params["mode"] == "inhibitor"
    for column in ("tl_responding_pct", "tl_nc_ratio", "tl_log2_nc", "tl_activation_pct",
                   "tl_fold_vs_unstimulated", "tl_cells", "tl_no_cytoplasm_pct"):  # fmt: skip
        assert resolve(column, "well") is not None, column
    for column in ("tl_nc_ratio", "tl_log2_nc", "tl_nuc_intensity", "tl_cyto_intensity"):
        assert resolve(column, "object") is not None, column
