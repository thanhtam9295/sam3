"""Phase 20e: proliferation markers, micronuclei and multinucleation with known answers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.proliferation import (
    Proliferation,
    ProliferationParams,
    find_micronuclei,
    mixture_threshold,
)
from hcs_studio.synthetic.proliferation import write_proliferation_run

VEHICLE: dict[str, float] = {"ki67": 0.7, "edu": 0.35, "ph3": 0.03, "mn": 0.02, "multi": 0.02}
NOCODAZOLE: dict[str, float] = {"ki67": 0.1, "edu": 0.02, "ph3": 0.0, "mn": 0.15, "multi": 0.3}
Wells = dict[str, tuple[str, Mapping[str, float], int]]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _plate(folder: Path, secondary: bool = True) -> dict[str, dict[str, float]]:
    wells: Wells = {
        "A01": ("negative_control", VEHICLE, 2000),
        "A02": ("negative_control", VEHICLE, 2000),
        "B01": ("positive_control", NOCODAZOLE, 2000),
        "B02": ("positive_control", NOCODAZOLE, 2000),
        "C01": ("sample", {"ki67": 0.4, "edu": 0.2, "ph3": 0.1, "mn": 0.05, "multi": 0.1}, 2000),
    }
    if secondary:
        wells["D01"] = ("unstained", {}, 1000)
    return write_proliferation_run(folder, wells, seed=1)


def _run(folder: Path, **params: Any) -> Any:
    from hcs_studio.analysis.table_steps import run_table_steps

    markers = {"ki67_channel": "Ki67", "edu_channel": "EdU", "ph3_channel": "pH3"}
    return run_table_steps(folder, [Proliferation(ProliferationParams(**{**markers, **params}))])


def test_find_micronuclei_geometry() -> None:
    yy, xx = np.mgrid[0:80, 0:120]
    nuclei = np.zeros((80, 120), dtype=np.int32)
    dna = np.zeros((80, 120))

    def disc(cy: float, cx: float, r: float, label: int = 0) -> None:
        inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        dna[inside] = 500
        if label:
            nuclei[inside] = label

    disc(40, 25, 9, 1)
    disc(40, 85, 9, 2)
    disc(40, 39, 2)  # 3 px from nucleus 1: a micronucleus
    disc(25, 85, 3, 3)  # segmented as a "nucleus", 3 px from nucleus 2: moved
    disc(40, 96, 2)  # touching nucleus 2 (a bleb): not a micronucleus
    disc(70, 55, 2)  # 20+ px from any nucleus: debris
    disc(10, 25, 0.5)  # 1 px: below the minimum size
    big, mn, parents, moved = find_micronuclei(
        nuclei, dna, max_area_fraction=0.2, min_area_px=3, search_px=8, threshold_fraction=0.4
    )
    assert moved == 1 and set(np.unique(big)) == {0, 1, 2}
    assert int(np.max(mn)) == 2
    found = {int(parents[i]): (np.nonzero(mn == i)[0].mean(), np.nonzero(mn == i)[1].mean())
             for i in (1, 2)}  # fmt: skip
    assert found[1] == pytest.approx((40, 39), abs=0.5)
    assert found[2] == pytest.approx((25, 85), abs=0.5)


def test_mixture_threshold_does_not_depend_on_the_positive_share() -> None:
    from hcs_studio.assays.cell_cycle import otsu_log

    rng = np.random.default_rng(0)
    thresholds, otsu = [], []
    for share in (0.15, 0.5, 0.85):
        k = int(share * 4000)
        values = np.concatenate([rng.lognormal(np.log(800), 0.35, k),
                                 rng.lognormal(np.log(80), 0.35, 4000 - k)])  # fmt: skip
        threshold, d = mixture_threshold(values)
        thresholds.append(threshold)
        otsu.append(otsu_log(values))
        assert d > 6
    # equal spreads: the populations are equally likely at the geometric mean, √(80·800) ≈ 253
    assert thresholds == pytest.approx([np.sqrt(80 * 800)] * 3, rel=0.05)
    assert max(otsu) / min(otsu) > max(thresholds) / min(thresholds)


def test_markers_micronuclei_and_multinucleation_from_images(home: Path) -> None:
    """Done-when (20e): known % Ki67-positive, micronucleus and multinucleation frequencies
    are recovered from images through the preset pipeline."""
    from hcs_studio.assays.proliferation import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.readouts import resolve
    from hcs_studio.synthetic import Background, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.proliferation import ProliferationCells

    fractions = {0: {"ki67": 0.7, "mn": 0.1, "multi": 0.05},
                 1: {"ki67": 0.15, "mn": 0.4, "multi": 0.4}}  # fmt: skip
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (1, 0), (1, 1)],
        features=[
            Background(level=100.0),
            ProliferationCells(fractions=lambda r, c: fractions[r]),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "Ki67", "CellMask"),
        size=256,
        n_fields=4,
        name="PF",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    layout = set_role(layout, [(1, 0), (1, 1)], "positive_control")
    pipeline = preset({"dna": "DAPI", "ki67": "Ki67", "cells": "CellMask"},
                      channels=["DAPI", "Ki67", "CellMask"], nucleus_diameter_um=14 * 0.65,
                      cell_diameter_um=32 * 0.65)  # fmt: skip
    result = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False)  # fmt: skip
    summary = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    values = summary.to_dict("index")
    for well, data in truth["wells"].items():
        cells = [c for f in data["fields"].values() for c in f["proliferation"]["cells"]]
        nuclei = sum(c["nuclei"] for c in cells)
        got = values[well]
        assert got["pf_nuclei"] == nuclei and got["pf_cells"] == len(cells), well
        ki67 = 100 * sum(c["ki67"] * c["nuclei"] for c in cells) / nuclei
        assert got["pf_ki67_pos_pct"] == pytest.approx(ki67, abs=1.5), well
        mn = 100 * sum(c["micronuclei"] for c in cells) / nuclei
        assert got["pf_mn_per_100"] == pytest.approx(mn, abs=1.0), well
        multi = 100 * float(np.mean([c["nuclei"] > 1 for c in cells]))
        assert got["pf_multinucleated_pct"] == pytest.approx(multi, abs=1.0), well
    numeric = summary.select_dtypes("number").columns
    unresolved = [c for c in numeric if resolve(c, "well") is None]
    assert not [c for c in unresolved if c.startswith(("pf_", "micronuclei"))], unresolved
    folder = result.folder / "assays" / "proliferation"
    calibration = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["markers"]["ki67"]["method"] == "vehicle_mixture"
    assert calibration["markers"]["ki67"]["ashman_d"] > 4 and calibration["warnings"] == []
    report = (folder / "report.html").read_text(encoding="utf-8")
    assert report.count("<img") == 1 and "Ki67" in report


def test_thresholds_from_secondary_only_and_vehicle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    truth = _plate(run)
    auto = _run(run)
    markers = auto.steps[0]["state"]["markers"]
    assert {m["method"] for m in markers.values()} == {"secondary_only"}
    for well, expected in truth.items():
        for key in ("ki67", "edu", "ph3"):
            got = auto.per_well[well][f"pf_{key}_pos_pct"]
            assert got == pytest.approx(expected[f"{key}_pos_pct"], abs=2.0), (well, key)
        assert auto.per_well[well]["pf_mn_pct"] == pytest.approx(expected["mn_pct"], abs=0.01)
        assert auto.per_well[well]["pf_multinucleated_pct"] == pytest.approx(
            expected["multinucleated_pct"], abs=0.01
        )
    z_prime = auto.steps[0]["state"]["z_prime"]
    assert z_prime["pf_ki67_pos_pct"] > 0.9 and z_prime["pf_mn_per_100"] > 0.9
    assert auto.per_well["B01"]["pf_ki67_pos_vs_vehicle"] == pytest.approx(10 / 70, abs=0.02)
    # without secondary-only wells: two populations in the vehicle; pH3 positives are rare
    vehicle_run = tmp_path / "vehicle"
    truth = _plate(vehicle_run, secondary=False)
    vehicle = _run(vehicle_run)
    methods = {k: m["method"] for k, m in vehicle.steps[0]["state"]["markers"].items()}
    assert methods == {"ki67": "vehicle_mixture", "edu": "vehicle_mixture", "ph3": "vehicle_robust"}
    for well, expected in truth.items():
        for key in ("ki67", "edu", "ph3"):
            got = vehicle.per_well[well][f"pf_{key}_pos_pct"]
            assert got == pytest.approx(expected[f"{key}_pos_pct"], abs=1.0), (well, key)
    fixed = _run(vehicle_run, thresholds={"ki67": 250.0})
    assert fixed.steps[0]["state"]["markers"]["ki67"]["method"] == "fixed"
    with pytest.raises(Exception, match="secondary-only"):
        _run(vehicle_run, threshold_method="secondary_only")


def test_unimodal_vehicle_warns(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_proliferation_run(run, {"A01": ("negative_control", {"ki67": 0.0}, 3000),
                                  "B01": ("sample", {"ki67": 0.5}, 3000)})  # fmt: skip
    state = _run(run, edu_channel="", ph3_channel="").steps[0]["state"]
    assert any("Ashman" in w for w in state["warnings"])


def test_mitotic_errors_needs_independent_cells(tmp_path: Path) -> None:
    from hcs_studio.assays.proliferation import MitoticErrors, MitoticErrorsParams
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData, LabelSet, ModuleError

    metadata = ImageMetadata(channels=(ChannelInfo(name="DAPI"),), pixel_size_um=0.65)
    nuclei = np.zeros((1, 40, 40), dtype=np.int32)
    nuclei[0, 10:20, 10:20] = 1
    image = ImageData.from_array(np.zeros((1, 1, 40, 40)), metadata)
    image = image.with_labels(LabelSet("nuclei", nuclei), "seg")
    grown = LabelSet(
        "cells", nuclei.copy(), parent="nuclei", parent_ids=np.arange(2, dtype=np.int32)
    )
    step = MitoticErrors(MitoticErrorsParams(channel=1, cells="cells"))
    with pytest.raises(ModuleError, match="right after"):
        step.apply(image.with_labels(grown, "expand"), None)
    cells = LabelSet("cells", (nuclei > 0).astype(np.int32) * 5)
    out = step.apply(image.with_labels(cells, "seg"), None)
    linked = out.label_set("nuclei")
    assert linked.parent == "cells" and linked.parent_ids is not None
    assert int(linked.parent_ids[1]) == 5
    assert [ls.name for ls in out.labels] == ["nuclei", "cells", "micronuclei"]


def test_proliferation_preset_cli_and_readouts(tmp_path: Path, home: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.assays.proliferation import preset
    from hcs_studio.readouts import resolve

    pipeline = preset({"dna": "Hoechst", "edu": "EdU"}, channels=["EdU", "Hoechst"],
                      micronuclei=False)  # fmt: skip
    modules = [s.module for s in pipeline.steps]
    assert modules == ["background", "segment_classical", "measure_objects", "qc", "proliferation"]
    params = pipeline.steps[-1].params
    assert (params["dna_channel"], params["edu_channel"], params["ki67_channel"]) == (
        "Hoechst", "EdU", "")  # fmt: skip
    with_cells = preset({"dna": "DAPI", "ki67": "Ki67", "cells": "Actin"},
                        channels=["DAPI", "Ki67", "Actin"])  # fmt: skip
    modules = [s.module for s in with_cells.steps]
    assert modules[1:4] == ["segment_classical", "segment_classical", "mitotic_errors"]
    assert with_cells.steps[2].params["channel"] == 3
    assert with_cells.steps[3].params["cells"] == "cells"
    run = tmp_path / "run"
    _plate(run)
    command = ["assay", "run", str(run), "proliferation", "--map", "dna=DAPI", "--map",
               "ki67=Ki67", "--map", "ph3=pH3", "--reason", "test"]  # fmt: skip
    assert main(command) == 0
    summary = pd.read_csv(run / "tables" / "per_well_summary.csv").set_index("well")
    ki67 = summary["pf_ki67_pos_pct"].astype(float).to_dict()
    assert ki67["A01"] > 60 > 20 > ki67["B01"]
    for column in ("pf_ki67_pos_pct", "pf_edu_median", "pf_ph3_pos_vs_vehicle", "pf_nuclei",
                   "pf_mn_pct", "pf_mn_per_100", "pf_cells", "pf_multinucleated_pct",
                   "pf_binucleated_pct", "pf_nuclei_per_cell"):  # fmt: skip
        assert resolve(column, "well") is not None, column
    assert resolve("pf_micronuclei", "object") is not None
