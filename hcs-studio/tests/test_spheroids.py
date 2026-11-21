"""Phase 21a: spheroid detection, size, volume, QC flags, gallery and a 384-well batch."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.spheroids import detect, spheroid_table


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _run(home: Path, specs: dict[tuple[int, int], dict[str, Any]], **options: Any) -> Any:
    from hcs_studio.assays.spheroids import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.spheroids import BrightfieldSpheroid

    plate = SyntheticPlate(
        wells=list(specs),
        features=[BrightfieldSpheroid(spec=lambda r, c: specs[(r, c)]), ShotNoise(sd=8.0)],
        channels=("Brightfield",),
        size=options.pop("size", 320),
        pixel_size_um=options.pop("pixel_size_um", 2.0),
        name="SPH",
    )
    path, truth = plate.write(home / "raw")
    pipeline = preset({"brightfield": "Brightfield"}, channels=["Brightfield"])
    result = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False, **options)  # fmt: skip
    return result.folder, truth


def test_volume_and_qc_flags_from_brightfield(home: Path) -> None:
    """Done-when (21a): volume within ±5 % (sphere) for spheroids of known radius, through the
    preset; each QC problem is flagged, and only that one."""
    specs: dict[tuple[int, int], dict[str, Any]] = {
        (0, 0): {"radius_um": 200},
        (0, 1): {"radius_um": 120},
        (0, 2): {"radius_um": 70},
        (0, 3): {"radius_um": 150, "blur_px": 8},
        (0, 4): {"radius_um": 120, "second_radius_um": 90},
        (0, 5): {"radius_um": 120, "debris": 4},
        (0, 6): {"radius_um": 120, "attached_um": 60},
        (0, 7): {"radius_um": 120, "offset_um": (150, 150)},
        (0, 8): {"radius_um": 260, "offset_um": (0, 200)},
        (0, 9): {"radius_um": 1},  # empty well: only the meniscus
    }
    folder, truth = _run(home, specs)
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    values = summary.to_dict("index")
    for well, data in truth["wells"].items():
        expected = data["fields"]["1"]["spheroid"]
        got = values[well]
        flags = set(str(got["sp_flags"]).split(",")) - {"nan", ""}
        if well == "A10":
            assert flags == {"no_spheroid"} and got["sp_objects"] == 0 and got["sp_qc_ok"] == 0
            continue
        assert flags == set(expected["flags"]), well
        if not flags & {"touching_edge", "irregular"}:  # cut or grown objects are not spheres
            error = got["sp_volume_um3"] / expected["volume_um3"] - 1
            assert abs(error) < (0.05 if "out_of_focus" in flags else 0.01), (well, error)
    for well in ("A01", "A02", "A03"):
        assert values[well]["sp_circularity"] > 0.85 and values[well]["sp_solidity"] > 0.98
        assert values[well]["sp_compactness"] == pytest.approx(1.0, abs=0.01)
        assert values[well]["sp_darkness"] == pytest.approx(0.6, abs=0.05)
        assert values[well]["sp_edge_sigma_um"] < 3
    assert values["A04"]["sp_edge_sigma_um"] > 10
    out = folder / "spheroids"
    assert {p.name for p in out.iterdir()} == {"calibration.json", "crops", "gallery.html",
                                               "spheroids.csv", "wells.csv"}  # fmt: skip
    assert len(list((out / "crops").iterdir())) == 9  # every well with a spheroid
    gallery = (out / "gallery.html").read_text()
    assert gallery.count('class="card"') == 10 and 'data-sp-volume-um3="' in gallery
    assert '<option value="sp-volume-um3">' in gallery
    per_spheroid = pd.read_csv(out / "spheroids.csv")
    assert per_spheroid["sp_main"].sum() == 9
    calibration = json.loads((out / "calibration.json").read_text())
    assert calibration["flag_counts"]["multiple"] == 1
    assert "spheroids/gallery.html" in (folder / "index.html").read_text()


def test_fluorescence_and_dark_methods() -> None:
    yy, xx = np.mgrid[0:200, 0:200]
    disc = (yy - 100) ** 2 + (xx - 90) ** 2 <= 40**2
    rng = np.random.default_rng(0)
    image = np.where(disc, 900.0, 100.0) + rng.normal(0, 5, disc.shape)
    for method, data in (("bright", image), ("dark", 1000 - image)):
        labels = detect(data, method=method, texture_px=3.0, min_area_px=100)
        assert int(np.max(labels)) == 1, method
        assert (labels > 0).sum() == pytest.approx(disc.sum(), rel=0.02), method


def test_spheroid_table_known_shapes() -> None:
    yy, xx = np.mgrid[0:200, 0:240]
    labels = np.zeros((200, 240), dtype=np.int32)
    labels[((yy - 100) / 60) ** 2 + ((xx - 120) / 30) ** 2 <= 1] = 1  # ellipse 120 × 60 px
    labels[(yy - 10) ** 2 + (xx - 230) ** 2 <= 15**2] = 2  # cut by the border
    table = spheroid_table(labels, None, 2.0).set_index("object_id")
    ellipse = table.loc[1].to_dict()
    assert ellipse["volume_ellipsoid_um3"] == pytest.approx(math.pi / 6 * 240 * 120**2, rel=0.02)
    assert ellipse["offcentre"] == pytest.approx(0.0, abs=0.01)
    assert not ellipse["touching_edge"] and table.loc[2, "touching_edge"]
    assert ellipse["compactness"] > 1.1  # elongated: more spread than a disc


def test_384_plate_batch_and_dose_response_on_volume(home: Path) -> None:
    """Done-when (21a): a 384-well plate runs in batch with a gallery, and IC50s on volume."""
    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology

    v0, vmin = 4 / 3 * math.pi * 180**3, 4 / 3 * math.pi * 80**3
    ic50 = {r: 10 ** (-7.5 + 0.1 * r) for r in range(16)}
    concentrations = [1e-9 * 3**i for i in range(10)]

    def volume(r: int, c: int) -> float:
        if c < 2:
            return v0
        if c >= 22:
            return vmin
        return float(vmin + (v0 - vmin) / (1 + concentrations[(c - 2) % 10] / ic50[r]))

    specs = {(r, c): {"radius_um": (3 * volume(r, c) / (4 * math.pi)) ** (1 / 3)}
             for r in range(16) for c in range(24)}  # fmt: skip
    layout = PlateLayout.for_plate_type(384)
    layout = set_role(layout, [(r, c) for r in range(16) for c in (0, 1)], "negative_control")
    layout = set_role(layout, [(r, c) for r in range(16) for c in (22, 23)], "positive_control")
    for r in range(16):
        for c in range(2, 22):
            treatment = Treatment(f"cpd{r:02d}", concentrations[(c - 2) % 10], "µM")
            layout = apply_fields(layout, [(r, c)], role="sample", cell_line="HCT116",
                                  treatments=(treatment,))  # fmt: skip
    folder, truth = _run(home, specs, size=256, pixel_size_um=2.5, layout=layout, workers=2)
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    errors = [summary.loc[w, "sp_volume_um3"] / d["fields"]["1"]["spheroid"]["volume_um3"] - 1
              for w, d in truth["wells"].items()]  # fmt: skip
    assert len(errors) == 384 and max(abs(e) for e in errors) < 0.02
    assert (summary["sp_qc_ok"] == 1).all()
    vehicle = summary.loc[[f"{r}01" for r in "ABCDEFGHIJKLMNOP"], "sp_volume_vs_vehicle_pct"]
    assert vehicle.between(98, 102).all()
    calibration = json.loads((folder / "spheroids" / "calibration.json").read_text())
    assert calibration["z_prime_volume"] > 0.9
    assert len(list((folder / "spheroids" / "crops").iterdir())) == 384
    options = PharmOptions(readouts=["sp_volume_um3"], normalization="percent_of_control",
                           figures=False, pdf=False)  # fmt: skip
    result = run_pharmacology([folder], options, write=False)
    curves = result.readouts["sp_volume_um3"].curves
    assert len(curves) == 16
    for curve in curves:
        expected = math.log10(ic50[int(curve.compound[3:])])
        assert curve.fit.log_ec50 == pytest.approx(expected, abs=0.05), curve.compound


def test_preset_readouts_and_brightfield_colour() -> None:
    from hcs_studio.assays.spheroids import preset
    from hcs_studio.model.plate import ChannelInfo
    from hcs_studio.readouts import resolve
    from hcs_studio.utils.display import channel_colormap

    pipeline = preset({"fluorescence": "Calcein"}, channels=["Hoechst", "Calcein"])
    assert [s.module for s in pipeline.steps] == ["detect_spheroids", "measure_spheroids",
                                                  "spheroid"]  # fmt: skip
    assert pipeline.steps[0].params["method"] == "bright"
    assert pipeline.steps[0].params["channel"] == 2
    for column in ("sp_volume_um3", "sp_volume_sphere_um3", "sp_volume_vs_vehicle_pct",
                   "sp_diameter_um", "sp_solidity", "sp_darkness", "sp_edge_sigma_um",
                   "sp_qc_ok", "spheroids_darkness_median"):  # fmt: skip
        assert resolve(column, "well") is not None, column
    assert resolve("edge_sigma_um", "object") is not None
    assert channel_colormap(ChannelInfo("Brightfield"), 0) == "gray"
    assert channel_colormap(ChannelInfo("BF"), 3) == "gray"
    assert channel_colormap(ChannelInfo("DAPI"), 0) == "blue"
