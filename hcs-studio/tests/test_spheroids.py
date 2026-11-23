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
                                               "spheroids.csv", "wells.csv", "radial_profiles.csv",
                                               "radial_profiles.png"}  # fmt: skip
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

    full = preset({"brightfield": "BF", "live": "Calcein", "dead": "EthD1"},
                  channels=["BF", "Calcein", "EthD1"], invasion=True, organoids=True)  # fmt: skip
    assert full.steps[0].params["protrusion_reach_um"] == 300
    assert (full.steps[1].params["live_channel"], full.steps[1].params["dead_channel"]) == (2, 3)
    assert full.steps[1].params["invasion"] and full.steps[2].params["mode"] == "organoids"
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


def test_core_viability_and_invasion_from_images(home: Path) -> None:
    """Done-when (21b): known necrotic core fraction within ±5 pp; dead fraction, invasion area,
    protrusion count and reach recovered from brightfield + live/dead images."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.spheroids import BrightfieldSpheroid, LiveDeadStains

    specs: dict[tuple[int, int], dict[str, Any]] = {
        (0, 0): {"radius_um": 200, "core_ratio": 0.3},
        (0, 1): {"radius_um": 200, "core_ratio": 0.5},
        (0, 2): {"radius_um": 200, "core_ratio": 0.7},
        (0, 3): {"radius_um": 150, "core_ratio": 0.0, "dead_ratio": 0.0},
        (0, 4): {"radius_um": 120, "core_ratio": 0.5,
                 "strands": [(0, 100, 12), (90, 60, 10), (200, 140, 14), (300, 40, 10)]},
        (0, 5): {"radius_um": 150, "core_ratio": 0.4, "strands": [(45, 80, 12), (160, 120, 12)]},
    }  # fmt: skip
    plate = SyntheticPlate(
        wells=list(specs),
        features=[
            BrightfieldSpheroid(spec=lambda r, c: specs[(r, c)]),
            LiveDeadStains(spec=lambda r, c: specs[(r, c)]),
            ShotNoise(sd=8.0),
        ],
        channels=("Brightfield", "Calcein", "EthD1"),
        size=400,
        pixel_size_um=2.0,
        name="SPH",
    )
    path, truth = plate.write(home / "raw")
    from hcs_studio.assays.spheroids import preset

    mapping = {"brightfield": "Brightfield", "live": "Calcein", "dead": "EthD1"}
    pipeline = preset(mapping, channels=["Brightfield", "Calcein", "EthD1"], invasion=True)
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False).folder  # fmt: skip
    values = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    got = values.to_dict("index")
    for well, data in truth["wells"].items():
        spheroid, stains = data["fields"]["1"]["spheroid"], data["fields"]["1"]["live_dead"]
        invading = spheroid["protrusions"] > 0
        tolerance = 0.05 if invading else 0.02  # the strands count in the spheroid's area
        assert got[well]["sp_core_fraction"] == pytest.approx(
            spheroid["core_fraction"], abs=tolerance
        ), well
        assert got[well]["sp_dead_fraction"] == pytest.approx(stains["dead_fraction"], abs=0.03), (
            well
        )
        assert got[well]["sp_protrusions"] == spheroid["protrusions"], well
        if invading:
            assert got[well]["sp_invasion_area_um2"] == pytest.approx(
                spheroid["invasion_area_um2"], rel=0.1
            )
            assert got[well]["sp_max_invasion_um"] == pytest.approx(
                spheroid["max_invasion_um"], abs=6
            )
            assert got[well]["sp_skeleton_length_um"] > 0.8 * sum(
                s[1] for s in specs[(0, int(well[1:]) - 1)]["strands"]
            )
        else:
            assert got[well]["sp_invasion_area_um2"] == 0
    assert got["A04"]["sp_core_area_um2"] == 0 and got["A04"]["sp_dead_area_um2"] == 0
    # the viable rim: radius minus core radius
    assert got["A02"]["sp_rim_um"] == pytest.approx(100, abs=4)
    profiles = pd.read_csv(folder / "spheroids" / "radial_profiles.csv")
    dead = profiles[(profiles["well"] == "A03") & (profiles["channel"] == "dead")]
    by_depth = dead.groupby("depth")["shell_mean"].mean()
    assert by_depth.iloc[-1] > 5 * by_depth.iloc[0]  # dead stain in the core, not at the edge
    assert (folder / "spheroids" / "radial_profiles.png").exists()


def test_organoid_mode_counts_sizes_and_efficiency(home: Path) -> None:
    from hcs_studio.assays.spheroids import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.spheroids import BrightfieldSpheroid

    counts = {(0, 0): 20, (0, 1): 20, (1, 0): 6, (1, 1): 6, (2, 0): 0}
    specs = {k: {"organoids": n, "organoid_radius_um": (20, 45), "core_ratio": 0}
             for k, n in counts.items()}  # fmt: skip
    plate = SyntheticPlate(
        wells=list(specs),
        features=[BrightfieldSpheroid(spec=lambda r, c: specs[(r, c)]), ShotNoise(sd=8.0)],
        channels=("Brightfield",),
        size=400,
        n_fields=2,
        pixel_size_um=2.0,
        name="ORG",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    layout = set_role(layout, [(1, 0), (1, 1)], "positive_control")
    layout = apply_fields(layout, list(counts), seeding_density=100)
    pipeline = preset({"brightfield": "Brightfield"}, channels=["Brightfield"], organoids=True,
                      min_area_um2=200)  # fmt: skip
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False).folder  # fmt: skip
    got = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well").to_dict("index")
    imaged_mm2 = 2 * (400 * 2.0) ** 2 / 1e6
    for well, data in truth["wells"].items():
        diameters = [d for f in data["fields"].values()
                     for d in f["spheroid"]["organoid_diameters_um"]]  # fmt: skip
        assert got[well]["sp_org_count"] == len(diameters), well
        assert got[well]["sp_ofe_pct"] == pytest.approx(len(diameters), abs=1e-9)  # 100 seeded
        assert got[well]["sp_org_per_mm2"] == pytest.approx(len(diameters) / imaged_mm2)
        if diameters:
            assert got[well]["sp_org_diameter_median_um"] == pytest.approx(
                np.median(diameters), rel=0.03
            )
            assert got[well]["sp_qc_ok"] == 1
    assert got["C01"]["sp_flags"] == "no_organoids" and got["C01"]["sp_qc_ok"] == 0
    assert got["B01"]["sp_volume_vs_vehicle_pct"] < 50
    out = folder / "spheroids"
    assert {"organoid_sizes.csv", "size_distribution.png"} <= {p.name for p in out.iterdir()}
    sizes = pd.read_csv(out / "organoid_sizes.csv")
    assert len(sizes) == 40 + 40 + 12 + 12
    assert "organoids" in (out / "gallery.html").read_text()


def test_invasion_core_and_shells_known_shapes() -> None:
    from hcs_studio.assays.spheroids import (
        ExtraOptions,
        invasion_measures,
        necrotic_core,
        radial_shells,
    )

    yy, xx = np.mgrid[0:200, 0:200]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 <= 50**2
    strand = (np.abs(yy - 100) <= 3) & (xx >= 100) & (xx <= 180)  # 30 px beyond the edge
    mask = disc | strand
    values = invasion_measures(mask, ExtraOptions(invasion_core_px=10, min_protrusion_px=5))
    assert values["protrusions"] == 1
    assert values["max_invasion_px"] == pytest.approx(30, abs=1.5)
    assert values["invasion_px2"] == pytest.approx(7 * 30, rel=0.1)
    assert values["skeleton_branches"] >= 1
    shells = radial_shells(disc, 5)
    assert set(np.unique(shells[disc])) == {0, 1, 2, 3, 4} and shells[100, 100] == 4
    plane = np.where((yy - 100) ** 2 + (xx - 100) ** 2 <= 25**2, 100.0, 400.0)
    core, contrast = necrotic_core(plane, disc, ExtraOptions(core_smooth_px=1.0))
    assert contrast > 0.5 and core.sum() == pytest.approx(np.pi * 25**2, rel=0.05)
    flat, low = necrotic_core(np.full(plane.shape, 400.0), disc, ExtraOptions())
    assert not flat.any() and low == pytest.approx(0)


@pytest.mark.parametrize("length", [60.0, 120.0, 300.0])
def test_attenuation_length_recovered(length: float) -> None:
    from types import SimpleNamespace

    from hcs_studio.processing.attenuation import correction_factors, plane_signal
    from hcs_studio.synthetic.spheroids import ConfocalSpheroid

    image = np.zeros((1, 60, 120, 120))
    rng = np.random.default_rng(0)
    ctx = SimpleNamespace(row=0, col=0, image=image, rng=rng, pixel_size_um=2.0, z_step_um=4.0)
    ConfocalSpheroid(spec=lambda r, c: {"radius_um": 90, "attenuation_um": length}).draw(ctx)
    stack = image[0] + rng.normal(0, 5, image[0].shape)
    signal, used, background = plane_signal(stack, 90, 0.5)
    _factors, info = correction_factors(signal, used, "exponential", 4.0)
    assert info["attenuation_length_um"] == pytest.approx(length, rel=0.03)
    assert background.mean() == pytest.approx(100, abs=3)
    flat, _per_plane = correction_factors(signal, used, "per_plane", 4.0)
    corrected = signal * flat
    assert np.nanstd(corrected[used]) / np.nanmean(corrected[used]) < 0.02


def test_attenuation_step_flat_stack_and_gain_cap() -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.attenuation import Attenuation, AttenuationParams
    from hcs_studio.processing.base import ImageData

    metadata = ImageMetadata(channels=(ChannelInfo("Hoechst"),), pixel_size_um=2.0, z_step_um=4.0)
    zz, yy, xx = np.mgrid[0:30, 0:60, 0:60]
    ball = ((zz - 15) * 4) ** 2 + ((yy - 30) * 2) ** 2 + ((xx - 30) * 2) ** 2 <= 50**2
    flat = ImageData.from_array((100 + 800 * ball)[None].astype(float), metadata)
    out = Attenuation(AttenuationParams()).apply(flat, None)
    assert np.allclose(out.data, flat.data) and out.linear  # no loss with depth: unchanged
    assert any("does not fall" in w for w in out.reports[-1]["warnings"])
    steep = (100 + 800 * ball * np.exp(-np.clip(zz * 4 - 10, 0, None) / 10.0))[None]
    capped = Attenuation(AttenuationParams(max_gain=5)).apply(
        ImageData.from_array(steep.astype(float), metadata), None
    )
    report = capped.reports[-1]
    assert report["channels"]["1"]["max_gain"] == pytest.approx(5)
    assert any("capped" in w for w in report["warnings"])


def test_confocal_3d_volume_core_viability_and_coverage(home: Path) -> None:
    """Done-when (21c): 3D volume within ±5 %, core fraction within ±5 pp, attenuation
    corrected, incomplete Z coverage flagged."""
    from hcs_studio.assays.spheroids import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.spheroids import ConfocalSpheroid

    specs: dict[tuple[int, int], dict[str, Any]] = {
        (0, 0): {"radius_um": 90, "core_ratio": 0.5},
        (0, 1): {"radius_um": 70, "core_ratio": 0.6},
        (0, 2): {"radius_um": 90, "core_ratio": 0.5, "attenuation_um": 80},
        (0, 3): {"radius_um": 90, "core_ratio": 0.5, "z_centre_um": 200},
    }
    plate = SyntheticPlate(
        wells=list(specs),
        features=[
            ConfocalSpheroid(spec=lambda r, c: specs[(r, c)], dead_channel=1),
            ShotNoise(sd=5.0),
        ],
        channels=("Hoechst", "EthD1"),
        size=128,
        planes=60,
        z_step_um=4.0,
        pixel_size_um=2.0,
        name="CONF",
    )
    path, truth = plate.write(home / "raw")
    pipeline = preset({"fluorescence": "Hoechst", "dead": "EthD1"},
                      channels=["Hoechst", "EthD1"], z_planes=60)  # fmt: skip
    modules = [s.module for s in pipeline.steps]
    assert modules == ["attenuation", "detect_spheroids", "measure_spheroids", "spheroid"]
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False).folder  # fmt: skip
    got = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well").to_dict("index")
    for well, data in truth["wells"].items():
        expected = data["fields"]["1"]["confocal_spheroid"]
        values = got[well]
        assert values["sp_core_fraction"] == pytest.approx(
            expected["core_volume_fraction"], abs=0.05
        )
        assert values["sp_dead_fraction"] == pytest.approx(
            expected["core_volume_fraction"], abs=0.03
        )
        if expected["complete_z"]:
            assert values["sp_volume_um3"] == pytest.approx(expected["volume_um3"], rel=0.02), well
            assert values["sp_volume_um3"] == values["sp_volume_3d_um3"]
            assert values["sp_sphericity"] > 0.95 and values["sp_z_coverage"] == 1.0
            assert pd.isna(values["sp_flags"]) and values["sp_qc_ok"] == 1
        else:
            assert values["sp_flags"] == "incomplete_z" and values["sp_qc_ok"] == 0
            assert values["sp_z_coverage"] == pytest.approx(expected["z_coverage"], abs=0.05)
    # without the attenuation correction the dim deep half of a spheroid is lost
    plain_pipeline = pipeline.model_copy(update={"steps": pipeline.steps[1:]})
    uncorrected = run_pipeline(open_plate(path), plain_pipeline, output_root=home / "plain",
                               save_images=False).folder  # fmt: skip
    plain = pd.read_csv(uncorrected / "tables" / "per_well_summary.csv").set_index("well")
    lost = (
        plain.loc["A03", "sp_volume_um3"]
        / truth["wells"]["A03"]["fields"]["1"]["confocal_spheroid"]["volume_um3"]
    )
    assert float(lost) < 0.8
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert "attenuation" in json.dumps(manifest["pipeline"])


def test_nuclei_counted_per_spheroid_and_3d_checks() -> None:
    from hcs_studio.assays.spheroids import (
        DetectSpheroids,
        DetectSpheroidsParams,
        MeasureSpheroids,
        MeasureSpheroidsParams,
    )
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData, LabelSet, MeasureContext, ModuleError

    metadata = ImageMetadata(channels=(ChannelInfo("Hoechst"),), pixel_size_um=2.0, z_step_um=4.0)
    zz, yy, xx = np.mgrid[0:20, 0:80, 0:80]
    ball = ((zz - 10) * 4) ** 2 + ((yy - 40) * 2) ** 2 + ((xx - 40) * 2) ** 2 <= 30**2
    image = ImageData.from_array((100 + 800 * ball)[None].astype(float), metadata)
    spheroids = LabelSet("spheroids", ball.astype(np.int32))
    nuclei = np.zeros(ball.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate([(10, 40, 40), (8, 35, 45), (12, 45, 35)], start=1):
        nuclei[z - 1 : z + 2, y - 2 : y + 3, x - 2 : x + 3] = i
    children = LabelSet("nuclei", nuclei, parent="spheroids",
                        parent_ids=np.array([0, 1, 1, 1], dtype=np.int32))  # fmt: skip
    labelled = image.with_labels(spheroids, "seg").with_labels(children, "nuclei")
    context = MeasureContext(raw=labelled, corrected=labelled, labels=labelled.labels)
    table = MeasureSpheroids(MeasureSpheroidsParams(channel=1)).measure(context)["objects"]
    row = table.iloc[0].to_dict()
    assert row["nuclei_count"] == 3
    volume = row["volume_3d_um3"]
    assert volume == pytest.approx(4 / 3 * np.pi * 30**3, rel=0.08)
    assert row["nuclei_per_mm3"] == pytest.approx(3 / (volume / 1e9))
    assert not row["z_incomplete"]
    brightfield = DetectSpheroids(DetectSpheroidsParams(channel=1))
    with pytest.raises(ModuleError, match="Z projection"):
        brightfield.apply(image, None)
