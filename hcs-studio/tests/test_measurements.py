"""Measurements, colocalization and QC against synthetic data with a known answer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage, stats

from hcs_studio.analysis import coloc, features
from hcs_studio.analysis.measure import QcParams, focus_score
from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.output.tables import merge_object_tables, qc_flags, well_summary
from hcs_studio.processing.base import (
    ImageData,
    LabelSet,
    MeasureContext,
    MeasurementModule,
    ModuleError,
    create_module,
)

META = ImageMetadata(
    channels=(
        ChannelInfo("DAPI", emission_nm=450),
        ChannelInfo("GFP", emission_nm=520),
        ChannelInfo("RFP"),
    ),
    pixel_size_um=0.5,
    objective_na=1.0,
    bit_depth=12,
)


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> Any:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2


def measurer(name: str, params: dict[str, Any] | None = None) -> MeasurementModule:
    module = create_module(name, params)
    assert isinstance(module, MeasurementModule)
    return module


def context(
    data: Any, labels: tuple[LabelSet, ...] = (), metadata: ImageMetadata = META, raw: Any = None
) -> MeasureContext:
    image = ImageData(np.asarray(data, dtype=np.float32), metadata, labels=labels)
    raw_image = ImageData(np.asarray(raw if raw is not None else data, dtype=np.float32), metadata)
    return MeasureContext(raw=raw_image, corrected=image, labels=labels)


# ------------------------------------------------------------ coloc basics


def test_identical_and_anticorrelated_channels() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 1000, (64, 64))
    same = coloc.colocalize(a, a)
    assert same.pearson == pytest.approx(1.0) and same.spearman == pytest.approx(1.0)
    assert same.icq == pytest.approx(0.5, abs=0.01)
    assert (same.m1, same.m2, same.tm1, same.tm2) == pytest.approx((1, 1, 1, 1))
    assert same.overlap == pytest.approx(1.0)
    anti = coloc.colocalize(a, 1001 - a)
    assert anti.pearson == pytest.approx(-1.0)
    assert any("no positive correlation" in w for w in anti.warnings)


def test_statistics_match_scipy() -> None:
    rng = np.random.default_rng(1)
    a = rng.gamma(2, 100, 5000)
    b = 0.4 * a + rng.gamma(2, 80, 5000)
    assert coloc.pearson(a, b) == pytest.approx(stats.pearsonr(a, b).statistic, abs=1e-12)
    assert coloc.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-12)


def test_manders_known_overlap() -> None:
    a = np.zeros((40, 40))
    b = np.zeros((40, 40))
    a[0:20, 0:20] = 100  # four 10×10 quadrants of A …
    b[0:10, 0:10] = 50  # … one of which B shares,
    b[30:40, 30:40] = 50  # plus one region of B alone.
    m1, m2 = coloc.manders(a, b)
    assert (m1, m2) == pytest.approx((0.25, 0.5))
    tm1, tm2 = coloc.thresholded_manders(a, b, 60, 10)
    assert tm1 == pytest.approx(0.25) and tm2 == pytest.approx(0.5)
    nothing_above, tm2_high = coloc.thresholded_manders(a, b, 100, 10)
    assert np.isnan(nothing_above) and tm2_high == 0.0  # no A pixel exceeds 100


def test_costes_threshold_matches_brute_force_and_separates_background() -> None:
    rng = np.random.default_rng(2)
    signal = np.zeros((96, 96))
    signal[24:72, 24:72] = rng.uniform(400, 1500, (48, 48))
    a = signal + rng.normal(50, 12, signal.shape)
    b = 0.7 * signal + rng.normal(40, 12, signal.shape)
    result = coloc.costes_threshold(a, b)
    assert not result.warning and result.slope == pytest.approx(0.7, abs=0.03)
    # Independent check: scan thresholds from the top; the first with r(below) <= 0.
    av, bv = a.ravel(), b.ravel()
    expected = None
    for t in np.linspace(float(np.max(av)), float(np.min(av)), 4000):
        below = (av < t) & (bv < result.slope * t + result.intercept)
        if below.sum() >= 3 and stats.pearsonr(av[below], bv[below]).statistic <= 0:
            expected = t
            break
    assert expected is not None
    # The first crossing is located to within one step of the downward scan (1/256 of the
    # range); within that step r(T) fluctuates around 0 at the noise floor.
    scan_step = (float(np.max(av)) - float(np.min(av))) / 256
    assert result.threshold_a == pytest.approx(expected, abs=scan_step)
    assert a[signal == 0].max() < result.threshold_a + 60 and result.threshold_a < 400 + 50
    full = coloc.colocalize(a, b)
    # With uncorrelated background, the threshold can settle anywhere in the background's
    # range (r of pure noise fluctuates around 0), so some noise enters tM1's denominator.
    assert full.tm1 > 0.9 and full.tm2 > 0.9 and full.threshold_method == "costes (and)"


def test_randomization_separates_real_from_chance_correlation() -> None:
    rng = np.random.default_rng(3)
    shared = ndimage.gaussian_filter(rng.uniform(0, 1, (96, 96)), 2) * 1000
    a = shared + rng.normal(0, 20, shared.shape)
    b = shared + rng.normal(0, 20, shared.shape)
    real = coloc.colocalize(a, b, randomizations=99, block=(5, 5))
    assert real.p_value == pytest.approx(0.01) and abs(real.random_r_mean) < 0.1
    other = ndimage.gaussian_filter(rng.uniform(0, 1, (96, 96)), 2) * 1000
    chance = coloc.colocalize(a, other, randomizations=99, block=(5, 5))
    assert chance.p_value > 0.05
    with pytest.raises(ValueError, match="block size"):
        coloc.colocalize(a, b, randomizations=5)


def test_block_scramble_permutes_blocks() -> None:
    image = np.arange(36).reshape(6, 6)
    scrambled = coloc.block_scramble(image, (3, 3), np.random.default_rng(0))
    assert sorted(scrambled.ravel()) == sorted(image.ravel())
    blocks = {tuple(image[i : i + 3, j : j + 3].ravel()) for i in (0, 3) for j in (0, 3)}
    assert {
        tuple(scrambled[i : i + 3, j : j + 3].ravel()) for i in (0, 3) for j in (0, 3)
    } == blocks


def test_mask_restricts_the_statistics() -> None:
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 100, (40, 40))
    b = a.copy()
    b[:, 20:] = 100 - a[:, 20:]  # anti-correlated on the right half
    mask = np.zeros_like(a, dtype=bool)
    mask[:, :20] = True
    assert coloc.colocalize(a, b, mask).pearson == pytest.approx(1.0)
    assert coloc.colocalize(a, b, ~mask).pearson == pytest.approx(-1.0)


def test_channel_shift_detection_and_registration(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    a = ndimage.gaussian_filter(rng.uniform(0, 1, (128, 128)), 2) * 1000
    b = ndimage.shift(a, (2.3, -1.6), order=3, mode="nearest")
    assert coloc.channel_shift(a, b) == pytest.approx((2.3, -1.6), abs=0.15)

    data = np.stack([a, b, a])[:, None].astype(np.float32)
    shifted = create_module("register_channels", {"shifts_um": {2: [1.15, -0.8]}})
    from types import SimpleNamespace

    state = shifted.fit(SimpleNamespace(plate=SimpleNamespace(metadata=META)))
    fixed = shifted.apply(ImageData(data, META), state)
    assert fixed.linear and coloc.channel_shift(a, fixed.data[1, 0]) == pytest.approx(
        (0, 0), abs=0.15
    )

    beads = tmp_path / "beads.ome.tif"
    tifffile.imwrite(
        beads, data[:, 0].astype(np.float32), metadata={"axes": "CYX", "PhysicalSizeX": 0.5}
    )
    from_beads = create_module("register_channels", {"bead_image": str(beads)})
    state = from_beads.fit(SimpleNamespace(plate=SimpleNamespace(metadata=META)))
    # The bead file is read as a (Z, Y, X) stack: compare the Y, X components.
    assert state[2][-2:] == pytest.approx((2.3, -1.6), abs=0.15)
    assert state[3][-2:] == pytest.approx((0, 0), abs=0.15)
    assert from_beads.describe_state(state)["shifts_px"]["2"][-2] == pytest.approx(2.3, abs=0.15)


# ------------------------------------------------------- coloc module


def _cells_image(shape: tuple[int, int] = (80, 160)) -> tuple[Any, tuple[LabelSet, ...]]:
    """Two cells: in cell 1 the channels colocalize, in cell 2 they are independent."""
    rng = np.random.default_rng(6)
    labels = np.zeros(shape, dtype=np.int32)
    labels[disc(shape, (40, 40), 30)] = 1
    labels[disc(shape, (40, 120), 30)] = 2
    texture = ndimage.gaussian_filter(rng.uniform(0, 1, shape), 1.5) * 2000
    other = ndimage.gaussian_filter(rng.uniform(0, 1, shape), 1.5) * 2000
    gfp = np.where(labels > 0, texture, 0) + 100 + rng.normal(0, 5, shape)
    rfp = (
        np.where(labels == 1, texture, np.where(labels == 2, other, 0))
        + 100
        + rng.normal(0, 5, shape)
    )
    dapi = np.full(shape, 100.0)
    data = np.stack([dapi, gfp, rfp])[:, None]
    cells = LabelSet("cells", labels[None])
    return data, (cells,)


def test_coloc_module_per_cell_and_per_field() -> None:
    data, labels = _cells_image()
    module = measurer(
        "colocalization",
        {"channel_a": 2, "channel_b": 3, "randomizations": 99, "block_size_um": 1.5},
    )
    table = module.measure(context(data, labels))["coloc"]
    assert list(table.object_id) == [1, 2] and set(table.channel_a) == {"GFP"}
    one, two = table.iloc[0], table.iloc[1]
    assert one.pearson > 0.95 and abs(two.pearson) < 0.3
    assert one.p_value == pytest.approx(0.01) and two.p_value > 0.05
    field = measurer("colocalization", {"channel_a": 2, "channel_b": 3, "scope": "field"}).measure(
        context(data, labels)
    )["coloc"]
    assert (
        len(field) == 1
        and 0.3 < field.pearson.iloc[0] < 0.9
        and field.n_pixels.iloc[0] == (labels[0].data > 0).sum()
    )


def test_coloc_warnings_saturation_and_shift() -> None:
    data, labels = _cells_image()
    raw = data.copy()
    raw[1, 0, 40:50, 40:50] = 4095  # saturated block in channel 2 (12-bit)
    table = measurer("colocalization", {"channel_a": 2, "channel_b": 3}).measure(
        context(data, labels, raw=raw)
    )["coloc"]
    assert all("channel 2: " in w and "saturated" in w for w in table.warnings)
    shifted = data.copy()
    shifted[2, 0] = ndimage.shift(data[1, 0], (3, 0), order=1, mode="nearest")
    table = measurer("colocalization", {"channel_a": 2, "channel_b": 3, "labels": None}).measure(
        context(shifted)
    )["coloc"]
    assert "chromatic shift of +3.0" in table.warnings.iloc[0]


def test_coloc_parameters_and_object_based() -> None:
    with pytest.raises(ValueError, match="two different channels"):
        measurer("colocalization", {"channel_a": 2, "channel_b": 2})
    with pytest.raises(ValueError, match="both object_labels"):
        measurer("colocalization", {"channel_a": 1, "channel_b": 2, "object_labels_a": "spots"})
    data, labels = _cells_image()
    needs_block = measurer("colocalization", {"channel_a": 3, "channel_b": 2, "randomizations": 5})
    with pytest.raises(ModuleError, match="block_size_um"):
        needs_block.measure(context(data, labels))  # channel 3 has no emission wavelength

    spots = np.zeros((1, 80, 160), dtype=np.int32)
    for i, (y, x) in enumerate([(40, 40), (40, 120), (5, 80), (75, 5)], start=1):
        spots[0, y - 1 : y + 2, x - 1 : x + 2] = i
    nuclei = np.zeros((1, 80, 160), dtype=np.int32)
    nuclei[0][disc((80, 160), (40, 40), 10)] = 1
    nuclei[0][disc((80, 160), (40, 120), 10)] = 2
    sets = (*labels, LabelSet("spots", spots), LabelSet("nuclei", nuclei))
    module = measurer(
        "colocalization",
        {
            "channel_a": 2,
            "channel_b": 3,
            "scope": "field",
            "object_labels_a": "spots",
            "object_labels_b": "nuclei",
        },
    )
    row = module.measure(context(data, sets))["coloc"].iloc[0]
    assert row.fraction_spots_near_nuclei == pytest.approx(0.5) and row.n_spots == 4
    near = (
        measurer(
            "colocalization",
            {
                "channel_a": 2,
                "channel_b": 3,
                "scope": "field",
                "object_labels_a": "spots",
                "object_labels_b": "nuclei",
                "object_distance_um": 20,
            },
        )
        .measure(context(data, sets))["coloc"]
        .iloc[0]
    )
    assert near.fraction_spots_near_nuclei == pytest.approx(0.75)


# ------------------------------------------------------------- features


def test_morphology_in_physical_units() -> None:
    labels = np.zeros((1, 100, 160), dtype=np.int32)
    labels[0][disc((100, 160), (50, 40), 20)] = 1
    yy, xx = np.mgrid[0:100, 0:160]
    labels[0][((yy - 50) / 10) ** 2 + ((xx - 115) / 30) ** 2 <= 1] = 2
    table = features.morphology(labels, META, features.FeatureOptions(zernike=True)).set_index(
        "object_id"
    )
    circle, ellipse = table.loc[1], table.loc[2]
    assert circle.area_um2 == pytest.approx(np.pi * 10**2, rel=0.02)  # radius 20 px = 10 µm
    assert circle.equivalent_diameter_um == pytest.approx(20, rel=0.01)
    assert circle.perimeter_um == pytest.approx(2 * np.pi * 10, rel=0.05)
    assert circle.form_factor == pytest.approx(1, abs=0.1) and circle.eccentricity < 0.1
    assert circle.centroid_x_um == pytest.approx(20) and circle.centroid_y_um == pytest.approx(25)
    assert ellipse.major_axis_um == pytest.approx(
        30, rel=0.03
    ) and ellipse.minor_axis_um == pytest.approx(10, rel=0.05)
    assert ellipse.eccentricity == pytest.approx(np.sqrt(1 - (10 / 30) ** 2), abs=0.02)
    assert abs(ellipse.orientation_deg) == pytest.approx(90, abs=2)
    assert "zernike_00" in table and table.zernike_00.notna().all()


def test_3d_volume_and_surface() -> None:
    meta = ImageMetadata(channels=(ChannelInfo("x"),), pixel_size_um=0.5, z_step_um=1.0)
    zz, yy, xx = np.mgrid[0:30, 0:80, 0:80]
    sphere = ((zz - 15) * 1.0) ** 2 + ((yy - 40) * 0.5) ** 2 + ((xx - 40) * 0.5) ** 2 <= 10**2
    table = features.morphology(sphere.astype(np.int32), meta, features.FeatureOptions()).iloc[0]
    assert table.volume_um3 == pytest.approx(4 / 3 * np.pi * 10**3, rel=0.03)
    assert table.surface_area_um2 == pytest.approx(4 * np.pi * 10**2, rel=0.08)
    assert table.equivalent_diameter_um == pytest.approx(20, rel=0.02)
    assert table.centroid_z_um == pytest.approx(15)


def test_intensity_radial_and_nuclear_cytoplasmic() -> None:
    shape = (80, 160)
    cells = np.zeros((1, *shape), dtype=np.int32)
    nuclei = np.zeros_like(cells)
    for i, centre in enumerate([(40, 40), (40, 120)], start=1):
        cells[0][disc(shape, centre, 30)] = i
        nuclei[0][disc(shape, centre, 12)] = i
    image = np.zeros((3, 1, *shape), dtype=np.float32)
    image[0][nuclei > 0] = 200  # channel 1: nucleus 200, cytoplasm 50
    image[0][(cells > 0) & (nuclei == 0)] = 50
    ring = disc(shape, (40, 120), 30) & ~disc(shape, (40, 120), 22)
    image[1, 0][ring] = 1000  # channel 2: bright rim in cell 2 only
    image[1, 0][disc(shape, (40, 40), 6)] = 1000  # and a bright centre in cell 1
    cell_set = LabelSet(
        "cells", cells, parent="nuclei", parent_ids=np.array([0, 1, 2], dtype=np.int32)
    )
    sets = (LabelSet("nuclei", nuclei), cell_set)
    options = features.FeatureOptions(texture=False, granularity=False)
    table = features.measure_label_set(cell_set, image, META, options, sets, {}).set_index(
        "object_id"
    )
    rows = table.to_dict("index")
    assert rows[1]["dapi_nc_ratio"] == pytest.approx(4.0) and rows[1][
        "dapi_cytoplasm_mean"
    ] == pytest.approx(50)
    assert rows[1]["dapi_integrated"] == pytest.approx(
        200 * (nuclei == 1).sum() + 50 * ((cells == 1) & (nuclei == 0)).sum()
    )
    assert rows[1]["dapi_p95"] == 200 and rows[1]["dapi_min"] == 50
    assert rows[2]["gfp_radial_outer"] > 0.9
    assert rows[1]["gfp_radial_inner"] > 0.9
    assert rows[2]["gfp_center_periphery_ratio"] < 0.1
    assert np.isnan(rows[1]["gfp_center_periphery_ratio"])  # nothing in the outer ring
    assert list(table.parent_id) == [1, 2]


def test_spots_per_cell_and_neighbours() -> None:
    shape = (60, 200)
    cells = np.zeros((1, *shape), dtype=np.int32)
    spots = np.zeros_like(cells)
    centres = [(30, 30), (30, 70), (30, 110), (30, 150)]  # 40 px = 20 µm apart
    for i, centre in enumerate(centres, start=1):
        cells[0][disc(shape, centre, 15)] = i
    spot_parents = [0]
    k = 0
    for i, (y, x) in enumerate(centres, start=1):
        for j in range(i):  # cell i holds i spots
            k += 1
            spots[0, y - 8 + 5 * j : y - 6 + 5 * j, x - 1 : x + 1] = k
            spot_parents.append(i)
    image = np.ones((3, 1, *shape), dtype=np.float32) * 10
    image[2][spots > 0] = 500
    spot_set = LabelSet(
        "spots", spots, parent="cells", parent_ids=np.array(spot_parents, dtype=np.int32)
    )
    cell_set = LabelSet("cells", cells)
    module = measurer(
        "measure_objects", {"texture": False, "granularity": False, "neighbor_radius_um": 25}
    )
    table = module.measure(context(image, (cell_set, spot_set)))["objects"]
    cells_table = table[table.object_type == "cells"].set_index("object_id")
    assert list(cells_table.spots_count) == [1, 2, 3, 4]
    assert cells_table.loc[4, "spots_rfp_total_intensity"] == pytest.approx(4 * 4 * 500)
    assert cells_table.spots_mean_area_um2.iloc[0] == pytest.approx(4 * 0.25)
    assert list(cells_table.nearest_neighbour_um) == pytest.approx([20, 20, 20, 20])
    assert list(cells_table.neighbours_within_radius) == [1, 2, 2, 1]
    assert (
        set(table.object_type) == {"cells", "spots"}
        and table[table.object_type == "spots"].parent_id.tolist() == spot_parents[1:]
    )


def test_texture_and_granularity() -> None:
    rng = np.random.default_rng(7)
    shape = (60, 140)
    labels = np.zeros((1, *shape), dtype=np.int32)
    labels[0, 10:50, 10:50] = 1
    labels[0, 10:50, 80:120] = 2
    image = np.zeros((1, 1, *shape), dtype=np.float32)
    image[0, 0, 10:50, 10:50] = ndimage.gaussian_filter(rng.uniform(0, 1000, (40, 40)), 4)  # smooth
    image[0, 0, 10:50, 80:120] = rng.uniform(0, 1000, (40, 40))  # rough
    meta = ImageMetadata(channels=(ChannelInfo("x"),), pixel_size_um=0.5)
    table = (
        features.texture(labels, image, ["x"], features.FeatureOptions())
        .set_index("object_id")
        .to_dict("index")
    )
    assert table[2]["x_haralick_contrast_s1"] > 5 * table[1]["x_haralick_contrast_s1"]
    assert table[1]["x_haralick_correlation_s1"] > table[2]["x_haralick_correlation_s1"]

    granules = np.zeros(shape, dtype=np.float32)
    for y in range(14, 48, 8):
        for x in range(14, 48, 8):
            granules[disc(shape, (y, x), 1.5)] = 1000  # small granules in object 1
    for y, x in [(22, 92), (22, 108), (38, 92), (38, 108)]:
        granules[disc(shape, (y, x), 5)] = 1000  # large granules in object 2
    spectrum = features.texture(
        labels, granules[None, None], ["x"], features.FeatureOptions(texture=False)
    ).set_index("object_id")
    small = [spectrum.loc[1, f"x_granularity_r{r}"] for r in range(1, 7)]
    large = [spectrum.loc[2, f"x_granularity_r{r}"] for r in range(1, 7)]
    assert int(np.argmax(small)) + 1 <= 2 and int(np.argmax(large)) + 1 >= 5
    del meta


def test_measure_objects_needs_labels_and_matching_shapes() -> None:
    module = measurer("measure_objects")
    with pytest.raises(ModuleError, match="No segmentation to measure"):
        module.measure(context(np.zeros((3, 1, 8, 8))))
    labels = LabelSet("nuclei", np.ones((1, 4, 4), dtype=np.int32))
    bad = MeasureContext(
        raw=ImageData(np.zeros((3, 1, 8, 8), dtype=np.float32), META),
        corrected=ImageData(np.zeros((3, 1, 8, 8), dtype=np.float32), META),
        labels=(labels,),
    )
    with pytest.raises(ModuleError, match="Z projection before"):
        bad.check_shapes()
    with pytest.raises(ModuleError, match="pixel size"):
        module.validate(ImageMetadata(channels=META.channels))
    assert features.column_name("Alexa Fluor 488 (nm)") == "alexa_fluor_488_nm"
    dup = ImageMetadata(channels=(ChannelInfo("GFP"), ChannelInfo("gfp")))
    assert features.channel_prefixes(dup) == ["gfp", "gfp_2"]


# -------------------------------------------------------------------- QC


def test_focus_score_and_plate_flags() -> None:
    from tests.test_segmentation import CENTRES, nuclei_image

    dense = nuclei_image(CENTRES).data[0, 0]
    sparse = nuclei_image(CENTRES[:1]).data[0, 0]
    sharp = focus_score(dense)
    # Independent of how many cells the field holds, and of brightness.
    assert focus_score(sparse) == pytest.approx(sharp, rel=0.05)
    assert focus_score(dense * 10) == pytest.approx(sharp, rel=0.01)
    # Blur lowers it steadily.
    blurred = [focus_score(ndimage.gaussian_filter(dense, s)) for s in (1, 2, 3, 5)]
    assert blurred == sorted(blurred, reverse=True) and blurred[2] < 0.5 * sharp
    assert np.isnan(focus_score(np.full((64, 64), 100.0)))  # no foreground: never flagged
    assert np.isnan(focus_score(np.zeros((64, 64))))

    rows = []
    for well, count in (("A01", 100), ("A02", 90), ("A03", 10)):
        for field in range(1, 10):
            gx, gy = (field - 1) % 3, (field - 1) // 3
            rows.append(
                {
                    "well": well,
                    "field": field,
                    "x_um": gx * 100.0,
                    "y_um": gy * 100.0,
                    "focus_score": 0.01 if (well, field) == ("A02", 5) else 1.0,
                    "saturated_fraction": 0.01 if (well, field) == ("A01", 2) else 0.0,
                    "area_mm2": 0.01,
                    "n_nuclei": count,
                    "count_labels": "nuclei",
                }
            )
    fields = pd.DataFrame(rows)
    flags = qc_flags(fields, QcParams())
    assert flags[flags.out_of_focus][["well", "field"]].values.tolist() == [["A02", 5]]
    assert flags[flags.saturated][["well", "field"]].values.tolist() == [["A01", 2]]
    assert flags.edge_field.sum() == 3 * 8  # every field but the centre one
    assert flags.excluded.sum() == 1
    summary = well_summary(None, flags, None, QcParams()).set_index("well")
    assert (
        summary.loc["A02", "nuclei_count"] == 8 * 90 and summary.loc["A02", "fields_excluded"] == 1
    )
    assert summary.loc["A01", "nuclei_per_mm2"] == pytest.approx(100 / 0.01)
    assert bool(summary.loc["A03", "low_count"]) and not bool(summary.loc["A01", "low_count"])
    edges = qc_flags(fields, QcParams(exclude_edge_fields=True))
    assert (
        well_summary(None, edges, None, QcParams()).set_index("well").loc["A01", "nuclei_count"]
        == 100
    )


def test_merge_object_tables_unifies_columns(tmp_path: Path) -> None:
    (tmp_path / "A01").mkdir()
    (tmp_path / "A02").mkdir()
    pd.DataFrame({"well": ["A01"], "object_id": [1], "area_um2": [5.0]}).to_parquet(
        tmp_path / "A01" / "objects.parquet"
    )
    pd.DataFrame({"well": ["A02"] * 2, "object_id": [1, 2], "gfp_mean": [3.0, 4.0]}).to_parquet(
        tmp_path / "A02" / "objects.parquet"
    )
    rows = merge_object_tables(
        [tmp_path / "A01", tmp_path / "A02", tmp_path / "A03"], tmp_path / "all.parquet"
    )
    merged = pd.read_parquet(tmp_path / "all.parquet")
    assert rows == 3 and set(merged.columns) == {"well", "object_id", "area_um2", "gfp_mean"}
    assert merged.area_um2.isna().sum() == 2 and merged.gfp_mean.isna().sum() == 1


# ----------------------------------------------------------- run folder


def test_run_writes_measurement_tables(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from tests.builders import make_imagexpress
    from tests.test_segmentation import CENTRES, nuclei_image

    def image(row: int, col: int, site: int, wave: int) -> Any:
        centres = CENTRES if col == 0 else CENTRES[:3]
        return np.clip(nuclei_image(centres, seed=site).data[wave, 0], 0, None).astype(np.uint16)

    root = make_imagexpress(tmp_path / "raw" / "ix", [(0, 0), (0, 1)], n_sites=2, image_fn=image)
    pipeline = Pipeline.from_text(
        """
name: measure
steps:
  - {module: segment_classical, params: {channel: 1}}
  - {module: expand_cells, params: {distance_um: 3}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: colocalization, params: {channel_a: 1, channel_b: 2, labels: cells}}
  - {module: qc}
"""
    )
    result = run_pipeline(open_plate(root), pipeline, checksums=False)
    assert not result.failed_wells
    well = result.folder / "wells" / "A01"
    objects = pd.read_parquet(well / "objects.parquet")
    assert set(objects.object_type) == {"nuclei", "cells"}
    assert objects[["well", "row", "col", "field", "timepoint"]].iloc[0].tolist() == [
        "A01",
        1,
        1,
        1,
        0,
    ]
    assert len(objects[objects.object_type == "nuclei"]) == 10  # 5 nuclei × 2 fields
    assert "dapi_nc_ratio" in objects and "area_um2" in objects
    coloc_table = pd.read_csv(well / "coloc.csv")
    assert len(coloc_table) == 10 and {"pearson", "tm1", "p_value", "warnings"} <= set(
        coloc_table.columns
    )
    tables = result.folder / "tables"
    everything = pd.read_parquet(tables / "per_object_all.parquet")
    assert len(everything) == 2 * (5 + 5) + 2 * (3 + 3)  # fields × (nuclei + cells), per well
    summary = pd.read_csv(tables / "per_well_summary.csv").set_index("well")
    assert summary.loc["A01", "nuclei_count"] == 10 and summary.loc["A02", "nuclei_count"] == 6
    area = 2 * 128 * 128 * 0.65**2 / 1e6  # two 128 × 128 px fields at 0.65 µm
    assert summary.loc["A01", "nuclei_per_mm2"] == pytest.approx(10 / area)
    assert "nuclei_area_um2_mean" in summary and "coloc_DAPI_FITC_pearson_median" in summary
    flags = pd.read_csv(tables / "qc_flags.csv")
    assert len(flags) == 4 and {"out_of_focus", "saturated", "excluded", "low_count_well"} <= set(
        flags.columns
    )
    assert (tables / "per_field.csv").exists() and (tables / "coloc_all.csv").exists()
    well_json = json.loads((well / "well_summary.json").read_text(encoding="utf-8"))
    assert (
        well_json["tables"]["objects"] == "objects.parquet"
        and well_json["summary"]["nuclei_count"] == 10
    )


def test_coloc_inputs_and_fiji_export(tmp_path: Path) -> None:
    from hcs_studio.analysis.coloc_export import export_for_fiji
    from hcs_studio.analysis.measure import coloc_inputs, colocalization_map

    data, labels = _cells_image()
    image = ImageData(np.asarray(data, dtype=np.float32), META)
    a, b, mask = coloc_inputs(image, labels, 1, 2, "cells", 1, 0)
    assert mask.sum() == (labels[0].data[0] == 1).sum() and a.min() >= 0
    assert np.median(b[labels[0].data[0] == 0]) == pytest.approx(0, abs=1)  # background removed
    module_row = (
        measurer("colocalization", {"channel_a": 2, "channel_b": 3})
        .measure(context(data, labels))["coloc"]
        .iloc[0]
    )
    assert coloc.colocalize(a, b, mask).pearson == pytest.approx(module_row.pearson)
    with pytest.raises(ModuleError, match="Object 9 is not on plane 0"):
        coloc_inputs(image, labels, 1, 2, "cells", 9, 0)

    nmdp = colocalization_map(a, b, mask)
    assert nmdp.max() <= 1 and nmdp.min() >= -1 and not nmdp[~mask].any()

    result = export_for_fiji(
        tmp_path / "fiji", a, b, mask, names=("GFP", "RFP"), psf_px=4, randomizations=20
    )
    assert tifffile.imread(tmp_path / "fiji" / "channel_a.tif").dtype == np.float32
    assert set(np.unique(tifffile.imread(tmp_path / "fiji" / "mask.tif"))) == {0, 255}
    saved = json.loads((tmp_path / "fiji" / "hcs_studio_results.json").read_text(encoding="utf-8"))
    assert saved["results"]["pearson"] == pytest.approx(result.pearson) and saved["psf_px"] == 4
    macro = (tmp_path / "fiji" / "coloc2_compare.ijm").read_text(encoding="utf-8")
    assert 'run("Coloc 2"' in macro and "psf=4 costes_randomisations=20" in macro
    with pytest.raises(ValueError, match="same shape"):
        export_for_fiji(tmp_path / "bad", a, b[:10], mask)


def test_qc_flag_summary_counts_fields_per_flag() -> None:
    import pandas as pd

    from hcs_studio.output.experiment import qc_flag_summary

    flags = pd.DataFrame(
        {"out_of_focus": [True, False, True], "saturated": [0, 0, 1], "well": ["A01"] * 3}
    )
    table = qc_flag_summary(flags)
    assert table.to_dict("records") == [
        {"flag": "out of focus", "fields": 2, "of": 3},
        {"flag": "saturated", "fields": 1, "of": 3},
    ]
