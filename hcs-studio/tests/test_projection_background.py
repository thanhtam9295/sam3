"""Phase 12: Z projection and background removal after projection (known answers)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from hcs_studio.model.annotation import WellAnnotation
from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing.background import estimate_shading
from hcs_studio.processing.base import ImageData, ModuleError, create_module
from hcs_studio.processing.projection import best_planes, focus_score


def meta(channels: int = 1, pixel: float = 0.5) -> ImageMetadata:
    return ImageMetadata(
        channels=tuple(ChannelInfo(f"c{i}") for i in range(channels)), pixel_size_um=pixel
    )


def czyx(data: Any) -> ImageData:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, None]
    elif array.ndim == 3:
        array = array[None]
    return ImageData(array, meta(array.shape[0]))


# --------------------------------------------------------------- projection


def test_projections_equal_known_values() -> None:
    rng = np.random.default_rng(0)
    stack = rng.uniform(0, 100, (2, 20, 32, 32)).astype(np.float32)
    # a known maximum hidden at a random plane of every pixel
    planes = rng.integers(0, 20, (2, 32, 32))
    np.put_along_axis(stack, planes[:, None], 5000.0, axis=1)
    data = ImageData(stack, meta(2))
    out = create_module("projection").apply(data, None)  # default: maximum
    assert out.data.shape == (2, 1, 32, 32) and out.linear
    assert np.array_equal(out.data[:, 0], np.full((2, 32, 32), 5000, np.float32))
    expected: dict[str, np.ndarray] = {
        "mean": stack.mean(axis=1),
        "sum": stack.sum(axis=1, dtype=np.float64),
        "min": stack.min(axis=1),
        "std": stack.std(axis=1, dtype=np.float64),
    }
    for method, value in expected.items():
        got = create_module("projection", {"method": method}).apply(data, None).data[:, 0]
        np.testing.assert_allclose(got, value, rtol=1e-5)
    assert out.metadata.z_step_um is None
    assert out.reports[-1]["planes"] == [list(range(1, 21))] * 2


def _focus_stack(best: int, n: int = 15, size: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = ndimage.gaussian_filter(rng.uniform(0, 1000, (size, size)), 1.0) + 200
    return np.stack(
        [ndimage.gaussian_filter(texture, 0.3 + 1.2 * abs(z - best)) for z in range(n)]
    ).astype(np.float32)


@pytest.mark.parametrize("edf_method", ["variance", "pyramid"])
def test_extended_depth_of_field_recovers_the_sharp_regions(edf_method: str) -> None:
    rng = np.random.default_rng(3)
    texture = ndimage.gaussian_filter(rng.uniform(0, 1000, (64, 64)), 0.7) + 100
    blurred = ndimage.gaussian_filter(texture, 3.0)
    # left half in focus in plane 0, right half in plane 2; plane 1 blurred everywhere
    plane0 = np.concatenate([texture[:, :32], blurred[:, 32:]], axis=1)
    plane2 = np.concatenate([blurred[:, :32], texture[:, 32:]], axis=1)
    stack = np.stack([plane0, blurred, plane2])
    out = create_module("projection", {"method": "edf", "edf_method": edf_method})
    fused = out.apply(czyx(stack), None).data[0, 0]
    inner = (slice(4, -4), np.r_[4:28, 36:60])
    correlation = np.corrcoef(fused[inner].ravel(), texture[inner].ravel())[0, 1]
    assert correlation > 0.97
    mip_corr = np.corrcoef(stack.max(axis=0)[inner].ravel(), texture[inner].ravel())[0, 1]
    assert correlation > mip_corr


def test_plane_range_and_auto_focus() -> None:
    stack = np.stack([_focus_stack(7), _focus_stack(11, seed=2)])  # (2, 15, 64, 64)
    data = ImageData(stack, meta(2))

    ranged = create_module("projection", {"planes": "range", "first_plane": 3, "last_plane": 5})
    out = ranged.apply(data, None)
    np.testing.assert_array_equal(out.data[:, 0], stack[:, 2:5].max(axis=1))
    assert out.reports[-1]["planes"] == [[3, 4, 5], [3, 4, 5]]

    for metric in ("normalized_variance", "brenner"):
        scores = [focus_score(p, metric) for p in stack[0]]
        assert int(np.argmax(scores)) == 7
        auto = create_module(
            "projection", {"planes": "auto", "n_planes": 3, "focus_metric": metric}
        ).apply(data, None)
        report = auto.reports[-1]
        assert report["planes"] == [[7, 8, 9], [11, 12, 13]]  # numbered from 1
        assert len(report["focus_scores"][0]) == 15
        np.testing.assert_array_equal(auto.data[1, 0], stack[1, 10:13].max(axis=0))

    shared = create_module(
        "projection", {"planes": "auto", "n_planes": 3, "focus_channel": 2}
    ).apply(data, None)
    assert shared.reports[-1]["planes"] == [[11, 12, 13], [11, 12, 13]]
    assert best_planes([0, 1, 5, 2], 3) == [1, 2, 3]
    assert best_planes([9, 1, 0, 0], 3) == [0, 1, 2]  # shifted at the start
    with pytest.raises(ModuleError, match="Focus channel 3"):
        create_module("projection", {"planes": "auto", "focus_channel": 3}).apply(data, None)
    with pytest.raises(ModuleError, match="First plane"):
        create_module("projection", {"planes": "range", "first_plane": 40}).apply(data, None)


# --------------------------------------------------------------- background


def _gradient_with_spots(
    n: int = 512, noise: float = 5.0, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """A realistic illumination gradient (≈ 50 % across the field plus a broad bump),
    40 Gaussian spots and noise. Returns the image, the spots alone, the centres."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    gradient = (
        300 + 0.25 * xx + 0.1 * yy + 150 * np.exp(-((yy - 200) ** 2 + (xx - 350) ** 2) / 64800)
    )
    spots = np.zeros((n, n))
    centres: list[tuple[float, float]] = []
    while len(centres) < 40:
        y, x = rng.uniform(20, n - 20, 2)
        if all((y - a) ** 2 + (x - b) ** 2 > 30**2 for a, b in centres):
            centres.append((y, x))
            spots += 2000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2.0**2))
    image = gradient + spots + rng.normal(0, noise, (n, n))
    return image.astype(np.float32), spots, centres


def _spot_sums(image: np.ndarray, centres: list[tuple[float, float]]) -> np.ndarray:
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    return np.array([image[(yy - y) ** 2 + (xx - x) ** 2 <= 6.5**2].sum() for y, x in centres])


@pytest.mark.parametrize(
    "params",
    [
        {"method": "tophat"},
        {"method": "rolling_ball"},
        {"method": "paraboloid"},
        {"method": "gaussian"},
        {"method": "median"},
        {"method": "polynomial", "degree": 3},
        {"method": "spline"},
    ],
    ids=lambda p: p["method"],
)
def test_background_keeps_spot_intensity_within_5_percent(params: dict[str, Any]) -> None:
    """Done when: gradient + spots → spot-integrated intensity within ± 5 %."""
    image, spots, centres = _gradient_with_spots()
    module = create_module("background", {"radius_um": 10, **params})  # 20 px
    out = module.apply(czyx(image), None)
    corrected = out.data[0, 0]
    truth = _spot_sums(spots, centres)
    error = _spot_sums(corrected, centres) / truth - 1
    assert np.abs(error).max() < 0.05, f"worst spot {error[np.argmax(np.abs(error))]:.3f}"
    assert out.linear and corrected.min() >= 0
    background = corrected[spots < 1]
    assert np.median(background) < 8  # the gradient (300–550) is gone
    record = out.reports[-1]["channels"][0]
    assert record["background_mean_change_pct"] < -95
    assert -60 < record["object_mean_change_pct"] < 0
    assert "warnings" not in out.reports[-1]


def test_divide_corrects_multiplicative_shading() -> None:
    _, spots, centres = _gradient_with_spots(noise=0)
    yy, xx = np.mgrid[0:512, 0:512]
    shading = 1 - 0.4 * (((yy - 256) / 362.0) ** 2 + ((xx - 256) / 362.0) ** 2)
    image = (shading * (500 + spots)).astype(np.float32)
    out = create_module(
        "background", {"method": "polynomial", "operation": "divide", "degree": 4}
    ).apply(czyx(image), None)
    corrected = out.data[0, 0]
    background = corrected[spots < 1]
    assert background.std() / background.mean() < 0.01  # shading was 40 %
    sums = _spot_sums(corrected, centres)
    assert sums.std() / sums.mean() < 0.02  # spots equally bright again


def test_constant_background_and_blank_wells() -> None:
    rng = np.random.default_rng(4)
    flat = (rng.normal(400, 5, (64, 64))).astype(np.float32)
    out = create_module("background", {"method": "constant", "percentile": 50}).apply(
        czyx(flat), None
    )
    assert abs(float(np.median(out.data))) < 0.5

    module = create_module("background", {"method": "constant", "constant_source": "blank_wells"})
    assert module.needs_fit
    blank = ImageData(np.full((2, 1, 16, 16), [[[[100]]], [[[30]]]], np.float32), meta(2))
    context = FakeContext([blank] * 3, layout={(0, 0): WellAnnotation(role="blank")})
    state = module.fit(context)
    assert module.describe_state(state)["blank_mean_per_channel"] == [100.0, 30.0]
    sample = ImageData(np.full((2, 1, 16, 16), 150, np.float32), meta(2))
    np.testing.assert_allclose(module.apply(sample, state).data[:, 0, 0, 0], [50, 120])
    with pytest.raises(ModuleError, match="No blank wells"):
        module.fit(FakeContext([blank], layout={}))
    with pytest.raises(ModuleError, match="not fitted"):
        module.apply(sample, None)


@dataclass
class FakeContext:
    images: list[ImageData]
    layout: dict[tuple[int, int], WellAnnotation] = field(default_factory=dict)
    plate: Any = field(default_factory=lambda: SimpleNamespace(wells={(0, 0): None}))

    def sample(
        self, max_fields: int, wells: Sequence[tuple[int, int]] | None = None
    ) -> Iterator[ImageData]:
        yield from self.images[:max_fields]


def _shaded_fields(n: int = 40, size: int = 96, seed: int = 2) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1) * 2 - 1
    flat = 1 - 0.3 * (yy**2 + xx**2) / 2  # vignetting, 1 at the centre
    flat = flat / flat.mean()
    dark = 60 + 25 * (xx + 1)  # a dark-field gradient
    fields, levels = [], []
    for _ in range(n):
        level = rng.uniform(150, 600)
        spots = np.zeros((size, size))
        for y, x in rng.uniform(5, size - 5, (6, 2)):
            spots += 1500 * np.exp(
                -((np.arange(size)[:, None] - y) ** 2 + (np.arange(size)[None] - x) ** 2) / 8
            )
        fields.append(flat * (level + spots) + dark + rng.normal(0, 2, (size, size)))
        levels.append(level)
    return np.stack(fields).astype(np.float32), flat, dark, np.array(levels)


def test_basic_style_flat_and_dark_field() -> None:
    fields, flat, dark, levels = _shaded_fields()
    est_flat, est_dark, info = estimate_shading(fields, darkfield=True)
    assert np.abs(est_flat / flat - 1)[8:-8, 8:-8].max() < 0.03
    assert np.abs(est_dark - dark)[8:-8, 8:-8].max() < 15
    assert info["n_fields"] == 40 and not info["notes"]

    module = create_module("background", {"method": "basic", "max_fields": 40})
    images = [ImageData(f[None, None], meta(1)) for f in fields]
    state = module.fit(FakeContext(images))
    corrected = module.apply(images[0], state).data[0, 0]
    background = corrected[8:-8, 8:-8][corrected[8:-8, 8:-8] < levels[0] + 30]
    before = fields[0][8:-8, 8:-8][fields[0][8:-8, 8:-8] < np.percentile(fields[0], 80)]
    assert background.std() / background.mean() < 0.03 < before.std() / before.mean()
    assert abs(np.median(background) / levels[0] - 1) < 0.05
    assert module.describe_state(state)["channels"][0]["n_fields"] == 40

    # equal backgrounds in all fields: no dark-field, with a note
    same = np.stack([flat * 300 + 50 for _ in range(10)]).astype(np.float32)
    _, dark_none, info = estimate_shading(same, darkfield=True)
    assert np.all(dark_none == 0) and "dark-field not estimated" in info["notes"][0]


def test_radius_warning_and_object_report() -> None:
    rng = np.random.default_rng(6)
    image = rng.normal(200, 3, (200, 200))
    yy, xx = np.mgrid[0:200, 0:200]
    for y, x in [(40, 40), (40, 150), (150, 40), (150, 150), (100, 100)]:
        image[(yy - y) ** 2 + (xx - x) ** 2 <= 8**2] += 800  # objects 16 px = 8 µm across
    for method, loss in (("rolling_ball", -20), ("tophat", -80)):
        small = create_module("background", {"method": method, "radius_um": 3})
        out = small.apply(czyx(image), None)
        record = out.reports[-1]["channels"][0]
        assert record["object_diameter"] == pytest.approx(8, rel=0.15)
        assert "less than 2× the typical object diameter" in out.reports[-1]["warnings"][0]
        assert record["object_mean_change_pct"] < loss  # the radius removes signal
    good = create_module("background", {"method": "rolling_ball", "radius_um": 20})
    out = good.apply(czyx(image), None)
    assert "warnings" not in out.reports[-1]
    assert out.reports[-1]["channels"][0]["object_mean_change_pct"] > -30
    given = create_module(
        "background", {"method": "tophat", "radius_um": 5, "object_diameter_um": 2}
    )
    assert "warnings" not in given.apply(czyx(image), None).reports[-1]
    with pytest.raises(ModuleError, match="pixel size is unknown"):
        create_module("background").validate(ImageMetadata(channels=(ChannelInfo("a"),)))
