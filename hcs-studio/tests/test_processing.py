"""Processing modules against synthetic images with a known answer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import ndimage

from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing.autofluorescence import robust_slope
from hcs_studio.processing.base import ImageData, ModuleError, create_module
from hcs_studio.processing.enhance import estimate_noise, gated_unsharp

META2 = ImageMetadata(
    channels=(ChannelInfo("a"), ChannelInfo("b")), pixel_size_um=0.5, z_step_um=1.0
)


def image(data: Any, metadata: ImageMetadata | None = None) -> ImageData:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:  # (Y, X)
        array = array[None, None]
    elif array.ndim == 3:  # (C, Y, X)
        array = array[:, None]
    if metadata is None:
        metadata = ImageMetadata(
            channels=tuple(ChannelInfo(f"c{i}") for i in range(array.shape[0])), pixel_size_um=0.5
        )
    return ImageData(array, metadata)


class FakeContext:
    """Serves prepared images to ``fit``; records which wells were asked for."""

    def __init__(
        self, images: list[ImageData], wells_present: Sequence[tuple[int, int]] = ()
    ) -> None:
        from types import SimpleNamespace

        self._images = images
        self.plate: Any = SimpleNamespace(
            wells=dict.fromkeys(wells_present, None), metadata=images[0].metadata
        )
        self.layout: dict[tuple[int, int], Any] = {}
        self.requested: Sequence[tuple[int, int]] | None = None

    def sample(
        self, max_fields: int, wells: Sequence[tuple[int, int]] | None = None
    ) -> Iterator[ImageData]:
        self.requested = wells
        yield from self._images[:max_fields]


def blobs(rng: np.random.Generator, shape: tuple[int, int], n: int, amplitude: float) -> Any:
    canvas = np.zeros(shape)
    ys, xs = rng.integers(0, shape[0], n), rng.integers(0, shape[1], n)
    canvas[ys, xs] = amplitude * 60
    return ndimage.gaussian_filter(canvas, 3.0)


# ----------------------------------------------------------------- flat-field


def test_flatfield_recovers_a_known_vignette() -> None:
    rng = np.random.default_rng(1)
    shape = (200, 240)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r2 = ((yy - 100) / 100) ** 2 + ((xx - 120) / 120) ** 2
    vignette = 1.0 - 0.3 * r2
    dark = 100.0
    fields = [
        image(dark + vignette * (300 + blobs(rng, shape, 12, 2000)) + rng.normal(0, 3, shape))
        for _ in range(40)
    ]
    module = create_module("flatfield", {"dark_offset": dark})
    state = module.fit(FakeContext(fields))
    assert state["n_fields"] == 40

    corrected = module.apply(image(dark + vignette * 300.0), state)
    assert corrected.linear and corrected.history == ("flatfield",)
    flat_field = corrected.data[0, 0]
    # A uniform 300-count sample imaged through the vignette becomes uniform again.
    assert flat_field.std() / flat_field.mean() < 0.02
    uncorrected = vignette * 300
    assert uncorrected.std() / uncorrected.mean() > 0.08
    assert module.describe_state(state)["profile_min"][0] < 0.85


@pytest.mark.parametrize("fit", ["polynomial", "gaussian"])
def test_flatfield_profile_is_accurate_up_to_the_border(fit: str) -> None:
    """An off-centre, tilted vignette seen only through background: the polynomial surface
    matches it everywhere (corners included); Gaussian smoothing flattens the corners."""
    rng = np.random.default_rng(3)
    shape = (256, 256)
    yy, xx = (np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5) / 256 - 0.5
    vignette = 1 - 0.6 * ((yy - 0.1) ** 2 + (xx + 0.05) ** 2) + 0.05 * xx
    fields = [image(100 + rng.poisson(400 * vignette)) for _ in range(30)]
    module = create_module("flatfield", {"dark_offset": 100, "profile_fit": fit})
    flat = module.fit(FakeContext(fields))["flat"][0]
    truth = vignette / vignette.mean()
    error = np.abs(flat - truth)
    if fit == "polynomial":
        assert error.max() < 0.005
    else:
        assert error.max() > 0.01  # the border effect the polynomial avoids
        assert error[32:-32, 32:-32].max() < 0.01


def test_flatfield_without_background_falls_back_to_smoothing() -> None:
    """Nuclei on an almost black background (as in the batch tests): the polynomial would
    dip below zero, so smoothing is used for that channel and recorded. With no background
    at all there is nothing to estimate from, and the step refuses."""
    from tests.test_batch import _image

    fields = [
        image(np.stack([_image(r, c, s, w) for w in range(2)]).astype(float))
        for r in range(2)
        for c in range(3)
        for s in range(2)
    ]
    module = create_module("flatfield", {"min_fields": 8})
    state = module.fit(FakeContext(fields))
    assert state["flat"].min() > 0
    assert module.describe_state(state)["gaussian_fallback_channels"]
    black = [image(blobs(np.random.default_rng(i), (64, 64), 3, 2000)) for i in range(10)]
    with pytest.raises(ModuleError, match="non-positive"):
        module.fit(FakeContext(black))


def test_flatfield_needs_enough_fields_and_a_fit() -> None:
    module = create_module("flatfield")
    with pytest.raises(ModuleError, match="at least 8"):
        module.fit(FakeContext([image(np.ones((32, 32)))] * 3))
    with pytest.raises(ModuleError, match="not fitted"):
        module.apply(image(np.ones((32, 32))), None)


# ----------------------------------------------------- reference-well AF


def test_af_reference_scalar_and_image_modes() -> None:
    rng = np.random.default_rng(2)
    ramp = np.linspace(50, 150, 64)[None, :].repeat(64, axis=0)
    refs = [
        image(np.stack([ramp + rng.normal(0, 2, ramp.shape), np.full_like(ramp, 40)]))
        for _ in range(6)
    ]
    sample = image(np.stack([ramp + 500, np.full_like(ramp, 30.0)]))

    scalar = create_module("af_reference", {"reference_wells": ["A01", "H12"]})
    context = FakeContext(refs, wells_present=[(0, 0)])
    state = scalar.fit(context)
    assert context.requested == [(0, 0)]  # only wells present on the plate
    assert state["scalar"][0] == pytest.approx(100, abs=2) and state["scalar"][1] == 40
    out = scalar.apply(sample, state)
    assert out.data[0].mean() == pytest.approx(500, abs=2)
    assert out.data[1].max() == 0, "subtraction must clip at zero"

    per_pixel = create_module(
        "af_reference", {"reference_wells": ["A01"], "mode": "image", "channels": [1]}
    )
    state = per_pixel.fit(FakeContext(refs, wells_present=[(0, 0)]))
    out = per_pixel.apply(sample, state)
    interior = out.data[0, 0, 8:-8, 8:-8]
    assert interior.mean() == pytest.approx(500, abs=3) and interior.std() < 3  # ramp removed
    assert np.array_equal(out.data[1], sample.data[1]), "unselected channel changed"


def test_af_reference_requires_reference_wells() -> None:
    refs = [image(np.ones((2, 16, 16)))]
    with pytest.raises(ModuleError, match="No reference wells"):
        create_module("af_reference").fit(FakeContext(refs))
    with pytest.raises(ModuleError, match="None of the reference wells"):
        create_module("af_reference", {"reference_wells": ["B02"]}).fit(FakeContext(refs))
    with pytest.raises(ModuleError, match="Not a well name"):
        create_module("af_reference", {"reference_wells": ["nope"]}).fit(FakeContext(refs))


# ----------------------------------------------------------------- AF channel


def test_robust_slope_ignores_signal_pixels() -> None:
    rng = np.random.default_rng(3)
    af = rng.uniform(100, 1000, 20000)
    target = 0.3 * af + 50 + rng.normal(0, 5, af.size)
    signal = rng.random(af.size) < 0.25
    target[signal] += rng.uniform(200, 3000, signal.sum())
    slope, intercept = robust_slope(af, target)
    assert slope == pytest.approx(0.3, abs=0.01) and intercept == pytest.approx(50, abs=5)
    naive = np.polyfit(af, target, 1)[1]
    assert abs(naive - 50) > 100, "the test data must actually defeat ordinary least squares"
    with pytest.raises(ModuleError, match="variation"):
        robust_slope(np.ones(100), np.ones(100))


def test_af_channel_subtraction() -> None:
    rng = np.random.default_rng(4)
    shape = (96, 96)
    fields = []
    for _ in range(4):
        af = 200 + blobs(rng, shape, 30, 800)
        signal = blobs(rng, shape, 5, 3000)
        target = 0.4 * af + signal + rng.normal(0, 2, shape)
        fields.append((image(np.stack([target, af])), signal))
    module = create_module("af_channel", {"af_channel": 2})
    state = module.fit(FakeContext([f for f, _ in fields]))
    assert state["k"] == {0: pytest.approx(0.4, abs=0.02)}
    field, signal = fields[0]
    out = module.apply(field, state)
    assert np.abs(out.data[0, 0] - signal).mean() < 8  # only the true signal remains
    assert np.array_equal(out.data[1], field.data[1])

    manual = create_module("af_channel", {"af_channel": 2, "coefficients": {1: 0.4}})
    assert manual.fit(FakeContext([field]))["k"] == {0: 0.4}
    with pytest.raises(ModuleError, match="No manual coefficient"):
        create_module("af_channel", {"af_channel": 2, "coefficients": {5: 1}}).fit(
            FakeContext([field])
        )
    with pytest.raises(ModuleError, match="does not exist"):
        create_module("af_channel", {"af_channel": 3}).validate(field.metadata)


# ----------------------------------------------------------------- background


@pytest.mark.parametrize("method", ["tophat", "rolling_ball"])
def test_background_removes_gradient_and_keeps_spots(method: str) -> None:
    yy, xx = np.mgrid[0:128, 0:128]
    gradient = 200 + (2.0 if method == "tophat" else 0.3) * xx
    spot = 1000 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 2.0**2))
    module = create_module("background", {"method": method, "radius_um": 10})  # 20 px
    out = module.apply(image(gradient + spot), None).data[0, 0]
    assert out[64, 64] == pytest.approx(1000, rel=0.1)
    assert np.median(out) < 5 and out.min() >= 0


def test_background_needs_pixel_size() -> None:
    uncalibrated = ImageMetadata(channels=(ChannelInfo("a"),))
    module = create_module("background")
    with pytest.raises(ModuleError, match="pixel size is unknown"):
        module.validate(uncalibrated)
    with pytest.raises(ModuleError, match="smaller than one pixel"):
        create_module("background", {"radius_um": 0.1}).validate(META2)


# ------------------------------------------------------------------- unmixing


def test_unmix_recovers_known_abundances() -> None:
    rng = np.random.default_rng(5)
    abundances = rng.uniform(0, 1000, (2, 1, 20, 20)).astype(np.float32)
    mixing = np.array([[1.0, 0.2], [0.3, 1.0], [0.05, 0.4]])
    mixed = np.tensordot(mixing.astype(np.float32), abundances, axes=([1], [0]))
    module = create_module(
        "unmix", {"mixing_matrix": mixing.tolist(), "fluorophores": ["GFP", "YFP"]}
    )
    out = module.apply(image(mixed), None)
    assert np.allclose(out.data, abundances, atol=0.05)
    assert [c.name for c in out.metadata.channels] == ["GFP", "YFP"] and out.linear
    with pytest.raises(ModuleError, match="3 rows"):
        module.validate(META2)
    for bad in (
        {"mixing_matrix": [[1, 1], [1, 1]], "fluorophores": ["a", "b"]},  # dependent
        {"mixing_matrix": [[1, 0, 0]], "fluorophores": ["a", "b", "c"]},  # under-determined
        {"mixing_matrix": [[1, 0], [0, 1]], "fluorophores": ["a"]},
    ):
        with pytest.raises(ValidationError):
            create_module("unmix", bad)


# ---------------------------------------------------------------- enhancement


@pytest.mark.parametrize("method", ["gaussian", "median", "nlm"])
def test_denoise_lowers_noise_and_keeps_the_mean(method: str) -> None:
    rng = np.random.default_rng(6)
    noisy = image(500 + rng.normal(0, 20, (64, 64)))
    out = create_module("denoise", {"method": method}).apply(noisy, None)
    assert out.linear
    assert out.data.std() < 0.7 * noisy.data.std()
    assert out.data.mean() == pytest.approx(500, abs=2)


def test_noise_estimate_matches_known_sd() -> None:
    rng = np.random.default_rng(10)
    yy, xx = np.mgrid[0:128, 0:128]
    smooth = 500 + 200 * np.sin(xx / 20.0) * np.cos(yy / 25.0)
    assert estimate_noise(smooth + rng.normal(0, 12, smooth.shape)) == pytest.approx(12, rel=0.1)
    assert estimate_noise(smooth) < 0.5


def test_contrast_is_display_only_and_linearity_is_never_regained() -> None:
    ramp = image(np.linspace(0, 1000, 101 * 4).reshape(4, 101))
    out = create_module(
        "contrast", {"low_percentile": 0, "high_percentile": 100, "gamma": 0.5}
    ).apply(ramp, None)
    assert not out.linear
    assert float(np.min(out.data)) == 0 and float(np.max(out.data)) == pytest.approx(1)
    assert out.data.ravel()[101] == pytest.approx(np.sqrt(0.25), abs=0.01)
    again = create_module("denoise").apply(out, None)
    assert not again.linear and again.history == ("contrast", "denoise")
    with pytest.raises(ValidationError):
        create_module("contrast", {"low_percentile": 50, "high_percentile": 10})


def test_clahe_boosts_local_contrast_in_dim_regions() -> None:
    rng = np.random.default_rng(7)
    data = np.concatenate(
        [100 + rng.normal(0, 2, (64, 64)), 1000 + rng.normal(0, 2, (64, 64))], axis=1
    )
    out = create_module("clahe", {"tile_um": 8}).apply(image(data), None)
    assert not out.linear and out.data.shape == (1, 1, 64, 128)
    assert float(np.min(out.data)) >= 0 and float(np.max(out.data)) <= 1
    global_scaled = (data - data.min()) / (data.max() - data.min())
    assert out.data[0, 0, :, :60].std() > 1.5 * global_scaled[:, :60].std()
    with pytest.raises(ModuleError, match="pixel size"):
        create_module("clahe").validate(ImageMetadata(channels=(ChannelInfo("a"),)))


@pytest.mark.parametrize("method", ["unsharp", "dog"])
def test_sharpen_steepens_edges(method: str) -> None:
    edge = ndimage.gaussian_filter(np.repeat([[100.0] * 32 + [600.0] * 32], 32, axis=0), 3.0)
    out = create_module("sharpen", {"method": method, "amount": 1.5}).apply(image(edge), None)
    assert not out.linear and float(np.min(out.data)) >= 0
    assert np.abs(np.diff(out.data[0, 0, 16])).max() > 1.2 * np.abs(np.diff(edge[16])).max()
    with pytest.raises(ValidationError):
        create_module("sharpen", {"method": "dog", "sigma_px": 5, "sigma_large_px": 2})


def test_clarity_enhances_structure_but_not_noise() -> None:
    rng = np.random.default_rng(8)
    _yy, xx = np.mgrid[0:96, 0:96]
    structure = 300 + 300 * (np.sin(xx / 4.0) > 0)
    noise = rng.normal(0, 6, structure.shape)

    # The unsharp stage alone: on pure noise the gate must suppress amplification.
    flat = 1000 + noise
    gated = gated_unsharp(flat, 0.7, 3.0, 2.0, noise_gate=3.0)
    ungated = gated_unsharp(flat, 0.7, 3.0, 2.0, noise_gate=0.0)
    assert gated.std() < 0.7 * ungated.std(), "the noise gate did not limit amplification"
    # ... while a real edge is still steepened.
    edges = gated_unsharp(structure + noise, 0.7, 3.0, 2.0, noise_gate=3.0)
    plain = ndimage.gaussian_filter(structure + noise, 0.7)
    assert np.abs(np.diff(edges[48])).max() > 1.3 * np.abs(np.diff(plain[48])).max()

    out = create_module("clarity").apply(image(structure + noise), None)
    assert not out.linear and np.isfinite(out.data).all()
    assert float(np.min(out.data)) >= 0 and float(np.max(out.data)) <= 1


def test_projection() -> None:
    rng = np.random.default_rng(9)
    stack = rng.uniform(0, 100, (2, 5, 16, 16)).astype(np.float32)
    data = ImageData(stack, META2)
    maximum = create_module("projection", {"method": "max"}).apply(data, None)
    assert maximum.data.shape == (2, 1, 16, 16) and maximum.linear
    assert np.array_equal(maximum.data[:, 0], stack.max(axis=1))
    assert maximum.metadata.z_step_um is None
    mean = create_module("projection", {"method": "mean"}).apply(data, None)
    assert np.allclose(mean.data[:, 0], stack.mean(axis=1))

    # EDF: the left half is in focus in plane 0, the right half in plane 2.
    texture = rng.uniform(0, 1000, (64, 64))
    blurred = ndimage.gaussian_filter(texture, 4.0)
    left = np.zeros((64, 64), dtype=bool)
    left[:, :32] = True
    planes = np.stack([np.where(left, texture, blurred), blurred, np.where(left, blurred, texture)])
    edf = create_module("projection", {"method": "edf"}).apply(image(planes[None]), None)
    assert (edf.data[0, 0] == texture.astype(np.float32)).mean() > 0.9
    single = image(np.ones((8, 8)))
    assert create_module("projection").apply(single, None).data.shape == (1, 1, 8, 8)


def test_channel_selection_and_parameter_validation() -> None:
    data = ImageData(
        np.random.default_rng(0).uniform(0, 100, (2, 1, 16, 16)).astype(np.float32), META2
    )
    out = create_module("denoise", {"channels": [2], "sigma_px": 2}).apply(data, None)
    assert np.array_equal(out.data[0], data.data[0]) and not np.array_equal(
        out.data[1], data.data[1]
    )
    with pytest.raises(ModuleError, match=r"Channel\(s\) \[3\]"):
        create_module("denoise", {"channels": [3]}).apply(data, None)
    with pytest.raises(ValidationError):
        create_module("denoise", {"sigma": 2})  # unknown parameter name
    with pytest.raises(ValidationError):
        create_module("denoise", {"sigma_px": -1})
    with pytest.raises(KeyError, match="Unknown module"):
        create_module("nope")
    with pytest.raises(ValueError, match="Channel count"):
        ImageData(np.zeros((3, 1, 4, 4), dtype=np.float32), META2)
