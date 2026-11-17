"""PSF tools and deconvolution against synthetic data with a known answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import signal

from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing import psf as psf_tools
from hcs_studio.processing.base import ImageData, ModuleError, create_module
from hcs_studio.processing.deconvolution import (
    NumpyBackend,
    TorchBackend,
    auto_tile_size,
    deconvolve,
    estimate_memory,
    get_backend,
    tile_slices,
)

CONFOCAL = ImageMetadata(
    channels=(ChannelInfo("GFP", excitation_nm=488, emission_nm=520),),
    pixel_size_um=0.1,
    z_step_um=0.25,
    objective_na=1.4,
    immersion_ri=1.515,
    magnification=60,
    modality="confocal",
    pinhole_um=50,
)


def gaussian_psf(sigma_zyx: tuple[float, float, float], shape: tuple[int, int, int]) -> Any:
    grids = np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij")
    exponent = sum((g / s) ** 2 for g, s in zip(grids, sigma_zyx, strict=True) if s > 0)
    return psf_tools.normalise(np.exp(-0.5 * exponent))


def blur(volume: Any, psf: Any) -> Any:
    return signal.fftconvolve(volume, psf, mode="same")


def bead_stack(
    seed: int = 0, shape: tuple[int, int, int] = (25, 96, 96), n: int = 6
) -> tuple[Any, Any, Any]:
    """Point sources blurred by a known PSF, with background and Poisson noise."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(shape)
    spots = [(12, 20, 20), (12, 20, 70), (12, 70, 20), (12, 70, 70), (11, 45, 45), (13, 46, 80)][:n]
    for z, y, x in spots:
        truth[z, y, x] = 60000
    psf = gaussian_psf((2.2, 2.0, 2.0), (17, 15, 15))
    noisy = rng.poisson(blur(truth, psf) + 100).astype(np.float32)
    return noisy, psf, spots


# ------------------------------------------------------------------------ PSF


def test_profile_fwhm_of_a_gaussian() -> None:
    x = np.arange(-30, 31)
    sigma = 3.7
    assert psf_tools.profile_fwhm(np.exp(-0.5 * (x / sigma) ** 2) + 5, 0.1) == pytest.approx(
        2.3548 * sigma * 0.1, rel=0.01
    )
    assert psf_tools.profile_fwhm(np.ones(10), 1.0) is None
    assert psf_tools.profile_fwhm(np.arange(10.0), 1.0) is None  # never falls on one side


def test_theoretical_widefield_psf_matches_diffraction_theory() -> None:
    spec = psf_tools.PsfSpec(
        na=1.4,
        emission_nm=520,
        pixel_size_um=0.04,
        immersion_ri=1.515,
        sample_ri=1.515,
        z_step_um=0.1,
    )
    psf = psf_tools.theoretical_psf(spec, three_d=True)
    assert psf.dtype == np.float32 and psf.sum() == pytest.approx(1, abs=1e-5)
    assert all(n % 2 == 1 for n in psf.shape)
    assert np.unravel_index(np.argmax(psf), psf.shape) == tuple(n // 2 for n in psf.shape)
    fwhm = psf_tools.measure_fwhm(psf, (0.1, 0.04, 0.04))
    # Lateral FWHM of an Airy pattern is 0.51 λ / NA; the vectorial model is slightly wider.
    assert fwhm["x"] == pytest.approx(0.51 * 0.52 / 1.4, rel=0.2)
    assert fwhm["x"] == pytest.approx(fwhm["y"], rel=0.02)
    assert fwhm["x"] is not None and fwhm["z"] is not None
    assert 2 * fwhm["x"] < fwhm["z"] < 6 * fwhm["x"]
    flat = psf_tools.theoretical_psf(spec, three_d=False)
    assert flat.shape[0] == 1


def test_confocal_psf_is_narrower_than_widefield() -> None:
    wide, _ = psf_tools.spec_from_metadata(
        ImageMetadata(**{**CONFOCAL.__dict__, "modality": "widefield"}), 0
    )
    conf, notes = psf_tools.spec_from_metadata(CONFOCAL, 0, pinhole_au=0.5)
    assert conf.confocal and conf.pinhole_au == 0.5 and not wide.confocal
    voxel = (0.25, 0.1, 0.1)
    f_wide = psf_tools.measure_fwhm(psf_tools.theoretical_psf(wide, True), voxel)
    f_conf = psf_tools.measure_fwhm(psf_tools.theoretical_psf(conf, True), voxel)
    assert f_conf["x"] < f_wide["x"] and f_conf["z"] < f_wide["z"]  # type: ignore[operator]

    auto, notes = psf_tools.spec_from_metadata(CONFOCAL, 0)
    # 50 µm pinhole / 60x = 0.833 µm in the sample; Airy diameter 1.22·0.52/1.4 = 0.453 µm.
    assert auto.pinhole_au == pytest.approx(1.84, abs=0.01)
    assert any("Airy units" in n for n in notes)


def test_spec_from_metadata_reports_what_is_missing() -> None:
    bare = ImageMetadata(channels=(ChannelInfo("x"),))
    with pytest.raises(ModuleError, match="objective NA, pixel size, emission wavelength"):
        psf_tools.spec_from_metadata(bare, 0)
    unknown = ImageMetadata(**{**CONFOCAL.__dict__, "modality": "unknown"})
    with pytest.raises(ModuleError, match="set it to confocal or widefield"):
        psf_tools.spec_from_metadata(unknown, 0)
    no_ri = ImageMetadata(**{**CONFOCAL.__dict__, "immersion_ri": None})
    with pytest.raises(ModuleError, match="immersion refractive index"):
        psf_tools.spec_from_metadata(no_ri, 0)
    air = ImageMetadata(
        **{**CONFOCAL.__dict__, "immersion_ri": None, "objective_na": 0.8, "modality": "widefield"}
    )
    spec, notes = psf_tools.spec_from_metadata(air, 0)
    assert spec.immersion_ri == 1.0 and any("assumed air" in n for n in notes)
    with pytest.raises(ModuleError, match="below the immersion refractive index"):
        psf_tools.PsfSpec(na=1.4, emission_nm=520, pixel_size_um=0.1, immersion_ri=1.0)


def test_measured_psf_from_beads_recovers_the_true_psf() -> None:
    stack, psf, spots = bead_stack()
    # Add a bead touching the edge, and a pair too close together: both must be rejected.
    extra = np.zeros(stack.shape)
    extra[12, 3, 50] = 60000
    extra[12, 45, 12] = 60000
    extra[12, 45, 19] = 50000
    stack = stack + blur(extra, psf).astype(np.float32)

    result = psf_tools.extract_psf_from_beads(stack, size_xy=15, size_z=17)
    assert result.n_beads == 6 and result.n_rejected >= 2
    assert sorted(result.positions_zyx) == sorted(spots)
    assert result.psf.shape == (17, 15, 15) and result.psf.sum() == pytest.approx(1, abs=1e-5)
    measured = psf_tools.measure_fwhm(result.psf, (1.0, 1.0, 1.0))
    truth = psf_tools.measure_fwhm(psf, (1.0, 1.0, 1.0))
    for axis in "zyx":
        assert measured[axis] == pytest.approx(truth[axis], rel=0.06)
    assert np.abs(result.psf - psf).max() < 0.1 * psf.max()

    with pytest.raises(ModuleError, match="No usable beads"):
        psf_tools.extract_psf_from_beads(stack, size_xy=15, size_z=17, saturation=300)
    with pytest.raises(ModuleError, match="odd"):
        psf_tools.extract_psf_from_beads(stack, size_xy=16)


def test_psf_file_round_trip_and_resampling(tmp_path: Path) -> None:
    psf = gaussian_psf((2.0, 3.0, 3.0), (9, 21, 21))
    psf_tools.save_psf(tmp_path / "psf.ome.tif", psf, 0.1, 0.25)
    loaded, pixel, z_step = psf_tools.load_psf(tmp_path / "psf.ome.tif")
    assert np.allclose(loaded, psf, atol=1e-7) and (pixel, z_step) == (0.1, 0.25)

    coarse = psf_tools.resample_psf(psf, (0.25, 0.1, 0.1), (0.25, 0.2, 0.2))
    assert coarse.shape == (9, 11, 11) and coarse.sum() == pytest.approx(1, abs=1e-5)
    before = psf_tools.measure_fwhm(psf, (0.25, 0.1, 0.1))["x"]
    after = psf_tools.measure_fwhm(coarse, (0.25, 0.2, 0.2))["x"]
    assert after == pytest.approx(before, rel=0.1)  # same physical width
    assert psf_tools.resample_psf(psf, (0.25, 0.1, 0.1), (0.25, 0.1, 0.1)).shape == psf.shape

    (tmp_path / "junk.tif").write_bytes(b"nope")
    with pytest.raises(ModuleError, match="Could not read"):
        psf_tools.load_psf(tmp_path / "junk.tif")


# -------------------------------------------------------------- deconvolution


@pytest.mark.parametrize("shape", [(9, 21, 21), (9, 20, 22), (8, 21, 20)])
def test_kernel_centre_lands_on_the_origin(shape: tuple[int, int, int]) -> None:
    psf = gaussian_psf((1.0, 1.5, 1.5), (5, 7, 7))
    placed = psf_tools.centre_crop_or_pad(psf, shape)
    assert np.unravel_index(np.argmax(placed), shape) == tuple(n // 2 for n in shape)
    assert tuple(int(i) for i in np.unravel_index(np.argmax(np.fft.ifftshift(placed)), shape)) == (
        0,
        0,
        0,
    )
    cropped = psf_tools.centre_crop_or_pad(psf, (3, 5, 4))
    assert tuple(int(i) for i in np.unravel_index(np.argmax(cropped), cropped.shape)) == (1, 2, 2)


def test_rl_reconcentrates_a_blurred_point_and_conserves_flux() -> None:
    truth = np.zeros((1, 65, 65))
    truth[0, 32, 32] = 10000
    psf = gaussian_psf((0, 3.0, 3.0), (1, 21, 21))
    blurred = blur(truth, psf).astype(np.float32)
    result = deconvolve(blurred, psf, iterations=60, tv_lambda=0, background=0)
    assert result.iterations == 60 and result.backend.startswith("cpu")
    assert float(np.min(result.image)) >= 0
    assert float(np.max(result.image)) > 8 * float(
        np.max(blurred)
    )  # intensity pulled back into the point
    assert result.image.sum() == pytest.approx(10000, rel=0.01)
    assert result.image[0, 30:35, 30:35].sum() > 0.9 * 10000
    assert result.residual < 0.05


def test_bead_stack_fwhm_is_reduced_in_3d() -> None:
    """Phase 3 acceptance: a bead stack shows reduced FWHM after deconvolution."""
    stack, psf, spots = bead_stack()
    result = deconvolve(stack, psf, iterations=40)
    assert result.background == pytest.approx(100, abs=5)  # true background is 100

    def bead_fwhm(volume: Any, spot: tuple[int, int, int]) -> dict[str, float | None]:
        z, y, x = spot
        crop = volume[z - 8 : z + 9, y - 7 : y + 8, x - 7 : x + 8]
        return psf_tools.measure_fwhm(crop - np.median(volume), (1.0, 1.0, 1.0))

    for spot in spots:
        raw, decon = bead_fwhm(stack, spot), bead_fwhm(result.image, spot)
        for axis in "zyx":
            assert raw[axis] is not None and decon[axis] is not None
            assert decon[axis] < 0.6 * raw[axis], f"FWHM {axis} not reduced at {spot}"  # type: ignore[operator]
    # Signal is conserved: each bead's crop holds its 60000 counts.
    for z, y, x in spots:
        crop = result.image[z - 8 : z + 9, y - 7 : y + 8, x - 7 : x + 8]
        assert float(crop.sum()) == pytest.approx(60000, rel=0.1)
    raw_mean = np.mean([float(bead_fwhm(stack, s)["x"] or 0) for s in spots])
    new_mean = np.mean([float(bead_fwhm(result.image, s)["x"] or 0) for s in spots])
    print(f"mean lateral FWHM: raw {raw_mean:.2f} px, deconvolved {new_mean:.2f} px")


def test_tv_regularisation_limits_noise_amplification() -> None:
    rng = np.random.default_rng(1)
    flat = rng.poisson(np.full((1, 96, 96), 200.0)).astype(np.float32)
    psf = gaussian_psf((0, 2.0, 2.0), (1, 15, 15))
    plain = deconvolve(flat, psf, iterations=50, tv_lambda=0, background=0).image
    regular = deconvolve(flat, psf, iterations=50, tv_lambda=0.01, background=0).image
    assert plain.std() > 2 * flat.std(), "unregularised RL should amplify noise in this test"
    assert regular.std() < 0.6 * plain.std()
    assert regular.mean() == pytest.approx(200, rel=0.02)


def test_acceleration_converges_faster() -> None:
    truth = np.zeros((1, 80, 80))
    truth[0, 20:60:8, 20:60:8] = 5000
    psf = gaussian_psf((0, 2.5, 2.5), (1, 17, 17))
    blurred = blur(truth, psf).astype(np.float32)
    kwargs: dict[str, Any] = {"iterations": 25, "tv_lambda": 0, "background": 0}
    slow = deconvolve(blurred, psf, accelerate=False, **kwargs)
    fast = deconvolve(blurred, psf, accelerate=True, **kwargs)
    twice = deconvolve(blurred, psf, accelerate=False, **{**kwargs, "iterations": 50})

    def error(result: Any) -> float:
        return float(np.abs(result.image - truth).sum())

    # 25 accelerated iterations beat 50 plain ones, in data fit and distance to the truth.
    assert fast.residual < twice.residual < slow.residual
    assert error(fast) < error(twice) < error(slow)
    converged = deconvolve(blurred, psf, **{**kwargs, "iterations": 200})
    assert float(np.max(converged.image)) == pytest.approx(
        5000, rel=0.02
    )  # noise-free: exact recovery

    stopped = deconvolve(
        blurred, psf, iterations=500, tv_lambda=0, background=0, stop_rel_change=0.01
    )
    assert stopped.iterations < 500
    assert stopped.relative_change is not None and stopped.relative_change < 0.01


def test_wiener_sharpens_quickly() -> None:
    stack, psf, spots = bead_stack(n=4)
    result = deconvolve(stack, psf, method="wiener", wiener_nsr=0.01)
    assert result.iterations == 1 and result.relative_change is None
    z, y, x = spots[0]
    assert result.image[z, y, x] > 2 * (stack[z, y, x] - 100)
    assert float(np.min(result.image)) >= 0


def test_torch_backend_matches_numpy() -> None:
    pytest.importorskip("torch")
    stack, psf, _ = bead_stack(n=3)
    a = deconvolve(stack, psf, iterations=15, backend=NumpyBackend())
    b = deconvolve(stack, psf, iterations=15, backend=TorchBackend("cpu"))
    assert b.backend.startswith("cpu (torch")
    assert np.allclose(a.image, b.image, rtol=2e-2, atol=0.02 * float(np.max(a.image)))
    assert b.residual == pytest.approx(a.residual, rel=0.02)


def test_backend_selection_falls_back_to_cpu() -> None:
    assert isinstance(get_backend("cpu"), NumpyBackend)
    auto = get_backend("auto")
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        assert isinstance(auto, NumpyBackend)
        with pytest.raises(ModuleError, match="No CUDA GPU"):
            get_backend("gpu")


def test_tiled_result_matches_whole_image() -> None:
    rng = np.random.default_rng(2)
    truth = np.zeros((1, 200, 230))
    truth[0, rng.integers(10, 190, 60), rng.integers(10, 220, 60)] = 4000
    psf = gaussian_psf((0, 2.0, 2.0), (1, 13, 13))
    blurred = (blur(truth, psf) + 50).astype(np.float32)
    whole = deconvolve(blurred, psf, iterations=20, background=50)
    tiled = deconvolve(blurred, psf, iterations=20, background=50, tile_size=96)
    assert tiled.tiles == 9 and whole.tiles == 1
    difference = np.abs(tiled.image - whole.image)
    assert float(np.max(difference)) < 0.05 * float(np.max(whole.image)), (
        "visible seams between tiles"
    )
    assert tiled.image.sum() == pytest.approx(whole.image.sum(), rel=0.01)

    assert tile_slices(100, 0, 16) == [(0, 100)]
    assert tile_slices(100, 200, 16) == [(0, 100)]
    assert tile_slices(200, 96, 16) == [(0, 96), (80, 176), (104, 200)]
    with pytest.raises(ModuleError, match="larger than the tile overlap"):
        tile_slices(200, 10, 16)


def test_deconvolution_module_3d_2d_and_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    spec, _ = psf_tools.spec_from_metadata(CONFOCAL, 0)
    psf = psf_tools.theoretical_psf(spec, True)
    truth = np.zeros((15, 64, 64))
    truth[7, 32, 32] = 50000
    truth[7, 20, 44] = 50000
    data = rng.poisson(blur(truth, psf) + 20).astype(np.float32)
    image = ImageData(data[None], CONFOCAL)

    module = create_module("deconvolution", {"iterations": 20})
    out = module.apply(image, None)
    assert out.linear and out.history == ("deconvolution",) and out.data.shape == image.data.shape
    assert out.data[0, 7, 32, 32] > 3 * data[7, 32, 32]
    report = out.reports[0]
    channel = report["channels"][0]
    assert report["method"] == "rl" and report["backend"].startswith("cpu")
    assert (channel["mode"], channel["iterations"], channel["tiles"]) == ("3D", 20, 1)
    assert channel["seconds"] > 0 and 0 < channel["residual"] < 0.5
    assert channel["psf"]["source"] == "theoretical" and channel["psf"]["confocal"]
    assert channel["psf"]["fwhm_um"]["x"] < channel["psf"]["fwhm_um"]["z"]

    flat = create_module("deconvolution", {"iterations": 5, "force_2d": True}).apply(image, None)
    assert flat.reports[0]["channels"][0]["mode"] == "2D"

    # A measured PSF saved at a different pixel size is resampled to the image.
    psf_tools.save_psf(tmp_path / "psf.ome.tif", psf, 0.1, 0.25)
    measured = create_module(
        "deconvolution", {"iterations": 20, "psf_file": str(tmp_path / "psf.ome.tif")}
    ).apply(image, None)
    assert measured.reports[0]["channels"][0]["psf"]["source"] == "measured"
    assert np.allclose(measured.data, out.data, rtol=0.05, atol=0.02 * float(np.max(out.data)))

    uncalibrated = ImageMetadata(channels=(ChannelInfo("x"),))
    with pytest.raises(ModuleError, match="pixel size"):
        create_module("deconvolution").validate(uncalibrated)
    no_z = ImageMetadata(**{**CONFOCAL.__dict__, "z_step_um": None})
    with pytest.raises(ModuleError, match="Z step"):
        create_module("deconvolution").apply(ImageData(data[None], no_z), None)
    with pytest.raises(ModuleError, match="Could not read the PSF file"):
        create_module("deconvolution", {"psf_file": str(tmp_path / "missing.tif")}).apply(
            image, None
        )


def test_automatic_tiling_respects_the_memory_budget() -> None:
    psf_shape = (19, 79, 79)
    whole = estimate_memory((30, 2048, 2048), psf_shape)
    assert 15 * 1024**3 < whole < 25 * 1024**3  # ~80 bytes per padded voxel, measured
    assert auto_tile_size((30, 2048, 2048), psf_shape, whole) == 0
    tile = auto_tile_size((30, 2048, 2048), psf_shape, 2 * 1024**3)
    assert 128 <= tile < 2048 and tile % 64 == 0
    assert estimate_memory((30, tile, tile), psf_shape) <= 2 * 1024**3
    assert estimate_memory((30, tile + 64, tile + 64), psf_shape) > 2 * 1024**3
    with pytest.raises(ModuleError, match="Not enough memory"):
        auto_tile_size((30, 2048, 2048), psf_shape, 10 * 1024**2)

    # A small budget makes deconvolve tile on its own, with the same result.
    rng = np.random.default_rng(4)
    truth = np.zeros((1, 300, 300))
    truth[0, rng.integers(10, 290, 40), rng.integers(10, 290, 40)] = 4000
    psf = gaussian_psf((0, 2.0, 2.0), (1, 13, 13))
    blurred = (blur(truth, psf) + 50).astype(np.float32)
    whole_run = deconvolve(blurred, psf, iterations=10, background=50)
    budget = estimate_memory((1, 200, 200), (1, 13, 13)) / 1024**3
    tiled_run = deconvolve(blurred, psf, iterations=10, background=50, max_memory_gb=budget)
    assert whole_run.tiles == 1 and tiled_run.tiles > 1
    assert float(np.max(np.abs(tiled_run.image - whole_run.image))) < 0.05 * float(
        np.max(whole_run.image)
    )


def test_benchmark_command(capsys: pytest.CaptureFixture[str]) -> None:
    from hcs_studio.app import main

    assert (
        main(["benchmark", "--shape", "5", "64", "64", "--iterations", "3", "--device", "cpu"]) == 0
    )
    out = capsys.readouterr().out
    assert "64×64×5 voxels" in out and "3 iterations" in out and "s per iteration" in out
