"""Known-answer tests for whole-well stitching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

from hcs_studio.analysis.stitching import StitchResult, mosaic, owner_fields, stitch

PX = 0.5  # µm per pixel
TILE = 128
STEP = 100


def _grid(
    stage_sign_y: int = -1, noise_um: float = 1.5, seed: int = 0
) -> tuple[dict[int, tuple[float, float]], dict[int, NDArray[Any]], dict[int, NDArray[Any]]]:
    """3×3 overlapping fields cut from one textured image. Returns stage positions (x, y),
    images and the true top-left (row, col) in µm."""
    rng = np.random.default_rng(seed)
    big = ndimage.gaussian_filter(rng.uniform(0, 1, (400, 400)), 2) * 1000
    positions, images, truth = {}, {}, {}
    k = 0
    for r in range(3):
        for c in range(3):
            k += 1
            y = 10 + r * STEP + int(rng.integers(-3, 4))
            x = 10 + c * STEP + int(rng.integers(-3, 4))
            truth[k] = np.array([y * PX, x * PX])
            images[k] = big[y : y + TILE, x : x + TILE]
            positions[k] = (
                x * PX + rng.normal(0, noise_um),
                stage_sign_y * y * PX + rng.normal(0, noise_um),
            )
    return positions, images, truth


def _max_error(result: StitchResult, truth: dict[int, NDArray[Any]]) -> float:
    ref = np.array(result.offsets_um[1]) - truth[1]
    return float(max(np.abs(np.array(result.offsets_um[k]) - truth[k] - ref).max() for k in truth))


@pytest.mark.parametrize("sign", [1, -1])
def test_detects_stage_orientation(sign: int) -> None:
    positions, images, _ = _grid(stage_sign_y=sign)
    result = stitch(positions, (TILE * PX, TILE * PX), images, PX)
    assert result.orientation == (sign, 1)


def test_refinement_reaches_subpixel_accuracy() -> None:
    positions, images, truth = _grid(noise_um=1.5)
    size = (TILE * PX, TILE * PX)
    unrefined = stitch(positions, size, images, PX, refine=False)
    refined = stitch(positions, size, images, PX)
    assert refined.refined_pairs >= 12  # all edge neighbours (small diagonal overlaps may fail)
    assert _max_error(unrefined, truth) > 2.0
    assert _max_error(refined, truth) < PX  # better than one pixel
    assert min(refined.offsets_um[k][0] for k in refined.offsets_um) == pytest.approx(0)


def test_no_overlap_falls_back_to_stage_positions() -> None:
    positions = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (0.0, 100.0)}
    images = {k: np.random.default_rng(k).uniform(size=(64, 64)) for k in positions}
    result = stitch(positions, (32.0, 32.0), images, 0.5)
    assert result.orientation == (1, 1)
    assert result.offsets_um[2] == (0.0, 100.0)
    assert result.offsets_um[3] == (100.0, 0.0)
    assert any("do not overlap" in note for note in result.notes)


def test_featureless_overlaps_are_rejected() -> None:
    positions = {1: (0.0, 0.0), 2: (50.0, 0.0)}
    flat = {k: np.full((128, 128), 100.0) for k in positions}
    result = stitch(positions, (64.0, 64.0), flat, 0.5, orientation=(1, 1))
    assert result.refined_pairs == 0
    assert result.offsets_um[2] == (0.0, 50.0)
    assert any("no reliable overlaps" in note for note in result.notes)


def test_single_field() -> None:
    result = stitch({7: (1234.0, -55.0)}, (100.0, 100.0))
    assert result.offsets_um == {7: (0.0, 0.0)}


def test_owner_fields_assigns_overlap_to_nearest_centre() -> None:
    result = StitchResult({1: (0.0, 0.0), 2: (0.0, 80.0)}, (1, 1))
    size = (100.0, 100.0)
    points = np.array([[50.0, 10.0], [50.0, 85.0], [50.0, 95.0], [50.0, 170.0], [150.0, 50.0]])
    # Overlap is x in [80, 100); the boundary between the centres (50, 130) is x = 90.
    assert owner_fields(points, result, size).tolist() == [1, 1, 2, 2, -1]


def test_mosaic_places_tiles() -> None:
    result = StitchResult({1: (0.0, 0.0), 2: (0.0, 20.0)}, (1, 1))
    tiles = {1: np.full((10, 10, 3), 50, np.uint8), 2: np.full((10, 10, 3), 200, np.uint8)}
    canvas, scale = mosaic(tiles, result, tile_scale_um=2.0)
    assert canvas.shape[:2] == (11, 21) and scale == 2.0
    assert canvas[5, 2, 0] == 50 and canvas[5, 15, 0] == 200
    small, small_scale = mosaic(tiles, result, tile_scale_um=2.0, max_size=10)
    assert max(small.shape[:2]) <= 11 and small_scale == pytest.approx(4.0)
