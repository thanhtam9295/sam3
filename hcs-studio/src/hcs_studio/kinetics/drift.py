"""Drift between time points (Phase 22a).

``register_timepoints`` is an image-scope step that ``process_well`` recognises (like
``stitch_well``): before a field's time points are processed, the shift of each time point
against the previous one is measured by phase cross-correlation (scikit-image, sub-pixel;
mean-subtracted, Hann-windowed) on a
maximum projection of ``channel`` (downsampled to ``max_size``), the shifts are accumulated, and
each time point is moved back onto the first (linear interpolation; edges repeat the nearest
pixel). Consecutive pairs are compared, not every time point with the first, because cells
move and divide: neighbouring frames look most alike. A shift larger than ``max_shift_um``
(a jump of the stage, not drift) is not applied and is reported.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    Module,
    ModuleError,
    ModuleParams,
    block_reduce_mean,
)


def pair_shift(
    reference: NDArray[Any], moving: NDArray[Any], upsample: int = 10
) -> tuple[float, float]:
    """(dy, dx) in pixels that moves ``moving`` onto ``reference``. Both are mean-subtracted
    and multiplied by a Hann window: the correlation is periodic, and the jump between
    opposite image edges would otherwise pull the shift towards zero."""
    from skimage.registration import phase_cross_correlation

    window = np.outer(np.hanning(reference.shape[-2]), np.hanning(reference.shape[-1]))

    def prepare(plane: NDArray[Any]) -> NDArray[np.float64]:
        x = np.asarray(plane, dtype=np.float64)
        return np.asarray((x - x.mean()) * window, dtype=np.float64)

    shift, _error, _phase = phase_cross_correlation(
        prepare(reference), prepare(moving), upsample_factor=upsample, normalization=None
    )
    return float(shift[0]), float(shift[1])


def drift_shifts(
    planes: list[NDArray[Any]], factor: float = 1.0, upsample: int = 10
) -> list[tuple[float, float]]:
    """Cumulative (dy, dx) of every time point against the first, from consecutive pairs;
    ``planes`` may be downsampled by ``factor`` (shifts are returned at full resolution)."""
    shifts = [(0.0, 0.0)]
    for previous, current in itertools.pairwise(planes):
        dy, dx = pair_shift(previous, current, upsample)
        y0, x0 = shifts[-1]
        shifts.append((y0 + dy * factor, x0 + dx * factor))
    return shifts


def shift_image(data: NDArray[Any], shift: tuple[float, float]) -> NDArray[np.float32]:
    """Shift (C, Z, Y, X) by (dy, dx) pixels (linear interpolation, nearest edge)."""
    from scipy import ndimage

    if shift == (0.0, 0.0):
        return np.asarray(data, dtype=np.float32)
    out = np.empty(data.shape, dtype=np.float32)
    for c in range(data.shape[0]):
        for z in range(data.shape[1]):
            out[c, z] = ndimage.shift(data[c, z], shift, order=1, mode="nearest")
    return out


def common_area(
    shifts: list[tuple[float, float]], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Pixels to remove (top, bottom, left, right) so that every shifted time point covers the
    rest: a shift by +d leaves the first ⌈d⌉ rows without data."""
    top = max(math.ceil(max(dy, 0.0) - 1e-6) for dy, _ in shifts)
    bottom = max(math.ceil(max(-dy, 0.0) - 1e-6) for dy, _ in shifts)
    left = max(math.ceil(max(dx, 0.0) - 1e-6) for _, dx in shifts)
    right = max(math.ceil(max(-dx, 0.0) - 1e-6) for _, dx in shifts)
    if top + bottom >= shape[0] or left + right >= shape[1]:
        raise ModuleError("The drift is larger than the field: no area is seen at every time point")
    return top, bottom, left, right


class RegisterTimepointsParams(ModuleParams):
    channel: int = Field(default=1, ge=1, description="Channel used to measure the drift.")
    max_size: int = Field(default=512, ge=64, description="Downsample to at most this (px).")
    max_shift_um: float = Field(
        default=50.0, gt=0, description="Larger jumps are stage moves, not drift: not applied."
    )
    upsample: int = Field(default=10, ge=1, le=100, description="Sub-pixel precision 1/n px.")


@MODULES.register("register_timepoints")
class RegisterTimepoints(Module):
    """Corrects drift between the time points of each field (see module docs)."""

    name = "register_timepoints"
    title = "Register time points (drift)"
    Params = RegisterTimepointsParams

    def validate(self, metadata: ImageMetadata) -> None:
        if self.params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return image  # process_well moves each time point before the steps run

    def shifts(self, load: Callable[[int], ImageData], n_timepoints: int) -> dict[str, Any]:
        """Shifts of every time point of one field (pixels, and µm when calibrated)."""
        params: RegisterTimepointsParams = self.params
        planes, factor, pixel = [], 1.0, None
        for t in range(n_timepoints):
            image = load(t)
            plane = image.data[params.channel - 1].max(axis=0)
            small = block_reduce_mean(plane, params.max_size)
            factor = plane.shape[-1] / small.shape[-1]
            pixel = image.metadata.pixel_size_um
            planes.append(small)
        shifts = drift_shifts(planes, factor, params.upsample)
        limit = params.max_shift_um / pixel if pixel else math.inf
        # a jump between consecutive time points larger than the limit is a stage move: the
        # time point is left in place (and the drift continues from it)
        applied, warnings = [(0.0, 0.0)], []
        for t in range(1, len(shifts)):
            step = (shifts[t][0] - shifts[t - 1][0], shifts[t][1] - shifts[t - 1][1])
            y0, x0 = applied[-1]
            if math.hypot(*step) > limit:
                size = math.hypot(*step) * (pixel or 1)
                unit = "µm" if pixel else "px"
                warnings.append(f"time point {t + 1}: jump of {size:.1f} {unit} not corrected "
                                "(stage move?)")  # fmt: skip
                step = (0.0, 0.0)
            applied.append((y0 + step[0], x0 + step[1]))
        height, width = planes[0].shape[-2:]
        crop = common_area(applied, (round(height * factor), round(width * factor)))
        return {
            "shifts_px": [[round(dy, 3), round(dx, 3)] for dy, dx in applied],
            "drift_um": [
                round(math.hypot(dy, dx) * pixel, 3) if pixel else None for dy, dx in applied
            ],
            "crop_px": list(crop),
            "warnings": warnings,
        }

    @staticmethod
    def correct(image: ImageData, shift: list[float], crop: list[int] | None = None) -> ImageData:
        """The time point moved onto the first and cropped to the area every time point sees."""
        moved = shift_image(image.data, (float(shift[0]), float(shift[1])))
        if crop is not None:
            top, bottom, left, right = crop
            moved = moved[..., top : moved.shape[-2] - bottom, left : moved.shape[-1] - right]
        return image.derive(np.ascontiguousarray(moved), "register_timepoints")
