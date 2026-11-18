"""Intensity along a line (for checking background removal before and after)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def line_profile(
    plane: NDArray[Any],
    start: tuple[float, float],
    end: tuple[float, float],
    pixel_um: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Values of a 2D image along the segment ``start`` → ``end`` ((y, x) in pixels),
    sampled once per pixel of length with bilinear interpolation. Returns the distance
    from ``start`` (µm when ``pixel_um`` is given, else pixels) and the values; points
    outside the image are left out."""
    from scipy import ndimage

    (y0, x0), (y1, x1) = start, end
    length = float(np.hypot(y1 - y0, x1 - x0))
    n = max(2, int(np.ceil(length)) + 1)
    t = np.linspace(0.0, 1.0, n)
    ys, xs = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
    inside = (ys >= 0) & (ys <= plane.shape[0] - 1) & (xs >= 0) & (xs <= plane.shape[1] - 1)
    values = ndimage.map_coordinates(
        np.asarray(plane, dtype=np.float64), [ys[inside], xs[inside]], order=1
    )
    distance = t[inside] * length * (pixel_um or 1.0)
    return distance, np.asarray(values, dtype=np.float64)
