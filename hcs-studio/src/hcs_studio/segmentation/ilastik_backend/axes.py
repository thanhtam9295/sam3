"""Axis order between HCS Studio and ilastik.

HCS Studio arrays are ``(T, C, Z, Y, X)``. ilastik describes data with an axis string made of
the letters ``t z y x c`` (for example ``yxc``, ``zyxc``, ``tyxc``, ``tzyxc``); the number of
letters equals the number of dimensions. A trained project only accepts data with the same
spatial dimensionality and the same channels as its training data, so the axis string comes
from the project (or, for new data, from :func:`ilastik_axes`), and every axis HCS Studio
has but the string lacks must have length 1.

This is the only place that converts between the two orders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

INTERNAL = "tczyx"  # HCS Studio's order (T, C, Z, Y, X)
CANONICAL = "tzyxc"  # ilastik's usual order
LETTERS = frozenset(INTERNAL)


def check_axes(axes: str) -> str:
    """Normalise and validate an ilastik axis string (lower case, unique letters of tzyxc,
    with y and x)."""
    axes = axes.strip().lower()
    if not axes:
        raise ValueError("Empty axis order")
    unknown = sorted(set(axes) - LETTERS)
    if unknown:
        raise ValueError(
            f"Axis order {axes!r}: unknown axis {''.join(unknown)!r} (use t, z, y, x, c)"
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"Axis order {axes!r} repeats an axis")
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"Axis order {axes!r} needs both y and x")
    return axes


def ilastik_axes(
    shape: tuple[int, int, int, int, int],
    *,
    time: bool | None = None,
    z: bool | None = None,
    channel: bool = True,
) -> str:
    """The ilastik axis string (canonical order) for a ``(T, C, Z, Y, X)`` shape.

    ``time`` and ``z`` force an axis in (True) or out (False); by default an axis is kept
    when it is longer than 1. The channel axis is kept unless ``channel=False`` (allowed only
    for one channel), as ilastik itself does for exported data.
    """
    t, c, zz, _, _ = shape
    keep_t = t > 1 if time is None else time
    keep_z = zz > 1 if z is None else z
    if not keep_t and t > 1:
        raise ValueError(f"The data has {t} timepoints but the axis order has no time axis")
    if not keep_z and zz > 1:
        raise ValueError(
            f"The data has {zz} Z planes but the project is 2D: add a Z projection step "
            "before the ilastik step (or train a 3D project)"
        )
    if not channel and c > 1:
        raise ValueError(f"The data has {c} channels but the axis order has no channel axis")
    return "".join(
        a for a, keep in zip(CANONICAL, (keep_t, keep_z, True, True, channel), strict=True) if keep
    )


def to_ilastik(data: NDArray[Any], axes: str) -> NDArray[Any]:
    """Reorder a ``(T, C, Z, Y, X)`` array to ``axes``; axes not in ``axes`` must be length 1."""
    axes = check_axes(axes)
    if data.ndim != 5:
        raise ValueError(f"Expected a (T, C, Z, Y, X) array, got {data.ndim} dimensions")
    names = {"t": "timepoints", "c": "channels", "z": "Z planes"}
    for letter, size in zip(INTERNAL, data.shape, strict=True):
        if letter not in axes and size != 1:
            raise ValueError(
                f"The data has {size} {names.get(letter, letter)} but the axis order {axes!r} "
                f"has no {letter!r} axis"
            )
    squeezed = [i for i, a in enumerate(INTERNAL) if a not in axes]
    reduced = np.squeeze(data, axis=tuple(squeezed)) if squeezed else data
    remaining = [a for a in INTERNAL if a in axes]
    return np.transpose(reduced, [remaining.index(a) for a in axes])


def from_ilastik(data: NDArray[Any], axes: str) -> NDArray[Any]:
    """Reorder an array with ilastik ``axes`` to ``(T, C, Z, Y, X)`` (missing axes become 1)."""
    axes = check_axes(axes)
    if data.ndim != len(axes):
        raise ValueError(
            f"Axis order {axes!r} has {len(axes)} letters but the data has {data.ndim} dimensions"
        )
    present = [a for a in INTERNAL if a in axes]
    ordered = np.transpose(data, [axes.index(a) for a in present])
    shape = [ordered.shape[present.index(a)] if a in axes else 1 for a in INTERNAL]
    return ordered.reshape(shape)


def shape_in(axes: str, shape_tczyx: tuple[int, ...]) -> tuple[int, ...]:
    """The shape a ``(T, C, Z, Y, X)`` array has after :func:`to_ilastik`."""
    axes = check_axes(axes)
    return tuple(shape_tczyx[INTERNAL.index(a)] for a in axes)


def describe(axes: str) -> str:
    """Human-readable dimensionality: '2D', '3D', '2D + time', '3D + time'."""
    axes = check_axes(axes)
    text = "3D" if "z" in axes else "2D"
    return text + (" + time" if "t" in axes else "")
