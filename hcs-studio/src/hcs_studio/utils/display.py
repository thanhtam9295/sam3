"""Display helpers: channel colours and quick-look thumbnails. Not for measurement."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.model.plate import ChannelInfo

_FALLBACK = ("blue", "green", "red", "magenta", "cyan", "yellow", "gray")
_RGB: dict[str, tuple[float, float, float]] = {
    "blue": (0.15, 0.35, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "gray": (1.0, 1.0, 1.0),
}


def channel_colormap(channel: ChannelInfo, index: int) -> str:
    """Colormap name for a channel: gray for transmitted light (brightfield, phase contrast),
    else from its emission wavelength when known."""
    name = channel.name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if name in ("bf", "tl", "dpc", "dic") or any(
        word in name for word in ("brightfield", "phase", "transmitted", "trans")
    ):
        return "gray"
    emission = channel.emission_nm
    if emission is None:
        return _FALLBACK[index % len(_FALLBACK)]
    if emission < 480:
        return "blue"
    if emission < 550:
        return "green"
    if emission < 600:
        return "yellow"
    if emission < 660:
        return "red"
    return "magenta"


def channel_colormaps(channels: tuple[ChannelInfo, ...]) -> list[str]:
    return [channel_colormap(c, i) for i, c in enumerate(channels)]


def make_thumbnail(data: Any, colormaps: list[str], max_size: int = 160) -> NDArray[np.uint8]:
    """RGB quick-look of a lazy (T, C, Z, Y, X) image: first timepoint, middle Z plane.

    Each channel is scaled between its 1st and 99.8th percentile, so the result is
    for orientation only.
    """
    if data.ndim != 5:
        raise ValueError(f"Expected (T, C, Z, Y, X), got {data.ndim} dimensions")
    n_c, n_z, height, width = data.shape[1:]
    if len(colormaps) != n_c:
        raise ValueError("One colormap per channel is required")
    step = max(1, -(-max(height, width) // max_size))
    planes = np.asarray(data[0, :, n_z // 2, ::step, ::step]).astype(np.float32)
    rgb = np.zeros((*planes.shape[1:], 3), dtype=np.float32)
    for plane, name in zip(planes, colormaps, strict=True):
        low, high = np.percentile(plane, (1.0, 99.8))
        if high <= low:
            continue
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        rgb += scaled[..., None] * np.asarray(_RGB.get(name, _RGB["gray"]), dtype=np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


#: Outline colours for label sets, in order of creation (nuclei, cells, spots, ...).
OUTLINE_COLOURS: tuple[tuple[int, int, int], ...] = (
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (255, 255, 255),
)


def add_outlines(rgb: NDArray[np.uint8], label_planes: list[NDArray[Any]]) -> NDArray[np.uint8]:
    """Draw object outlines (full-resolution 2D label planes) onto a possibly smaller RGB image."""
    from skimage.segmentation import find_boundaries

    out = rgb.copy()
    for index, plane in enumerate(label_planes):
        if plane.size == 0 or not plane.any():
            continue
        edges = find_boundaries(plane, mode="inner")
        step = max(1, -(-max(plane.shape) // max(rgb.shape[:2])))
        if step > 1:
            height, width = rgb.shape[:2]
            padded = np.zeros((height * step, width * step), dtype=bool)
            h, w = min(edges.shape[0], height * step), min(edges.shape[1], width * step)
            padded[:h, :w] = edges[:h, :w]
            edges = padded.reshape(height, step, width, step).any(axis=(1, 3))
        edges = edges[: rgb.shape[0], : rgb.shape[1]]
        out[edges] = OUTLINE_COLOURS[index % len(OUTLINE_COLOURS)]
    return out


def outline_overlay(
    shape: tuple[int, int], label_plane: NDArray[Any], colour: tuple[int, int, int]
) -> NDArray[np.uint8]:
    """Transparent RGBA image of a label plane's outlines, at a (smaller) preview size."""
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    drawn = add_outlines(np.zeros((*shape, 3), dtype=np.uint8), [label_plane])
    edges = drawn.any(axis=-1)
    rgba[edges, :3] = colour
    rgba[edges, 3] = 255
    return rgba
