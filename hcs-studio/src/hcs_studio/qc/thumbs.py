"""Field thumbnails and a small image embedding for QC review and learned QC (Phase 18).

- :func:`thumbnail`: the content channel scaled to 8 bits (1st–99.8th percentile of the
  field) and shrunk to at most 192 px, written as ``qc/thumbs/<well>_f<field>.png``.
- :func:`embedding`: 28 numbers describing the field's appearance independently of its
  brightness: the 4 × 4 block means of the normalised field (16) and the radial power
  spectrum of a 64 × 64 version in 12 log-spaced rings (12), as ``emb_00`` … ``emb_27``.
  With the QC metrics it is the input of learned QC.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

THUMBS = "thumbs"
SIZE = 192
N_EMBEDDING = 28


def thumb_name(well: str, field: int) -> str:
    return f"{well}_f{int(field):03d}.png"


def _shrink(plane: NDArray[Any], size: int) -> NDArray[np.float64]:
    h, w = plane.shape
    factor = max(1, int(np.ceil(max(h, w) / size)))
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    blocks = np.asarray(plane[:h2, :w2], dtype=np.float64).reshape(
        h2 // factor, factor, w2 // factor, factor
    )
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float64)


def thumbnail(plane: NDArray[Any], path: Path) -> Path:
    from PIL import Image

    small = _shrink(plane, SIZE)
    low, high = np.percentile(small, [1, 99.8])
    scaled = np.clip((small - low) / (high - low if high > low else 1.0), 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype(np.uint8)).save(path, optimize=True)
    return path


def embedding(plane: NDArray[Any]) -> dict[str, float]:
    small = _shrink(plane, 64)
    centred = small - np.median(small)
    scale = float(np.percentile(np.abs(centred), 99)) or 1.0
    normalised = centred / scale
    h, w = normalised.shape
    blocks = [
        float(normalised[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4].mean())
        for i in range(4)
        for j in range(4)
    ]
    power = np.abs(np.fft.fftshift(np.fft.fft2(normalised))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - h / 2, xx - w / 2)
    edges = np.geomspace(1, min(h, w) / 2, 13)
    total = float(power.sum()) or 1.0
    rings = [
        float(np.log10(power[(radius >= a) & (radius < b)].sum() / total + 1e-12))
        for a, b in itertools.pairwise(edges)
    ]
    values = blocks + rings
    return {f"emb_{i:02d}": v for i, v in enumerate(values)}
