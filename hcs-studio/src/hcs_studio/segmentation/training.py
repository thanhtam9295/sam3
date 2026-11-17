"""Training data from edited segmentations, in Cellpose's ``*_img.tif`` / ``*_masks.tif`` format."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.processing.base import ModuleError
from hcs_studio.segmentation.objects import relabel_sequential


def save_training_pair(
    folder: Path,
    name: str,
    image_czyx: NDArray[Any],
    labels_zyx: NDArray[Any],
    channels: list[int],
) -> list[Path]:
    """Write one image/mask pair per Z plane that has objects. Returns the image files."""
    if image_czyx.ndim != 4 or labels_zyx.ndim != 3 or image_czyx.shape[1:] != labels_zyx.shape:
        raise ModuleError("The image and the labels do not have matching shapes")
    bad = [c for c in channels if not 0 <= c < image_czyx.shape[0]]
    if bad or not channels:
        raise ModuleError("Choose existing channels for the training images")
    import tifffile

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "field"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for z in range(labels_zyx.shape[0]):
        masks = relabel_sequential(labels_zyx[z])
        if not masks.any():
            continue
        stem = f"{safe}_z{z:03d}" if labels_zyx.shape[0] > 1 else safe
        image = np.asarray(image_czyx[channels, z], dtype=np.float32)
        tifffile.imwrite(folder / f"{stem}_img.tif", image if len(channels) > 1 else image[0])
        tifffile.imwrite(
            folder / f"{stem}_masks.tif",
            masks.astype(np.uint16 if int(np.max(masks)) < 65536 else np.uint32),
        )
        written.append(folder / f"{stem}_img.tif")
    if not written:
        raise ModuleError("The labels are empty: nothing to save")
    return written
