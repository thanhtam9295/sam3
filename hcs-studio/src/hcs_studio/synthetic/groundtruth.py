"""Synthetic annotated fields for training and validating segmentation (Phase 25).

``nuclei_field`` draws elliptical nuclei with exact instance masks. Two "cell lines" differ the
way real ones do: ``easy`` = small, well separated, even nuclei; ``hard`` = larger nuclei in
touching clusters with uneven brightness and a dim halo, which a default threshold +
watershed merges. ``ground_truth_items`` makes (Item, image, masks) triples for
``training.groundtruth.create_set`` with plate/well/field provenance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

CELL_LINES: dict[str, dict[str, float]] = {
    "easy": {"radius": 5.0, "spacing": 3.0, "jitter": 0.12, "cv": 0.1, "clusters": 0.0},
    "hard": {"radius": 7.0, "spacing": 1.55, "jitter": 0.12, "cv": 0.45, "clusters": 1.0},
}


def nuclei_field(
    rng: np.random.Generator, cell_line: str = "easy", size: int = 128, noise: float = 6.0
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """(image (1, Y, X), masks (Y, X)) of one field."""
    p = CELL_LINES[cell_line]
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    masks = np.zeros((size, size), np.int32)
    image = np.zeros((size, size))
    pitch = p["radius"] * 2 * p["spacing"] / 2
    centres = []
    for cy in np.arange(pitch, size - pitch / 2, pitch):
        for cx in np.arange(pitch, size - pitch / 2, pitch):
            if p["clusters"] and rng.random() < 0.25:  # gaps between clusters
                continue
            centres.append(
                (cy + rng.normal(0, p["jitter"] * pitch), cx + rng.normal(0, p["jitter"] * pitch))
            )
    label = 0
    for cy, cx in centres:
        a = p["radius"] * rng.uniform(0.85, 1.2)
        b = p["radius"] * rng.uniform(0.75, 1.0)
        theta = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = (dx * np.cos(theta) + dy * np.sin(theta)) / a
        v = (-dx * np.sin(theta) + dy * np.cos(theta)) / b
        inside = (u**2 + v**2 <= 1) & (masks == 0)
        if inside.sum() < 12:
            continue
        label += 1
        masks[inside] = label
        brightness = 600 * max(0.25, rng.normal(1, p["cv"]))
        image[inside] += brightness * (1 - 0.35 * (u[inside] ** 2 + v[inside] ** 2))
    from scipy import ndimage

    halo = ndimage.gaussian_filter(image, 2.0) * (0.25 if p["clusters"] else 0.0)
    image = ndimage.gaussian_filter(image, 0.8) + halo + 100 + rng.normal(0, noise, image.shape)
    return image[None].astype(np.float32), masks


def ground_truth_items(
    cell_line: str = "easy",
    plates: int = 2,
    wells: int = 3,
    fields: int = 2,
    seed: int = 0,
    size: int = 128,
    pixel_size_um: float = 0.65,
) -> list[tuple[Any, NDArray[np.float32], NDArray[np.int32]]]:
    from hcs_studio.training.groundtruth import Item

    rng = np.random.default_rng([seed, len(cell_line)])
    out = []
    for p in range(plates):
        for w in range(wells):
            for f in range(fields):
                image, masks = nuclei_field(rng, cell_line, size)
                well = f"B{w + 2:02d}"
                item = Item(
                    id=f"{cell_line}_P{p + 1}_{well}_f{f + 1}",
                    plate=f"P{p + 1}",
                    well=well,
                    field=f + 1,
                    cell_line=cell_line,
                    stain="Hoechst",
                    annotator="synthetic",
                    source="synthetic truth",
                    pixel_size_um=pixel_size_um,
                )
                out.append((item, image, masks))
    return out
