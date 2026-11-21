"""Image artifacts with known positions for QC tests (Phase 18).

:class:`Artifacts` is a :class:`~hcs_studio.synthetic.SyntheticPlate` feature. Put it after
the background and nuclei and before the noise; ``kind(row, col, field)`` names the artifact
of each field (``None`` = clean), and the truth records it per field.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

#: Artifact classes the QC rules are tested on.
KINDS = (
    "blur",
    "saturation",
    "bubble",
    "debris",
    "empty",
    "gradient",
    "out_of_well",
    "scratch",
    "fiber",
)


@dataclass
class Artifacts:
    kind: Callable[[int, int, int], str | None]
    level: float = 100.0  # the plate background drawn before (Background.level)
    name: str = "artifact"
    channel: int = 0

    def draw(self, ctx: Any) -> dict[str, Any]:
        kind = self.kind(ctx.row, ctx.col, ctx.field)
        if kind is None:
            return {"kind": None}
        if kind not in KINDS:
            raise ValueError(f"Unknown artifact {kind!r}")
        image = ctx.image  # (C, Z, Y, X)
        rng: np.random.Generator = ctx.rng
        h, w = image.shape[-2:]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        detail: dict[str, Any] = {"kind": kind}
        if kind == "blur":
            sigma = float(rng.uniform(2.0, 3.0))
            for c in range(image.shape[0]):
                for z in range(image.shape[1]):
                    image[c, z] = ndimage.gaussian_filter(image[c, z], sigma)
            detail["sigma_px"] = round(sigma, 2)
        elif kind == "saturation":
            plane = image[self.channel].max(axis=0)
            labels, n = ndimage.label(plane > self.level + 200)
            chosen: Any = (
                rng.choice(np.arange(1, n + 1), size=max(1, n // 2), replace=False) if n else []
            )
            hit = np.isin(labels, chosen)
            image[self.channel][:, hit] = 70000.0  # clipped to 65535 when written
            detail["pixels"] = int(hit.sum())
        elif kind == "bubble":
            radius = float(rng.uniform(0.15, 0.22) * min(h, w))
            cy = float(rng.uniform(radius + 8, h - radius - 8))
            cx = float(rng.uniform(radius + 8, w - radius - 8))
            dist = np.hypot(yy - cy, xx - cx)
            inside = dist < radius
            rim = np.abs(dist - radius) <= 1.5
            image[:, :, inside] *= 0.55
            image[:, :, rim] += 0.8 * self.level
            detail.update(centre_px=[round(cy, 1), round(cx, 1)], radius_px=round(radius, 1))
        elif kind == "debris":
            cy, cx = rng.uniform(0.2, 0.8, 2) * (h, w)
            blob = np.zeros((h, w), dtype=bool)
            for _ in range(5):
                oy, ox = rng.normal(0, 6, 2)
                r = rng.uniform(5, 8)
                blob |= np.hypot(yy - cy - oy, xx - cx - ox) <= r
            image[self.channel][:, blob] += 6000.0
            detail.update(
                centre_px=[round(float(cy), 1), round(float(cx), 1)], area_px=int(blob.sum())
            )
        elif kind == "empty":
            image[self.channel] = self.level
        elif kind == "gradient":
            angle = float(rng.uniform(0, 2 * np.pi))
            t = ((xx - w / 2) * np.cos(angle) + (yy - h / 2) * np.sin(angle)) / (
                0.5 * np.hypot(h, w)
            )
            image *= 1.0 - 0.35 * (t + 1)  # 1.0 at one corner, 0.3 at the other
            detail["angle_rad"] = round(angle, 3)
        elif kind == "out_of_well":
            angle = float(rng.uniform(0, 2 * np.pi))
            big = 4.0 * max(h, w)  # well edge: an arc of a large circle crossing the field
            depth = float(rng.uniform(0.25, 0.4))
            cy = h / 2 - np.sin(angle) * (big - (0.5 - depth) * h)
            cx = w / 2 - np.cos(angle) * (big - (0.5 - depth) * w)
            outside = np.hypot(yy - cy, xx - cx) > big
            image[:, :, outside] = 0.5 * self.level
            detail["fraction"] = round(float(outside.mean()), 3)
        elif kind == "scratch":
            angle = float(rng.uniform(0, np.pi))
            width = float(rng.uniform(0.22, 0.3) * min(h, w))
            d = (xx - w / 2) * np.sin(angle) - (yy - h / 2) * np.cos(angle)
            band = np.abs(d) < width / 2
            image[self.channel][:, band] = self.level
            detail["width_px"] = round(width, 1)
        elif kind == "fiber":
            angle = float(rng.uniform(0, np.pi))
            length = 0.6 * min(h, w)
            cy, cx = h / 2, w / 2
            along = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            across = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            line = (np.abs(along) < length / 2) & (np.abs(across) < 1.6)
            image[self.channel][:, line] += 1200.0
            detail["length_px"] = round(length, 1)
        return detail


def qc_plate(
    root: Any,
    *,
    per_kind: int = 30,
    clean: int = 150,
    kinds: tuple[str, ...] = KINDS,
    fields_per_well: int = 6,
    size: int = 256,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """A labeled OME-Zarr plate for QC: ``clean`` clean fields (rounded up to whole wells) and
    ``per_kind`` fields of each artifact, shuffled over the plate, 70–140 nuclei per field.
    Returns (plate path, truth); ``truth["labels"]`` maps ``"<well>/<field from 1>"`` to the
    artifact (``None`` = clean)."""
    from pathlib import Path

    from hcs_studio.model.plate import well_name
    from hcs_studio.synthetic import Background, Nuclei, ShotNoise, SyntheticPlate

    rng = np.random.default_rng(seed)
    n_artifacts = per_kind * len(kinds)
    clean += -(clean + n_artifacts) % fields_per_well  # whole wells: round the clean fields up
    labels: list[str | None] = [None] * clean + [k for k in kinds for _ in range(per_kind)]
    rng.shuffle(labels)
    n_wells = -(-len(labels) // fields_per_well)
    wells = [(r, c) for r in range(16) for c in range(24)][:n_wells]
    assigned: dict[tuple[int, int, int], str | None] = {}
    for i, label in enumerate(labels):
        r, c = wells[i // fields_per_well]
        assigned[(r, c, i % fields_per_well)] = label
    counts = {key: int(rng.integers(70, 141)) for key in assigned}
    plate = SyntheticPlate(
        wells=wells,
        features=[
            Background(level=100.0),
            Nuclei(count=lambda r, c, f: counts.get((r, c, f), 0), grid=12),
            Artifacts(kind=lambda r, c, f: assigned.get((r, c, f))),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "GFP"),
        size=size,
        n_fields=fields_per_well,
        name="QC artifacts",
        seed=seed,
    )
    path, truth = plate.write(Path(root))
    truth["labels"] = {
        f"{well_name(r, c)}/{f + 1}": label for (r, c, f), label in sorted(assigned.items())
    }
    truth["nucleus_diameter_um"] = round(2 * 3.5 * plate.pixel_size_um, 2)
    return path, truth
