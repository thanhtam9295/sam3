"""Brightfield spheroids with known size and QC problems (Phase 21).

:class:`BrightfieldSpheroid` draws into channel ``channel`` of each field: a bright background
(``background``) that darkens smoothly and radially towards the field corners (well edge /
meniscus shadow, ``meniscus``), and a spheroid: a disc of radius ``spec['radius_um']`` (darker,
textured: cells, with a darker core), blurred by ``spec['blur_px']``. Optional per well
(``spec``): ``offset_um`` (dy, dx), ``second_radius_um`` (a second spheroid), ``debris`` (small
dark specks), ``attached_um`` (a blob stuck to the spheroid's edge). The truth records the
radius, area, sphere volume and the expected QC flags.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BrightfieldSpheroid:
    spec: Callable[[int, int], Mapping[str, Any]]
    name: str = "spheroid"
    channel: int = 0
    background: float = 1000.0
    spheroid: float = 420.0
    core: float = 300.0
    texture: float = 35.0
    meniscus: float = 0.6

    def _disc(self, yy: Any, xx: Any, cy: float, cx: float, r: float, blur: float) -> Any:
        from scipy import ndimage

        inside = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(float)
        return ndimage.gaussian_filter(inside, blur) if blur > 0 else inside

    def draw(self, ctx: Any) -> dict[str, Any]:
        from scipy import ndimage

        spec = dict(self.spec(ctx.row, ctx.col))
        image = ctx.image
        h, w = image.shape[-2:]
        px = ctx.pixel_size_um
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        # meniscus / well edge: smooth radial darkening towards the field corners
        rho = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)
        u = np.clip((rho - 0.75) / 0.7, 0, 1)
        field = self.background * (1 - self.meniscus * u * u * (3 - 2 * u))
        texture = ndimage.gaussian_filter(ctx.rng.normal(0, 1, (h, w)), 1.5)
        texture *= self.texture / max(float(texture.std()), 1e-9)
        blur = float(spec.get("blur_px", 0.8))
        dy, dx = spec.get("offset_um", (0.0, 0.0))
        cy, cx = (h - 1) / 2 + dy / px, (w - 1) / 2 + dx / px
        radius = float(spec["radius_um"]) / px
        spheres = [(cy, cx, radius)]
        if spec.get("second_radius_um"):
            r2 = float(spec["second_radius_um"]) / px
            spheres.append((cy, cx + radius + r2 + 10, r2))
        content = np.zeros((h, w))
        alpha = np.zeros((h, w))
        for y, x, r in spheres:
            a = self._disc(yy, xx, y, x, r, blur)
            core = self._disc(yy, xx, y, x, 0.5 * r, blur)
            value = self.spheroid + (self.core - self.spheroid) * core
            content = np.where(a > alpha, value, content)
            alpha = np.maximum(alpha, a)
        if spec.get("attached_um"):
            ra = float(spec["attached_um"]) / px
            a = self._disc(yy, xx, cy, cx - radius - 0.6 * ra, ra, blur)
            content = np.where(a > alpha, self.spheroid, content)
            alpha = np.maximum(alpha, a)
        for _ in range(int(spec.get("debris", 0))):
            y, x = ctx.rng.uniform(0.2 * h, 0.8 * h), ctx.rng.uniform(0.2 * w, 0.8 * w)
            if math.hypot(y - cy, x - cx) < radius + 15:
                y = cy + radius + 20
            a = self._disc(yy, xx, y, x, 6, blur)
            content = np.where(a > alpha, self.spheroid, content)
            alpha = np.maximum(alpha, a)
        image[self.channel] += (field * (1 - alpha) + (content + texture) * alpha)[None]
        area = math.pi * float(spec["radius_um"]) ** 2
        flags = []
        if spec.get("second_radius_um"):
            flags.append("multiple")
        if spec.get("debris"):
            flags.append("debris")
        if spec.get("attached_um"):
            flags.append("irregular")
        if blur >= 6:
            flags.append("out_of_focus")
        if min(cy - radius, cx - radius, h - 1 - cy - radius, w - 1 - cx - radius) < 0:
            flags.append("touching_edge")
        elif math.hypot(dy, dx) / px > 0.5 * min(h, w) / 2:
            flags.append("off_centre")
        return {
            "count": 1,
            "radius_um": float(spec["radius_um"]),
            "area_um2": area,
            "volume_um3": 4 / 3 * math.pi * float(spec["radius_um"]) ** 3,
            "centre_px": [cy, cx],
            "flags": flags,
        }
