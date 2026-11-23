"""Brightfield spheroids with known size and QC problems (Phase 21).

:class:`BrightfieldSpheroid` draws into channel ``channel`` of each field: a bright background
(``background``) that darkens smoothly and radially towards the field corners (well edge /
meniscus shadow, ``meniscus``), and a spheroid: a disc of radius ``spec['radius_um']`` (darker,
textured: cells, with a darker core), blurred by ``spec['blur_px']``. Optional per well
(``spec``): ``offset_um`` (dy, dx), ``second_radius_um`` (a second spheroid), ``debris`` (small
dark specks), ``attached_um`` (a blob stuck to the spheroid's edge), ``core_ratio`` (radius of
the darker necrotic core / radius; 0.5, 0 = none), ``strands`` ([(angle °, length µm, width µm)]:
invading strands of cells from the edge into the matrix), ``organoids`` + ``organoid_radius_um``
(many organoids instead of one spheroid). The truth records the radius, area, sphere volume,
core fraction, invasion and the expected QC flags. :class:`LiveDeadStains` draws a live stain
(calcein) in the viable rim and a dead stain (EthD-1, PI) in the core of the same spheroid.
:class:`ConfocalSpheroid` draws a 3D spheroid in a confocal Z stack (blur, depth attenuation,
necrotic core, partial Z coverage).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def geometry(spec: Mapping[str, Any], h: int, w: int, px: float) -> tuple[float, float, float]:
    """(centre y, centre x, radius) in pixels of the spheroid described by ``spec``."""
    dy, dx = spec.get("offset_um", (0.0, 0.0))
    return (h - 1) / 2 + dy / px, (w - 1) / 2 + dx / px, float(spec.get("radius_um", 0)) / px


def strand_mask(yy: Any, xx: Any, cy: float, cx: float, radius: float,
                strands: Any, px: float) -> Any:  # fmt: skip
    """Pixels of the invading strands (rectangles from inside the edge outwards)."""
    mask = np.zeros(yy.shape, dtype=bool)
    for angle, length, width in strands:
        a = math.radians(angle)
        along = (yy - cy) * math.sin(a) + (xx - cx) * math.cos(a)
        across = -(yy - cy) * math.cos(a) + (xx - cx) * math.sin(a)
        mask |= (
            (along >= radius - 3)
            & (along <= radius + length / px)
            & (np.abs(across) <= width / px / 2)
        )
    return mask


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
        cy, cx, radius = geometry(spec, h, w, px)
        spheres = [(cy, cx, radius)]
        organoids: list[float] = []
        if "organoids" in spec:
            spheres = []
            low, high = spec.get("organoid_radius_um", (15, 40))
            tries = 0
            while len(spheres) < int(spec["organoids"]) and tries < 5000:
                tries += 1
                r = float(ctx.rng.uniform(low, high)) / px
                y, x = (
                    ctx.rng.uniform(0.15 * h + r, 0.85 * h - r),
                    ctx.rng.uniform(0.15 * w + r, 0.85 * w - r),
                )
                if all(math.hypot(y - a, x - b) > r + c + 6 for a, b, c in spheres):
                    spheres.append((float(y), float(x), r))
            organoids = [2 * r * px for _, _, r in spheres]
        if spec.get("second_radius_um"):
            r2 = float(spec["second_radius_um"]) / px
            spheres.append((cy, cx + radius + r2 + 10, r2))
        content = np.zeros((h, w))
        alpha = np.zeros((h, w))
        for y, x, r in spheres:
            a = self._disc(yy, xx, y, x, r, blur)
            core = self._disc(yy, xx, y, x, float(spec.get("core_ratio", 0.5)) * r, blur)
            value = self.spheroid + (self.core - self.spheroid) * core
            content = np.where(a > alpha, value, content)
            alpha = np.maximum(alpha, a)
        if spec.get("attached_um"):
            ra = float(spec["attached_um"]) / px
            a = self._disc(yy, xx, cy, cx - radius - 0.6 * ra, ra, blur)
            content = np.where(a > alpha, self.spheroid, content)
            alpha = np.maximum(alpha, a)
        strands = spec.get("strands") or []
        invasion = np.zeros((h, w), dtype=bool)
        if strands:
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            invasion = strand_mask(yy, xx, cy, cx, radius, strands, px) & ~disc
            a = ndimage.gaussian_filter(
                strand_mask(yy, xx, cy, cx, radius, strands, px).astype(float), blur
            )
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
        if "organoids" in spec:
            return {"count": len(organoids), "organoid_diameters_um": organoids}
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
            "core_fraction": float(spec.get("core_ratio", 0.5)) ** 2,
            "invasion_area_um2": float(invasion.sum()) * px * px,
            "protrusions": len(strands),
            "max_invasion_um": max((s[1] for s in strands), default=0.0),
        }


@dataclass
class LiveDeadStains:
    """Live stain in the viable rim, dead stain in the necrotic core (radius ratio
    ``spec['core_ratio']``) of the spheroid in ``spec``; ``spec['dead_ratio']`` overrides the
    radius of the dead region (e.g. 0 for an untreated, fully viable spheroid)."""

    spec: Callable[[int, int], Mapping[str, Any]]
    name: str = "live_dead"
    live_channel: int = 1
    dead_channel: int = 2
    background: float = 100.0
    live: float = 800.0
    dead: float = 900.0

    def draw(self, ctx: Any) -> dict[str, Any]:
        from scipy import ndimage

        spec = dict(self.spec(ctx.row, ctx.col))
        image = ctx.image
        h, w = image.shape[-2:]
        cy, cx, radius = geometry(spec, h, w, ctx.pixel_size_um)
        ratio = float(spec.get("dead_ratio", spec.get("core_ratio", 0.5)))
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        distance = np.hypot(yy - cy, xx - cx)
        blur = float(spec.get("blur_px", 0.8))
        dead = ndimage.gaussian_filter((distance <= ratio * radius).astype(float), blur)
        whole = ndimage.gaussian_filter((distance <= radius).astype(float), blur)
        image[self.live_channel] += (self.background + self.live * np.clip(whole - dead, 0, 1))[
            None
        ]
        image[self.dead_channel] += (self.background + self.dead * dead)[None]
        return {
            "dead_fraction": ratio**2,
            "dead_area_um2": math.pi * (ratio * radius * ctx.pixel_size_um) ** 2,
        }


@dataclass
class ConfocalSpheroid:
    """A 3D spheroid in a confocal Z stack: a sphere of radius ``spec['radius_um']`` centred at
    ``spec['z_centre_um']`` (default: the middle of the stack) and ``offset_um``, stained
    uniformly in ``channel`` (e.g. Hoechst), a dead stain in its core (``core_ratio``) in
    ``dead_channel``, blurred by ``blur_xy_um`` / ``blur_z_um`` (an elongated confocal PSF) and
    dimmed with depth into the spheroid as exp(−depth / ``spec['attenuation_um']``) (none if
    absent). The truth records the volume, core volume fraction and whether the stack covers the
    whole spheroid."""

    spec: Callable[[int, int], Mapping[str, Any]]
    name: str = "confocal_spheroid"
    channel: int = 0
    dead_channel: int | None = None
    background: float = 100.0
    intensity: float = 800.0
    dead: float = 900.0
    blur_xy_um: float = 2.0
    blur_z_um: float = 4.0

    def draw(self, ctx: Any) -> dict[str, Any]:
        from scipy import ndimage

        spec = dict(self.spec(ctx.row, ctx.col))
        image = ctx.image
        planes, h, w = image.shape[-3:]
        px, dz = ctx.pixel_size_um, ctx.z_step_um
        cy, cx, _ = geometry(spec, h, w, px)
        radius = float(spec["radius_um"])
        zc = float(spec.get("z_centre_um", (planes - 1) / 2 * dz))
        zz, yy, xx = np.mgrid[0:planes, 0:h, 0:w].astype(float)
        distance = np.sqrt(((zz * dz - zc) ** 2) + ((yy - cy) * px) ** 2 + ((xx - cx) * px) ** 2)
        sigma = (self.blur_z_um / dz, self.blur_xy_um / px, self.blur_xy_um / px)
        sphere = ndimage.gaussian_filter((distance <= radius).astype(float), sigma)
        ratio = float(spec.get("core_ratio", 0.0))
        core = ndimage.gaussian_filter((distance <= ratio * radius).astype(float), sigma)
        attenuation = 1.0
        if spec.get("attenuation_um"):
            depth = np.clip(zz * dz - (zc - radius), 0, None)
            attenuation = np.exp(-depth / float(spec["attenuation_um"]))
        image[self.channel] += self.background + self.intensity * sphere * attenuation
        if self.dead_channel is not None:
            image[self.dead_channel] += self.background + self.dead * core * attenuation
        top, bottom = zc - radius, zc + radius
        complete = top >= 0 and bottom <= (planes - 1) * dz
        inside = min(bottom, (planes - 1) * dz + dz / 2) - max(top, -dz / 2)
        return {
            "count": 1,
            "radius_um": radius,
            "volume_um3": 4 / 3 * math.pi * radius**3,
            "core_volume_fraction": ratio**3,
            "complete_z": bool(complete),
            "z_coverage": float(min(1.0, max(inside, 0) / (2 * radius))),
            "attenuation_um": float(spec.get("attenuation_um", 0) or 0),
        }
