"""Growing cells on a time-lapse plate with known counts, confluence and drift (Phase 22).

:class:`GrowingCells` draws, at each time point, ``count(row, col, hours)`` cells of a fixed
pool of positions (the same cells stay where they are; new ones appear): a nucleus (disc of
radius ``nucleus_px`` in ``nucleus_channel``) inside a textured, darker cell body (disc of
``body_px`` in ``body_channel`` — a brightfield/phase-like image on a bright background). The
whole field moves by ``drift(row, col, hours)`` (µm, stage drift). The truth per time point:
cells, confluence (% of the field covered by bodies) and the drift in pixels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def exponential(n0: float, doubling_h: float, cap: int | None = None) -> Callable[..., int]:
    """count(row, col, hours) for exponential growth from ``n0`` (capped at ``cap``)."""

    def count(row: int, col: int, hours: float) -> int:
        n = round(n0 * 2 ** (hours / doubling_h))
        return min(n, cap) if cap is not None else n

    return count


@dataclass
class GrowingCells:
    count: Callable[[int, int, float], int]
    name: str = "cells"
    nucleus_channel: int = 0
    body_channel: int | None = 1
    nucleus_px: float = 5.0
    body_px: float = 16.0
    #: Cells stay this far from the field's edge (room for the drift).
    margin_px: float = 0.0
    drift: Callable[[int, int, float], tuple[float, float]] | None = None
    background: float = 1000.0
    body: float = 650.0
    texture: float = 60.0
    nucleus: float = 900.0

    def _pool(self, ctx: Any) -> np.ndarray:
        """Cell positions of this field, in the order cells appear (same at every time)."""
        h, w = ctx.image.shape[-2:]
        pitch = 2 * self.body_px + 4
        edge = self.body_px + 6 + self.margin_px
        ys = np.arange(edge, h - edge, pitch)
        xs = np.arange(edge, w - edge, pitch)
        grid = np.array([(y, x) for y in ys for x in xs], dtype=float)
        rng = np.random.default_rng([ctx.row, ctx.col, ctx.field, 7])
        grid += rng.uniform(-1.0, 1.0, grid.shape)
        return grid[rng.permutation(len(grid))]

    def draw(self, ctx: Any) -> dict[str, Any]:
        image = ctx.image
        h, w = image.shape[-2:]
        hours = ctx.time_s / 3600
        pool = self._pool(ctx)
        n = min(int(self.count(ctx.row, ctx.col, hours)), len(pool))
        dy_um, dx_um = self.drift(ctx.row, ctx.col, hours) if self.drift else (0.0, 0.0)
        dy, dx = dy_um / ctx.pixel_size_um, dx_um / ctx.pixel_size_um
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        bodies = np.zeros((h, w), dtype=bool)
        nuclei = np.zeros((h, w), dtype=bool)
        for y, x in pool[:n]:
            distance = (yy - (y + dy)) ** 2 + (xx - (x + dx)) ** 2
            bodies |= distance <= self.body_px**2
            nuclei |= distance <= self.nucleus_px**2
        image[self.nucleus_channel, :, nuclei] += self.nucleus
        if self.body_channel is not None:
            from scipy import ndimage

            noise = np.random.default_rng([ctx.row, ctx.col, ctx.field, 3]).normal(0, 1, (h, w))
            texture = ndimage.shift(ndimage.gaussian_filter(noise, 1.0), (dy, dx), order=1)
            texture *= self.texture / max(float(texture.std()), 1e-9)
            # the optics blur outlines (σ 1 px): no pixel-sharp steps
            plane = ndimage.gaussian_filter(np.where(bodies, self.body, self.background), 1.0)
            plane = plane + np.where(bodies, texture, 0.0)
            image[self.body_channel] += plane[None]
        return {"count": n, "confluence_pct": 100 * float(bodies.mean()),
                "drift_px": [round(dy, 4), round(dx, 4)]}  # fmt: skip


def drug_rate(rate_per_h: float, conc: float, emax: float, ec50: float, hill: float = 1.0) -> float:
    """Growth rate under a drug that lowers it by ``emax`` × the Hill occupancy (emax > 1:
    the cells die at high doses): μ(c) = μ₀ · (1 − emax · cʰ / (cʰ + EC50ʰ))."""
    occupancy = conc**hill / (conc**hill + ec50**hill) if conc > 0 else 0.0
    return rate_per_h * (1 - emax * occupancy)


def response_course(
    wells: dict[str, tuple[str, float]],
    times_h: list[float],
    *,
    n0: float = 400.0,
    doubling_h: float = 24.0,
    emax: float = 1.5,
    ec50: float = 1e-7,
    hill: float = 1.0,
    treated_h: float = 0.0,
    onset_h: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A time-course table with known answers: each well (role, concentration in M) grows
    exponentially until ``treated_h`` + ``onset_h`` (the drug acts after a delay), then at the
    drug's rate (Poisson counts). Exponential
    growth gives a GR that does not change with time:
    GR(c) = 2^(μ(c)/μ₀) − 1, and % of control exp((μ(c) − μ₀)(t − treated))."""
    rng = np.random.default_rng(seed)
    rate0 = math.log(2) / doubling_h
    rows = []
    for well, (role, conc) in wells.items():
        rate = drug_rate(rate0, conc, emax, ec50, hill) if role == "sample" else rate0
        for i, t in enumerate(times_h):
            start = treated_h + onset_h
            before = min(t, start)
            after = max(t - start, 0.0)
            mean = n0 * math.exp(rate0 * before + rate * after)
            rows.append({"well": well, "timepoint": i, "time_h": t,
                         "hours_since_treatment": t - treated_h,
                         "nuclei_objects_measured": float(rng.poisson(mean))})  # fmt: skip
    return pd.DataFrame(rows)


def simulate_lineages(
    times_h: list[float],
    size_um: tuple[float, float],
    *,
    n0: int,
    speed_um_per_h: float,
    cycle_h: float,
    radius_um: float,
    seed: int,
    margin_um: float = 15.0,
    min_spacing_um: float = 30.0,
) -> list[list[dict[str, Any]]]:
    """Cells per time point: ``id``, ``parent`` (0: none), ``y``, ``x`` (µm), ``radius`` (µm).

    Cells move with a persistent random walk (heading changes by N(0, 0.5 rad) per hour, speed
    ``speed_um_per_h``) inside the field (reflected at ``margin_um``), grow from area A to 2A over
    their cycle (``cycle_h`` ± 10 %, the first cells at random ages) and divide into two cells of
    area A placed on either side of the mother. Steps are of at most 0.25 h between the times
    asked for, so positions do not depend on the sampling."""
    rng = np.random.default_rng(seed)
    height, width = size_um
    cells: list[dict[str, Any]] = []
    next_id = 1
    placed: list[tuple[float, float]] = []
    while len(cells) < n0:
        y = rng.uniform(margin_um, height - margin_um)
        x = rng.uniform(margin_um, width - margin_um)
        if all(math.hypot(y - a, x - b) >= min_spacing_um for a, b in placed):
            placed.append((y, x))
            cycle = cycle_h * rng.uniform(0.9, 1.1)
            cells.append({"id": next_id, "parent": 0, "y": y, "x": x,
                          "heading": rng.uniform(0, 2 * math.pi), "age": rng.uniform(0, cycle),
                          "cycle": cycle})  # fmt: skip
            next_id += 1
        elif len(placed) + 1 > 10 * n0:
            break
    out: list[list[dict[str, Any]]] = []
    now = 0.0

    def snapshot() -> list[dict[str, Any]]:
        return [
            {"id": c["id"], "parent": c["parent"], "y": c["y"], "x": c["x"],
             "radius": radius_um * math.sqrt(1 + c["age"] / c["cycle"])}
            for c in cells
        ]  # fmt: skip

    for target in times_h:
        while now < target - 1e-9:
            dt = min(0.25, target - now)
            born: list[dict[str, Any]] = []
            for c in cells:
                c["heading"] += rng.normal(0, 0.5 * math.sqrt(dt))
                c["y"] += speed_um_per_h * dt * math.sin(c["heading"])
                c["x"] += speed_um_per_h * dt * math.cos(c["heading"])
                for key, limit in (("y", height), ("x", width)):
                    if c[key] < margin_um:
                        c[key] = 2 * margin_um - c[key]
                        c["heading"] += math.pi
                    elif c[key] > limit - margin_um:
                        c[key] = 2 * (limit - margin_um) - c[key]
                        c["heading"] += math.pi
                c["age"] += dt
                if c["age"] >= c["cycle"]:
                    axis = rng.uniform(0, math.pi)
                    offset = 1.6 * radius_um
                    for sign in (1, -1):
                        y = c["y"] + sign * offset * math.sin(axis)
                        x = c["x"] + sign * offset * math.cos(axis)
                        born.append({"id": next_id, "parent": c["id"],
                                     "y": min(max(y, margin_um), height - margin_um),
                                     "x": min(max(x, margin_um), width - margin_um),
                                     "heading": axis if sign > 0 else axis + math.pi, "age": 0.0,
                                     "cycle": cycle_h * rng.uniform(0.9, 1.1)})  # fmt: skip
                        next_id += 1
                    c["dead"] = True
            cells = [c for c in cells if not c.get("dead")] + born
            now += dt
        out.append(snapshot())
    return out


@dataclass
class MovingCells:
    """Nuclei that move and divide (``simulate_lineages``); truth per time point: every cell's
    id, parent and position, so links can be checked."""

    times_h: list[float]
    n0: int = 12
    speed_um_per_h: float = 10.0
    cycle_h: float = 20.0
    radius_um: float = 4.0
    name: str = "moving"
    channel: int = 0
    intensity: float = 1500.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._cache: dict[tuple[int, int, int], list[list[dict[str, Any]]]] = {}

    def draw(self, ctx: Any) -> dict[str, Any]:
        h, w = ctx.image.shape[-2:]
        px = ctx.pixel_size_um
        key = (ctx.row, ctx.col, ctx.field)
        if key not in self._cache:
            self._cache[key] = simulate_lineages(
                self.times_h, (h * px, w * px), n0=self.n0, speed_um_per_h=self.speed_um_per_h,
                cycle_h=self.cycle_h, radius_um=self.radius_um,
                seed=hash((self.seed, *key)) % 2**32,
            )  # fmt: skip
        cells = self._cache[key][ctx.timepoint]
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        for cell in cells:
            disc = (yy * px - cell["y"]) ** 2 + (xx * px - cell["x"]) ** 2 <= cell["radius"] ** 2
            ctx.image[self.channel, :, disc] += self.intensity
        rounded = [{k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()}
                   for c in cells]  # fmt: skip
        return {"count": len(cells), "cells": rounded}
