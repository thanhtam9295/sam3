"""Cell tracking over the time points of a field (Phase 22d), with scipy only.

``track_objects(objects, times_h, params)`` links the objects (nuclei) of one field from time
point to time point:

1. **frame to frame**: optimal assignment (``scipy.optimize.linear_sum_assignment``) of the
   objects at t to those at t + 1 on the squared distance from each object's predicted position
   (``predict``: its last displacement carried on — cells move persistently; objects without a
   history use their position), in the augmented form of Jaqaman et al.
   2008 — every object may instead end (or start) a track at a cost equal to the gate, so a
   cheaper overall solution may leave a far pair unlinked. Pairs farther than the gate
   (``max_speed_um_per_h`` × the interval, at least ``min_gate_um``) are never linked;
2. **divisions**: a track starting at t within reach of an object at t − 1 (the larger of
   ``division_radius_um`` and the gate) whose track continues into a sister at t, where both are
   smaller than ``max_daughter_ratio`` × the mother's area at t − 1, is a daughter: the mother's
   track ends at t − 1 and both daughters get new tracks with the mother as parent. Two guards
   against a merge of two touching nuclei splitting up again, which looks the same: a new object
   within the gate of a track lost in the last ``max_gap`` time points is that track coming
   back (left to gap closing), and a cell born less than ``min_cycle_h`` ago does not divide;
3. **gap closing**: a track that ends early and one that starts up to ``max_gap`` time points
   later, within the gate for that time, are joined (a missed detection or a nucleus out of
   focus at one time point) — again by optimal assignment.

Tracks are numbered per field from 1. ``track_table`` summarises each track: start and end,
duration, displacement, path length, mean speed (path / duration), straightness (displacement /
path), parent, whether it divides, how it ends (division, end of the movie, lost) and, for a
track born from a division that divides again, the cell-cycle time — birth and division are
placed halfway between the time points around them (they happened in those intervals), so the
cycle is not short by one interval. Complete cycles must fit in the movie: in a movie not much
longer than the cycle, only the faster cycles are complete (censoring). Speeds
are of the nucleus centroid in the drift-corrected frame (use ``register_timepoints``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.table_steps import TableModule, WellContext
from hcs_studio.processing.base import MODULES, ModuleError, ModuleParams

BIG = 1e12


@dataclass
class TrackingOptions:
    max_speed_um_per_h: float = 30.0
    min_gate_um: float = 5.0
    max_gap: int = 2
    division_radius_um: float = 20.0
    max_daughter_ratio: float = 0.8
    divisions: bool = True
    predict: bool = True
    min_cycle_h: float = 4.0


def _assign(cost: NDArray[np.float64], alternative: float) -> list[tuple[int, int]]:
    """Pairs (i, j) of the augmented assignment (Jaqaman et al. 2008): ``cost`` n × m with
    ``inf`` for forbidden pairs; not linking an object costs ``alternative``."""
    from scipy.optimize import linear_sum_assignment

    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    allowed = np.isfinite(cost)
    top_left = np.where(allowed, cost, BIG)
    top_right = np.full((n, n), BIG)
    np.fill_diagonal(top_right, alternative)
    bottom_left = np.full((m, m), BIG)
    np.fill_diagonal(bottom_left, alternative)
    bottom_right = np.where(allowed.T, 0.0, BIG)  # lets the unused alternatives pair up
    full = np.block([[top_left, top_right], [bottom_left, bottom_right]])
    rows, cols = linear_sum_assignment(full)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True)
            if i < n and j < m and allowed[i, j]]  # fmt: skip


def _distances(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2), dtype=np.float64)


def _reappears(
    points: dict[int, NDArray[Any]],
    nxt: dict[tuple[int, int], tuple[int, int]],
    recent: list[int],
    t: int,
    j: int,
    gate: Any,
    parent_of: dict[tuple[int, int], tuple[int, int]],
) -> bool:
    """Whether a track ended (not by division) at one of the ``recent`` time points within the
    gate of object j at t — a nucleus merged with a neighbour or missed, now back."""
    mothers = set(parent_of.values())
    for t0 in recent[:-1] if len(recent) > 1 else []:
        for i in range(len(points[t0])):
            if (t0, i) in nxt or (t0, i) in mothers:
                continue
            if float(((points[t0][i] - points[t][j]) ** 2).sum()) <= gate(t0, t) ** 2:
                return True
    return False


def _birth(
    node: tuple[int, int],
    prv: dict[tuple[int, int], tuple[int, int]],
    parent_of: dict[tuple[int, int], tuple[int, int]],
) -> int | None:
    """Time point at which the track through ``node`` was born from a division (None if it was
    not)."""
    while node in prv:
        node = prv[node]
    return node[0] if node in parent_of else None


def track_objects(
    objects: pd.DataFrame, times_h: list[float], options: TrackingOptions | None = None
) -> pd.DataFrame:
    """Track ids for the objects of one field: a frame with ``timepoint``, ``centroid_y_um``,
    ``centroid_x_um`` (and ``area_um2`` for divisions). Returns ``trk_id``, ``trk_parent``
    (0: none) and ``trk_divided`` (born from a division), indexed like ``objects``."""
    options = options or TrackingOptions()
    out = pd.DataFrame({"trk_id": 0, "trk_parent": 0, "trk_divided": False}, index=objects.index)
    if objects.empty:
        return out
    frames = {int(str(t)): g for t, g in objects.groupby("timepoint")}
    order = sorted(frames)
    points = {t: frames[t][["centroid_y_um", "centroid_x_um"]].to_numpy(float) for t in order}
    areas = {t: frames[t]["area_um2"].to_numpy(float) if "area_um2" in frames[t] else
             np.ones(len(frames[t])) for t in order}  # fmt: skip

    def gate(t1: int, t2: int) -> float:
        dt = abs(times_h[t2] - times_h[t1]) if t2 < len(times_h) else 1.0
        return max(options.max_speed_um_per_h * dt, options.min_gate_um)

    # 1. frame-to-frame links: (t, i) → (t', j)
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    prv: dict[tuple[int, int], tuple[int, int]] = {}
    for k, (t1, t2) in enumerate(itertools.pairwise(order)):
        limit = gate(t1, t2)
        cost = _distances(points[t1], points[t2])
        allowed = cost <= limit**2
        if options.predict and k > 0:
            # cells keep moving the way they moved: cost to the predicted position
            # (last velocity × interval); the gate stays on the actual distance
            predicted = points[t1].copy()
            t0 = order[k - 1]
            scale = (times_h[t2] - times_h[t1]) / max(times_h[t1] - times_h[t0], 1e-9)
            for i in range(len(predicted)):
                before = prv.get((t1, i))
                if before is not None:
                    predicted[i] += (points[t1][i] - points[before[0]][before[1]]) * scale
            cost = _distances(predicted, points[t2])
        cost[~allowed] = np.inf
        for i, j in _assign(cost, limit**2):
            nxt[(t1, i)] = (t2, j)
            prv[(t2, j)] = (t1, i)

    # 2. divisions: a new object near a mother whose track continues into its sister
    parent_of: dict[tuple[int, int], tuple[int, int]] = {}  # daughter start → mother end
    if options.divisions:
        for k, t in enumerate(order[1:], start=1):
            previous = order[k - 1]
            reach = max(options.division_radius_um, gate(previous, t))
            recent = [order[i] for i in range(max(k - 1 - options.max_gap, 0), k)]
            for j in [j for j in range(len(points[t])) if (t, j) not in prv]:
                if _reappears(points, nxt, recent, t, j, gate, parent_of):
                    continue  # a track lost just before: gap closing, not a division
                d = np.sqrt(((points[previous] - points[t][j]) ** 2).sum(axis=1))
                for m in np.argsort(d):
                    if d[m] > reach:
                        break
                    mother = (previous, int(m))
                    sister = nxt.get(mother)
                    if sister is None or sister in parent_of:
                        continue
                    birth_t = _birth(mother, prv, parent_of)
                    if birth_t is not None and (
                        times_h[previous] - times_h[birth_t] < options.min_cycle_h
                    ):
                        continue  # too young to divide: a merge splitting up
                    limit = options.max_daughter_ratio * areas[previous][int(m)]
                    if areas[t][j] < limit and areas[t][sister[1]] < limit:
                        del nxt[mother]  # the mother's track ends; both daughters start
                        del prv[sister]
                        parent_of[(t, j)] = mother
                        parent_of[sister] = mother
                        break

    # 3. gap closing: ends before the last time point → starts up to max_gap later
    ends = [(t, i) for t in order[:-1] for i in range(len(points[t]))
            if (t, i) not in nxt and not any(m == (t, i) for m in parent_of.values())]  # fmt: skip
    starts = [(t, j) for t in order[1:] for j in range(len(points[t]))
              if (t, j) not in prv and (t, j) not in parent_of]  # fmt: skip
    if ends and starts and options.max_gap > 0:
        position = {t: k for k, t in enumerate(order)}
        cost = np.full((len(ends), len(starts)), np.inf)
        for a, (t1, i) in enumerate(ends):
            for b, (t2, j) in enumerate(starts):
                skipped = position[t2] - position[t1]
                if 2 <= skipped <= options.max_gap + 1:
                    d2 = float(((points[t1][i] - points[t2][j]) ** 2).sum())
                    if d2 <= gate(t1, t2) ** 2:
                        cost[a, b] = d2
        alternative = max(gate(order[0], order[min(1, len(order) - 1)]), options.min_gate_um) ** 2
        for a, b in _assign(cost, alternative):
            nxt[ends[a]] = starts[b]
            prv[starts[b]] = ends[a]

    # number the tracks: follow links from every start
    track_of: dict[tuple[int, int], int] = {}
    parents: dict[int, int] = {}
    born: set[int] = set()
    number = 0
    for t in order:
        for i in range(len(points[t])):
            node = (t, i)
            if node in prv or node in track_of:
                continue
            number += 1
            while node is not None and node not in track_of:
                track_of[node] = number
                node = nxt.get(node)  # type: ignore[assignment]
    for daughter, mother in parent_of.items():
        parents[track_of[daughter]] = track_of[mother]
        born.add(track_of[daughter])
    for t in order:
        index = frames[t].index
        ids = [track_of[(t, i)] for i in range(len(index))]
        out.loc[index, "trk_id"] = ids
        out.loc[index, "trk_parent"] = [parents.get(k, 0) for k in ids]
        out.loc[index, "trk_divided"] = [k in born for k in ids]
    return out


def track_table(objects: pd.DataFrame, times_h: list[float]) -> pd.DataFrame:
    """One row per track (objects with ``trk_id`` / ``trk_parent`` / ``trk_divided``)."""
    if objects.empty:
        return pd.DataFrame()
    last = max(int(t) for t in objects["timepoint"])
    frames = sorted(int(t) for t in objects["timepoint"].unique())
    position = {t: k for k, t in enumerate(frames)}

    def between(t: int, step: int) -> float | None:
        """Halfway to the neighbouring time point (an event happened in that interval)."""
        k = position[t] + step
        return (times_h[t] + times_h[frames[k]]) / 2 if 0 <= k < len(frames) else None

    mothers = set(objects.loc[objects["trk_parent"] > 0, "trk_parent"].astype(int))
    rows = []
    for track, group in objects.sort_values("timepoint").groupby("trk_id"):
        t = [times_h[int(k)] for k in group["timepoint"]]
        y, x = group["centroid_y_um"].to_numpy(float), group["centroid_x_um"].to_numpy(float)
        steps = np.hypot(np.diff(y), np.diff(x))
        path = float(steps.sum())
        duration = float(t[-1] - t[0])
        displacement = float(math.hypot(y[-1] - y[0], x[-1] - x[0]))
        divides = int(str(track)) in mothers
        end = (
            "division"
            if divides
            else ("end" if int(group["timepoint"].iloc[-1]) == last else "lost")
        )
        born = bool(group["trk_divided"].iloc[0])
        birth = between(int(group["timepoint"].iloc[0]), -1) if born else None
        division = between(int(group["timepoint"].iloc[-1]), 1) if divides else None
        cycle = division - birth if birth is not None and division is not None else math.nan
        rows.append({
            "track": int(str(track)), "parent": int(group["trk_parent"].iloc[0]),
            "born_by_division": bool(group["trk_divided"].iloc[0]),
            "start_h": t[0], "end_h": t[-1], "frames": len(group), "duration_h": duration,
            "displacement_um": displacement, "path_um": path,
            "speed_um_per_h": path / duration if duration > 0 else math.nan,
            "straightness": displacement / path if path > 0 else math.nan,
            "divides": divides, "ends": end, "cycle_h": cycle,
        })  # fmt: skip
    return pd.DataFrame(rows)


def well_summary(tracks: pd.DataFrame, min_frames: int = 3) -> dict[str, Any]:
    """Per-well values from the track table(s) of the well's fields."""
    if tracks.empty:
        return {"trk_tracks": 0}
    long = tracks[tracks["frames"] >= min_frames]
    cell_hours = float(tracks["duration_h"].sum())
    divisions = int(tracks["divides"].sum())
    cycles = tracks["cycle_h"].dropna() if "cycle_h" in tracks else pd.Series(dtype=float)
    return {
        "trk_tracks": len(long),
        "trk_speed_um_per_h": float(long["speed_um_per_h"].median())
        if not long.empty
        else math.nan,
        "trk_straightness": float(long["straightness"].median()) if not long.empty else math.nan,
        "trk_divisions": divisions,
        "trk_division_rate_per_h": divisions / cell_hours if cell_hours > 0 else math.nan,
        "trk_cycle_h": float(cycles.median()) if not cycles.empty else math.nan,
        "trk_lost_fraction": float((tracks["ends"] == "lost").mean()),
    }


# ----------------------------------------------------------------- table step


class TrackingParams(ModuleParams):
    object_type: str = Field(default="nuclei", description="Objects to track (one per cell).")
    max_speed_um_per_h: float = Field(
        default=30.0, gt=0, description="Fastest plausible movement; farther pairs never link."
    )
    min_gate_um: float = Field(default=5.0, ge=0, description="Smallest linking distance.")
    max_gap: int = Field(default=2, ge=0, le=10, description="Time points a track may skip.")
    divisions: bool = Field(default=True, description="Detect divisions (parent tracks).")
    division_radius_um: float = Field(default=20.0, gt=0, description="Daughters within this.")
    max_daughter_ratio: float = Field(
        default=0.8, gt=0, le=1.5, description="Daughters smaller than this × the mother's area."
    )
    min_track_frames: int = Field(default=3, ge=1, description="Shorter tracks are not summarised.")
    min_cycle_h: float = Field(
        default=4.0, ge=0, description="A cell born less than this ago does not divide."
    )


@MODULES.register("tracking")
class Tracking(TableModule):
    """Tracks cells over the time points of each field (see module docs)."""

    name = "tracking"
    title = "Cell tracking"
    Params = TrackingParams

    def __init__(self, params: Any = None) -> None:
        super().__init__(params)
        self._tracks: list[pd.DataFrame] = []
        self._times: list[float] | None = None

    def times(self) -> list[float]:
        if self._times is None:
            import json

            if self.run_folder is None:
                raise ModuleError("tracking needs the run folder")
            manifest = json.loads((self.run_folder / "run_manifest.json").read_text("utf-8"))
            seconds = (manifest.get("plate") or {}).get("time_points_s")
            if not seconds or len(seconds) < 2:
                raise ModuleError(
                    "tracking needs a time-lapse run with times (set the time interval in the "
                    "Metadata panel)"
                )
            self._times = [float(s) / 3600 for s in seconds]
        return self._times

    def _options(self) -> TrackingOptions:
        p: TrackingParams = self.params
        return TrackingOptions(p.max_speed_um_per_h, p.min_gate_um, p.max_gap,
                               p.division_radius_um, p.max_daughter_ratio, p.divisions,
                               min_cycle_h=p.min_cycle_h)  # fmt: skip

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        times = self.times()
        out = pd.DataFrame({"trk_id": 0, "trk_parent": 0, "trk_divided": False},
                           index=objects.index)  # fmt: skip
        chosen = objects[self.selected(objects)]
        if chosen.empty:
            return out
        for field, group in chosen.groupby("field"):
            ids = track_objects(group, times, self._options())
            # unique within the well: field × 100000 + track
            offset = int(str(field)) * 100_000
            ids.loc[ids["trk_id"] > 0, "trk_id"] += offset
            ids.loc[ids["trk_parent"] > 0, "trk_parent"] += offset
            out.loc[ids.index, ["trk_id", "trk_parent", "trk_divided"]] = ids
            table = track_table(pd.concat([group, ids], axis=1), times)
            if not table.empty:
                table.insert(0, "field", int(str(field)))
                table.insert(0, "well", well.well)
                self._tracks.append(table)
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mine = [t for t in self._tracks if not t.empty and t["well"].iloc[0] == well.well]
        tracks = pd.concat(mine, ignore_index=True) if mine else pd.DataFrame()
        return well_summary(tracks, self.params.min_track_frames), {}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        out = Path(folder) / "tracking"
        out.mkdir(parents=True, exist_ok=True)
        tracks = pd.concat(self._tracks, ignore_index=True) if self._tracks else pd.DataFrame()
        tracks.to_csv(out / "tracks.csv", index=False)
        self._tracks = []
