"""Spatial statistics of object positions in a well or field.

Point-pattern statistics follow the standard definitions (Baddeley, Rubak & Turner,
*Spatial Point Patterns*, 2015):

* Ripley's K with border (reduced-sample) correction: only objects at least r from the edge
  of the imaged area are used as centres at radius r. L = √(K/π); H = L − r (0 under complete
  spatial randomness, CSR; > 0 clustered; < 0 regular).
* Pair correlation g(r): the same counts in rings, divided by the CSR expectation (1 = CSR).
* CSR envelopes: pointwise min/max of H and g over simulated uniform patterns with the same
  number of points in the same imaged area (39 simulations → two-sided 5 % pointwise test).
* Clark–Evans ratio R = mean nearest-neighbour distance / its CSR expectation, with Donnelly's
  edge correction; R ≈ 1 random, < 1 clustered, up to 2.15 for a hexagonal lattice.
* Voronoi cell area (local density) and Delaunay neighbour count; objects whose Voronoi
  cell is not closed inside the imaged area get NaN (their true neighbourhood is unknown).
* Colonies: DBSCAN (eps, min_samples) or HDBSCAN clusters of positions.

The imaged area (window) is the union of the field rectangles, rasterised for edge
distances and simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.table_steps import (
    TableModule,
    TableParams,
    WellContext,
    unique_rows,
)
from hcs_studio.processing.base import MODULES, ModuleError

Rect = tuple[float, float, float, float]  # (y0, x0, y1, x1) in µm


@dataclass(frozen=True)
class Window:
    """The imaged area as a raster: ``mask[r, c]`` covers µm ``origin + (r, c) * res``."""

    mask: NDArray[np.bool_]
    origin: tuple[float, float]
    res: float
    #: Distance (µm) from each pixel centre to the outside of the window.
    edge_distance: NDArray[np.float64]

    @classmethod
    def from_rects(cls, rects: list[Rect], max_pixels: int = 1_000_000) -> Window:
        if not rects:
            raise ValueError("a window needs at least one rectangle")
        array = np.asarray(rects, dtype=float)
        y0, x0 = array[:, 0].min(), array[:, 1].min()
        y1, x1 = array[:, 2].max(), array[:, 3].max()
        res = max(math.sqrt((y1 - y0) * (x1 - x0) / max_pixels), 1e-6)
        shape = (max(1, math.ceil((y1 - y0) / res)), max(1, math.ceil((x1 - x0) / res)))
        mask = np.zeros(shape, dtype=bool)
        for ry0, rx0, ry1, rx1 in array:
            r0, r1 = round((ry0 - y0) / res), round((ry1 - y0) / res)
            c0, c1 = round((rx0 - x0) / res), round((rx1 - x0) / res)
            mask[r0:r1, c0:c1] = True
        from scipy import ndimage

        distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        edge = np.maximum(np.asarray(distance, dtype=float) - 0.5, 0) * res
        return cls(mask, (float(y0), float(x0)), res, edge)

    @property
    def area_um2(self) -> float:
        return float(self.mask.sum()) * self.res**2

    @property
    def perimeter_um(self) -> float:
        padded = np.pad(self.mask, 1).astype(np.int8)
        edges = np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum()
        return float(edges) * self.res

    def _index(self, points: NDArray[Any]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        rows = np.floor((points[:, 0] - self.origin[0]) / self.res).astype(np.intp)
        cols = np.floor((points[:, 1] - self.origin[1]) / self.res).astype(np.intp)
        return np.clip(rows, 0, self.mask.shape[0] - 1), np.clip(cols, 0, self.mask.shape[1] - 1)

    def contains(self, points: NDArray[Any]) -> NDArray[np.bool_]:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        inside_box = (
            (points[:, 0] >= self.origin[0])
            & (points[:, 1] >= self.origin[1])
            & (points[:, 0] < self.origin[0] + self.mask.shape[0] * self.res)
            & (points[:, 1] < self.origin[1] + self.mask.shape[1] * self.res)
        )
        return np.asarray(inside_box & self.mask[self._index(points)], dtype=bool)

    def border_distance(self, points: NDArray[Any]) -> NDArray[np.float64]:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        distance = self.edge_distance[self._index(points)]
        return np.where(self.contains(points), distance, 0.0)

    def random_points(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """``n`` points uniformly distributed in the window."""
        cells = np.flatnonzero(self.mask)
        pick = cells[rng.integers(0, len(cells), n)]
        rows, cols = np.unravel_index(pick, self.mask.shape)
        jitter = rng.uniform(0, 1, (n, 2))
        return np.column_stack(
            [
                self.origin[0] + (rows + jitter[:, 0]) * self.res,
                self.origin[1] + (cols + jitter[:, 1]) * self.res,
            ]
        )


def nn_histogram(nn: Any, bins: int = 30) -> tuple[Any, Any]:
    """Histogram of nearest-neighbour distances. Distances that are all (almost) equal (a
    regular arrangement) get a 1 µm wide range instead of failing in numpy."""
    values = np.asarray(nn, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.zeros(0), np.zeros(0)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-9 * max(1.0, abs(hi)):
        return np.histogram(values, bins=bins, range=(lo - 0.5, hi + 0.5))
    return np.histogram(values, bins=bins)


def neighbour_counts(points: NDArray[Any], radii: NDArray[Any]) -> NDArray[np.float64]:
    """``counts[i, k]`` = number of other points within ``radii[k]`` of point i."""
    from scipy.spatial import cKDTree

    n = len(points)
    counts = np.zeros((n, len(radii) + 1))
    if n < 2:
        return counts[:, :-1]
    pairs = cKDTree(points).query_pairs(float(radii[-1]), output_type="ndarray")
    if len(pairs):
        distance = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        bins = np.searchsorted(radii, distance, side="left")
        np.add.at(counts, (pairs[:, 0], bins), 1)
        np.add.at(counts, (pairs[:, 1], bins), 1)
    return np.cumsum(counts, axis=1)[:, :-1]


def ripley(points: NDArray[Any], window: Window, radii: NDArray[Any]) -> dict[str, NDArray[Any]]:
    """Border-corrected K, L, H and pair correlation g at ``radii`` (ascending, > 0).

    g at ``radii[k]`` uses the ring ``(radii[k-1], radii[k]]`` (``(0, radii[0]]`` for k=0)
    and is reported at the ring's middle radius ``g_r``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float)
    n = len(points)
    nan = np.full(len(radii), np.nan)
    if n < 2:
        return {"K": nan, "L": nan, "H": nan, "g": nan, "g_r": radii.copy(), "n_inner": nan}
    lam = n / window.area_um2
    border = window.border_distance(points)
    counts = neighbour_counts(points, radii)
    inner_r = np.concatenate([[0.0], radii[:-1]])
    k_values, g_values, n_inner = nan.copy(), nan.copy(), np.zeros(len(radii))
    for k, r in enumerate(radii):
        inner = border >= r
        n_inner[k] = inner.sum()
        if not inner.any():
            continue
        k_values[k] = counts[inner, k].mean() / lam
        ring = counts[inner, k] - (counts[inner, k - 1] if k else 0)
        g_values[k] = ring.mean() / (lam * math.pi * (r**2 - inner_r[k] ** 2))
    l_values = np.sqrt(np.maximum(k_values, 0) / math.pi)
    return {
        "K": k_values,
        "L": l_values,
        "H": l_values - radii,
        "g": g_values,
        "g_r": (radii + inner_r) / 2,
        "n_inner": n_inner,
    }


def csr_envelope(
    n: int, window: Window, radii: NDArray[Any], simulations: int, rng: np.random.Generator
) -> dict[str, NDArray[Any]]:
    """Pointwise min/max of H and g over ``simulations`` uniform patterns of ``n`` points."""
    h_values, g_values = [], []
    for _ in range(simulations):
        stats = ripley(window.random_points(n, rng), window, radii)
        h_values.append(stats["H"])
        g_values.append(stats["g"])
    h, g = np.asarray(h_values), np.asarray(g_values)
    with np.errstate(all="ignore"):
        return {
            "H_low": np.nanmin(h, axis=0),
            "H_high": np.nanmax(h, axis=0),
            "g_low": np.nanmin(g, axis=0),
            "g_high": np.nanmax(g, axis=0),
        }


def nearest_neighbour(points: NDArray[Any]) -> NDArray[np.float64]:
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return np.full(len(points), np.nan)
    distance, _ = cKDTree(points).query(points, k=2)
    return np.asarray(distance[:, 1], dtype=float)


def clark_evans(points: NDArray[Any], window: Window) -> float:
    """Clark–Evans R with Donnelly's edge correction."""
    n = len(points)
    if n < 2:
        return math.nan
    area, perimeter = window.area_um2, window.perimeter_um
    expected = 0.5 * math.sqrt(area / n) + (0.0514 + 0.041 / math.sqrt(n)) * perimeter / n
    return float(np.mean(nearest_neighbour(points)) / expected)


def voronoi_delaunay(
    points: NDArray[Any], window: Window
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Voronoi cell area (µm²) and Delaunay neighbour count per point; NaN where the Voronoi
    cell is open or extends outside the window."""
    from scipy.spatial import Delaunay, QhullError, Voronoi

    n = len(points)
    areas, neighbours = np.full(n, np.nan), np.full(n, np.nan)
    if n < 4:
        return areas, neighbours
    try:
        voronoi = Voronoi(points)
        triangulation = Delaunay(points)
    except (QhullError, ValueError):
        return areas, neighbours
    indptr, _ = triangulation.vertex_neighbor_vertices
    vertex_inside = window.contains(voronoi.vertices)
    for i in range(n):
        region = voronoi.regions[voronoi.point_region[i]]
        if not region or -1 in region or not vertex_inside[region].all():
            continue
        polygon = voronoi.vertices[region]
        y, x = polygon[:, 0], polygon[:, 1]
        areas[i] = 0.5 * abs(float(np.dot(y, np.roll(x, 1)) - np.dot(x, np.roll(y, 1))))
        neighbours[i] = float(indptr[i + 1] - indptr[i])
    return areas, neighbours


def colonies(
    points: NDArray[Any],
    method: Literal["dbscan", "hdbscan"],
    eps_um: float,
    min_samples: int,
    min_cluster_size: int,
) -> NDArray[np.int64]:
    """Colony label per point, 1 = largest colony, -1 = not in a colony."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(points)
    else:
        if len(points) < max(2, min_cluster_size):
            return np.full(len(points), -1, dtype=np.int64)
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(points)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    mapping = {int(old): new for new, old in enumerate(order, start=1)}
    return np.array([mapping.get(int(v), -1) for v in labels], dtype=np.int64)


def hull_area(points: NDArray[Any]) -> float:
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # 2D "volume" is the area
    except (QhullError, ValueError):
        return 0.0


# ------------------------------------------------------------------ pipeline step


class SpatialParams(TableParams):
    scope: Literal["well", "field"] = Field(
        default="well",
        description="Whole well (needs the Stitch well step) or each field separately.",
    )
    method: Literal["dbscan", "hdbscan"] = Field(
        default="dbscan", description="Colony detection method."
    )
    eps_um: float = Field(
        default=20.0, gt=0, description="DBSCAN: objects closer than this join a colony."
    )
    min_samples: int = Field(default=5, ge=1, description="DBSCAN: neighbours to start a colony.")
    min_cluster_size: int = Field(default=10, ge=2, description="HDBSCAN: smallest colony.")
    radii_um: list[float] = Field(
        default=[10.0, 25.0, 50.0, 100.0], description="Radii for Ripley's H and g in the summary."
    )
    simulations: int = Field(
        default=39, ge=0, le=999, description="CSR simulations for the envelopes; 0 = none."
    )
    seed: int = Field(default=0, description="Random seed for the simulations.")


SPATIAL_COLUMNS = (
    "spatial_colony",
    "spatial_colony_size",
    "spatial_nn_um",
    "spatial_voronoi_area_um2",
    "spatial_local_density_per_mm2",
    "spatial_delaunay_neighbours",
)


@MODULES.register("spatial_clusters")
class SpatialClusters(TableModule):
    """Whole-well spatial organisation: colonies (DBSCAN/HDBSCAN), Ripley's K/L/H and pair
    correlation against CSR envelopes, nearest-neighbour distances, Clark–Evans ratio,
    Voronoi/Delaunay local density."""

    name = "spatial_clusters"
    title = "Spatial clusters"
    Params = SpatialParams

    def __init__(self, params: Any = None) -> None:
        super().__init__(params)
        self._groups: dict[str, list[dict[str, Any]]] = {}

    def _patterns(
        self, objects: pd.DataFrame, well: WellContext
    ) -> list[tuple[str, pd.Index, NDArray[np.float64], Window]]:
        """(group name, row index, points, window) per field or per timepoint of the well."""
        params: SpatialParams = self.params
        rows = objects[(objects["object_type"] == params.object_type) & unique_rows(objects)]
        out = []
        if params.scope == "well":
            if "well_x_um" not in objects or not well.field_offsets_um or not well.field_size_um:
                raise ModuleError(
                    "whole-well statistics need well coordinates: add the 'Stitch well' step "
                    "to the pipeline, or use scope 'field'"
                )
            h, w = well.field_size_um
            rects = [(y, x, y + h, x + w) for y, x in well.field_offsets_um.values()]
            window = Window.from_rects(rects)
            for t, group in rows.groupby("timepoint"):
                points = group[["well_y_um", "well_x_um"]].to_numpy(float)
                out.append((f"t{t}", group.index, points, window))
            return out
        for (field_no, timepoint), group in rows.groupby(["field", "timepoint"]):
            points = group[["centroid_y_um", "centroid_x_um"]].to_numpy(float)
            if well.field_size_um:
                rect = (0.0, 0.0, *well.field_size_um)
            else:  # the field size is unknown: use the objects' bounding box
                rect = (*points.min(axis=0), *(points.max(axis=0) + 1e-6))
            name = f"f{field_no}_t{timepoint}"
            out.append((name, group.index, points, Window.from_rects([rect])))
        return out

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        params: SpatialParams = self.params
        out = pd.DataFrame(index=objects.index)
        for column in SPATIAL_COLUMNS:
            integer = column in ("spatial_colony", "spatial_colony_size")
            out[column] = pd.Series(pd.NA if integer else np.nan, index=objects.index)
            if integer:
                out[column] = out[column].astype("Int64")
        radii = np.unique(np.asarray(params.radii_um, dtype=float))
        if not len(radii) or radii[0] <= 0:
            raise ModuleError("radii must be positive")
        grid = np.unique(np.concatenate([np.linspace(radii[-1] / 50, radii[-1], 50), radii]))
        rng = np.random.default_rng(params.seed)
        groups = []
        for name, index, points, window in self._patterns(objects, well):
            n = len(points)
            labels = colonies(
                points, params.method, params.eps_um, params.min_samples, params.min_cluster_size
            )
            sizes = pd.Series(labels).map(pd.Series(labels[labels > 0]).value_counts())
            nn = nearest_neighbour(points)
            areas, neighbours = voronoi_delaunay(points, window)
            out.loc[index, "spatial_colony"] = labels
            out.loc[index, "spatial_colony_size"] = sizes.fillna(0).astype(int).to_numpy()
            out.loc[index, "spatial_nn_um"] = nn
            out.loc[index, "spatial_voronoi_area_um2"] = areas
            with np.errstate(divide="ignore"):
                out.loc[index, "spatial_local_density_per_mm2"] = 1e6 / areas
            out.loc[index, "spatial_delaunay_neighbours"] = neighbours
            stats = ripley(points, window, grid)
            envelope = (
                csr_envelope(n, window, grid, params.simulations, rng)
                if params.simulations and n >= 2
                else None
            )
            colony_ids = sorted(set(labels[labels > 0].tolist()))
            colony_sizes = [int((labels == c).sum()) for c in colony_ids]
            groups.append(
                {
                    "group": name,
                    "n": n,
                    "area_um2": window.area_um2,
                    "labels": labels,
                    "points": points,
                    "colony_sizes": colony_sizes,
                    "colony_areas": [hull_area(points[labels == c]) for c in colony_ids],
                    "nn": nn,
                    "clark_evans": clark_evans(points, window),
                    "grid": grid,
                    "stats": stats,
                    "envelope": envelope,
                }
            )
        self._groups[well.well] = groups
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        params: SpatialParams = self.params
        groups = self._groups.pop(well.well, [])
        radii = np.unique(np.asarray(params.radii_um, dtype=float))
        rows: list[dict[str, float]] = []
        details = []
        for group in groups:
            n, sizes = group["n"], group["colony_sizes"]
            nn = group["nn"][np.isfinite(group["nn"])]
            row: dict[str, float] = {
                "spatial_objects": n,
                "spatial_density_per_mm2": n / group["area_um2"] * 1e6,
                "spatial_colonies": len(sizes),
                "spatial_colony_size_mean": float(np.mean(sizes)) if sizes else math.nan,
                "spatial_colony_size_median": float(np.median(sizes)) if sizes else math.nan,
                "spatial_colony_area_um2_mean": float(np.mean(group["colony_areas"]))
                if sizes
                else math.nan,
                "spatial_fraction_in_colonies": sum(sizes) / n if n else math.nan,
                "spatial_nn_mean_um": float(nn.mean()) if len(nn) else math.nan,
                "spatial_nn_median_um": float(np.median(nn)) if len(nn) else math.nan,
                "spatial_clark_evans_R": group["clark_evans"],
            }
            grid, stats, envelope = group["grid"], group["stats"], group["envelope"]
            for r in radii:
                k = int(np.searchsorted(grid, r))
                tag = f"{r:g}um"
                row[f"spatial_H_{tag}"] = float(stats["H"][k])
                row[f"spatial_g_{tag}"] = float(np.interp(r, stats["g_r"], stats["g"]))
                if envelope is not None:
                    low, high = float(envelope["H_low"][k]), float(envelope["H_high"][k])
                    row[f"spatial_H_{tag}_csr_low"] = low
                    row[f"spatial_H_{tag}_csr_high"] = high
                    h = row[f"spatial_H_{tag}"]
                    # +1 clustered, -1 regular, 0 consistent with CSR at this radius
                    pattern = 1.0 if h > high else (-1.0 if h < low else 0.0)
                    row[f"spatial_pattern_{tag}"] = pattern if math.isfinite(h) else math.nan
            rows.append(row)
            histogram, edges = nn_histogram(nn)
            details.append(
                {
                    "group": group["group"],
                    "n": n,
                    "area_um2": group["area_um2"],
                    "r_um": grid,
                    "K": stats["K"],
                    "L": stats["L"],
                    "H": stats["H"],
                    "g_r_um": stats["g_r"],
                    "g": stats["g"],
                    "envelope": envelope,
                    "colony_sizes": sizes,
                    "nn_histogram": {"counts": histogram, "edges_um": edges},
                }
            )
        if not rows:
            return {"spatial_objects": 0}, {}
        frame = pd.DataFrame(rows)
        summary = {c: float(frame[c].mean()) for c in frame.columns}
        summary["spatial_objects"] = int(frame["spatial_objects"].sum())
        if len(rows) > 1:  # several fields or timepoints: values are means over them
            summary["spatial_groups"] = len(rows)
        return summary, {"scope": params.scope, "groups": details}
