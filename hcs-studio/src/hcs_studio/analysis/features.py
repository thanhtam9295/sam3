"""Per-object measurements in physical units: morphology, intensity, texture, neighbours.

Intensities are always measured on linear data (see ``MeasureContext.corrected``).
Column names: ``<feature>`` for morphology, ``<channel>_<feature>`` for intensity and
texture, ``<child label set>_<feature>`` for counts of contained objects (e.g. spots).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import LabelSet

HARALICK_NAMES = (
    "asm",
    "contrast",
    "correlation",
    "variance",
    "idm",
    "sum_avg",
    "sum_var",
    "sum_entropy",
    "entropy",
    "diff_var",
    "diff_entropy",
    "imc1",
    "imc2",
)


def column_name(text: str) -> str:
    """Channel name → safe column prefix ('Alexa 488' → 'alexa_488')."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "channel"


def channel_prefixes(metadata: ImageMetadata) -> list[str]:
    names = [column_name(c.name) for c in metadata.channels]
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return unique


@dataclass(frozen=True)
class FeatureOptions:
    channels: list[int] | None = None  # zero-based; None = all
    percentiles: tuple[float, ...] = (5, 25, 75, 95)
    radial_bins: int = 3
    texture: bool = True
    texture_scales_px: tuple[int, ...] = (1, 3)
    granularity: bool = True
    granularity_max_radius_px: int = 6
    zernike: bool = False
    zernike_degree: int = 8
    neighbor_radius_um: float = 20.0
    nucleus_labels: str | None = "nuclei"


def _spacing(metadata: ImageMetadata, three_d: bool) -> tuple[float, ...]:
    pixel = metadata.pixel_size_um or 1.0
    if three_d:
        return (metadata.z_step_um or pixel, pixel, pixel)
    return (pixel, pixel)


def morphology(
    labels: NDArray[Any], metadata: ImageMetadata, options: FeatureOptions
) -> pd.DataFrame:
    """Shape features of each object. 2D when the stack has one plane."""
    from skimage.measure import regionprops

    three_d = labels.shape[0] > 1
    data = labels if three_d else labels[0]
    spacing = _spacing(metadata, three_d)
    unit = float(np.prod(spacing))
    rows = []
    for region in regionprops(data, spacing=spacing):
        row: dict[str, Any] = {"object_id": int(region.label)}
        centre = region.centroid  # physical units with spacing
        if three_d:
            row.update(centroid_z_um=centre[0], centroid_y_um=centre[1], centroid_x_um=centre[2])
            row["volume_um3"] = region.num_pixels * unit
            row["surface_area_um2"] = _surface_area(region.image, spacing)
            row["equivalent_diameter_um"] = (6 * row["volume_um3"] / np.pi) ** (1 / 3)
        else:
            row.update(centroid_y_um=centre[0], centroid_x_um=centre[1])
            area = region.num_pixels * unit
            perimeter = float(region.perimeter)
            row.update(
                area_um2=area,
                perimeter_um=perimeter,
                equivalent_diameter_um=2 * np.sqrt(area / np.pi),
                eccentricity=float(region.eccentricity),
                orientation_deg=float(np.degrees(region.orientation)),
                form_factor=4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan,
            )
        row.update(
            solidity=float(region.solidity),
            extent=float(region.extent),
            major_axis_um=float(region.axis_major_length),
            minor_axis_um=float(region.axis_minor_length),
        )
        if options.zernike and not three_d:
            row.update(_zernike(region.image, options.zernike_degree))
        rows.append(row)
    return pd.DataFrame(rows)


def _surface_area(mask: NDArray[Any], spacing: tuple[float, ...]) -> float:
    from skimage.measure import marching_cubes, mesh_surface_area

    padded = np.pad(mask.astype(np.float32), 2)
    if np.min(padded) == np.max(padded):
        return float("nan")
    # Light smoothing (in µm, equal on all axes) removes most of the voxel staircase,
    # which otherwise inflates the area by 10-15 %.
    sigma = [min(spacing) / s * 0.7 for s in spacing]
    smooth = ndimage.gaussian_filter(padded, sigma)
    verts, faces, _, _ = marching_cubes(smooth, 0.5, spacing=spacing)
    return float(mesh_surface_area(verts, faces))


def _zernike(mask: NDArray[Any], degree: int) -> dict[str, float]:
    import mahotas

    radius = max(mask.shape) / 2
    values = mahotas.features.zernike_moments(mask.astype(np.uint8), radius, degree=degree)
    return {f"zernike_{i:02d}": float(v) for i, v in enumerate(values)}


def intensity(
    labels: NDArray[Any],
    image: NDArray[Any],
    prefixes: list[str],
    options: FeatureOptions,
) -> pd.DataFrame:
    """Intensity statistics per object and channel; image is (C, Z, Y, X) linear data."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    table = pd.DataFrame({"object_id": ids.astype(int)})
    if ids.size == 0:
        return table
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    starts = np.searchsorted(sorted_ids, ids, side="left")
    ends = np.searchsorted(sorted_ids, ids, side="right")
    depth = _normalised_depth(labels) if options.radial_bins > 1 else None
    channels = options.channels if options.channels is not None else list(range(image.shape[0]))
    for c in channels:
        prefix = prefixes[c]
        values = image[c].ravel()[order]
        columns: dict[str, list[float]] = {
            key: [] for key in ("mean", "median", "integrated", "min", "max", "sd")
        }
        for p in options.percentiles:
            columns[f"p{int(p):02d}"] = []
        for start, end in zip(starts, ends, strict=True):
            v = values[start:end]
            columns["mean"].append(float(v.mean()))
            columns["median"].append(float(np.median(v)))
            columns["integrated"].append(float(v.sum()))
            columns["min"].append(float(v.min()))
            columns["max"].append(float(v.max()))
            columns["sd"].append(float(v.std()))
            for p, q in zip(
                options.percentiles, np.percentile(v, options.percentiles), strict=True
            ):
                columns[f"p{int(p):02d}"].append(float(q))
        for key, column in columns.items():
            table[f"{prefix}_{key}"] = column
        if depth is not None:
            table = table.join(
                _radial(labels, image[c], depth, ids, prefix, options.radial_bins), on="object_id"
            )
    return table


def _normalised_depth(labels: NDArray[Any]) -> NDArray[np.float64]:
    """0 at an object's edge, 1 at its deepest point (per plane, so 2D in Z stacks)."""
    depth = np.zeros(labels.shape, dtype=np.float64)
    for z in range(labels.shape[0]):
        plane = labels[z]
        if not plane.any():
            continue
        # Distance to the nearest pixel of a different label (background or neighbour).
        edges = np.zeros(plane.shape, dtype=bool)
        edges[:-1] |= plane[:-1] != plane[1:]
        edges[1:] |= plane[:-1] != plane[1:]
        edges[:, :-1] |= plane[:, :-1] != plane[:, 1:]
        edges[:, 1:] |= plane[:, :-1] != plane[:, 1:]
        distance = ndimage.distance_transform_edt(~edges | (plane == 0)) * (plane > 0)
        distance = np.asarray(distance, dtype=np.float64) + (plane > 0)
        maxima = ndimage.maximum(distance, plane, np.arange(int(plane.max()) + 1))
        maxima = np.asarray(maxima, dtype=np.float64)
        maxima[maxima == 0] = 1
        depth[z] = distance / maxima[plane]
    return depth


def _radial(
    labels: NDArray[Any],
    channel: NDArray[Any],
    depth: NDArray[Any],
    ids: NDArray[Any],
    prefix: str,
    bins: int,
) -> pd.DataFrame:
    """Fraction of each object's intensity in rings from the edge (bin 0) to the centre."""
    ring = np.minimum((depth * bins).astype(int), bins - 1)
    index = labels.astype(np.int64) * bins + ring
    sums = np.bincount(
        index.ravel(), weights=channel.ravel(), minlength=(int(labels.max()) + 1) * bins
    )
    sums = sums.reshape(-1, bins)[ids]
    totals = sums.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = sums / totals
    names = ["outer", "middle", "inner"] if bins == 3 else [f"ring{i}" for i in range(bins)]
    frame = pd.DataFrame(fractions, columns=[f"{prefix}_radial_{n}" for n in names])
    frame[f"{prefix}_center_periphery_ratio"] = fractions[:, -1] / np.where(
        fractions[:, 0] > 0, fractions[:, 0], np.nan
    )
    frame.index = ids.astype(int)
    return frame


def texture(
    labels: NDArray[Any], image: NDArray[Any], prefixes: list[str], options: FeatureOptions
) -> pd.DataFrame:
    """Haralick features (mean over directions) per scale, and granularity spectrum."""
    import mahotas
    from skimage.measure import regionprops

    data = labels if labels.shape[0] > 1 else labels[0]
    channels = options.channels if options.channels is not None else list(range(image.shape[0]))
    rows = []
    for region in regionprops(data):
        row: dict[str, Any] = {"object_id": int(region.label)}
        mask = region.image
        for c in channels:
            crop = image[c] if labels.shape[0] > 1 else image[c, 0]
            values = crop[region.slice]
            inside = values[mask]
            low, high = float(inside.min()), float(inside.max())
            if options.texture:
                # 1..64 grey levels inside the object; 0 marks outside (ignored).
                levels = np.zeros(values.shape, dtype=np.uint8)
                if high > low:
                    levels[mask] = 1 + np.round((values[mask] - low) / (high - low) * 63).astype(
                        np.uint8
                    )
                else:
                    levels[mask] = 1
                for scale in options.texture_scales_px:
                    try:
                        features = mahotas.features.haralick(
                            levels, distance=scale, ignore_zeros=True, return_mean=True
                        )
                    except ValueError:  # object too small for this distance
                        features = np.full(13, np.nan)
                    for name, value in zip(HARALICK_NAMES, features, strict=True):
                        row[f"{prefixes[c]}_haralick_{name}_s{scale}"] = float(value)
            if options.granularity:
                row.update(
                    _granularity(values, mask, prefixes[c], options.granularity_max_radius_px)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _granularity(
    values: NDArray[Any], mask: NDArray[Any], prefix: str, max_radius: int
) -> dict[str, float]:
    """% of the object's signal removed by grey opening at each radius (CellProfiler-style)."""
    from skimage.morphology import ball, disk

    work = np.where(mask, values - values[mask].min(), 0).astype(np.float64)
    total = work[mask].sum()
    out = {}
    previous = total
    for radius in range(1, max_radius + 1):
        footprint = disk(radius) if work.ndim == 2 else ball(radius)
        opened = ndimage.grey_opening(work, footprint=footprint)
        current = opened[mask].sum()
        out[f"{prefix}_granularity_r{radius}"] = (
            float(100 * (previous - current) / total) if total > 0 else np.nan
        )
        previous = current
    return out


def children(
    parent: LabelSet, child: LabelSet, child_table: pd.DataFrame, prefixes: list[str]
) -> pd.DataFrame:
    """Count, area and intensity of child objects (e.g. spots) per parent object."""
    if child.parent_ids is None:
        return pd.DataFrame({"object_id": []})
    links = pd.DataFrame(
        {"object_id": np.arange(len(child.parent_ids)), "parent": child.parent_ids}
    )
    links = links[(links.object_id > 0) & (links.parent > 0)]
    merged = links.merge(child_table, on="object_id", how="inner")
    name = child.name
    grouped = merged.groupby("parent")
    result = pd.DataFrame({f"{name}_count": grouped.size()})
    size_column = "area_um2" if "area_um2" in merged else "volume_um3"
    if size_column in merged:
        result[f"{name}_mean_{size_column}"] = grouped[size_column].mean()
    for prefix in prefixes:
        column = f"{prefix}_integrated"
        if column in merged:
            result[f"{name}_{prefix}_total_intensity"] = grouped[column].sum()
            result[f"{name}_{prefix}_mean_intensity"] = grouped[f"{prefix}_mean"].mean()
    result.index.name = "object_id"
    return result.reset_index()


def nuclear_cytoplasmic(
    cells: LabelSet, nuclei: LabelSet, image: NDArray[Any], prefixes: list[str], channels: list[int]
) -> pd.DataFrame:
    """Mean intensity in the nucleus / mean in the cytoplasm (cell minus nucleus)."""
    if cells.parent != nuclei.name or cells.parent_ids is None:
        return pd.DataFrame({"object_id": []})
    nucleus_of_cell = cells.parent_ids[cells.data]  # per pixel: the cell's nucleus id
    in_nucleus = (nuclei.data > 0) & (nuclei.data == nucleus_of_cell) & (cells.data > 0)
    in_cytoplasm = (cells.data > 0) & ~in_nucleus
    ids = np.arange(int(np.max(cells.data)) + 1)
    result = pd.DataFrame({"object_id": ids[1:]})
    for c in channels:
        nucleus_mean = ndimage.mean(image[c], cells.data * in_nucleus, ids)
        cytoplasm_mean = ndimage.mean(image[c], cells.data * in_cytoplasm, ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.asarray(nucleus_mean, dtype=float) / np.asarray(cytoplasm_mean, dtype=float)
        result[f"{prefixes[c]}_nc_ratio"] = ratio[1:]
        result[f"{prefixes[c]}_cytoplasm_mean"] = np.asarray(cytoplasm_mean, dtype=float)[1:]
    present = np.unique(cells.data)
    return result[result.object_id.isin(present[present > 0])]


def neighbours(table: pd.DataFrame, radius_um: float) -> pd.DataFrame:
    """Neighbours within ``radius_um`` (centroid distance) and nearest-neighbour distance."""
    from scipy.spatial import cKDTree

    columns = [c for c in ("centroid_z_um", "centroid_y_um", "centroid_x_um") if c in table]
    points = table[columns].to_numpy(dtype=float)
    result = pd.DataFrame({"object_id": table.object_id.to_numpy()})
    if len(points) < 2:
        result["neighbours_within_radius"] = 0
        result["nearest_neighbour_um"] = np.nan
        return result
    tree = cKDTree(points)
    distance, _ = tree.query(points, k=2)
    result["nearest_neighbour_um"] = distance[:, 1]
    result["neighbours_within_radius"] = [
        len(n) - 1 for n in tree.query_ball_point(points, radius_um)
    ]
    return result


def measure_label_set(
    label_set: LabelSet,
    image: NDArray[Any],
    metadata: ImageMetadata,
    options: FeatureOptions,
    all_sets: tuple[LabelSet, ...],
    child_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Every feature for one label set, one row per object."""
    prefixes = channel_prefixes(metadata)
    table = morphology(label_set.data, metadata, options)
    if table.empty:
        return table
    table = table.merge(
        intensity(label_set.data, image, prefixes, options), on="object_id", how="left"
    )
    if options.texture or options.granularity:
        table = table.merge(
            texture(label_set.data, image, prefixes, options), on="object_id", how="left"
        )
    if label_set.parent_ids is not None:
        parent_ids = label_set.parent_ids
        table.insert(
            1,
            "parent_id",
            [int(parent_ids[i]) if i < len(parent_ids) else 0 for i in table.object_id],
        )
    channels = options.channels if options.channels is not None else list(range(image.shape[0]))
    for other in all_sets:
        if other.parent == label_set.name and other.name in child_tables:
            table = table.merge(
                children(label_set, other, child_tables[other.name], prefixes),
                on="object_id",
                how="left",
            )
            count = f"{other.name}_count"
            if count in table:
                table[count] = table[count].fillna(0).astype(int)
    if options.nucleus_labels:
        nuclei = next((s for s in all_sets if s.name == options.nucleus_labels), None)
        if nuclei is not None and label_set.parent == nuclei.name:
            table = table.merge(
                nuclear_cytoplasmic(label_set, nuclei, image, prefixes, channels),
                on="object_id",
                how="left",
            )
    return table.merge(neighbours(table, options.neighbor_radius_um), on="object_id", how="left")
