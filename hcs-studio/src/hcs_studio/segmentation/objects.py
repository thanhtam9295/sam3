"""Object tools: parent links, cells from nuclei, spot detection, filters and label editing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy import ndimage

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)

Labels = NDArray[np.int32]


# ------------------------------------------------------------------ helpers


def relabel_sequential(labels: NDArray[Any]) -> Labels:
    """IDs 1..N in order of first appearance of each old ID (sorted), background 0."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int32)
    lookup[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return np.asarray(lookup[labels], dtype=np.int32)


def overlap_parents(child: NDArray[Any], parent: NDArray[Any]) -> NDArray[np.int32]:
    """For each child ID, the parent ID it overlaps most (0 when it overlaps none)."""
    n_child = int(child.max()) if child.size else 0
    result = np.zeros(n_child + 1, dtype=np.int32)
    inside = (child > 0) & (parent > 0)
    if not inside.any():
        return result
    pairs = np.stack([child[inside], parent[inside]], axis=1).astype(np.int64)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-counts, unique[:, 0]))  # by child, then largest overlap first
    unique = unique[order]
    first = np.ones(len(unique), dtype=bool)
    first[1:] = unique[1:, 0] != unique[:-1, 0]
    result[unique[first, 0]] = unique[first, 1]
    return result


def point_parents(child: NDArray[Any], parent: NDArray[Any]) -> NDArray[np.int32]:
    """For each child ID, the parent ID under the child's centroid (0 when outside)."""
    n_child = int(child.max()) if child.size else 0
    result = np.zeros(n_child + 1, dtype=np.int32)
    if n_child == 0:
        return result
    ids = np.arange(1, n_child + 1)
    centres: list[Any] = list(ndimage.center_of_mass(np.ones_like(child), child, ids))
    for i, centre in zip(ids, centres, strict=True):
        if np.isnan(centre[0]):
            continue
        index = tuple(
            min(max(round(c), 0), n - 1) for c, n in zip(centre, child.shape, strict=True)
        )
        result[i] = int(parent[index])
    return result


def touches_edge(labels: NDArray[Any]) -> set[int]:
    """IDs touching the Y/X border (Z borders are not edges for a Z stack)."""
    border = np.concatenate(
        [
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    return {int(v) for v in np.unique(border) if v > 0}


def remove_ids(labels: NDArray[Any], ids: set[int]) -> Labels:
    if not ids:
        return labels.astype(np.int32, copy=False)
    out = labels.astype(np.int32, copy=True)
    out[np.isin(out, list(ids))] = 0
    return out


def object_areas(labels: NDArray[Any]) -> NDArray[np.int64]:
    """Voxel count per ID (index = ID)."""
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1 if labels.size else 1)


def voxel_volume_um(metadata: ImageMetadata, three_d: bool) -> float | None:
    """µm² per pixel (2D) or µm³ per voxel (3D); None if uncalibrated."""
    if metadata.pixel_size_um is None:
        return None
    area = metadata.pixel_size_um**2
    if three_d:
        return area * metadata.z_step_um if metadata.z_step_um is not None else None
    return area


def link_children(image: ImageData, parent_name: str, removed: set[int]) -> tuple[LabelSet, ...]:
    """After removing parent objects, clear the links of their children."""
    updated = []
    for label_set in image.labels:
        if label_set.parent == parent_name and label_set.parent_ids is not None:
            ids = label_set.parent_ids.copy()
            ids[np.isin(ids, list(removed))] = 0
            label_set = LabelSet(
                label_set.name, label_set.data, label_set.parent, ids, label_set.source
            )
        updated.append(label_set)
    return tuple(updated)


# ----------------------------------------------------------- label editing


def merge_labels(labels: NDArray[Any], ids: list[int]) -> Labels:
    """Give all ``ids`` the smallest of them."""
    ids = sorted({int(i) for i in ids if i > 0})
    if len(ids) < 2:
        raise ModuleError("Choose at least two different objects to merge")
    out = labels.astype(np.int32, copy=True)
    out[np.isin(out, ids[1:])] = ids[0]
    return out


def split_label(labels: NDArray[Any], label_id: int, seeds: NDArray[Any]) -> Labels:
    """Split one object by watershed on its distance transform, from 2+ seed points (Z, Y, X)."""
    from skimage.segmentation import watershed

    mask = labels == label_id
    if not mask.any():
        raise ModuleError(f"Object {label_id} does not exist")
    points = np.asarray(seeds, dtype=float).reshape(-1, labels.ndim)
    markers = np.zeros(labels.shape, dtype=np.int32)
    count = 0
    for point in points:
        index = tuple(round(v) for v in point)
        if all(0 <= i < n for i, n in zip(index, labels.shape, strict=True)) and mask[index]:
            count += 1
            markers[index] = count
    if count < 2:
        raise ModuleError("Place at least two seed points inside the object to split")
    distance = ndimage.distance_transform_edt(mask)
    parts = watershed(-distance, markers, mask=mask)
    out = labels.astype(np.int32, copy=True)
    next_id = int(np.max(out)) + 1
    for part in range(2, count + 1):
        out[parts == part] = next_id
        next_id += 1
    return out


def delete_labels(labels: NDArray[Any], ids: list[int]) -> Labels:
    return remove_ids(labels, {int(i) for i in ids if i > 0})


# ------------------------------------------------------------ expand cells


class ExpandCellsParams(ModuleParams):
    nuclei: str = Field(default="nuclei", description="Label set to grow from.")
    output: str = Field(default="cells", description="Name of the new label set.")
    distance_um: float = Field(default=5.0, gt=0, description="Maximum growth in µm.")
    mask_channel: int | None = Field(
        default=None, ge=1, description="Limit growth to pixels where this channel is bright."
    )
    mask_threshold: float | None = Field(
        default=None, ge=0, description="Threshold for mask_channel; empty = Otsu."
    )


@MODULES.register("expand_cells")
class ExpandCells(Module):
    """Cells as nuclei grown outward without overlap; each cell keeps its nucleus' ID."""

    name = "expand_cells"
    title = "Cells from nuclei (expansion)"
    Params = ExpandCellsParams

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.distance_um, metadata, "The expansion distance")
        if self.params.mask_channel and self.params.mask_channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.mask_channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        from skimage.filters import threshold_otsu
        from skimage.segmentation import expand_labels

        params: ExpandCellsParams = self.params
        nuclei = image.label_set(params.nuclei)
        distance = um_to_px(params.distance_um, image.metadata, "The expansion distance")
        cells = np.zeros_like(nuclei.data)
        for z in range(nuclei.data.shape[0]):
            cells[z] = expand_labels(nuclei.data[z], distance=distance)
        if params.mask_channel:
            channel = image.data[params.mask_channel - 1]
            level = params.mask_threshold
            if level is None:
                level = (
                    float(threshold_otsu(channel)) if np.ptp(channel) > 0 else float(channel.max())
                )
            cells[(channel < level) & (nuclei.data == 0)] = 0
        ids = np.arange(int(np.max(cells)) + 1, dtype=np.int32)
        result = LabelSet(
            params.output, cells, parent=params.nuclei, parent_ids=ids, source="expansion"
        )
        report = {"module": self.name, "output": params.output, "objects": result.count}
        return image.with_labels(result, self.name, report)


# ------------------------------------------------------------------- spots


class SpotParams(ModuleParams):
    channel: int = Field(ge=1, description="Channel with the spots.")
    output: str = Field(default="spots", description="Name of the new label set.")
    parent: str | None = Field(
        default="cells", description="Label set the spots belong to; empty = none."
    )
    diameter_um: float = Field(default=0.6, gt=0, description="Typical spot diameter in µm.")
    method: Literal["log", "dog"] = Field(
        default="log", description="Laplacian or difference of Gaussians."
    )
    threshold_sd: float = Field(
        default=5.0,
        gt=0,
        description="Detection threshold in robust noise SDs of the filtered image.",
    )
    only_inside_parent: bool = Field(
        default=True, description="Drop spots outside any parent object."
    )


@MODULES.register("detect_spots")
class DetectSpots(Module):
    """Sub-cellular spots: band-pass filter, threshold at k noise SDs, one label per spot."""

    name = "detect_spots"
    title = "Spot detection (LoG / DoG)"
    Params = SpotParams

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.diameter_um, metadata, "The spot diameter")
        if self.params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {self.params.channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: SpotParams = self.params
        diameter = um_to_px(params.diameter_um, image.metadata, "The spot diameter")
        sigma = max(diameter / (2 * np.sqrt(2)), 0.5)
        data = image.data[params.channel - 1]
        spots = np.zeros(data.shape, dtype=np.int32)
        next_id = 1
        for z in range(data.shape[0]):
            plane = data[z]
            if params.method == "log":
                response = -ndimage.gaussian_laplace(plane, sigma) * sigma**2
            else:
                response = ndimage.gaussian_filter(plane, sigma) - ndimage.gaussian_filter(
                    plane, 1.6 * sigma
                )
            centre = np.median(response)
            noise = 1.4826 * np.median(np.abs(response - centre))
            if noise <= 0:
                continue
            mask = response > centre + params.threshold_sd * noise
            labelled, n = ndimage.label(mask)
            labelled[labelled > 0] += next_id - 1
            spots[z] = labelled
            next_id += n
        spots = relabel_sequential(spots)
        parent_ids = None
        if params.parent:
            parent = image.label_set(params.parent)
            parent_ids = point_parents(spots, parent.data)
            if params.only_inside_parent:
                outside = {int(i) for i in np.nonzero(parent_ids == 0)[0] if i > 0}
                spots = relabel_sequential(remove_ids(spots, outside))
                parent_ids = point_parents(spots, parent.data)
        result = LabelSet(
            params.output,
            spots,
            parent=params.parent if parent_ids is not None else None,
            parent_ids=parent_ids,
            source=f"{params.method} spots",
        )
        report = {"module": self.name, "output": params.output, "objects": result.count}
        return image.with_labels(result, self.name, report)


# ------------------------------------------------------------------ filters


class FilterParams(ModuleParams):
    labels: str = Field(default="nuclei", description="Label set to filter.")
    remove_edge: bool = Field(default=True, description="Remove objects touching the image border.")
    min_size: float | None = Field(default=None, ge=0, description="Minimum area (µm², µm³ in 3D).")
    max_size: float | None = Field(default=None, ge=0, description="Maximum area (µm², µm³ in 3D).")
    min_solidity: float | None = Field(
        default=None, ge=0, le=1, description="2D only: area / convex area."
    )
    intensity_channel: int | None = Field(
        default=None, ge=1, description="Channel for the intensity gate."
    )
    min_mean_intensity: float | None = Field(
        default=None, description="Keep objects at least this bright."
    )
    max_mean_intensity: float | None = Field(
        default=None, description="Keep objects at most this bright."
    )
    remove_children: bool = Field(
        default=True, description="Also remove objects of child label sets whose parent is removed."
    )


@MODULES.register("filter_objects")
class FilterObjects(Module):
    """Edge, size, shape and intensity gates. Removed IDs leave gaps; IDs are not renumbered."""

    name = "filter_objects"
    title = "Filter objects"
    Params = FilterParams

    def validate(self, metadata: ImageMetadata) -> None:
        params: FilterParams = self.params
        if (params.min_size or params.max_size) and metadata.pixel_size_um is None:
            raise ModuleError("Size gates are in µm; set the pixel size in the Metadata panel")
        if params.intensity_channel and params.intensity_channel > len(metadata.channels):
            raise ModuleError(f"Channel {params.intensity_channel} does not exist")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: FilterParams = self.params
        target = image.label_set(params.labels)
        data = target.data
        removed: dict[str, set[int]] = {
            "edge": set(),
            "size": set(),
            "shape": set(),
            "intensity": set(),
        }
        if params.remove_edge:
            removed["edge"] = touches_edge(data)
        three_d = data.shape[0] > 1
        unit = voxel_volume_um(image.metadata, three_d)
        if params.min_size is not None or params.max_size is not None:
            if unit is None:
                raise ModuleError("Size gates need the pixel size (and Z step in 3D)")
            areas = object_areas(data) * unit
            for i in range(1, len(areas)):
                if areas[i] == 0:
                    continue
                if (params.min_size is not None and areas[i] < params.min_size) or (
                    params.max_size is not None and areas[i] > params.max_size
                ):
                    removed["size"].add(i)
        if params.min_solidity is not None:
            if three_d:
                raise ModuleError("The solidity gate is only available for 2D images")
            from skimage.measure import regionprops

            for region in regionprops(data[0]):
                if region.solidity < params.min_solidity:
                    removed["shape"].add(int(region.label))
        if params.intensity_channel and (
            params.min_mean_intensity is not None or params.max_mean_intensity is not None
        ):
            if not image.linear:
                raise ModuleError(
                    "Intensity gates need linear data; put this filter before contrast, CLAHE, "
                    "sharpen or clarity steps"
                )
            channel = image.data[params.intensity_channel - 1]
            object_ids = [int(v) for v in np.arange(1, int(np.max(data)) + 1)]
            means = list(ndimage.mean(channel, data, object_ids)) if object_ids else []
            for i, mean in zip(object_ids, means, strict=True):
                if np.isnan(mean):
                    continue
                if (params.min_mean_intensity is not None and mean < params.min_mean_intensity) or (
                    params.max_mean_intensity is not None and mean > params.max_mean_intensity
                ):
                    removed["intensity"].add(int(i))
        gone = set().union(*removed.values())
        parent_ids = target.parent_ids
        kept = LabelSet(
            target.name, remove_ids(data, gone), target.parent, parent_ids, target.source
        )
        others = tuple(ls if ls.name != target.name else kept for ls in image.labels)
        if params.remove_children:
            cleaned = []
            for ls in others:
                if ls.parent == target.name and ls.parent_ids is not None and gone:
                    orphans = {
                        int(i) for i in np.nonzero(np.isin(ls.parent_ids, list(gone)))[0] if i > 0
                    }
                    links = np.asarray(ls.parent_ids, dtype=np.int32).copy()
                    links[list(orphans)] = 0
                    ls = LabelSet(
                        ls.name, remove_ids(ls.data, orphans), ls.parent, links, ls.source
                    )
                cleaned.append(ls)
            others = tuple(cleaned)
        report = {
            "module": self.name,
            "labels": target.name,
            "removed": {reason: len(ids) for reason, ids in removed.items()},
            "remaining": kept.count,
        }
        result = ImageData(
            data=image.data,
            metadata=image.metadata,
            linear=image.linear,
            history=(*image.history, self.name),
            reports=(*image.reports, report),
            labels=others,
        )
        return result
