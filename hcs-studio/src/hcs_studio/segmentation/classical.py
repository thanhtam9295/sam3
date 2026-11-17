"""Classical segmentation: smoothing, global or local threshold, watershed for touching objects."""

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
from hcs_studio.segmentation.objects import overlap_parents, relabel_sequential


def threshold_plane(
    plane: NDArray[Any], method: str, block_px: int, offset: float
) -> NDArray[np.bool_]:
    from skimage import filters

    if np.ptp(plane) == 0:
        return np.zeros(plane.shape, dtype=bool)
    if method == "local":
        # Remove the slowly varying background, then threshold the remainder globally.
        # (Comparing each pixel with its local mean would mark half the noise as objects.)
        background = ndimage.gaussian_filter(plane, block_px / 3.0, mode="reflect")
        detrended = plane - background
        if np.ptp(detrended) == 0:
            return np.zeros(plane.shape, dtype=bool)
        return np.asarray(detrended > filters.threshold_otsu(detrended) + offset)
    function = {
        "otsu": filters.threshold_otsu,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
    }[method]
    return np.asarray(plane > function(plane) + offset)


def split_touching(mask: NDArray[np.bool_], min_distance_px: float) -> NDArray[np.int32]:
    """Watershed on the distance transform, one seed per local maximum ≥ min_distance apart."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    distance = ndimage.distance_transform_edt(mask)
    smooth = ndimage.gaussian_filter(distance, max(min_distance_px / 4, 0.5))
    regions, _ = ndimage.label(mask)
    peaks = peak_local_max(
        smooth,
        min_distance=max(1, round(min_distance_px)),
        labels=regions,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # Regions without a peak (very small) keep a marker of their own.
    missing = set(np.unique(regions[mask])) - set(np.unique(regions[markers > 0]))
    next_id = len(peaks) + 1
    for region in sorted(missing):
        if region == 0:
            continue
        index = np.argwhere(regions == region)[0]
        markers[tuple(index)] = next_id
        next_id += 1
    return np.asarray(watershed(-smooth, markers, mask=mask), dtype=np.int32)


class ClassicalParams(ModuleParams):
    channel: int = Field(ge=1, description="Channel to segment (e.g. the nuclear stain).")
    output: str = Field(default="nuclei", description="Name of the new label set.")
    method: Literal["otsu", "li", "triangle", "local"] = Field(
        default="otsu", description="Threshold method; local adapts to uneven background."
    )
    local_block_um: float = Field(
        default=50.0, gt=0, description="Neighbourhood for 'local', in µm."
    )
    offset: float = Field(default=0.0, description="Added to the threshold (intensity units).")
    smoothing_um: float = Field(
        default=1.0, ge=0, description="Gaussian smoothing before thresholding."
    )
    split_touching: bool = Field(
        default=True, description="Separate touching objects by watershed."
    )
    min_distance_um: float = Field(
        default=5.0,
        gt=0,
        description="Smallest distance between object centres (≈ nucleus radius).",
    )
    min_size_px: int = Field(default=20, ge=0, description="Remove objects smaller than this.")
    fill_holes: bool = Field(default=True, description="Fill holes inside objects.")
    parent: str | None = Field(
        default=None,
        description="Link each object to the object it overlaps most in this label set.",
    )


@MODULES.register("segment_classical")
class SegmentClassical(Module):
    name = "segment_classical"
    title = "Segmentation: threshold + watershed"
    Params = ClassicalParams

    def validate(self, metadata: ImageMetadata) -> None:
        params: ClassicalParams = self.params
        if params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {params.channel} does not exist")
        um_to_px(params.min_distance_um, metadata, "The minimum distance")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ClassicalParams = self.params
        self.validate(image.metadata)
        meta = image.metadata
        data = image.data[params.channel - 1]
        sigma = um_to_px(params.smoothing_um, meta, "The smoothing") if params.smoothing_um else 0
        block = max(3, round(um_to_px(params.local_block_um, meta, "The local block")))
        min_distance = um_to_px(params.min_distance_um, meta, "The minimum distance")
        out = np.zeros(data.shape, dtype=np.int32)
        next_id = 0
        for z in range(data.shape[0]):
            plane = ndimage.gaussian_filter(data[z], sigma) if sigma else data[z]
            mask = threshold_plane(plane, params.method, block, params.offset)
            if params.fill_holes:
                mask = ndimage.binary_fill_holes(mask)
            labelled = (
                split_touching(mask, min_distance)
                if params.split_touching
                else ndimage.label(mask)[0]
            )
            if params.min_size_px:
                sizes = np.bincount(labelled.ravel())
                small = np.nonzero(sizes < params.min_size_px)[0]
                labelled[np.isin(labelled, small[small > 0])] = 0
            labelled = relabel_sequential(labelled)
            labelled[labelled > 0] += next_id
            next_id = int(np.max(labelled)) if np.max(labelled) else next_id
            out[z] = labelled
        parent_ids = (
            overlap_parents(out, image.label_set(params.parent).data) if params.parent else None
        )
        result = LabelSet(
            params.output, out, params.parent, parent_ids, source=f"{params.method} threshold"
        )
        report = {"module": self.name, "output": params.output, "objects": result.count}
        return image.with_labels(result, self.name, report)
