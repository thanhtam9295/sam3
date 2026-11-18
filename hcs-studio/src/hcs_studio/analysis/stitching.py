"""Whole-well stitching from stage positions, refined by phase correlation.

Stage axes do not always run the same way as image axes (it differs between instruments),
so when fields overlap, the orientation whose overlaps agree best is chosen automatically.
Positions are then refined pair by pair and solved globally by least squares.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field as _Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    Module,
    ModuleError,
    ModuleParams,
)

Orientation = tuple[int, int]  # (sign of stage y → image row, sign of stage x → image column)


@dataclass
class StitchResult:
    #: Top-left corner of each field in the well, in µm (well coordinates: x right, y down).
    offsets_um: dict[int, tuple[float, float]]
    orientation: Orientation
    refined_pairs: int = 0
    notes: list[str] = field(default_factory=list)

    def field_rect(
        self, index: int, size_um: tuple[float, float]
    ) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of a field in µm; ``size_um`` is (height, width)."""
        y, x = self.offsets_um[index]
        return x, y, x + size_um[1], y + size_um[0]


def _nominal(
    positions: dict[int, tuple[float, float]], orientation: Orientation
) -> dict[int, NDArray[Any]]:
    """Stage (x, y) in µm → top-left (row, col) in µm with the given axis signs, min at 0."""
    sy, sx = orientation
    raw = {i: np.array([sy * y, sx * x], dtype=float) for i, (x, y) in positions.items()}
    origin = np.min(np.stack(list(raw.values())), axis=0)
    return {i: p - origin for i, p in raw.items()}


def _overlap(
    a: NDArray[Any], b: NDArray[Any], size: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]] | None:
    """Overlap of two fields as (start, stop) in µm, or None."""
    start = np.maximum(a, b)
    stop = np.minimum(a + size, b + size)
    if np.any(stop - start <= 0.05 * size):
        return None
    return start, stop


def _crop_at(
    image: NDArray[Any], origin: NDArray[Any], start: NDArray[Any], stop: NDArray[Any], scale: float
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Part of a field image covering [start, stop) µm, and where it starts in the field (µm).

    ``origin`` is the field's (assumed) position; ``scale`` µm per pixel.
    """
    r0, c0 = (max(0, int(v)) for v in np.floor((start - origin) / scale))
    r1, c1 = (max(0, int(v)) for v in np.ceil((stop - origin) / scale))
    return image[r0:r1, c0:c1], np.array([r0, c0], dtype=float) * scale


def _crop(
    image: NDArray[Any], origin: NDArray[Any], start: NDArray[Any], stop: NDArray[Any], scale: float
) -> NDArray[Any]:
    return _crop_at(image, origin, start, stop, scale)[0]


def _ncc(a: NDArray[Any], b: NDArray[Any]) -> float:
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    if h < 4 or w < 4:
        return float("nan")
    a, b = a[:h, :w].astype(float), b[:h, :w].astype(float)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def stitch(
    positions_um: dict[int, tuple[float, float]],
    size_um: tuple[float, float],
    images: dict[int, NDArray[Any]] | None = None,
    image_scale_um: float | None = None,
    *,
    orientation: Orientation | None = None,
    refine: bool = True,
    max_correction_fraction: float = 0.1,
    min_correlation: float = 0.3,
) -> StitchResult:
    """Place fields in a well.

    ``positions_um``: stage (x, y) per field index. ``size_um``: field (height, width).
    ``images``: one 2D image per field (e.g. a downsampled channel), ``image_scale_um`` µm
    per pixel, used to choose the orientation and refine the positions.
    """
    size = np.asarray(size_um, dtype=float)
    notes: list[str] = []
    if len(positions_um) == 1:
        (only,) = positions_um
        return StitchResult({only: (0.0, 0.0)}, orientation or (1, 1))
    candidates = [orientation] if orientation else [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    if images is None or image_scale_um is None:
        refine = False
        if orientation is None:
            notes.append("no images to check the stage orientation: assumed stage x right, y down")
            candidates = [(1, 1)]

    def pairs(
        nominal: dict[int, NDArray[Any]],
    ) -> list[tuple[int, int, NDArray[Any], NDArray[Any]]]:
        found = []
        for i, j in itertools.combinations(sorted(nominal), 2):
            region = _overlap(nominal[i], nominal[j], size)
            if region is not None:
                found.append((i, j, *region))
        return found

    best: tuple[float, Orientation] | None = None
    for candidate in candidates:
        nominal = _nominal(positions_um, candidate)
        overlapping = pairs(nominal)
        if not overlapping or images is None or image_scale_um is None:
            score = float("nan")
        else:
            scores = [
                _ncc(
                    _crop(images[i], nominal[i], start, stop, image_scale_um),
                    _crop(images[j], nominal[j], start, stop, image_scale_um),
                )
                for i, j, start, stop in overlapping
            ]
            finite = [s for s in scores if np.isfinite(s)]
            score = float(np.mean(finite)) if finite else float("nan")
        if best is None or (np.isfinite(score) and (not np.isfinite(best[0]) or score > best[0])):
            best = (score, candidate)
    assert best is not None
    chosen = best[1]
    if len(candidates) > 1 and not np.isfinite(best[0]):
        notes.append("fields do not overlap: stage orientation assumed (x right, y down)")
        chosen = (1, 1)
    nominal = _nominal(positions_um, chosen)
    result = StitchResult(
        {i: (float(p[0]), float(p[1])) for i, p in nominal.items()}, chosen, notes=notes
    )
    if not refine or images is None or image_scale_um is None:
        return result

    from skimage.registration import phase_cross_correlation

    order = sorted(nominal)
    index = {f: k for k, f in enumerate(order)}
    equations, targets = [], []
    limit = max_correction_fraction * float(size.max())
    for i, j, start, stop in pairs(nominal):
        a, offset_a = _crop_at(images[i], nominal[i], start, stop, image_scale_um)
        b, offset_b = _crop_at(images[j], nominal[j], start, stop, image_scale_um)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        if h < 8 or w < 8:
            continue
        a, b = a[:h, :w].astype(float), b[:h, :w].astype(float)
        if a.std() == 0 or b.std() == 0:  # nothing to register on
            continue
        window = np.outer(np.hanning(h), np.hanning(w))  # suppress the crop edges
        shift, _, _ = phase_cross_correlation(
            (a - a.mean()) * window, (b - b.mean()) * window, upsample_factor=10, normalization=None
        )
        # Crop a starts at offset_a inside field i, crop b at offset_b inside field j. If
        # shifting b by d registers it onto a, then position_j - position_i = offset_a -
        # offset_b + d (exact, whatever the rounding of the crop boundaries).
        measured = offset_a - offset_b + np.asarray(shift) * image_scale_um
        if np.abs(measured - (nominal[j] - nominal[i])).max() > limit:
            continue
        moved = np.roll(b, tuple(round(s) for s in shift), axis=(0, 1))
        margin = int(np.ceil(np.abs(shift).max())) + 1
        quality = _ncc(
            a[margin : -margin or None, margin : -margin or None],
            moved[margin : -margin or None, margin : -margin or None],
        )
        if not np.isfinite(quality) or quality < min_correlation:
            continue
        row = np.zeros(len(order))
        row[index[j]], row[index[i]] = 1, -1
        equations.append(row)
        targets.append(measured)
    if not equations:
        result.notes.append("no reliable overlaps: stage positions used without refinement")
        return result
    anchor = np.zeros(len(order))
    anchor[0] = 1
    matrix = np.vstack([*equations, anchor])
    solved = np.zeros((len(order), 2))
    connected = set()
    for row in equations:
        connected |= {k for k in np.nonzero(row)[0]}
    for axis in range(2):
        rhs = np.array([t[axis] for t in targets] + [nominal[order[0]][axis]])
        solved[:, axis] = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    for f in order:
        # Fields with no reliable overlap keep their stage position.
        point: NDArray[Any] = (
            solved[index[f]] if index[f] in connected or index[f] == 0 else nominal[f]
        )
        result.offsets_um[f] = (float(point[0]), float(point[1]))
    origin = np.min(np.array(list(result.offsets_um.values())), axis=0)
    result.offsets_um = {
        f: (p[0] - origin[0], p[1] - origin[1]) for f, p in result.offsets_um.items()
    }
    result.refined_pairs = len(equations)
    return result


def union_area(rects: list[tuple[float, float, float, float]]) -> float:
    """Exact area covered by axis-aligned rectangles (y0, x0, y1, x1), overlaps once."""
    if not rects:
        return 0.0
    ys = sorted({v for r in rects for v in (r[0], r[2])})
    xs = sorted({v for r in rects for v in (r[1], r[3])})
    covered = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for y0, x0, y1, x1 in rects:
        covered[ys.index(y0) : ys.index(y1), xs.index(x0) : xs.index(x1)] = True
    heights, widths = np.diff(ys), np.diff(xs)
    return float((covered * np.outer(heights, widths)).sum())


def owner_fields(
    points_um: NDArray[Any], result: StitchResult, size_um: tuple[float, float]
) -> NDArray[np.int64]:
    """For points (y, x) in well µm, the field that owns each: the nearest field centre among
    the fields containing it (-1 when no field contains it)."""
    fields = sorted(result.offsets_um)
    centres = np.array(
        [
            [result.offsets_um[f][0] + size_um[0] / 2, result.offsets_um[f][1] + size_um[1] / 2]
            for f in fields
        ]
    )
    corners = np.array([result.offsets_um[f] for f in fields])
    points = np.asarray(points_um, dtype=float).reshape(-1, 2)
    inside = (
        (points[:, None, 0] >= corners[None, :, 0])
        & (points[:, None, 0] < corners[None, :, 0] + size_um[0])
        & (points[:, None, 1] >= corners[None, :, 1])
        & (points[:, None, 1] < corners[None, :, 1] + size_um[1])
    )
    distance = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=-1)
    distance[~inside] = np.inf
    owner = np.asarray(fields)[np.argmin(distance, axis=1)]
    owner[~inside.any(axis=1)] = -1
    return np.asarray(owner, dtype=np.int64)


def mosaic(
    tiles: dict[int, NDArray[np.uint8]],
    result: StitchResult,
    tile_scale_um: float,
    max_size: int = 4096,
) -> tuple[NDArray[np.uint8], float]:
    """RGB mosaic of preview tiles placed at the stitched positions (brightest wins in
    overlaps), and its µm per pixel."""
    from scipy import ndimage

    extent = np.zeros(2)
    for f, tile in tiles.items():
        y, x = result.offsets_um[f]
        extent = np.maximum(
            extent, [y / tile_scale_um + tile.shape[0], x / tile_scale_um + tile.shape[1]]
        )
    factor = min(1.0, max_size / float(np.max(extent)))
    shape = (int(np.ceil(extent[0] * factor)) + 1, int(np.ceil(extent[1] * factor)) + 1, 3)
    canvas = np.zeros(shape, dtype=np.uint8)
    for f, tile in tiles.items():
        y, x = result.offsets_um[f]
        small = (
            tile
            if factor == 1
            else np.stack(
                [ndimage.zoom(tile[..., k], factor, order=1) for k in range(3)], axis=-1
            ).astype(np.uint8)
        )
        r0, c0 = round(y / tile_scale_um * factor), round(x / tile_scale_um * factor)
        region = canvas[r0 : r0 + small.shape[0], c0 : c0 + small.shape[1]]
        np.maximum(region, small[: region.shape[0], : region.shape[1]], out=region)
    return canvas, tile_scale_um / factor


def float_mosaic(
    tiles: dict[int, NDArray[np.float32]],
    result: StitchResult,
    tile_scale_um: float,
) -> NDArray[np.float32]:
    """(C, H, W) mosaic of linear (C, Y, X) tiles at the stitched positions (the top-left
    field at 0, 0); overlaps are averaged, so intensities stay comparable with the single
    fields."""
    y_min = min(result.offsets_um[f][0] for f in tiles)
    x_min = min(result.offsets_um[f][1] for f in tiles)
    offsets = {f: (result.offsets_um[f][0] - y_min, result.offsets_um[f][1] - x_min) for f in tiles}
    extent = np.zeros(2)
    for f, tile in tiles.items():
        y, x = offsets[f]
        extent = np.maximum(
            extent, [y / tile_scale_um + tile.shape[1], x / tile_scale_um + tile.shape[2]]
        )
    n_c = next(iter(tiles.values())).shape[0]
    shape = (int(np.ceil(extent[0])) + 1, int(np.ceil(extent[1])) + 1)
    total = np.zeros((n_c, *shape), dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    for f, tile in tiles.items():
        y, x = offsets[f]
        r0, c0 = round(y / tile_scale_um), round(x / tile_scale_um)
        h = min(tile.shape[1], shape[0] - r0)
        w = min(tile.shape[2], shape[1] - c0)
        total[:, r0 : r0 + h, c0 : c0 + w] += tile[:, :h, :w]
        count[r0 : r0 + h, c0 : c0 + w] += 1
    return np.asarray(total / np.maximum(count, 1), dtype=np.float32)


# ------------------------------------------------------------------ pipeline step

ORIENTATIONS: dict[str, Orientation | None] = {
    "auto": None,
    "x right, y down": (1, 1),
    "x right, y up": (-1, 1),
    "x left, y down": (1, -1),
    "x left, y up": (-1, -1),
}


class StitchParams(ModuleParams):
    channel: int = _Field(default=1, ge=1, description="Channel used to register overlaps.")
    refine: bool = _Field(
        default=True, description="Refine stage positions by phase correlation of the overlaps."
    )
    stage_orientation: str = _Field(
        default="auto",
        description="How stage axes map to the image: auto (from the overlaps), or "
        + ", ".join(k for k in ORIENTATIONS if k != "auto"),
    )
    max_correction_fraction: float = _Field(
        default=0.1, gt=0, le=0.5, description="Largest accepted correction, × the field size."
    )
    min_correlation: float = _Field(
        default=0.3, ge=0, le=1, description="Overlaps that correlate less are not used."
    )
    image_max_size: int = _Field(
        default=1024, ge=128, le=4096, description="Fields are downsampled to this for matching."
    )
    save_mosaic: bool = _Field(
        default=False,
        description="Also write the well as one linear image (mosaic.ome.tif, all channels; "
        "overlaps averaged), e.g. of projected and background-corrected fields.",
    )
    mosaic_downsample: int = _Field(
        default=2,
        ge=1,
        le=16,
        description="Mosaic image: block-average by this factor (1 = full resolution; needs "
        "all fields of a well in memory).",
    )


@MODULES.register("stitch_well")
class StitchWell(Module):
    """Places the fields of each well in one coordinate system (whole-well positions).

    Objects get ``well_x_um`` / ``well_y_um`` and ``in_overlap_duplicate`` (True when an
    overlapping field's centre is nearer, so the object is counted once); each well gets
    ``previews/mosaic.png``. The images themselves are not changed.
    """

    name = "stitch_well"
    title = "Stitch well"
    Params = StitchParams

    def validate(self, metadata: ImageMetadata) -> None:
        params: StitchParams = self.params
        if params.stage_orientation not in ORIENTATIONS:
            raise ModuleError(
                f"Unknown stage orientation '{params.stage_orientation}'; use one of "
                + ", ".join(ORIENTATIONS)
            )
        if params.channel > len(metadata.channels):
            raise ModuleError(f"Channel {params.channel} does not exist")
        if metadata.pixel_size_um is None:
            raise ModuleError("Stitching needs the pixel size; set it in the Metadata panel")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return image

    def orientation(self) -> Orientation | None:
        return ORIENTATIONS[self.params.stage_orientation]
