"""Field-level image QC metrics (Phase 18a), computed on raw data, one field at a time.

Per channel (columns ``c<n>_<metric>``, channels from 1):

- ``plls``: power log-log slope (Bray et al. 2012), as CellProfiler's MeasureImageQuality
  computes it (``centrosome.radial_power_spectrum.rps``, read 2026-11-21): the image divided by
  its median absolute deviation from the mean, the FFT power summed in rings of integer
  radius 2 … min(H, W)/8, and the least-squares slope of log power against log radius.
  Blur removes high frequencies, so the slope becomes more negative.
- ``normalized_variance``: variance / mean (CellProfiler's FocusScore).
- ``laplacian_variance``: variance of the Laplacian of the (σ = 1 px) smoothed image divided
  by the image variance; scale-free, so bright and dim fields compare.
- ``saturated_fraction``: fraction of pixels at the detector maximum (``saturation_level``).
- ``background``, ``p50``, ``p99``, ``p999``: background (median of non-object pixels) and
  intensity percentiles.
- ``gradient``: (max − min) / mean of a quadratic surface fitted to the background.
- ``best_z``: plane (from 1) with the highest Laplacian focus when the field has a Z stack.

On the content channel only (plain column names): ``object_count``, ``area_fraction``,
``largest_gap_fraction`` (scratches, cell-free bands), ``debris_count`` /
``debris_area_fraction``, ``fiber_count``, ``bubble_fraction`` (round dark or bright regions
of the background inside the field) and ``out_of_well_fraction`` (dark regions at the field
border: well edge, meniscus shadow).

Thresholds are not decided here: :mod:`hcs_studio.qc.rules` compares metrics with absolute
limits and with the plate's own distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

#: Detector maxima recognised when the metadata's bit depth is only the container's (16 bit).
DETECTOR_MAXIMA = (255, 1023, 4095, 16383, 65535)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MetricSettings:
    content_channel: int = 1  # from 1
    nucleus_diameter_um: float = 10.0
    pixel_size_um: float | None = None
    bit_depth: int | None = None
    brightfield_channel: int | None = None  # from 1

    @property
    def diameter_px(self) -> float:
        pixel = self.pixel_size_um or 0.65
        return max(3.0, self.nucleus_diameter_um / pixel)


# ------------------------------------------------------------------ focus


def plls(plane: NDArray[Any]) -> float:
    """Power log-log slope (CellProfiler / centrosome definition, see module docstring)."""
    img = np.asarray(plane, dtype=np.float64)
    if img.ndim != 2 or np.ptp(img) == 0:
        return float("nan")
    h, w = img.shape
    yy = np.arange(h)[:, None] ** 2 + np.zeros((1, w), dtype=np.int64)
    radii2 = yy + (np.arange(w)[None, :] ** 2)
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    maxwidth = min(h, w) / 8.0
    scale = np.median(np.abs(img - img.mean()))
    if scale > 0:
        img = img / scale
    power = np.abs(np.fft.fft2(img - img.mean())) ** 2
    radii = np.floor(np.sqrt(radii2)).astype(int) + 1
    labels = np.arange(2, int(np.floor(maxwidth)))
    if labels.size < 2:
        return float("nan")
    power_sum = np.asarray(ndimage.sum(power, radii, labels), dtype=np.float64)
    good = power_sum > 0
    if good.sum() < 2:
        return float("nan")
    x = np.log(labels[good].astype(np.float64))
    y = np.log(power_sum[good])
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def normalized_variance(plane: NDArray[Any]) -> float:
    img = np.asarray(plane, dtype=np.float64)
    mean = float(img.mean())
    return float(img.var() / mean) if mean > 0 else float("nan")


def laplacian_variance(plane: NDArray[Any]) -> float:
    img = np.asarray(plane, dtype=np.float64)
    var = float(img.var())
    if var <= 0:
        return float("nan")
    return float(ndimage.laplace(ndimage.gaussian_filter(img, 1.0)).var() / var)


# ------------------------------------------------------------- intensity


def saturation_level(data: NDArray[Any], bit_depth: int | None) -> float:
    """The detector maximum: 2^bits − 1 from metadata; when the metadata only gives the
    container (16 bit) and the field's maximum is a common camera maximum (e.g. 4095 for a
    12-bit camera) that value is used."""
    peak = float(np.max(data))
    if bit_depth and bit_depth < 16:
        return float(2**bit_depth - 1)
    integer = np.issubdtype(np.asarray(data).dtype, np.integer) or peak.is_integer()
    if integer and peak in DETECTOR_MAXIMA:
        return peak
    return float(2 ** (bit_depth or 16) - 1)


def _fit_surface(values: FloatArray, mask: NDArray[np.bool_], degree: int = 2) -> FloatArray | None:
    """Least-squares polynomial surface through ``values`` where ``mask`` holds."""
    rows, cols = values.shape
    y, x = np.mgrid[:rows, :cols].astype(np.float64)
    y = (y + 0.5) / rows * 2 - 1
    x = (x + 0.5) / cols * 2 - 1
    terms = [x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    design = np.stack([t.ravel() for t in terms], axis=1)
    good = mask.ravel() & np.isfinite(values.ravel())
    if good.sum() < 2 * len(terms):
        return None
    coefficients, *_ = np.linalg.lstsq(design[good], values.ravel()[good], rcond=None)
    return np.asarray((design @ coefficients).reshape(rows, cols), dtype=np.float64)


def robust_surface(
    values: FloatArray, keep: NDArray[np.bool_] | None = None, degree: int = 2
) -> tuple[FloatArray | None, NDArray[np.bool_]]:
    """Quadratic (``degree``) surface refitted without regions that deviate from it (bubbles,
    out-of-well regions, debris), so it follows the illumination only."""
    keep = np.ones(values.shape, dtype=bool) if keep is None else keep.copy()
    surface = _fit_surface(values, keep, degree)
    for _ in range(5):
        if surface is None:
            break
        residual = values / np.maximum(surface, 1e-6) - 1
        spread = max(_robust_sd(residual[keep]), 0.01)
        new = np.abs(residual - np.median(residual[keep])) < max(0.08, 3 * spread)
        if new.mean() < 0.4 or np.array_equal(new, keep):
            break
        keep = new
        surface = _fit_surface(values, keep, degree)
    return surface, keep


def _bin(plane: FloatArray, factor: int) -> FloatArray:
    if factor <= 1:
        return plane
    h, w = (plane.shape[0] // factor) * factor, (plane.shape[1] // factor) * factor
    blocks = plane[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float64)


def _robust_sd(values: FloatArray) -> float:
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


# --------------------------------------------------------------- objects


@dataclass
class _Objects:
    mask: NDArray[np.bool_]
    labels: NDArray[np.int32]
    count: int
    background: float
    noise: float


def pixel_noise(plane: FloatArray) -> float:
    """Pixel noise SD from neighbouring-pixel differences (robust to structure and gradients)."""
    diff = np.diff(plane, axis=1).ravel()
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2))


def _objects(plane: FloatArray, diameter_px: float) -> _Objects:
    """Bright objects: Otsu threshold on log contrast above the background (so very bright
    debris does not hide the cells), kept only when clearly above the noise (an empty field has
    no objects, not noise blobs)."""
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(plane, 1.0)
    noise = max(pixel_noise(plane), 1e-6)
    background = float(np.median(smooth))
    empty = _Objects(
        np.zeros(plane.shape, bool), np.zeros(plane.shape, np.int32), 0, background, noise
    )
    contrast = np.log1p(np.maximum(smooth - background, 0) / noise)
    if np.ptp(contrast) == 0:
        return empty
    threshold = float(np.expm1(threshold_otsu(contrast)) * noise + background)
    threshold = max(threshold, background + 6 * noise)
    mask = smooth > threshold
    min_area = max(4, int(0.15 * np.pi * (diameter_px / 2) ** 2))
    labels, n = ndimage.label(mask)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas >= min_area
        keep[0] = False
        mask = keep[labels]
        labels, n = ndimage.label(mask)
    if n:
        background = float(np.median(plane[~ndimage.binary_dilation(mask, iterations=2)]))
    return _Objects(mask, labels.astype(np.int32), int(n), background, noise)


def _properties(labels: NDArray[np.int32], n: int, plane: FloatArray) -> dict[str, FloatArray]:
    """Area, mean intensity and ellipse axis lengths (as regionprops) of labels 1 … n."""
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
    safe = np.maximum(area, 1)
    total = np.bincount(flat, weights=plane.ravel(), minlength=n + 1)[1:]
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    sy = np.bincount(flat, weights=yy.ravel(), minlength=n + 1)[1:] / safe
    sx = np.bincount(flat, weights=xx.ravel(), minlength=n + 1)[1:] / safe
    syy = np.bincount(flat, weights=(yy * yy).ravel(), minlength=n + 1)[1:] / safe - sy**2
    sxx = np.bincount(flat, weights=(xx * xx).ravel(), minlength=n + 1)[1:] / safe - sx**2
    sxy = np.bincount(flat, weights=(xx * yy).ravel(), minlength=n + 1)[1:] / safe - sx * sy
    root = np.sqrt(np.maximum(((sxx - syy) / 2) ** 2 + sxy**2, 0))
    major = 4 * np.sqrt(np.maximum((sxx + syy) / 2 + root, 0))
    minor = 4 * np.sqrt(np.maximum((sxx + syy) / 2 - root, 0))
    return {"area": area, "mean": total / safe, "major": major, "minor": minor}


def _content(plane: FloatArray, settings: MetricSettings) -> dict[str, float]:
    diameter = settings.diameter_px
    objects = _objects(plane, diameter)
    out: dict[str, float] = {
        "object_count": float(objects.count),
        "area_fraction": float(objects.mask.mean()),
        "largest_gap_fraction": float("nan"),
        "debris_count": 0.0,
        "debris_area_fraction": 0.0,
        "fiber_count": 0.0,
    }
    if objects.count:
        props = _properties(objects.labels, objects.count, plane)
        areas = props["area"]
        means = props["mean"] - objects.background
        nominal = np.pi * (diameter / 2) ** 2
        ref_area = float(np.median(areas)) if len(areas) >= 5 else nominal
        ref_area = min(max(ref_area, 0.3 * nominal), 3 * nominal)
        ref_mean = float(np.median(means)) if len(means) >= 5 else float(np.max(means))
        elongated = (props["major"] >= 4 * diameter) & (props["minor"] <= 0.25 * props["major"])
        bright = (ref_mean > 0) & (means >= 2.5 * ref_mean) & (areas >= 3 * ref_area)
        huge = (areas >= 12 * ref_area) & (means >= 0.8 * ref_mean)
        debris = ~elongated & (bright | huge)
        out["fiber_count"] = float(elongated.sum())
        out["debris_count"] = float(debris.sum())
        out["debris_area_fraction"] = float(areas[debris].sum()) / plane.size
    if objects.count >= 8:
        spacing = np.sqrt(plane.size / objects.count)
        radius = max(2, round(min(0.75 * spacing, 3 * diameter)))
        # within ``radius`` of an object (a distance transform: fast for any radius)
        covered = ndimage.distance_transform_edt(~objects.mask) <= radius
        gaps, n = ndimage.label(~covered)
        largest = int(np.max(np.bincount(gaps.ravel())[1:])) if n else 0
        out["largest_gap_fraction"] = largest / plane.size
    out.update(_background_artifacts(plane, objects, diameter))
    return out


def background_map(plane: FloatArray, mask: NDArray[np.bool_], diameter: float) -> FloatArray:
    """The background of a field at about 64 × 64: objects removed, gaps filled by a median of
    the surrounding background."""
    factor = max(1, min(plane.shape) // 64)
    small = _bin(plane, factor)
    covered = _bin(ndimage.binary_dilation(mask, iterations=2).astype(np.float64), factor) > 0
    valid = (~covered).astype(np.float64)
    size = max(3, round(2 * diameter / factor) | 1)
    # normalized convolution: mean of the surrounding background, objects left out
    total = ndimage.uniform_filter(small * valid, size, mode="nearest")
    weight = ndimage.uniform_filter(valid, size, mode="nearest")
    fallback = float(np.median(small[valid > 0])) if valid.any() else float(np.median(small))
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(weight > 0.05, total / np.maximum(weight, 1e-12), np.nan)
    return np.asarray(np.where(np.isfinite(smooth), smooth, fallback), dtype=np.float64)


def _background_artifacts(
    plane: FloatArray, objects: _Objects, diameter: float
) -> dict[str, float]:
    """Bubbles and out-of-well regions: regions of the background map that deviate by more
    than 20 % from a robust quadratic surface (a smooth illumination gradient fits the
    surface and is not counted). Regions touching the field border and darker than the
    surface are out-of-well (well edge, meniscus shadow); other regions of ≥ 1 % of the field
    are bubbles."""
    smooth = background_map(plane, objects.mask, diameter)
    surface, _ = robust_surface(smooth)
    if surface is None:
        return {"bubble_fraction": 0.0, "out_of_well_fraction": 0.0}
    residual = smooth / np.maximum(surface, 1e-6) - 1
    noise = objects.noise / max(1, min(plane.shape) // 64)
    limit = max(0.2, 6 * noise / max(float(np.median(surface)), 1e-6))
    deviant = np.abs(residual) > limit
    labels, n = ndimage.label(deviant)
    bubble = 0.0
    out_of_well = 0.0
    if n:
        h, w = deviant.shape
        for index, region in enumerate(ndimage.find_objects(labels), start=1):
            if region is None:
                continue
            inside = labels[region] == index
            fraction = float(inside.sum()) / deviant.size
            if fraction < 0.01:
                continue
            border = (
                region[0].start == 0
                or region[1].start == 0
                or region[0].stop == h
                or region[1].stop == w
            )
            darker = float(residual[region][inside].mean()) < 0
            if border and darker:
                out_of_well += fraction
            else:
                bubble += fraction
    return {"bubble_fraction": float(bubble), "out_of_well_fraction": float(out_of_well)}


# ------------------------------------------------------------------ field


def _channel_metrics(plane: FloatArray, saturation: float) -> dict[str, float]:
    objects_free = plane[plane <= np.percentile(plane, 60)]
    background = float(np.median(objects_free))
    sample = plane[::2, ::2]
    p50, p99, p999 = (float(v) for v in np.percentile(sample, [50, 99, 99.9]))
    small = _bin(plane, max(1, min(plane.shape) // 64))
    surface, _ = robust_surface(small, small <= np.percentile(small, 60))
    gradient = float("nan")
    if surface is not None and float(np.mean(surface)) > 0:
        gradient = float((np.max(surface) - np.min(surface)) / np.mean(surface))
    clipped = np.minimum(plane, np.percentile(plane, 99))
    return {
        "plls": plls(clipped),
        "normalized_variance": normalized_variance(plane),
        "laplacian_variance": laplacian_variance(clipped),
        "saturated_fraction": float((plane >= saturation).mean()),
        "background": background,
        "p50": p50,
        "p99": p99,
        "p999": p999,
        "gradient": gradient,
    }


def field_metrics(stack: NDArray[Any], settings: MetricSettings) -> dict[str, float]:
    """QC metrics of one field, ``stack`` = raw (C, Z, Y, X) of one timepoint."""
    data = np.asarray(stack)
    if data.ndim != 4:
        raise ValueError(f"Expected a (C, Z, Y, X) field, got shape {data.shape}")
    channels = data.shape[0]
    content = settings.content_channel
    if not 1 <= content <= channels:
        raise ValueError(f"Content channel {content} does not exist ({channels} channels)")
    row: dict[str, float] = {}
    best_planes: list[int] = []
    for c in range(channels):
        planes = data[c].astype(np.float64)
        best = 0
        if planes.shape[0] > 1:
            scores = [laplacian_variance(p) for p in planes]
            finite = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
            best = int(np.argmax(finite))
        best_planes.append(best)
        saturation = saturation_level(data[c], settings.bit_depth)
        for key, value in _channel_metrics(planes[best], saturation).items():
            row[f"c{c + 1}_{key}"] = value
        row[f"c{c + 1}_best_z"] = float(best + 1) if planes.shape[0] > 1 else float("nan")
    plane = data[content - 1, best_planes[content - 1]].astype(np.float64)
    row.update(_content(plane, settings))
    bright = settings.brightfield_channel
    if bright is not None:
        if not 1 <= bright <= channels:
            raise ValueError(f"Brightfield channel {bright} does not exist ({channels} channels)")
        row["particle_fraction"] = particle_fraction(
            data[bright - 1, best_planes[bright - 1]].astype(np.float64)
        )
    return row


def particle_fraction(plane: FloatArray, sigma_px: float = 1.5) -> float:
    """Fraction of a brightfield plane covered by small high-contrast spots (precipitate,
    refractile particles): |Laplacian of Gaussian| above 8 × the pixel noise, in connected
    spots no larger than a disc of radius 3σ. Larger structures (cells) are left out."""
    noise = max(pixel_noise(plane), 1e-6)
    response = -ndimage.gaussian_laplace(plane, sigma_px) * sigma_px**2
    # the σ²-scaled LoG of white noise has an SD of about 0.27 × the pixel noise (σ = 1.5)
    strong = np.abs(response) > 8 * 0.27 * noise
    labels, n = ndimage.label(strong)
    if not n:
        return 0.0
    areas = np.bincount(labels.ravel())[1:]
    small = areas <= np.pi * (3 * sigma_px) ** 2
    return float(areas[small].sum() / plane.size)
