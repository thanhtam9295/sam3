"""Background removal, typically after Z projection.

Methods (one per step; stack steps to use different methods for different channels):

- ``tophat``: white top-hat (morphological opening with a disk), fast.
- ``rolling_ball`` / ``paraboloid``: a ball or a paraboloid of radius R rolled under the
  intensity surface (Sternberg 1983; ImageJ's sliding paraboloid). Large radii are computed
  on a shrunk image, as ImageJ does.
- ``gaussian`` / ``median``: a large-kernel blur of the image, subtracted or divided.
- ``polynomial`` / ``spline``: a smooth surface fitted to background pixels only (objects
  are excluded iteratively), subtracted or divided; for illumination gradients.
- ``constant``: a percentile of each field, or the mean of blank (no-cell) wells of the
  plate layout.
- ``basic``: BaSiC-style flat-field and dark-field estimated from many (projected) fields of
  the plate (Peng et al. 2017): I = S·B + D; corrected = (I − D) / S.

Negative values are clipped to 0; output stays float and linear. Each field's report
records, per channel, the change of the mean intensity inside objects and in the
background, the typical object diameter and any warning (e.g. a radius small enough to
erase signal).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.model.plate import ImageMetadata, parse_well_name
from hcs_studio.processing.base import (
    MODULES,
    ChannelParams,
    FitContext,
    FloatImage,
    ImageData,
    Module,
    ModuleError,
    block_reduce_mean,
    resize_to,
    um_to_px,
)
from hcs_studio.processing.flatfield import polynomial_surface

LOCAL = ("tophat", "rolling_ball", "paraboloid", "gaussian", "median")
SURFACES = ("polynomial", "spline")
FITTED = ("basic",)


class BackgroundParams(ChannelParams):
    method: Literal[
        "tophat",
        "rolling_ball",
        "paraboloid",
        "gaussian",
        "median",
        "polynomial",
        "spline",
        "constant",
        "basic",
    ] = Field(
        default="tophat",
        description="tophat: fast opening; rolling_ball / paraboloid: ImageJ-style; gaussian / "
        "median: large blur; polynomial / spline: surface fitted to background pixels; "
        "constant: percentile or blank wells; basic: flat- and dark-field from the plate.",
    )
    radius_um: float = Field(
        default=25.0,
        gt=0,
        description="tophat, rolling_ball, paraboloid: radius; gaussian: sigma; median: "
        "window radius. Choose larger than the largest object (at least 2× its diameter).",
    )
    operation: Literal["subtract", "divide"] = Field(
        default="subtract",
        description="gaussian, median, polynomial, spline: subtract the background, or divide "
        "by it (rescaled to its mean; corrects multiplicative shading).",
    )
    degree: int = Field(default=2, ge=0, le=6, description="polynomial: degree of the surface.")
    grid: int = Field(
        default=8, ge=4, le=64, description="spline: number of control points along each side."
    )
    constant_source: Literal["percentile", "blank_wells"] = Field(
        default="percentile",
        description="constant: a percentile of each field, or the mean of blank wells.",
    )
    percentile: float = Field(
        default=5.0, ge=0, le=100, description="constant: percentile of each field."
    )
    blank_wells: list[str] = Field(
        default_factory=list,
        description="constant (blank_wells): wells without cells, e.g. [A01, P24]. Empty = "
        "wells marked 'blank' in the plate layout.",
    )
    darkfield: bool = Field(default=True, description="basic: also estimate a dark-field.")
    max_fields: int = Field(
        default=100, ge=4, description="blank_wells, basic: fields sampled across the plate."
    )
    object_diameter_um: float | None = Field(
        default=None,
        gt=0,
        description="Typical object diameter for the radius check; empty = estimated from "
        "each field.",
    )


# ------------------------------------------------------------ local backgrounds


def _robust_sigma(values: NDArray[Any]) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _shrink_factor(radius_px: float) -> int:
    """ImageJ's shrinking for rolling ball / paraboloid (keeps large radii fast)."""
    return 1 if radius_px <= 10 else 2 if radius_px <= 30 else 4 if radius_px <= 100 else 8


def _block_mean(plane: NDArray[Any], factor: int) -> FloatImage:
    """Shrink by block averaging (ImageJ uses the block minimum, which sits on noise
    minima; the rolled ball already follows the lower envelope)."""
    if factor == 1:
        return plane.astype(np.float32)
    h, w = plane.shape
    ph, pw = -(-h // factor) * factor, -(-w // factor) * factor
    padded = np.pad(plane, ((0, ph - h), (0, pw - w)), mode="edge")
    blocks = padded.reshape(ph // factor, factor, pw // factor, factor)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def _ball_kernel(radius: float) -> FloatImage:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    d2 = yy**2 + xx**2
    kernel = np.sqrt(np.clip(radius**2 - d2, 0, None))
    kernel[d2 > radius**2] = np.inf
    return np.asarray(kernel, dtype=np.float32)


def _paraboloid_kernel(radius: float) -> FloatImage:
    """A paraboloid of curvature radius ``radius`` (height r²/2R below its apex)."""
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    d2 = yy**2 + xx**2
    kernel = radius - d2 / (2 * radius)
    kernel[d2 > radius**2] = np.inf
    return np.asarray(kernel, dtype=np.float32)


def rolling_background(plane: FloatImage, radius_px: float, shape: str = "ball") -> FloatImage:
    """Background under a rolled ball / paraboloid; computed on a shrunk image for large
    radii (block mean, then bilinear expansion) and never above the image.

    As in ImageJ, the ball's height is in intensity units: on a slope of g counts per
    pixel it stays R·(√(1+g²) − 1) below the surface (negligible for g ≪ 1, i.e. for
    real illumination gradients; use ``tophat`` or a surface fit for steep ones)."""
    from skimage.restoration import rolling_ball

    factor = _shrink_factor(radius_px)
    small = _block_mean(plane, factor)
    radius = max(1.0, radius_px / factor)
    kernel = _ball_kernel(radius) if shape == "ball" else _paraboloid_kernel(radius)
    # heights in intensity units stay those of the full-size ball (only x, y shrink)
    kernel = kernel * np.float32(factor)
    background = rolling_ball(small, kernel=kernel)
    if factor > 1:
        import cv2

        h, w = plane.shape
        background = cv2.resize(
            background,
            (background.shape[1] * factor, background.shape[0] * factor),
            interpolation=cv2.INTER_LINEAR,
        )[:h, :w]
        # the block-minimum image sits half a block early: smooth the steps
        background = np.minimum(background, plane)
    return np.asarray(background, dtype=np.float32)


def _downsampled_filter(plane: FloatImage, size_px: float, kind: str) -> FloatImage:
    """Large Gaussian (sigma) or median (radius) background, computed at reduced size.
    The Gaussian is made robust: pixels well above the blur (objects) are replaced by the
    blur and the image is blurred again, so bright objects do not raise their own
    background."""
    import cv2
    from scipy import ndimage

    factor = max(1, int(size_px // 8))
    small = block_reduce_mean(plane, max(8, max(plane.shape) // factor))
    scale = plane.shape[0] / small.shape[0]
    size = max(1.0, size_px / scale)
    if kind == "gaussian":
        work = small.copy()
        blurred = cv2.GaussianBlur(work, (0, 0), size, borderType=cv2.BORDER_REFLECT)
        for _ in range(4):
            residual = small - blurred
            sigma = _robust_sigma(residual)
            work = np.where(residual > 2.5 * max(sigma, 1e-9), blurred, small).astype(np.float32)
            blurred = cv2.GaussianBlur(work, (0, 0), size, borderType=cv2.BORDER_REFLECT)
    else:
        footprint = int(2 * round(size) + 1)
        blurred = ndimage.median_filter(small, size=footprint, mode="reflect")
    return resize_to(np.asarray(blurred), plane.shape)


# ---------------------------------------------------------- fitted surfaces


def surface_background(
    plane: FloatImage, method: str, degree: int = 2, grid: int = 8, iterations: int = 6
) -> FloatImage:
    """A smooth surface through the background pixels: pixels more than 2.5 robust SDs
    above the current surface are left out and the surface is refitted."""
    small = block_reduce_mean(plane, 256).astype(np.float64)
    work = small.copy()
    surface = np.full_like(small, np.median(small))
    for _ in range(iterations):
        if method == "polynomial":
            surface = polynomial_surface(work, degree)
        else:
            surface = _grid_surface(work, grid)
        residual = small - surface
        sigma = _robust_sigma(residual[np.isfinite(work)])
        objects = residual > 2.5 * max(sigma, 1e-9)
        new = np.where(objects, np.nan, small)
        if np.array_equal(np.isnan(new), np.isnan(work)):
            break
        work = new
    return resize_to(surface.astype(np.float32), plane.shape)


def _grid_surface(image: NDArray[Any], grid: int) -> NDArray[Any]:
    """Bicubic surface through the medians of ``grid × grid`` blocks (NaN = left out)."""
    import cv2

    h, w = image.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    nodes = np.full((grid, grid), np.nan)
    for i in range(grid):
        for j in range(grid):
            block = image[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            if np.isfinite(block).sum() >= max(3, block.size // 10):
                nodes[i, j] = np.nanmedian(block)
    if not np.isfinite(nodes).any():
        return np.full(image.shape, np.nanmedian(image))
    # fill blocks without background from their neighbours
    while np.isnan(nodes).any():
        padded = np.pad(nodes, 1, constant_values=np.nan)
        neighbours = np.stack(
            [
                padded[1 + dy : 1 + dy + grid, 1 + dx : 1 + dx + grid]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dy or dx
            ]
        )
        fill = np.nanmean(np.where(np.isfinite(neighbours), neighbours, np.nan), axis=0)
        nodes = np.where(np.isnan(nodes), fill, nodes)
    return np.asarray(cv2.resize(nodes, (w, h), interpolation=cv2.INTER_CUBIC), dtype=np.float64)


# --------------------------------------------------------------- BaSiC-style


def estimate_shading(
    fields: NDArray[Any], darkfield: bool = True, smoothing: float = 0.08
) -> tuple[FloatImage, FloatImage, dict[str, Any]]:
    """Flat-field S (mean 1) and dark-field D from fields (n, Y, X) of one channel, using
    I_i(x) = S(x)·B_i + D(x) on background pixels (objects left out per field).

    B_i is each field's median background. With B varying across fields, S and D are the
    per-pixel slope and intercept of I against B; without enough variation, D cannot be
    told apart from S·B and is set to 0. Both are smoothed (Gaussian, ``smoothing`` × the
    field size), as BaSiC's low-frequency constraint does. D is only defined up to a
    multiple of S; the multiple that makes D uncorrelated with S is used (exact for
    uniform and linear dark-fields)."""
    from scipy import ndimage

    data = np.asarray(fields, dtype=np.float64)
    masked = data.copy()
    for i, f in enumerate(data):
        med = np.median(f)
        sigma = _robust_sigma(f - med)
        masked[i][f > med + 3 * max(sigma, 1e-9)] = np.nan
    baseline = np.nanmedian(masked.reshape(len(data), -1), axis=1)
    notes: list[str] = []
    spread = float(np.std(baseline) / max(np.mean(baseline), 1e-9))
    if darkfield and len(data) >= 4 and spread >= 0.05:
        b = baseline[:, None, None]
        valid = np.isfinite(masked)
        n = valid.sum(axis=0)
        bb = np.where(valid, b, np.nan)
        mean_b = np.nanmean(bb, axis=0)
        mean_i = np.nanmean(masked, axis=0)
        cov = np.nanmean((bb - mean_b) * (masked - mean_i), axis=0)
        var = np.nanmean((bb - mean_b) ** 2, axis=0)
        slope = np.where((n >= 3) & (var > 0), cov / np.where(var > 0, var, 1), np.nan)
        intercept = mean_i - slope * mean_b
    else:
        if darkfield:
            notes.append(
                "dark-field not estimated: field backgrounds vary too little "
                f"(coefficient of variation {spread:.3f} < 0.05)"
            )
        slope = np.nanmedian(masked / baseline[:, None, None], axis=0)
        intercept = np.zeros_like(slope)
    for array in (slope, intercept):
        if np.isnan(array).any():
            array[np.isnan(array)] = np.nanmedian(array)
    sigma = smoothing * max(slope.shape)
    slope = ndimage.gaussian_filter(slope, sigma, mode="nearest")
    intercept = ndimage.gaussian_filter(intercept, sigma, mode="nearest")
    scale = float(np.mean(slope))
    flat = slope / scale
    dark = intercept
    if np.any(dark):
        # D is defined only up to a multiple of S (a constant added to every baseline
        # moves −c·S into D): choose the multiple that leaves D uncorrelated with S. This is
        # exact for uniform dark-fields (camera offsets) and for linear gradients.
        centred = flat - flat.mean()
        variance = float(np.mean(centred**2))
        if variance > 1e-12:
            dark = dark - float(np.mean((dark - dark.mean()) * centred)) / variance * flat
    flat32 = np.asarray(flat, dtype=np.float32)
    dark32 = np.asarray(dark, dtype=np.float32)
    info = {
        "n_fields": len(data),
        "background_cv": round(spread, 4),
        "flat_min": round(float(np.min(flat32)), 4),
        "flat_max": round(float(np.max(flat32)), 4),
        "dark_mean": round(float(dark32.mean()), 3),
        "notes": notes,
    }
    return flat32, dark32, info


# ---------------------------------------------------------------- guard rails


def object_mask(plane: FloatImage) -> NDArray[np.bool_]:
    """Bright objects of a plane: Otsu threshold after removing a smooth gradient."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(plane.astype(np.float32), 1.0)
    flat = smooth - resize_to(
        polynomial_surface(block_reduce_mean(smooth, 128), 2).astype(np.float32), plane.shape
    )
    if float(np.max(flat)) <= float(np.min(flat)):
        return np.zeros(plane.shape, dtype=bool)
    return np.asarray(flat > threshold_otsu(flat))


def typical_diameter_px(mask: NDArray[np.bool_]) -> float | None:
    """Median equivalent diameter of the objects in a mask (None with < 3 objects)."""
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n < 3:
        return None
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas >= 4]
    if areas.size < 3:
        return None
    return float(np.median(2 * np.sqrt(areas / np.pi)))


# --------------------------------------------------------------------- module


@MODULES.register("background")
class Background(Module):
    name = "background"
    title = "Background removal"
    Params = BackgroundParams

    @property
    def needs_fit(self) -> bool:  # type: ignore[override]
        params: BackgroundParams = self.params
        return params.method == "basic" or (
            params.method == "constant" and params.constant_source == "blank_wells"
        )

    def validate(self, metadata: ImageMetadata) -> None:
        params: BackgroundParams = self.params
        if params.method in LOCAL:
            radius = um_to_px(params.radius_um, metadata, "The background radius")
            if radius < 1:
                raise ModuleError("The background radius is smaller than one pixel")

    # ---------------------------------------------------------------- fit

    def _blank_wells(self, context: FitContext) -> list[tuple[int, int]]:
        params: BackgroundParams = self.params
        if params.blank_wells:
            try:
                wells = [parse_well_name(w) for w in params.blank_wells]
            except ValueError as exc:
                raise ModuleError(str(exc)) from None
        else:
            wells = [k for k, v in context.layout.items() if v.role == "blank"]
        present = [w for w in wells if w in context.plate.wells]
        if not present:
            raise ModuleError(
                "No blank wells with images: list them in 'blank_wells' or mark wells as "
                "'blank' in the plate layout"
            )
        return present

    def fit(self, context: FitContext) -> Any:
        params: BackgroundParams = self.params
        if params.method == "constant" and params.constant_source == "blank_wells":
            means = [
                image.data.mean(axis=(1, 2, 3))
                for image in context.sample(params.max_fields, self._blank_wells(context))
            ]
            if not means:
                raise ModuleError("The blank wells contain no readable fields")
            return {"constant": np.mean(means, axis=0).astype(np.float32), "n_fields": len(means)}
        if params.method == "basic":
            fields: list[NDArray[Any]] = []
            for image in context.sample(params.max_fields):
                fields.append(np.stack([block_reduce_mean(p.max(axis=0), 128) for p in image.data]))
            if len(fields) < 4:
                raise ModuleError(
                    f"BaSiC needs at least 4 fields across the plate; found {len(fields)}"
                )
            stack = np.stack(fields)  # (n, C, y, x)
            flats, darks, infos = [], [], []
            for c in range(stack.shape[1]):
                flat, dark, info = estimate_shading(stack[:, c], params.darkfield)
                flats.append(flat)
                darks.append(dark)
                infos.append(info)
            return {"flat": flats, "dark": darks, "info": infos}
        return None

    def describe_state(self, state: Any) -> dict[str, Any]:
        if not state:
            return {}
        if "constant" in state:
            return {
                "n_fields": state["n_fields"],
                "blank_mean_per_channel": [float(v) for v in state["constant"]],
            }
        return {"channels": state["info"]}

    # -------------------------------------------------------------- apply

    def background(
        self, plane: FloatImage, channel: int, radius_px: float, state: Any
    ) -> FloatImage | float:
        """The background of one plane (an image or a constant)."""
        import cv2

        method = self.params.method
        if method in ("tophat", "rolling_ball", "paraboloid"):
            # estimated on a 3 × 3 mean (as ImageJ does): an opening of the noisy image
            # sits on noise minima and leaves a positive floor
            smooth = np.asarray(
                cv2.blur(plane, (3, 3), borderType=cv2.BORDER_REFLECT), dtype=np.float32
            )
            if method == "tophat":
                size = 2 * round(radius_px) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                return np.asarray(
                    cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel), dtype=np.float32
                )
            shape = "ball" if method == "rolling_ball" else "paraboloid"
            return rolling_background(smooth, radius_px, shape)
        if method in ("gaussian", "median"):
            return _downsampled_filter(plane, radius_px, method)
        if method in SURFACES:
            return surface_background(plane, method, self.params.degree, self.params.grid)
        if method == "constant":
            if state is not None:
                return float(state["constant"][channel])
            return float(np.percentile(plane, self.params.percentile))
        raise ModuleError(f"Unknown background method {method!r}")  # pragma: no cover

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: BackgroundParams = self.params
        if self.needs_fit and state is None:
            raise ModuleError(f"Background ({params.method}) was not fitted on the plate")
        self.validate(image.metadata)
        local = params.method in LOCAL
        pixel = image.metadata.pixel_size_um
        radius_px = um_to_px(params.radius_um, image.metadata, "The radius") if local else 0.0
        out = image.data.copy()
        channels_report = []
        for c in params.selected(out.shape[0]):
            before = image.data[c]
            for z in range(out.shape[1]):
                plane = before[z]
                if params.method == "basic":
                    flat = resize_to(state["flat"][c], plane.shape)
                    dark = resize_to(state["dark"][c], plane.shape)
                    out[c, z] = (plane - dark) / np.maximum(flat, 1e-3)
                    continue
                bg = self.background(plane, c, radius_px, state)
                divide = params.operation == "divide" and params.method in (
                    "gaussian",
                    "median",
                    *SURFACES,
                )
                if divide and isinstance(bg, np.ndarray):
                    level = float(np.mean(bg))
                    out[c, z] = plane / np.maximum(bg, 1e-6) * level
                else:
                    out[c, z] = plane - bg
            np.clip(out[c], 0, None, out=out[c])
            channels_report.append(
                self._check(before, out[c], c, radius_px, pixel if pixel else None)
            )
        warnings = [r["warning"] for r in channels_report if r.get("warning")]
        report: dict[str, Any] = {
            "module": self.name,
            "method": params.method,
            "channels": channels_report,
        }
        if warnings:
            report["warnings"] = warnings
        return image.derive(out, self.name, report=report)

    def _check(
        self,
        before: FloatImage,
        after: FloatImage,
        channel: int,
        radius_px: float,
        pixel_um: float | None,
    ) -> dict[str, Any]:
        """Change of the mean inside objects and in the background; radius warning."""
        params: BackgroundParams = self.params
        b, a = before.max(axis=0), after.max(axis=0)
        mask = object_mask(b)
        record: dict[str, Any] = {"channel": channel + 1}
        if mask.any() and (~mask).any():
            for where, sel in (("object", mask), ("background", ~mask)):
                mean_before = float(b[sel].mean())
                mean_after = float(a[sel].mean())
                change = (mean_after - mean_before) / mean_before * 100 if mean_before else 0.0
                record[f"{where}_mean_before"] = round(mean_before, 3)
                record[f"{where}_mean_after"] = round(mean_after, 3)
                record[f"{where}_mean_change_pct"] = round(change, 2)
        if params.method in LOCAL:
            if params.object_diameter_um is not None and pixel_um:
                diameter_px: float | None = params.object_diameter_um / pixel_um
            else:
                diameter_px = typical_diameter_px(mask)
            if diameter_px is not None:
                scale = pixel_um or 1.0
                unit = "µm" if pixel_um else "px"
                record["object_diameter"] = round(diameter_px * scale, 3)
                record["object_diameter_unit"] = unit
                if radius_px < 2 * diameter_px:
                    record["warning"] = (
                        f"channel {channel + 1}: background radius {params.radius_um:g} µm is "
                        f"less than 2× the typical object diameter "
                        f"({diameter_px * scale:.3g} {unit}); it can remove signal. Use at "
                        f"least {2 * diameter_px * scale:.3g} {unit}."
                    )
        return record
