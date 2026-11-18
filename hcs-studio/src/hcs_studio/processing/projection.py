"""Z projection: maximum (default), mean, sum, minimum, standard deviation or extended depth
of field, over all planes, a plane range or automatically chosen in-focus planes.

The planes used for every field and channel go into the step's report (``well_summary.json``
→ ``fields[].reports``), with the focus score of each plane when they were chosen
automatically.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy import ndimage

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    FloatImage,
    ImageData,
    Module,
    ModuleError,
    ModuleParams,
)

METHODS = ("max", "mean", "sum", "min", "std", "edf")


class ProjectionParams(ModuleParams):
    method: Literal["max", "mean", "sum", "min", "std", "edf"] = Field(
        default="max",
        description="max (default), mean, sum, min, std (standard deviation), or edf = "
        "extended depth of field (in-focus detail of every plane).",
    )
    edf_method: Literal["variance", "pyramid"] = Field(
        default="variance",
        description="edf: variance = sharpest plane per pixel (local variance); pyramid = "
        "multi-scale (Laplacian pyramid, wavelet-like) fusion, smoother at plane boundaries.",
    )
    planes: Literal["all", "range", "auto"] = Field(
        default="all",
        description="all planes; a range (first/last plane); or auto = the planes around "
        "the best focus.",
    )
    first_plane: int = Field(default=1, ge=1, description="range: first plane (from 1).")
    last_plane: int | None = Field(
        default=None, ge=1, description="range: last plane (empty = the last one)."
    )
    focus_metric: Literal["normalized_variance", "brenner"] = Field(
        default="normalized_variance",
        description="auto: focus score of each plane (normalized variance or Brenner gradient).",
    )
    n_planes: int = Field(
        default=5, ge=1, description="auto: number of planes kept around the best focus."
    )
    focus_channel: int | None = Field(
        default=None,
        ge=1,
        description="auto: channel whose focus chooses the planes for all channels; empty = "
        "each channel chooses its own planes.",
    )

    @model_validator(mode="after")
    def _check(self) -> ProjectionParams:
        if self.last_plane is not None and self.last_plane < self.first_plane:
            raise ValueError("last_plane must not be before first_plane")
        return self


# ------------------------------------------------------------------------ focus


def focus_score(plane: NDArray[Any], metric: str = "normalized_variance") -> float:
    """Focus of one plane: higher is sharper. Both metrics ignore uniform offsets."""
    p = np.asarray(plane, dtype=np.float64)
    if metric == "normalized_variance":
        mean = float(p.mean())
        return float(p.var() / mean) if mean > 0 else 0.0
    if metric == "brenner":
        dx = p[:, 2:] - p[:, :-2]
        dy = p[2:, :] - p[:-2, :]
        return float((np.mean(dx * dx) + np.mean(dy * dy)) / 2)
    raise ValueError(f"Unknown focus metric {metric!r}")


def best_planes(scores: list[float], n: int) -> list[int]:
    """``n`` consecutive planes centred on the best score (shifted at the stack's ends)."""
    n = min(n, len(scores))
    best = int(np.argmax(scores))
    start = min(max(0, best - n // 2), len(scores) - n)
    return list(range(start, start + n))


# ------------------------------------------------------------------- projections


def _edf_variance(stack: FloatImage) -> FloatImage:
    """Per pixel, the plane with the highest local variance (smoothed choice map)."""
    focus = np.stack([ndimage.gaussian_filter(ndimage.laplace(plane) ** 2, 2.0) for plane in stack])
    best = ndimage.median_filter(np.argmax(focus, axis=0), size=5)
    return np.asarray(np.take_along_axis(stack, best[None], axis=0)[0], dtype=np.float32)


def _pyramid(image: FloatImage, levels: int) -> list[FloatImage]:
    import cv2

    gaussian = [image]
    for _ in range(levels):
        gaussian.append(np.asarray(cv2.pyrDown(gaussian[-1]), dtype=np.float32))
    laplacian = []
    for k in range(levels):
        size = (gaussian[k].shape[1], gaussian[k].shape[0])
        laplacian.append(gaussian[k] - cv2.pyrUp(gaussian[k + 1], dstsize=size))
    laplacian.append(gaussian[-1])
    return laplacian


def _edf_pyramid(stack: FloatImage) -> FloatImage:
    """Laplacian-pyramid fusion: at each scale keep the plane with the strongest detail
    (absolute band value); the coarsest level is averaged. Equivalent in spirit to the
    wavelet EDF of Forster et al. (2004)."""
    import cv2

    levels = max(1, min(5, int(np.log2(min(stack.shape[1:]))) - 3))
    pyramids = [_pyramid(np.ascontiguousarray(p, dtype=np.float32), levels) for p in stack]
    fused: list[FloatImage] = []
    for k in range(levels):
        bands = np.stack([pyr[k] for pyr in pyramids])
        energy = np.stack([ndimage.uniform_filter(np.abs(b), 3) for b in bands])
        choice = np.argmax(energy, axis=0)
        fused.append(np.take_along_axis(bands, choice[None], axis=0)[0])
    fused.append(np.mean([pyr[-1] for pyr in pyramids], axis=0, dtype=np.float32))
    out = fused[-1]
    for k in range(levels - 1, -1, -1):
        size = (fused[k].shape[1], fused[k].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + fused[k]
    # fused detail can overshoot: keep values within the stack's range per pixel
    return np.clip(out, stack.min(axis=0), stack.max(axis=0)).astype(np.float32)


def project(stack: FloatImage, method: str, edf_method: str = "variance") -> FloatImage:
    """Project a (Z, Y, X) stack to (Y, X)."""
    if method == "max":
        return stack.max(axis=0)
    if method == "mean":
        return stack.mean(axis=0, dtype=np.float64).astype(np.float32)
    if method == "sum":
        return stack.sum(axis=0, dtype=np.float64).astype(np.float32)
    if method == "min":
        return stack.min(axis=0)
    if method == "std":
        return stack.std(axis=0, dtype=np.float64).astype(np.float32)
    if method == "edf":
        if stack.shape[0] == 1:
            return np.asarray(stack[0], dtype=np.float32)
        fused = _edf_pyramid(stack) if edf_method == "pyramid" else _edf_variance(stack)
        return np.asarray(fused, dtype=np.float32)
    raise ModuleError(f"Unknown projection method {method!r}")


@MODULES.register("projection")
class Projection(Module):
    name = "projection"
    title = "Z projection"
    Params = ProjectionParams

    def output_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return replace(metadata, z_step_um=None)

    def validate(self, metadata: ImageMetadata) -> None:
        channel = self.params.focus_channel
        if channel is not None and channel > len(metadata.channels):
            raise ModuleError(
                f"Focus channel {channel} does not exist (the image has "
                f"{len(metadata.channels)} channels)"
            )

    def choose_planes(self, data: FloatImage) -> tuple[list[list[int]], dict[str, Any]]:
        """Planes (0-based) used for each channel, and what to report about the choice."""
        params: ProjectionParams = self.params
        n_c, n_z = data.shape[:2]
        if params.planes == "all":
            return [list(range(n_z))] * n_c, {}
        if params.planes == "range":
            first = params.first_plane - 1
            last = n_z - 1 if params.last_plane is None else params.last_plane - 1
            if first >= n_z:
                raise ModuleError(
                    f"First plane {params.first_plane} > the {n_z} planes of the stack"
                )
            return [list(range(first, min(last, n_z - 1) + 1))] * n_c, {}
        scores = [
            [focus_score(data[c, z], params.focus_metric) for z in range(n_z)] for c in range(n_c)
        ]
        if params.focus_channel is not None:
            chosen = best_planes(scores[params.focus_channel - 1], params.n_planes)
            planes = [chosen] * n_c
        else:
            planes = [best_planes(s, params.n_planes) for s in scores]
        return planes, {"focus_scores": [[round(v, 6) for v in s] for s in scores]}

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: ProjectionParams = self.params
        self.validate(image.metadata)
        data = image.data
        planes, extra = self.choose_planes(data)
        out = np.stack(
            [project(data[c, p], params.method, params.edf_method) for c, p in enumerate(planes)]
        )[:, None]
        report = {
            "module": self.name,
            "method": params.method if params.method != "edf" else f"edf ({params.edf_method})",
            # planes as the user counts them (from 1), per channel
            "planes": [[z + 1 for z in p] for p in planes],
            **extra,
        }
        return image.derive(
            out, self.name, metadata=self.output_metadata(image.metadata), report=report
        )
