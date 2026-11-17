"""Point-spread functions: theoretical models, extraction from bead stacks, FWHM, file I/O.

All PSFs are float32 ``(Z, Y, X)`` arrays normalised to sum 1, with odd sizes and the
peak at the centre. A 2D PSF has ``Z == 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import FloatImage, ModuleError

PsfModel = Literal["vectorial", "scalar", "gaussian"]
_MAX_XY, _MAX_Z = 129, 65


@dataclass(frozen=True)
class PsfSpec:
    """Optics needed to compute a theoretical PSF. Lengths in µm, wavelengths in nm."""

    na: float
    emission_nm: float
    pixel_size_um: float
    immersion_ri: float
    sample_ri: float = 1.33
    z_step_um: float | None = None
    confocal: bool = False
    excitation_nm: float | None = None
    pinhole_au: float = 1.0
    model: PsfModel = "vectorial"

    def __post_init__(self) -> None:
        if not 0 < self.na < self.immersion_ri:
            raise ModuleError(
                f"NA {self.na} must be positive and below the immersion refractive index "
                f"{self.immersion_ri}"
            )
        for name in ("emission_nm", "pixel_size_um", "sample_ri", "pinhole_au"):
            if getattr(self, name) <= 0:
                raise ModuleError(f"{name} must be positive")
        if self.z_step_um is not None and self.z_step_um <= 0:
            raise ModuleError("z_step_um must be positive")

    @property
    def lateral_resolution_um(self) -> float:
        return 0.61 * self.emission_nm / 1000 / self.na

    @property
    def axial_resolution_um(self) -> float:
        return 2 * self.immersion_ri * self.emission_nm / 1000 / self.na**2


def _odd(value: float, maximum: int) -> int:
    size = 2 * math.ceil(value) + 1
    return min(size, maximum)


def psf_shape(spec: PsfSpec, three_d: bool) -> tuple[int, int]:
    """(nz, nxy) large enough to hold the PSF: ±3 axial resolutions and the light cone."""
    half_angle = math.asin(min(0.999, spec.na / spec.immersion_ri))
    z_extent = 3 * spec.axial_resolution_um if three_d else 0.0
    xy_extent = 4 * spec.lateral_resolution_um + 0.5 * z_extent * math.tan(half_angle)
    nxy = max(9, _odd(xy_extent / spec.pixel_size_um, _MAX_XY))
    if not three_d:
        return 1, nxy
    assert spec.z_step_um is not None
    return max(3, _odd(z_extent / spec.z_step_um, _MAX_Z)), nxy


def normalise(psf: NDArray[Any]) -> FloatImage:
    psf = np.clip(np.asarray(psf, dtype=np.float64), 0, None)
    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ModuleError("The PSF is empty or not finite")
    return np.asarray(psf / total, dtype=np.float32)


def theoretical_psf(spec: PsfSpec, three_d: bool) -> FloatImage:
    """Gibson-Lanni (scalar) or vectorial PSF via ``psfmodels``; pinhole-aware for confocal."""
    if three_d and spec.z_step_um is None:
        raise ModuleError("A 3D PSF needs the Z step; set it in the Metadata panel")
    try:
        import psfmodels
    except ImportError:
        raise ModuleError("The 'psfmodels' package is required for theoretical PSFs") from None
    nz, nxy = psf_shape(spec, three_d)
    common: dict[str, Any] = {
        "dxy": spec.pixel_size_um,
        "dz": spec.z_step_um if three_d and spec.z_step_um else spec.axial_resolution_um,
        "NA": spec.na,
        "ns": spec.sample_ri,
        "ni": spec.immersion_ri,
        "ni0": spec.immersion_ri,
        "model": spec.model,
    }
    if spec.confocal:
        excitation = spec.excitation_nm or spec.emission_nm - 30.0
        psf = psfmodels.confocal_psf(
            z=nz,
            nx=nxy,
            pinhole_au=spec.pinhole_au,
            ex_wvl=excitation / 1000,
            em_wvl=spec.emission_nm / 1000,
            **common,
        )
    else:
        psf = psfmodels.make_psf(z=nz, nx=nxy, wvl=spec.emission_nm / 1000, **common)
    return normalise(psf)


def spec_from_metadata(
    metadata: ImageMetadata,
    channel: int,
    *,
    sample_ri: float = 1.33,
    pinhole_au: float | None = None,
    model: PsfModel = "vectorial",
) -> tuple[PsfSpec, list[str]]:
    """Build a PSF spec for one channel. Returns the spec and the assumptions made."""
    info = metadata.channels[channel]
    missing = [
        label
        for label, value in (
            ("objective NA", metadata.objective_na),
            ("pixel size", metadata.pixel_size_um),
            (f"emission wavelength of channel '{info.name}'", info.emission_nm),
        )
        if value is None
    ]
    if missing:
        raise ModuleError(f"A theoretical PSF needs: {', '.join(missing)}")
    if metadata.modality not in ("confocal", "widefield"):
        raise ModuleError(
            f"The modality is '{metadata.modality}'; set it to confocal or widefield in the "
            "Metadata panel"
        )
    assert metadata.objective_na and metadata.pixel_size_um and info.emission_nm
    notes: list[str] = []
    immersion = metadata.immersion_ri
    if immersion is None:
        if metadata.objective_na >= 0.95:
            raise ModuleError(
                "The immersion refractive index is unknown; set it in the Metadata panel"
            )
        immersion = 1.0
        notes.append("immersion RI unknown: assumed air (1.0)")
    confocal = metadata.modality == "confocal"
    au = 1.0
    if confocal:
        if pinhole_au is not None:
            au = pinhole_au
        elif metadata.pinhole_um and metadata.magnification:
            airy = 1.22 * info.emission_nm / 1000 / metadata.objective_na
            au = (metadata.pinhole_um / metadata.magnification) / airy
            notes.append(f"pinhole {au:.2f} Airy units, from the pinhole size and magnification")
        else:
            notes.append("pinhole size unknown: assumed 1 Airy unit")
        if info.excitation_nm is None:
            notes.append("excitation wavelength unknown: assumed emission - 30 nm")
    spec = PsfSpec(
        na=metadata.objective_na,
        emission_nm=info.emission_nm,
        pixel_size_um=metadata.pixel_size_um,
        immersion_ri=immersion,
        sample_ri=sample_ri,
        z_step_um=metadata.z_step_um,
        confocal=confocal,
        excitation_nm=info.excitation_nm,
        pinhole_au=au,
        model=model,
    )
    return spec, notes


# ----------------------------------------------------------------------- FWHM


def profile_fwhm(profile: NDArray[Any], spacing: float) -> float | None:
    """Full width at half maximum of a single-peaked profile, by linear interpolation.

    Returns ``None`` when the profile does not fall to half maximum on both sides.
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 3:
        return None
    values = values - values.min()
    peak = int(np.argmax(values))
    half = values[peak] / 2
    if half <= 0:
        return None

    def crossing(step: int) -> float | None:
        i = peak
        while 0 <= i + step < values.size:
            if values[i + step] <= half:
                span = values[i] - values[i + step]
                return abs(i - peak) + (values[i] - half) / span if span > 0 else None
            i += step
        return None

    left, right = crossing(-1), crossing(1)
    if left is None or right is None:
        return None
    return (left + right) * spacing


def measure_fwhm(
    volume: NDArray[Any], voxel_zyx: tuple[float | None, float, float]
) -> dict[str, float | None]:
    """FWHM (µm) along Z, Y and X through the brightest voxel of a (Z, Y, X) volume."""
    z, y, x = np.unravel_index(int(np.argmax(volume)), volume.shape)
    result: dict[str, float | None] = {
        "y": profile_fwhm(volume[z, :, x], voxel_zyx[1]),
        "x": profile_fwhm(volume[z, y, :], voxel_zyx[2]),
        "z": None,
    }
    if volume.shape[0] > 1 and voxel_zyx[0] is not None:
        result["z"] = profile_fwhm(volume[:, y, x], voxel_zyx[0])
    return result


# ---------------------------------------------------------------- bead stacks


@dataclass(frozen=True)
class BeadPsf:
    psf: FloatImage
    n_beads: int
    n_rejected: int
    positions_zyx: tuple[tuple[int, int, int], ...]


def extract_psf_from_beads(
    stack: NDArray[Any],
    *,
    size_xy: int = 33,
    size_z: int | None = None,
    threshold_rel: float = 0.2,
    max_beads: int = 100,
    saturation: float | None = None,
) -> BeadPsf:
    """Average isolated sub-resolution beads of a (Z, Y, X) stack into a measured PSF.

    Beads are local maxima brighter than ``threshold_rel`` of the brightest one. Beads
    that are saturated, too close to the edge, or have a neighbour inside their crop
    are rejected. Each crop is background-subtracted, centred to sub-pixel precision
    and normalised before averaging.
    """
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ModuleError(f"A bead stack must be (Z, Y, X); got shape {data.shape}")
    if size_xy % 2 == 0 or size_xy < 5:
        raise ModuleError("size_xy must be an odd number of at least 5")
    nz = data.shape[0]
    if size_z is None:
        size_z = nz if nz % 2 else nz - 1
        size_z = max(1, min(size_z, _MAX_Z))
    if size_z % 2 == 0 or size_z > nz:
        raise ModuleError("size_z must be odd and no larger than the stack")
    hz, hxy = size_z // 2, size_xy // 2

    background = float(np.median(data))
    smooth = ndimage.gaussian_filter(data, (0.5 if nz > 1 else 0, 1.0, 1.0))
    footprint = (min(size_z, nz), size_xy, size_xy)
    is_peak = smooth == ndimage.maximum_filter(smooth, size=footprint, mode="constant")
    level = background + threshold_rel * (float(smooth.max()) - background)
    candidates = np.argwhere(is_peak & (smooth > level))
    order = np.argsort([-smooth[tuple(c)] for c in candidates])

    crops: list[NDArray[Any]] = []
    positions: list[tuple[int, int, int]] = []
    rejected = 0
    for index in order:
        z, y, x = (int(v) for v in candidates[index])
        inside = (
            hz <= z < nz - hz and hxy <= y < data.shape[1] - hxy and hxy <= x < data.shape[2] - hxy
        )
        if not inside:
            rejected += 1
            continue
        crop = data[z - hz : z + hz + 1, y - hxy : y + hxy + 1, x - hxy : x + hxy + 1]
        if saturation is not None and float(np.max(crop)) >= saturation:
            rejected += 1
            continue
        border = np.concatenate(
            [crop[:, 0].ravel(), crop[:, -1].ravel(), crop[:, :, 0].ravel(), crop[:, :, -1].ravel()]
        )
        local = crop - float(np.median(border))
        peak = float(np.max(local))
        # A second bead inside the crop shows up as bright border signal.
        if peak <= 0 or float(np.percentile(border - np.median(border), 99.5)) > 0.25 * peak:
            rejected += 1
            continue
        weights = np.clip(local - peak / 2, 0, None)
        centre = np.array(ndimage.center_of_mass(weights))
        shift = np.array([hz, hxy, hxy]) - centre
        if np.any(np.abs(shift) > 1.5):
            rejected += 1
            continue
        aligned = ndimage.shift(local, shift, order=1, mode="nearest")
        total = float(np.clip(aligned, 0, None).sum())
        if total <= 0:
            rejected += 1
            continue
        crops.append(aligned / total)
        positions.append((z, y, x))
        if len(crops) >= max_beads:
            break
    if not crops:
        raise ModuleError(
            "No usable beads found: they must be isolated, unsaturated and away from the edges"
        )
    return BeadPsf(
        psf=normalise(np.mean(crops, axis=0)),
        n_beads=len(crops),
        n_rejected=rejected,
        positions_zyx=tuple(positions),
    )


# ------------------------------------------------------------------- file I/O


def save_psf(path: Path, psf: NDArray[Any], pixel_size_um: float, z_step_um: float | None) -> None:
    """Save a PSF as a float32 OME-TIFF that records its voxel size."""
    import tifffile

    metadata: dict[str, Any] = {
        "axes": "ZYX",
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "µm",
    }
    if z_step_um is not None:
        metadata.update(PhysicalSizeZ=z_step_um, PhysicalSizeZUnit="µm")
    tifffile.imwrite(path, np.asarray(psf, dtype=np.float32), metadata=metadata, ome=True)


def load_psf(path: Path) -> tuple[FloatImage, float | None, float | None]:
    """Load a PSF file. Returns (psf, pixel size µm, z step µm); sizes may be ``None``."""
    import re

    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            data = np.asarray(tif.asarray(), dtype=np.float32)
            xml = tif.ome_metadata or ""
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise ModuleError(f"Could not read the PSF file {path}: {exc}") from None
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ModuleError(f"The PSF file must hold a 2D or 3D image; got shape {data.shape}")

    def size(axis: str) -> float | None:
        match = re.search(rf'PhysicalSize{axis}="([^"]+)"', xml)
        return float(match.group(1)) if match else None

    return normalise(data), size("X"), size("Z") if data.shape[0] > 1 else None


def resample_psf(
    psf: NDArray[Any],
    from_zyx: tuple[float | None, float, float],
    to_zyx: tuple[float | None, float, float],
) -> FloatImage:
    """Resample a PSF to another voxel size (linear interpolation), keeping odd sizes."""
    factors = [
        1.0 if a is None or b is None or psf.shape[i] == 1 else a / b
        for i, (a, b) in enumerate(zip(from_zyx, to_zyx, strict=True))
    ]
    if all(abs(f - 1) < 0.02 for f in factors):
        return normalise(psf)
    shape = [max(1, round(n * f)) for n, f in zip(psf.shape, factors, strict=True)]
    shape = [n if n % 2 or n == 1 else n + 1 for n in shape]
    zoom = [n / m for n, m in zip(shape, psf.shape, strict=True)]
    return normalise(ndimage.zoom(psf, zoom, order=1, mode="nearest", grid_mode=True))


def centre_crop_or_pad(psf: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    """Centre a PSF in an array of ``shape``, cropping where the PSF is larger."""
    out = np.zeros(shape, dtype=psf.dtype)
    src, dst = [], []
    for have, want in zip(psf.shape, shape, strict=True):
        # Keep the PSF centre (index have // 2) at the centre of the output (want // 2),
        # so that ifftshift moves it exactly to the origin for odd and even sizes alike.
        if have >= want:
            start = have // 2 - want // 2
            src.append(slice(start, start + want))
            dst.append(slice(0, want))
        else:
            start = want // 2 - have // 2
            src.append(slice(0, have))
            dst.append(slice(start, start + have))
    out[tuple(dst)] = psf[tuple(src)]
    return out
