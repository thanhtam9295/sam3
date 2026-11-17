"""Synthetic fluorescence images with a known ground truth, for tests and demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticField:
    """A (C, Y, X) uint16 image plus the nucleus centres used to draw it."""

    image: NDArray[np.uint16]
    channel_names: tuple[str, ...]
    centers_yx: NDArray[np.float64]
    pixel_size_um: float


def make_cell_field(
    shape: tuple[int, int] = (512, 512),
    n_cells: int = 40,
    pixel_size_um: float = 0.325,
    seed: int = 0,
) -> SyntheticField:
    """Draw ``n_cells`` cells: Gaussian nuclei (channel 0) inside wider cytoplasm (channel 1).

    Deterministic for a given seed. Includes a constant camera offset and Poisson noise.
    """
    if n_cells < 0:
        raise ValueError("n_cells must be non-negative")
    rng = np.random.default_rng(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    margin = 12.0
    centers = np.column_stack(
        [
            rng.uniform(margin, height - margin, n_cells),
            rng.uniform(margin, width - margin, n_cells),
        ]
    )
    nuclei = np.zeros(shape, dtype=np.float64)
    cytoplasm = np.zeros(shape, dtype=np.float64)
    for cy, cx in centers:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        nuclei += rng.uniform(1500, 4000) * np.exp(-r2 / (2 * rng.uniform(5.0, 8.0) ** 2))
        cytoplasm += rng.uniform(500, 1500) * np.exp(-r2 / (2 * rng.uniform(12.0, 18.0) ** 2))
    offset = 100.0
    stack = np.stack([nuclei, cytoplasm]) + offset
    noisy = rng.poisson(stack).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    return SyntheticField(
        image=noisy,
        channel_names=("Nuclei (synthetic)", "Cytoplasm (synthetic)"),
        centers_yx=centers,
        pixel_size_um=pixel_size_um,
    )


# ------------------------------------------------------- synthetic plate on disk


def _metaseries_description(wave_name: str, wavelength: int, pixel_size: float) -> str:
    return f"""<MetaData>
<ApplicationName>MetaMorph</ApplicationName>
<ApplicationVersion>6.5</ApplicationVersion>
<PlaneInfo>
<prop id="plane-type" type="string" value="plane"/>
<prop id="bits-per-pixel" type="int" value="16"/>
<prop id="spatial-calibration-state" type="bool" value="on"/>
<prop id="spatial-calibration-x" type="float" value="{pixel_size}"/>
<prop id="spatial-calibration-y" type="float" value="{pixel_size}"/>
<prop id="spatial-calibration-units" type="string" value="um"/>
<prop id="image-name" type="string" value="{wave_name}"/>
<prop id="wavelength" type="float" value="{wavelength}"/>
<custom-prop id="_MagNA_" type="float" value="0.75"/>
<custom-prop id="_MagSetting_" type="string" value="20X Plan Apo"/>
<custom-prop id="Exposure Time" type="string" value="150 ms"/>
</PlaneInfo>
</MetaData>"""


def write_imagexpress_plate(
    root: Path,
    wells: list[tuple[int, int]],
    image_fn: Callable[[int, int, int, int, int], NDArray[Any]],
    *,
    n_sites: int = 2,
    z_steps: int = 1,
    pixel_size_um: float = 0.65,
    plate_size: tuple[int, int] = (8, 12),
) -> Path:
    """Write an ImageXpress (MetaXpress) export: ``.HTD`` + GUID-suffixed TIFFs with MetaSeries
    descriptions + thumbnails, and ``ZStep_n`` folders for Z stacks. ``image_fn(row, col,
    site, wave, z)`` returns a 2D uint16 image; waves are DAPI (447 nm) and FITC (520 nm)."""
    import tifffile

    from hcs_studio.model.plate import well_name

    root.mkdir(parents=True)
    waves = [("DAPI", 447), ("FITC", 520)]
    guid = "0A1B2C3D-1111-2222-3333-444455556666"

    def write(folder: Path, row: int, col: int, s: int, w: int, z: int) -> None:
        stem = f"Plate1_{well_name(row, col)}_s{s + 1}_w{w + 1}"
        tifffile.imwrite(
            folder / f"{stem}{guid}.tif",
            image_fn(row, col, s, w, z),
            description=_metaseries_description(waves[w][0], waves[w][1], pixel_size_um),
            software="MetaSeries",
            metadata=None,
        )
        tifffile.imwrite(folder / f"{stem}_thumb{guid}.tif", np.zeros((4, 4), dtype=np.uint8))

    for row, col in wells:
        for s in range(n_sites):
            for w in range(len(waves)):
                if z_steps == 1:
                    write(root, row, col, s, w, 0)
                else:
                    # Projection at ZStep_0 (must be ignored), planes in ZStep_1..n.
                    for z in range(z_steps + 1):
                        folder = root / f"ZStep_{z}"
                        folder.mkdir(exist_ok=True)
                        write(folder, row, col, s, w, z - 1 if z else 99)
    lines = [
        '"HTSInfoFile", Version 1.0',
        '"Description", "Screen 7"',
        '"TimePoints", 1',
        f'"ZSeries", {"TRUE" if z_steps > 1 else "FALSE"}',
        f'"ZSteps", {z_steps}',
        f'"XWells", {plate_size[1]}',
        f'"YWells", {plate_size[0]}',
        '"Sites", TRUE',
        f'"XSites", {n_sites}',
        '"YSites", 1',
        '"Waves", TRUE',
        f'"NWavelengths", {len(waves)}',
        *[f'"WaveName{i + 1}", "{name}"' for i, (name, _) in enumerate(waves)],
        '"EndFile"',
    ]
    (root / "Plate1.HTD").write_text("\r\n".join(lines) + "\r\n", encoding="latin-1")
    return root
