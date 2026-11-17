"""Image writers for run folders. Files are written to a temporary name, then renamed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.model.plate import ImageMetadata


def write_ome_tiff(path: Path, data: NDArray[Any], metadata: ImageMetadata) -> None:
    """Write a (T, C, Z, Y, X) array as a compressed OME-TIFF with physical sizes."""
    import tifffile

    if data.ndim != 5:
        raise ValueError(f"Expected (T, C, Z, Y, X), got shape {data.shape}")
    ome: dict[str, Any] = {
        "axes": "TCZYX",
        "Channel": {"Name": [c.name for c in metadata.channels]},
    }
    if metadata.pixel_size_um is not None:
        ome.update(
            PhysicalSizeX=metadata.pixel_size_um,
            PhysicalSizeXUnit="µm",
            PhysicalSizeY=metadata.pixel_size_um,
            PhysicalSizeYUnit="µm",
        )
    if metadata.z_step_um is not None:
        ome.update(PhysicalSizeZ=metadata.z_step_um, PhysicalSizeZUnit="µm")
    tmp = path.with_name(path.name + ".part")
    tifffile.imwrite(tmp, data, metadata=ome, compression="zlib", ome=True)
    tmp.replace(path)


def write_png(path: Path, rgb: NDArray[np.uint8]) -> None:
    import imagecodecs

    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(imagecodecs.png_encode(np.ascontiguousarray(rgb)))
    tmp.replace(path)
