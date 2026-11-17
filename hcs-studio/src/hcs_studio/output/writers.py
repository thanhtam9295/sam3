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
    """Write an RGB or RGBA uint8 image as PNG."""
    import imagecodecs

    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(imagecodecs.png_encode(np.ascontiguousarray(rgb)))
    tmp.replace(path)


def write_labels(path: Path, labels: NDArray[Any], metadata: ImageMetadata, name: str) -> None:
    """Write a (T, Z, Y, X) label stack as a compressed OME-TIFF (uint16, or uint32 if needed)."""
    import tifffile

    if labels.ndim != 4:
        raise ValueError(f"Expected (T, Z, Y, X) labels, got shape {labels.shape}")
    dtype = np.uint16 if int(np.max(labels, initial=0)) < 65536 else np.uint32
    ome: dict[str, Any] = {"axes": "TZYX", "Name": name}
    if metadata.pixel_size_um is not None:
        ome.update(PhysicalSizeX=metadata.pixel_size_um, PhysicalSizeY=metadata.pixel_size_um)
        ome.update(PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm")
    if metadata.z_step_um is not None:
        ome.update(PhysicalSizeZ=metadata.z_step_um, PhysicalSizeZUnit="µm")
    tmp = path.with_name(path.name + ".part")
    tifffile.imwrite(tmp, labels.astype(dtype), metadata=ome, compression="zlib", ome=True)
    tmp.replace(path)


def write_parent_links(path: Path, parent_ids: NDArray[Any], parent: str) -> None:
    """CSV of object ID → parent object ID (0 = no parent), one row per object."""
    tmp = path.with_name(path.name + ".part")
    lines = [f"object_id,{parent}_id"] + [
        f"{i},{int(parent_ids[i])}" for i in range(1, len(parent_ids))
    ]
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
