"""Reader abstraction, registry and format auto-detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import dask
import dask.array as da
import numpy as np

from hcs_studio.model.plate import Plate
from hcs_studio.registry import Registry


class ReaderError(Exception):
    """The data looks like this reader's format but could not be read."""


class PlateReader(ABC):
    """Maps one vendor format onto the unified data model. Never writes to the source."""

    #: Short vendor badge shown in the browser.
    vendor: ClassVar[str]
    #: Higher priority readers are asked first during detection.
    priority: ClassVar[int] = 0

    @classmethod
    @abstractmethod
    def detect(cls, path: Path) -> bool:
        """Cheap check (directory listing only) whether ``path`` is in this format."""

    @classmethod
    @abstractmethod
    def read(cls, path: Path) -> Plate:
        """Build the plate model. Pixel data stays lazy."""


READERS: Registry[type[PlateReader]] = Registry("reader")


def _load_builtin_readers() -> None:
    from hcs_studio.io import generic, imagexpress, nikon, ome_zarr, operetta  # noqa: F401


def detect_reader(path: Path) -> type[PlateReader] | None:
    """Return the reader that recognises ``path``, or ``None``."""
    _load_builtin_readers()
    candidates = sorted((READERS.get(n) for n in READERS), key=lambda r: -r.priority)
    for reader in candidates:
        try:
            if reader.detect(path):
                return reader
        except OSError:
            continue
    return None


def open_plate(path: Path, reader: str | None = None) -> Plate:
    """Open a plate with the named reader, or auto-detect the format."""
    _load_builtin_readers()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cls = READERS.get(reader) if reader else detect_reader(path)
    if cls is None:
        raise ReaderError(
            f"Could not recognise the format of {path}. "
            "Use the generic reader with a filename pattern."
        )
    return cls.read(path)


def lazy_tiff_stack(
    files: dict[tuple[int, int, int], Path],
    shape_tcz: tuple[int, int, int],
    plane_shape: tuple[int, int],
    dtype: np.dtype[Any],
) -> Any:
    """Lazy (T, C, Z, Y, X) array from one single-plane TIFF per (t, c, z).

    Missing planes read as zeros so an incomplete acquisition still opens.
    """
    import tifffile

    def read(path: Path) -> Any:
        plane = np.asarray(tifffile.imread(path))
        if plane.shape != plane_shape:
            raise ReaderError(f"{path.name}: shape {plane.shape}, expected {plane_shape}")
        return plane.astype(dtype, copy=False)

    n_t, n_c, n_z = shape_tcz
    t_blocks = []
    for t in range(n_t):
        c_blocks = []
        for c in range(n_c):
            z_blocks = []
            for z in range(n_z):
                path = files.get((t, c, z))
                if path is None:
                    z_blocks.append(da.zeros(plane_shape, dtype=dtype, chunks=plane_shape))
                else:
                    z_blocks.append(
                        da.from_delayed(dask.delayed(read)(path), shape=plane_shape, dtype=dtype)
                    )
            c_blocks.append(da.stack(z_blocks))
        t_blocks.append(da.stack(c_blocks))
    return da.stack(t_blocks)
