"""ilastik-ready HDF5 files.

ilastik reads HDF5 block by block; it learns the axis order from the dataset attribute
``axistags``, a vigra JSON document such as::

    {"axes": [{"key": "y", "typeFlags": 2, "resolution": 0, "description": ""}, ...]}

(``typeFlags``: 1 channels, 2 space, 8 time). This is exactly what ilastik 1.4.2 writes on
export (dataset ``exported_data``) and accepts on input (``file.h5/data``); checked with the
headless Data Conversion workflow.

Files are written to a temporary name and renamed, so a killed run never leaves a half-written
file under the final name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.segmentation.ilastik_backend.axes import check_axes

TYPE_FLAGS = {"c": 1, "t": 8, "z": 2, "y": 2, "x": 2}
INPUT_DATASET = "data"
EXPORT_DATASET = "exported_data"  # ilastik's default --output_internal_path
CHUNK_XY = 256
CHUNK_Z = 32


def axistags_json(axes: str, resolution: dict[str, float] | None = None) -> str:
    """The vigra ``axistags`` attribute for ``axes`` (``resolution`` per axis, default 0)."""
    axes = check_axes(axes)
    resolution = resolution or {}
    return json.dumps(
        {
            "axes": [
                {
                    "key": a,
                    "typeFlags": TYPE_FLAGS[a],
                    "resolution": resolution.get(a, 0),
                    "description": "",
                }
                for a in axes
            ]
        }
    )


def parse_axistags(text: str | bytes) -> str:
    """The axis string of a vigra ``axistags`` attribute."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        data = json.loads(text)
        axes = "".join(str(a["key"]) for a in data["axes"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Not a valid axistags attribute: {text[:80]!r}") from exc
    return check_axes(axes)


def default_chunks(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    """One timepoint, all channels, up to 32 planes and 256 × 256 pixels per chunk."""
    limit = {"t": 1, "z": CHUNK_Z, "y": CHUNK_XY, "x": CHUNK_XY}
    return tuple(
        max(1, min(n, limit[a])) if a in limit else n for a, n in zip(axes, shape, strict=True)
    )


def write_h5(
    path: Path,
    data: NDArray[Any],
    axes: str,
    *,
    dataset: str = INPUT_DATASET,
    compression: int | None = 1,
    resolution: dict[str, float] | None = None,
) -> Path:
    """Write ``data`` (already in ``axes`` order) as an ilastik-ready HDF5 file.

    ``compression`` is the gzip level (0–9) or None for none; ilastik4ij recommends 0 for raw
    images (speed) and 9 for label images (they compress well).
    """
    import h5py

    axes = check_axes(axes)
    if data.ndim != len(axes):
        raise ValueError(f"Axis order {axes!r} does not match {data.ndim}-dimensional data")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    kwargs: dict[str, Any] = {"chunks": default_chunks(data.shape, axes)}
    if compression:
        if not 0 <= compression <= 9:
            raise ValueError("gzip level must be 0–9")
        kwargs.update(compression="gzip", compression_opts=int(compression))
    try:
        with h5py.File(tmp, "w") as f:
            ds = f.create_dataset(dataset, data=np.ascontiguousarray(data), **kwargs)
            ds.attrs["axistags"] = axistags_json(axes, resolution)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def dataset_names(path: Path) -> list[str]:
    import h5py

    names: list[str] = []
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
        )
    return names


def _pick_dataset(path: Path, dataset: str | None) -> str:
    if dataset:
        return dataset.lstrip("/")
    names = dataset_names(path)
    for preferred in (EXPORT_DATASET, INPUT_DATASET):
        if preferred in names:
            return preferred
    if len(names) == 1:
        return names[0]
    raise ValueError(f"{path} holds several datasets ({', '.join(names)}); name one")


def read_axes(path: Path, dataset: str | None = None) -> str | None:
    """The axis string stored with a dataset, or None when it has no ``axistags``."""
    import h5py

    name = _pick_dataset(Path(path), dataset)
    with h5py.File(path, "r") as f:
        tags = f[name].attrs.get("axistags")
    return parse_axistags(tags) if tags is not None else None


def read_h5(
    path: Path, dataset: str | None = None, axes: str | None = None
) -> tuple[NDArray[Any], str]:
    """Read a dataset and its axis string (from ``axistags``, else ``axes`` must be given).

    Without ``dataset``: ``exported_data`` (ilastik output), else ``data``, else the only one.
    """
    import h5py

    path = Path(path)
    name = _pick_dataset(path, dataset)
    with h5py.File(path, "r") as f:
        ds = f[name]
        tags = ds.attrs.get("axistags")
        found = parse_axistags(tags) if tags is not None else None
        array = ds[()]
    if found is None:
        if axes is None:
            raise ValueError(f"{path}/{name} has no axistags; give the axis order")
        found = check_axes(axes)
    elif axes is not None and check_axes(axes) != found:
        raise ValueError(f"{path}/{name} has axes {found!r}, expected {axes!r}")
    if array.ndim != len(found):
        raise ValueError(f"{path}/{name}: axes {found!r} but {array.ndim} dimensions")
    return array, found


def input_spec(path: Path, dataset: str = INPUT_DATASET) -> str:
    """How ilastik's command line names a dataset inside a file: ``<file>/<dataset>``."""
    return f"{Path(path)}/{dataset.lstrip('/')}"


def is_ilastik_ready(path: Path, dataset: str | None = None) -> bool:
    """True for an HDF5 dataset with ``axistags`` (usable as ilastik input without copying)."""
    try:
        return read_axes(Path(path), dataset) is not None
    except (OSError, ValueError, KeyError):
        return False


def export_plate(
    plate: Any,
    wells: list[tuple[int, int]],
    folder: Path,
    *,
    channels: list[int] | None = None,
    time_stack: bool = False,
) -> list[Path]:
    """Raw fields → ilastik-ready HDF5 (``<well>_f<field>_t<t>.h5``, or ``<well>_f<field>.h5``
    with every time point when ``time_stack``), for training in ilastik or a Fiji macro.
    ``channels`` from 1. Never writes into the raw data's folder."""
    from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes, to_ilastik

    folder = Path(folder).absolute()
    raw = Path(plate.path).absolute()
    if folder == raw or raw in folder.parents:
        raise ValueError("Write the HDF5 files outside the raw data")
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key in wells:
        well = plate.wells[key]
        for item in well.fields:
            data = np.asarray(item.load())  # (T, C, Z, Y, X), one field at a time
            if channels:
                data = data[:, [c - 1 for c in channels]]
            stacks = (
                [(None, data)] if time_stack else [(t, data[t : t + 1]) for t in range(len(data))]
            )
            for t, stack in stacks:
                axes = ilastik_axes(stack.shape, time=True if time_stack else None)
                name = f"{well.name}_f{item.index:03d}" + ("" if t is None else f"_t{t}")
                written.append(write_h5(folder / f"{name}.h5", to_ilastik(stack, axes), axes))
    return written
