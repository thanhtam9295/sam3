"""Generic readers: a single OME-TIFF, or a folder of TIFFs with a filename pattern."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np

from hcs_studio.io.base import READERS, PlateReader, ReaderError, lazy_tiff_stack
from hcs_studio.model.plate import (
    ChannelInfo,
    Field,
    ImageMetadata,
    Plate,
    Well,
    infer_plate_shape,
    parse_well_name,
)

#: Patterns tried during auto-detection. Named groups: well (required), field, channel, z, t.
DEFAULT_PATTERNS = (
    r"(?P<well>[A-Z]{1,2}\d{1,2})_f(?P<field>\d+)_c(?P<channel>[^_.]+)(?:_z(?P<z>\d+))?(?:_t(?P<t>\d+))?",
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<z>\d+)-ch(?P<channel>\d+)",
)
_TIFF_SUFFIXES = (".tif", ".tiff")
_OME_UM = {"µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}


def _tiffs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in _TIFF_SUFFIXES and p.is_file())


def _is_ome_tiff(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith((".ome.tif", ".ome.tiff"))


def _ome_metadata(xml: str | None, n_channels: int, has_z: bool) -> ImageMetadata:
    channels = [ChannelInfo(name=f"Channel {i + 1}") for i in range(n_channels)]
    if not xml:
        return ImageMetadata(channels=tuple(channels))
    root = ET.fromstring(xml)

    def find(tag: str) -> ET.Element | None:
        return next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == tag), None)

    def size(pixels: ET.Element, axis: str) -> float | None:
        value = pixels.get(f"PhysicalSize{axis}")
        if value is None:
            return None
        unit = pixels.get(f"PhysicalSize{axis}Unit", "µm")
        if unit not in _OME_UM:
            raise ReaderError(f"Unsupported OME unit {unit!r}")
        return float(value) * _OME_UM[unit]

    def number(element: ET.Element | None, key: str) -> float | None:
        value = element.get(key) if element is not None else None
        return float(value) if value else None

    pixels = find("Pixels")
    objective = find("Objective")
    found = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "Channel"]
    if len(found) == n_channels:
        channels = [
            ChannelInfo(
                name=e.get("Name") or f"Channel {i + 1}",
                excitation_nm=number(e, "ExcitationWavelength"),
                emission_nm=number(e, "EmissionWavelength"),
            )
            for i, e in enumerate(found)
        ]
    bits = pixels.get("SignificantBits") if pixels is not None else None
    return ImageMetadata(
        channels=tuple(channels),
        pixel_size_um=size(pixels, "X") if pixels is not None else None,
        z_step_um=size(pixels, "Z") if pixels is not None and has_z else None,
        objective_na=number(objective, "LensNA"),
        magnification=number(objective, "NominalMagnification"),
        bit_depth=int(bits) if bits else None,
    )


@READERS.register("ome-tiff")
class OmeTiffReader(PlateReader):
    """A single OME-TIFF file, shown as one well with one field."""

    vendor = "OME-TIFF"
    priority = 5

    @classmethod
    def detect(cls, path: Path) -> bool:
        return _is_ome_tiff(path)

    @classmethod
    def read(cls, path: Path) -> Plate:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = tuple(int(s) for s in series.shape)
            dtype = np.dtype(series.dtype)
            xml = tif.ome_metadata
        unknown = set(axes) - set("TCZYX")
        if unknown:
            raise ReaderError(f"{path.name}: unsupported axes {axes!r}")
        sizes = dict(zip(axes, shape, strict=True))
        tczyx = (sizes.get("T", 1), sizes.get("C", 1), sizes.get("Z", 1), sizes["Y"], sizes["X"])

        def load() -> Any:
            def read_all() -> Any:
                data = np.asarray(tifffile.imread(path, series=0))
                for axis in "TCZ":
                    if axis not in axes:
                        data = data[None]
                order = [a for a in "TCZ" if a not in axes][::-1] + list(axes)
                return data.transpose([order.index(a) for a in "TCZYX"])

            return da.from_delayed(dask.delayed(read_all)(), shape=tczyx, dtype=dtype)

        metadata = _ome_metadata(xml, tczyx[1], tczyx[2] > 1)
        if metadata.bit_depth is None:
            from dataclasses import replace

            metadata = replace(metadata, bit_depth=int(dtype.itemsize * 8))
        rows, cols = infer_plate_shape(1, 1)
        return Plate(
            name=path.name,
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells={
                (0, 0): Well(
                    row=0,
                    col=0,
                    fields=(Field(index=1, load=load, sources=(path,)),),
                    label=path.stem,
                )
            },
            metadata=metadata,
            shape_tczyx=tczyx,
            notes=("Single image file; shown as well A01.",),
        )


def _compile(pattern: str) -> re.Pattern[str]:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ReaderError(f"Invalid filename pattern: {exc}") from None
    groups = set(compiled.groupindex)
    if "well" not in groups and not {"row", "col"} <= groups:
        raise ReaderError("The pattern needs a 'well' group, or both 'row' and 'col' groups")
    return compiled


@READERS.register("tiff-folder")
class TiffFolderReader(PlateReader):
    """Folder of single-plane TIFFs named by a regular expression with named groups."""

    vendor = "TIFF"
    priority = 1

    @classmethod
    def detect(cls, path: Path) -> bool:
        if not path.is_dir():
            return False
        names = [p.name for p in _tiffs(path)[:20]]
        return any(
            all(re.search(pattern, n) for n in names) for pattern in DEFAULT_PATTERNS if names
        )

    @classmethod
    def read(cls, path: Path) -> Plate:
        names = [p.name for p in _tiffs(path)[:20]]
        for pattern in DEFAULT_PATTERNS:
            if names and all(re.search(pattern, n) for n in names):
                return cls.read_with_pattern(path, pattern)
        raise ReaderError(f"No built-in filename pattern matches the TIFFs in {path}")

    @classmethod
    def read_with_pattern(
        cls, path: Path, pattern: str, pixel_size_um: float | None = None
    ) -> Plate:
        """Read with a user-defined pattern. Files that do not match are reported, not hidden."""
        import tifffile

        compiled = _compile(pattern)
        parsed: list[tuple[tuple[int, int], int, str, int, int, Path]] = []
        unmatched = 0
        for tiff in _tiffs(path):
            match = compiled.search(tiff.name)
            if match is None:
                unmatched += 1
                continue
            groups = match.groupdict()
            if groups.get("well"):
                position = parse_well_name(groups["well"])
            else:
                position = (int(groups["row"]) - 1, int(groups["col"]) - 1)
            parsed.append(
                (
                    position,
                    int(groups.get("field") or 1),
                    groups.get("channel") or "1",
                    int(groups.get("z") or 1),
                    int(groups.get("t") or 1),
                    tiff,
                )
            )
        if not parsed:
            raise ReaderError(f"The pattern matches none of the TIFF files in {path}")

        def channel_key(value: str) -> tuple[int, int | str]:
            return (0, int(value)) if value.isdigit() else (1, value)

        channel_ids = sorted({p[2] for p in parsed}, key=channel_key)
        c_index = {c: i for i, c in enumerate(channel_ids)}
        z_index = {z: i for i, z in enumerate(sorted({p[3] for p in parsed}))}
        t_index = {t: i for i, t in enumerate(sorted({p[4] for p in parsed}))}
        shape_tcz = (len(t_index), len(c_index), len(z_index))

        with tifffile.TiffFile(parsed[0][5]) as tif:
            page = tif.pages[0]
            plane_shape = (int(page.imagelength), int(page.imagewidth))  # type: ignore[union-attr]
            dtype: np.dtype[Any] = np.dtype(page.dtype)

        grouped: dict[tuple[int, int], dict[int, dict[tuple[int, int, int], Path]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for position, field_id, channel, z, t, tiff in parsed:
            grouped[position][field_id][(t_index[t], c_index[channel], z_index[z])] = tiff

        wells: dict[tuple[int, int], Well] = {}
        for position, by_field in sorted(grouped.items()):
            fields = []
            for field_id, planes in sorted(by_field.items()):

                def load(planes: dict[tuple[int, int, int], Path] = planes) -> Any:
                    return lazy_tiff_stack(planes, shape_tcz, plane_shape, dtype)

                fields.append(
                    Field(index=field_id, load=load, sources=tuple(sorted(planes.values())))
                )
            wells[position] = Well(row=position[0], col=position[1], fields=tuple(fields))

        notes = ["Optical metadata is not available for plain TIFF folders; enter it manually."]
        if unmatched:
            notes.append(f"{unmatched} TIFF file(s) did not match the pattern and were skipped.")
        rows, cols = infer_plate_shape(max(r for r, _ in wells) + 1, max(c for _, c in wells) + 1)
        return Plate(
            name=path.name,
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells=wells,
            metadata=ImageMetadata(
                channels=tuple(
                    ChannelInfo(name=c if not c.isdigit() else f"Channel {c}") for c in channel_ids
                ),
                pixel_size_um=pixel_size_um,
                bit_depth=int(dtype.itemsize * 8),
            ),
            shape_tczyx=(*shape_tcz, *plane_shape),
            notes=tuple(notes),
            source={
                "reader": "tiff-folder",
                "path": str(path),
                "pattern": pattern,
                "pixel_size_um": pixel_size_um,
            },
        )
