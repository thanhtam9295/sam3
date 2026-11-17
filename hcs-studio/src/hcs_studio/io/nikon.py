"""Nikon NIS-Elements ``.nd2`` reader: one multi-position file or a folder of files.

Well names are taken from the XY position names when they look like wells, otherwise
from the file name (JOBS exports typically write ``WellB02_..._Seq0001.nd2``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcs_studio.io.base import READERS, PlateReader, ReaderError
from hcs_studio.model.plate import (
    ChannelInfo,
    Field,
    ImageMetadata,
    Modality,
    Plate,
    Well,
    infer_plate_shape,
    parse_well_name,
)

# Rows A-P (384) or AA-AF (1536); avoids reading "T01" or "Z03" as a well.
_WELL_TOKEN = r"(A[A-F]|[A-P])0*(\d{1,2})"
_EXPLICIT_WELL = re.compile(rf"Well[_\s-]?{_WELL_TOKEN}(?!\d)", re.IGNORECASE)
_BARE_WELL = re.compile(rf"(?:^|[_\s-]){_WELL_TOKEN}(?=$|[_\s.-])")


def well_from_text(text: str) -> tuple[int, int] | None:
    """Find a well name in a position or file name, or return ``None``."""
    match = _EXPLICIT_WELL.search(text) or _BARE_WELL.search(text)
    if match is None:
        return None
    try:
        row, col = parse_well_name(f"{match.group(1).upper()}{match.group(2)}")
    except ValueError:
        return None
    return (row, col) if col < 48 else None


def _open_nd2(path: Path) -> Any:
    import nd2

    return nd2.ND2File(path)


@dataclass(frozen=True)
class _Position:
    name: str
    x_um: float | None
    y_um: float | None


@dataclass(frozen=True)
class _FileInfo:
    path: Path
    sizes: dict[str, int]
    positions: tuple[_Position, ...]
    metadata: ImageMetadata


def _modality(flags: list[str]) -> Modality:
    lowered = [str(f).lower() for f in flags]
    if any("confocal" in f for f in lowered):
        return "confocal"
    if any("brightfield" in f for f in lowered):
        return "brightfield"
    if any("fluorescence" in f for f in lowered):
        return "widefield"
    return "unknown"


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _describe(path: Path) -> _FileInfo:
    with _open_nd2(path) as handle:
        sizes = {str(k): int(v) for k, v in handle.sizes.items()}
        if "S" in sizes:
            raise ReaderError(f"{path.name}: RGB .nd2 files are not supported")
        unknown = set(sizes) - set("TPZCYX")
        if unknown:
            raise ReaderError(f"{path.name}: unsupported axes {sorted(unknown)}")

        points: list[_Position] = []
        for loop in handle.experiment or []:
            if getattr(loop, "type", "") == "XYPosLoop":
                for point in loop.parameters.points:
                    stage = getattr(point, "stagePositionUm", None)
                    x, y = (stage[0], stage[1]) if stage is not None else (None, None)
                    points.append(_Position(name=point.name or "", x_um=x, y_um=y))
        n_positions = sizes.get("P", 1)
        if len(points) != n_positions:
            points = [_Position("", None, None)] * n_positions

        channels: list[ChannelInfo] = []
        microscope = None
        meta_channels = getattr(handle.metadata, "channels", None) or []
        for item in meta_channels:
            channels.append(
                ChannelInfo(
                    name=item.channel.name,
                    excitation_nm=_positive(item.channel.excitationLambdaNm),
                    emission_nm=_positive(item.channel.emissionLambdaNm),
                )
            )
            microscope = microscope or item.microscope
        n_channels = sizes.get("C", 1)
        if len(channels) != n_channels:
            channels = [ChannelInfo(name=f"Channel {i + 1}") for i in range(n_channels)]

        voxel = handle.voxel_size()
        bits = getattr(handle.attributes, "bitsPerComponentSignificant", None)
        flags = [f for item in meta_channels for f in (item.microscope.modalityFlags or [])]
        metadata = ImageMetadata(
            channels=tuple(channels),
            pixel_size_um=_positive(voxel.x),
            z_step_um=_positive(voxel.z) if sizes.get("Z", 1) > 1 else None,
            objective_na=_positive(getattr(microscope, "objectiveNumericalAperture", None)),
            objective_name=getattr(microscope, "objectiveName", None),
            magnification=_positive(getattr(microscope, "objectiveMagnification", None)),
            immersion_ri=_positive(getattr(microscope, "immersionRefractiveIndex", None)),
            modality=_modality(flags),
            pinhole_um=_positive(getattr(microscope, "pinholeDiameterUm", None)),
            bit_depth=int(bits) if bits else None,
        )
    return _FileInfo(path=path, sizes=sizes, positions=tuple(points), metadata=metadata)


def _loader(path: Path, position: int) -> Any:
    def load() -> Any:
        with _open_nd2(path) as handle:
            axes = [str(a) for a in handle.sizes]
            array = handle.to_dask()  # re-opens the file on demand when chunks are computed
        if "P" in axes:
            index: list[Any] = [slice(None)] * len(axes)
            index[axes.index("P")] = position
            array = array[tuple(index)]
            axes.remove("P")
        for axis in "TCZ":
            if axis not in axes:
                array = array[None]
                axes.insert(0, axis)
        return array.transpose([axes.index(a) for a in "TCZYX"])

    return load


@READERS.register("nikon")
class NikonReader(PlateReader):
    vendor = "Nikon"
    priority = 10

    @classmethod
    def _files(cls, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() == ".nd2" else []
        return sorted(p for p in path.iterdir() if p.suffix.lower() == ".nd2" and p.is_file())

    @classmethod
    def detect(cls, path: Path) -> bool:
        return bool(cls._files(path))

    @classmethod
    def read(cls, path: Path) -> Plate:
        files = cls._files(path)
        if not files:
            raise ReaderError(f"No .nd2 files in {path}")
        infos = [_describe(f) for f in files]
        reference = infos[0]
        notes: list[str] = []

        def tczyx(info: _FileInfo) -> tuple[int, int, int, int, int]:
            s = info.sizes
            return (s.get("T", 1), s.get("C", 1), s.get("Z", 1), s["Y"], s["X"])

        shape = tczyx(reference)
        differing = [i.path.name for i in infos if tczyx(i) != shape]
        if differing:
            notes.append(
                f"{len(differing)} file(s) have different dimensions than {reference.path.name} "
                f"(first: {differing[0]}); plate-level dimensions describe the first file."
            )

        # (well or None, file info, position index)
        entries: list[tuple[tuple[int, int] | None, _FileInfo, int]] = []
        for info in infos:
            from_file = well_from_text(info.path.stem)
            for p, position in enumerate(info.positions):
                entries.append((well_from_text(position.name) or from_file, info, p))

        wells: dict[tuple[int, int], Well] = {}
        if all(well is not None for well, _, _ in entries):
            grouped: dict[tuple[int, int], list[Field]] = {}
            for well, info, p in entries:
                assert well is not None
                fields = grouped.setdefault(well, [])
                pos = info.positions[p]
                fields.append(
                    Field(
                        index=len(fields) + 1,
                        load=_loader(info.path, p),
                        x_um=pos.x_um,
                        y_um=pos.y_um,
                    )
                )
            wells = {
                key: Well(row=key[0], col=key[1], fields=tuple(value))
                for key, value in sorted(grouped.items())
            }
            rows, cols = infer_plate_shape(
                max(r for r, _ in wells) + 1, max(c for _, c in wells) + 1
            )
        else:
            notes.append(
                "No well names found in the position or file names; "
                "each file is shown as one well, in file-name order."
            )
            rows, cols = infer_plate_shape(1, 1)
            for candidate in ((8, 12), (16, 24), (32, 48)):
                rows, cols = candidate
                if len(infos) <= rows * cols:
                    break
            else:
                raise ReaderError(f"{len(infos)} files without well names do not fit on a plate")
            for i, info in enumerate(infos):
                row, col = divmod(i, cols)
                fields_ = tuple(
                    Field(index=p + 1, load=_loader(info.path, p), x_um=pos.x_um, y_um=pos.y_um)
                    for p, pos in enumerate(info.positions)
                )
                wells[(row, col)] = Well(row=row, col=col, fields=fields_, label=info.path.stem)

        return Plate(
            name=path.stem if path.is_file() else path.name,
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells=wells,
            metadata=reference.metadata,
            shape_tczyx=shape,
            notes=tuple(notes),
        )
