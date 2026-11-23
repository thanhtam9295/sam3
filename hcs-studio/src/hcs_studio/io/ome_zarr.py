"""OME-Zarr reader for the HCS (plate) layout, NGFF 0.4 and 0.5 metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da

from hcs_studio.io.base import READERS, PlateReader, ReaderError
from hcs_studio.model.plate import ChannelInfo, Field, ImageMetadata, Plate, Well, parse_well_name

_UNIT_UM = {"micrometer": 1.0, "nanometer": 1e-3, "millimeter": 1e3, "meter": 1e6}
_UNIT_S = {"second": 1.0, "millisecond": 1e-3, "minute": 60.0, "hour": 3600.0, "day": 86400.0}


def _attrs(group: Any) -> dict[str, Any]:
    attrs = dict(group.attrs)
    ome = attrs.get("ome")
    return dict(ome) if isinstance(ome, dict) else attrs


def _is_zarr_group(path: Path) -> bool:
    return path.is_dir() and any((path / n).is_file() for n in (".zattrs", "zarr.json", ".zgroup"))


@READERS.register("ome-zarr")
class OmeZarrPlateReader(PlateReader):
    vendor = "OME-Zarr"
    priority = 40

    @classmethod
    def detect(cls, path: Path) -> bool:
        if not _is_zarr_group(path):
            return False
        for name in (".zattrs", "zarr.json"):
            file = path / name
            if file.is_file() and '"plate"' in file.read_text(encoding="utf-8", errors="ignore"):
                return True
        return False

    @classmethod
    def read(cls, path: Path) -> Plate:
        import zarr

        root = zarr.open_group(str(path), mode="r")
        plate_meta = _attrs(root).get("plate")
        if not isinstance(plate_meta, dict):
            raise ReaderError(f"{path.name}: no OME-Zarr plate metadata")
        rows, cols = len(plate_meta.get("rows", [])), len(plate_meta.get("columns", []))

        wells: dict[tuple[int, int], Well] = {}
        metadata: ImageMetadata | None = None
        shape: tuple[int, int, int, int, int] | None = None
        time_step: float | None = None
        for entry in plate_meta.get("wells", []):
            well_path = str(entry["path"])
            row_name, col_name = well_path.split("/")
            position = parse_well_name(f"{row_name}{col_name}")
            well_group = root[well_path]
            images = _attrs(well_group).get("well", {}).get("images", [])
            fields = []
            for i, image in enumerate(images, start=1):
                image_path = f"{well_path}/{image['path']}"
                array_path, axes, scale, channels = cls._image_info(root[image_path])
                full = f"{path}/{image_path}/{array_path}"

                def load(full: str = full, axes: tuple[str, ...] = axes) -> Any:
                    array = da.from_zarr(full)
                    names = list(axes)
                    for axis in "tcz":
                        if axis not in names:
                            array = array[None]
                            names.insert(0, axis)
                    return array.transpose([names.index(a) for a in "tczyx"])

                fields.append(Field(index=i, load=load))
                if metadata is None:
                    sizes = dict(zip(axes, root[f"{image_path}/{array_path}"].shape, strict=True))
                    shape = (
                        sizes.get("t", 1),
                        sizes.get("c", 1),
                        sizes.get("z", 1),
                        sizes["y"],
                        sizes["x"],
                    )
                    if len(channels) != shape[1]:
                        channels = [f"Channel {c + 1}" for c in range(shape[1])]
                    metadata = ImageMetadata(
                        channels=tuple(ChannelInfo(name=c) for c in channels),
                        pixel_size_um=scale.get("x"),
                        z_step_um=scale.get("z") if shape[2] > 1 else None,
                    )
                    time_step = scale.get("t")
            if fields:
                wells[position] = Well(row=position[0], col=position[1], fields=tuple(fields))
        if not wells or metadata is None:
            raise ReaderError(f"{path.name}: the plate contains no images")
        # times: an explicit list (irregular intervals) or the NGFF scale of the t axis
        extra = _attrs(root).get("hcs_studio") or {}
        times = extra.get("time_points_s")
        n_t = shape[0] if shape else 1
        time_points: tuple[float, ...] | None = None
        if isinstance(times, list) and len(times) == n_t:
            time_points = tuple(float(x) for x in times)
        elif n_t > 1 and time_step:
            time_points = tuple(i * time_step for i in range(n_t))
        return Plate(
            name=str(plate_meta.get("name") or path.stem),
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells=wells,
            metadata=metadata,
            shape_tczyx=shape,
            time_points_s=time_points,
            acquired_at=str(extra["acquired_at"]) if extra.get("acquired_at") else None,
        )

    @staticmethod
    def _image_info(group: Any) -> tuple[str, tuple[str, ...], dict[str, float], list[str]]:
        attrs = _attrs(group)
        try:
            multiscale = attrs["multiscales"][0]
            dataset = multiscale["datasets"][0]
        except (KeyError, IndexError):
            raise ReaderError("OME-Zarr image without multiscales metadata") from None
        axes_meta = multiscale.get("axes") or []
        axes = tuple(str(a["name"] if isinstance(a, dict) else a).lower() for a in axes_meta)
        if not axes or set(axes) - set("tczyx"):
            raise ReaderError(f"Unsupported OME-Zarr axes {axes!r}")
        scale: dict[str, float] = {}
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                for axis, value in zip(axes_meta, transform["scale"], strict=True):
                    if isinstance(axis, dict) and axis.get("type") == "time":
                        unit = axis.get("unit", "second")
                        if unit in _UNIT_S:
                            scale["t"] = float(value) * _UNIT_S[unit]
                    if isinstance(axis, dict) and axis.get("type") == "space":
                        unit = axis.get("unit", "micrometer")
                        if unit not in _UNIT_UM:
                            raise ReaderError(f"Unsupported OME-Zarr unit {unit!r}")
                        scale[str(axis["name"]).lower()] = float(value) * _UNIT_UM[unit]
        channels = [str(c.get("label", "")) for c in attrs.get("omero", {}).get("channels", [])]
        return str(dataset["path"]), axes, scale, channels
