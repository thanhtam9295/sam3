"""Revvity / PerkinElmer Operetta CLS reader (Harmony export, ``Index.idx.xml``)."""

from __future__ import annotations

import itertools
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.io.base import READERS, PlateReader, ReaderError, lazy_tiff_stack
from hcs_studio.model.plate import (
    ChannelInfo,
    Field,
    ImageMetadata,
    Modality,
    Plate,
    Well,
    infer_plate_shape,
)

INDEX_NAMES = ("Index.idx.xml", "Index.ref.xml", "Index.xml")

_TO_UM = {"m": 1e6, "mm": 1e3, "µm": 1.0, "um": 1.0, "nm": 1e-3}
_TO_MS = {"s": 1e3, "ms": 1.0}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_index(path: Path) -> Path | None:
    """Locate the Harmony index in ``path`` or its ``Images`` subfolder."""
    if path.is_file():
        return path if path.name in INDEX_NAMES else None
    for folder in (path / "Images", path):
        for name in INDEX_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def _number(element: ET.Element | None, scale: dict[str, float] | None = None) -> float | None:
    if element is None or element.text is None or not element.text.strip():
        return None
    try:
        value = float(element.text)
    except ValueError:
        return None
    if scale is None:
        return value
    unit = element.get("Unit", "")
    if unit not in scale:
        raise ReaderError(f"<{_local(element.tag)}>: unsupported unit {unit!r}")
    return value * scale[unit]


def _record(element: ET.Element) -> dict[str, ET.Element]:
    return {_local(child.tag): child for child in element}


def _text(record: dict[str, ET.Element], key: str) -> str | None:
    element = record.get(key)
    return element.text.strip() if element is not None and element.text else None


def _modality(acquisition_type: str | None, channel_type: str | None) -> Modality:
    acq = (acquisition_type or "").lower()
    if "nonconfocal" in acq or "widefield" in acq:
        return "brightfield" if (channel_type or "").lower() == "brightfield" else "widefield"
    if "confocal" in acq:
        return "confocal"
    return "unknown"


@READERS.register("operetta")
class OperettaReader(PlateReader):
    vendor = "Operetta"
    priority = 30

    @classmethod
    def detect(cls, path: Path) -> bool:
        return find_index(path) is not None

    @classmethod
    def read(cls, path: Path) -> Plate:
        index = find_index(path)
        if index is None:
            raise ReaderError(f"No Harmony index file found in {path}")
        image_dir = index.parent

        plate_info: dict[str, str] = {}
        channel_defaults: dict[str, dict[str, ET.Element]] = defaultdict(dict)
        images: list[dict[str, Any]] = []
        in_images = False

        # Streamed parse: a full 384-well index can be hundreds of MB.
        for event, element in ET.iterparse(index, events=("start", "end")):
            tag = _local(element.tag)
            if event == "start":
                if tag == "Images":
                    in_images = True
                continue
            if tag == "Images":
                in_images = False
            elif tag == "Plate" and len(element):
                record = _record(element)
                for key in ("PlateID", "PlateTypeName", "PlateRows", "PlateColumns"):
                    value = _text(record, key)
                    if value:
                        plate_info[key] = value
            elif tag == "Entry" and element.get("ChannelID"):
                # Harmony ≥ 5 stores per-channel values once, under <Maps>.
                channel_defaults[element.get("ChannelID", "")].update(_record(element))
            elif tag == "Image" and in_images:
                record = _record(element)
                merged = {**channel_defaults.get(_text(record, "ChannelID") or "", {}), **record}
                parsed = cls._parse_image(merged)
                if parsed is not None:
                    images.append(parsed)
                element.clear()

        if not images:
            raise ReaderError(f"{index.name} lists no usable images")
        return cls._build_plate(path, image_dir, plate_info, images)

    @staticmethod
    def _parse_image(record: dict[str, ET.Element]) -> dict[str, Any] | None:
        url = _text(record, "URL")
        state = _text(record, "State")
        if not url or (state is not None and state.lower() != "ok"):
            return None
        try:
            return {
                "url": url,
                "row": int(_text(record, "Row") or ""),
                "col": int(_text(record, "Col") or ""),
                "field": int(_text(record, "FieldID") or "1"),
                "plane": int(_text(record, "PlaneID") or "1"),
                "time": int(_text(record, "TimepointID") or "0"),
                "channel": int(_text(record, "ChannelID") or "1"),
                "channel_name": _text(record, "ChannelName"),
                "acquisition": _text(record, "AcquisitionType"),
                "channel_type": _text(record, "ChannelType"),
                "res_x": _number(record.get("ImageResolutionX"), _TO_UM),
                "res_y": _number(record.get("ImageResolutionY"), _TO_UM),
                "size_x": _number(record.get("ImageSizeX")),
                "size_y": _number(record.get("ImageSizeY")),
                "pos_x": _number(record.get("PositionX"), _TO_UM),
                "pos_y": _number(record.get("PositionY"), _TO_UM),
                "pos_z": _number(record.get("PositionZ"), _TO_UM),
                "excitation": _number(record.get("MainExcitationWavelength")),
                "emission": _number(record.get("MainEmissionWavelength")),
                "magnification": _number(record.get("ObjectiveMagnification")),
                "na": _number(record.get("ObjectiveNA")),
                "exposure_ms": _number(record.get("ExposureTime"), _TO_MS),
                "max_intensity": _number(record.get("MaxIntensity")),
            }
        except ValueError as exc:
            raise ReaderError(f"Image entry {url!r} has a malformed index value: {exc}") from None

    @classmethod
    def _build_plate(
        cls, path: Path, image_dir: Path, plate_info: dict[str, str], images: list[dict[str, Any]]
    ) -> Plate:
        first = images[0]
        if first["size_x"] is None or first["size_y"] is None:
            raise ReaderError("Index has no ImageSizeX / ImageSizeY")
        plane_shape = (int(first["size_y"]), int(first["size_x"]))
        channel_ids = sorted({im["channel"] for im in images})
        plane_ids = sorted({im["plane"] for im in images})
        time_ids = sorted({im["time"] for im in images})
        c_index = {cid: i for i, cid in enumerate(channel_ids)}
        z_index = {pid: i for i, pid in enumerate(plane_ids)}
        t_index = {tid: i for i, tid in enumerate(time_ids)}
        shape_tcz = (len(time_ids), len(channel_ids), len(plane_ids))

        max_intensity = max((im["max_intensity"] or 0) for im in images)
        dtype: np.dtype[Any] = np.dtype(np.uint8 if 0 < max_intensity <= 255 else np.uint16)

        channels = []
        for cid in channel_ids:
            sample = next(im for im in images if im["channel"] == cid)
            channels.append(
                ChannelInfo(
                    name=sample["channel_name"] or f"Channel {cid}",
                    excitation_nm=sample["excitation"],
                    emission_nm=sample["emission"],
                    exposure_ms=sample["exposure_ms"],
                )
            )

        # Z step: median spacing between consecutive planes of one field and channel.
        z_step = None
        if len(plane_ids) > 1:
            key = (first["row"], first["col"], first["field"], first["channel"], first["time"])
            zs = sorted(
                (im["plane"], im["pos_z"])
                for im in images
                if (im["row"], im["col"], im["field"], im["channel"], im["time"]) == key
                and im["pos_z"] is not None
            )
            steps = [abs(b[1] - a[1]) for a, b in itertools.pairwise(zs) if b[0] - a[0] == 1]
            if steps:
                z_step = statistics.median(steps)

        notes: list[str] = []
        if first["res_x"] and first["res_y"] and abs(first["res_x"] - first["res_y"]) > 1e-9:
            notes.append("Pixels are not square; the X pixel size is used.")
        modalities = {_modality(im["acquisition"], im["channel_type"]) for im in images}
        fluor = modalities - {"brightfield"}
        if len(fluor) > 1:
            notes.append("Channels were acquired in different modes (confocal and non-confocal).")
        modality: Modality = next(iter(fluor)) if len(fluor) == 1 else "unknown"
        if not fluor and modalities:
            modality = "brightfield"

        metadata = ImageMetadata(
            channels=tuple(channels),
            pixel_size_um=first["res_x"],
            z_step_um=z_step,
            objective_na=first["na"],
            magnification=first["magnification"],
            modality=modality,
            bit_depth=int(dtype.itemsize * 8),
        )

        grouped: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for im in images:
            grouped[(im["row"] - 1, im["col"] - 1)][im["field"]].append(im)

        wells: dict[tuple[int, int], Well] = {}
        for (row, col), by_field in sorted(grouped.items()):
            fields = []
            for field_id, entries in sorted(by_field.items()):
                files = {
                    (t_index[e["time"]], c_index[e["channel"]], z_index[e["plane"]]): image_dir
                    / e["url"]
                    for e in entries
                }

                def load(files: dict[tuple[int, int, int], Path] = files) -> Any:
                    return lazy_tiff_stack(files, shape_tcz, plane_shape, dtype)

                fields.append(
                    Field(
                        index=field_id,
                        load=load,
                        x_um=entries[0]["pos_x"],
                        y_um=entries[0]["pos_y"],
                        sources=tuple(sorted(files.values())),
                    )
                )
            wells[(row, col)] = Well(row=row, col=col, fields=tuple(fields))

        try:
            rows, cols = int(plate_info["PlateRows"]), int(plate_info["PlateColumns"])
        except (KeyError, ValueError):
            rows, cols = infer_plate_shape(
                max(r for r, _ in wells) + 1, max(c for _, c in wells) + 1
            )
            notes.append("Plate size not stated in the index; inferred from the wells present.")

        root = path if path.is_dir() else path.parent
        if root.name == "Images":
            root = root.parent
        return Plate(
            name=root.name,
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells=wells,
            metadata=metadata,
            barcode=plate_info.get("PlateID"),
            shape_tczyx=(*shape_tcz, *plane_shape),
            notes=tuple(notes),
        )
