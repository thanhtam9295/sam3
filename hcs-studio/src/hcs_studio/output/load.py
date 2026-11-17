"""Read a field's images and masks back from a run folder (for the viewer)."""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing.base import ImageData, LabelSet


def run_metadata(folder: Path) -> ImageMetadata:
    """The plate metadata recorded in the run manifest (with the user's overrides)."""
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    stored = dict(manifest["plate"]["metadata"])
    stored["channels"] = tuple(ChannelInfo(**c) for c in stored.get("channels", []))
    known = {f.name for f in dataclass_fields(ImageMetadata)}
    return ImageMetadata(**{k: v for k, v in stored.items() if k in known})


def _read_tczyx(path: Path) -> tuple[np.ndarray[Any, Any], list[str]]:
    import re

    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = np.asarray(series.asarray())
        axes = series.axes
        names = re.findall(r'<Channel[^>]*Name="([^"]*)"', tif.ome_metadata or "")
    for axis in "TCZYX":
        if axis not in axes:
            data = data[None]
            axes = axis + axes
    order = [axes.index(a) for a in "TCZYX"]
    return data.transpose(order), names


def load_field_output(folder: Path, well: str, index: int, stage: str = "Corrected") -> ImageData:
    """The corrected (linear) or enhanced image of a field, with its masks, first timepoint."""
    summary = json.loads(
        (folder / "wells" / well / "well_summary.json").read_text(encoding="utf-8")
    )
    entry = next(f for f in summary["fields"] if int(f["field"]) == index)
    files = entry["files"]
    if "corrected" not in files:
        raise ValueError(
            "this run did not save images (run option 'save corrected images' was off); "
            "open the raw field from the Plate tab instead"
        )
    key = (
        "enhanced_display_only"
        if stage == "Enhanced" and "enhanced_display_only" in files
        else "corrected"
    )
    data, names = _read_tczyx(folder / "wells" / well / files[key])
    metadata = run_metadata(folder)
    if names and len(names) == data.shape[1]:
        channels = tuple(
            next((c for c in metadata.channels if c.name == n), ChannelInfo(n)) for n in names
        )
        metadata = ImageMetadata(**{**metadata.__dict__, "channels": channels})
    if data.shape[2] == 1:
        metadata = ImageMetadata(**{**metadata.__dict__, "z_step_um": metadata.z_step_um})
    labels = []
    for name, value in files.items():
        if name.startswith("mask_"):
            masks, _ = _read_tczyx(folder / "wells" / well / value)
            labels.append(LabelSet(name[5:], np.asarray(masks[0, 0], dtype=np.int32)))
    image = ImageData(np.asarray(data[0], dtype=np.float32), metadata, linear=key == "corrected")
    for label_set in labels:
        if label_set.data.shape == image.data.shape[1:]:
            image = image.with_labels(label_set, "loaded")
    return image
