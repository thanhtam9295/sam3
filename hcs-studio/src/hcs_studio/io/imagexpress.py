"""Molecular Devices ImageXpress reader (MetaXpress export: ``.HTD`` + TIFFs)."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.io.base import READERS, PlateReader, ReaderError, lazy_tiff_stack
from hcs_studio.model.plate import ChannelInfo, Field, ImageMetadata, Plate, Well, parse_well_name

# <Plate>_<Well>[_s<site>][_w<wave>][_z<n>][<GUID>].tif ; thumbnails contain "_thumb".
_FILE_RE = re.compile(
    r"^(?P<plate>.+)_(?P<well>[A-Z]{1,2}\d{2})"
    r"(?:_s(?P<site>\d+))?(?:_w(?P<wave>\d+))?(?:_z(?P<z>\d+))?"
    r"(?P<guid>[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12})?\.tiff?$",
    re.IGNORECASE,
)
_SUBDIR_RE = re.compile(r"^(TimePoint|ZStep)_(\d+)$", re.IGNORECASE)


def find_htd(path: Path) -> Path | None:
    if path.is_file():
        return path if path.suffix.lower() == ".htd" else None
    matches = sorted(p for p in path.iterdir() if p.suffix.lower() == ".htd" and p.is_file())
    return matches[0] if matches else None


def parse_htd(path: Path) -> dict[str, list[str]]:
    """Parse the HTD descriptor into ``{key: [values...]}``."""
    entries: dict[str, list[str]] = {}
    with path.open(newline="", encoding="latin-1") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if not row:
                continue
            entries[row[0].strip()] = [v.strip() for v in row[1:]]
    if "XWells" not in entries or "YWells" not in entries:
        raise ReaderError(f"{path.name}: not a MetaXpress HTD file (no XWells / YWells)")
    return entries


def _int(entries: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(entries[key][0])
    except (KeyError, IndexError, ValueError):
        return default


def _flag(entries: dict[str, list[str]], key: str) -> bool:
    return bool(entries.get(key)) and entries[key][0].upper() == "TRUE"


def _plane_metadata(path: Path) -> dict[str, Any]:
    """MetaSeries ``PlaneInfo`` of one TIFF, or an empty dict."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        info: dict[str, Any] = {
            "shape": (int(page.imagelength), int(page.imagewidth)),  # type: ignore[union-attr]
            "dtype": np.dtype(page.dtype),
        }
        meta = tif.metaseries_metadata if tif.is_metaseries else None
    if meta:
        info.update(meta.get("PlaneInfo", {}))
    return info


@READERS.register("imagexpress")
class ImageXpressReader(PlateReader):
    vendor = "ImageXpress"
    priority = 20

    @classmethod
    def detect(cls, path: Path) -> bool:
        return (path.is_dir() or path.suffix.lower() == ".htd") and find_htd(path) is not None

    @classmethod
    def read(cls, path: Path) -> Plate:
        htd = find_htd(path)
        if htd is None:
            raise ReaderError(f"No .HTD file found in {path}")
        root = htd.parent
        entries = parse_htd(htd)
        cols, rows = _int(entries, "XWells", 0), _int(entries, "YWells", 0)
        n_waves = _int(entries, "NWavelengths", 1) if _flag(entries, "Waves") else 1
        wave_names = [
            (entries.get(f"WaveName{i}") or [f"w{i}"])[0] or f"w{i}" for i in range(1, n_waves + 1)
        ]

        # files[(row, col)][site][(t, c, z)] = path
        files: dict[tuple[int, int], dict[int, dict[tuple[int, int, int], Path]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        time_ids: set[int] = set()
        z_ids: set[int] = set()
        found: list[tuple[tuple[int, int], int, int, int, int, Path]] = []
        for tiff, time_id, z_dir in cls._scan(root):
            match = _FILE_RE.match(tiff.name)
            if match is None or "_thumb" in tiff.name.lower():
                continue
            position = parse_well_name(match["well"])
            site = int(match["site"] or 1)
            wave = int(match["wave"] or 1)
            z_id = int(match["z"]) if match["z"] else z_dir
            found.append((position, site, time_id, wave, z_id, tiff))
            time_ids.add(time_id)
            z_ids.add(z_id)
        if not found:
            raise ReaderError(f"{htd.name}: no image files matching the MetaXpress naming pattern")

        notes: list[str] = []
        # With a saved Z series MetaXpress also writes a projection (ZStep_0 / no step); skip it.
        if len(z_ids) > 1 and 0 in z_ids:
            z_ids.discard(0)
            found = [f for f in found if f[4] != 0]
            notes.append("Z-series found; the saved 2D projection images are ignored.")
        t_index = {t: i for i, t in enumerate(sorted(time_ids))}
        z_index = {z: i for i, z in enumerate(sorted(z_ids))}
        for position, site, time_id, wave, z_id, tiff in found:
            if not 1 <= wave <= n_waves:
                raise ReaderError(f"{tiff.name}: wavelength {wave} is not declared in {htd.name}")
            files[position][site][(t_index[time_id], wave - 1, z_index[z_id])] = tiff
        shape_tcz = (len(t_index), n_waves, len(z_index))

        # Optical metadata: one representative TIFF per wavelength.
        per_wave: dict[int, dict[str, Any]] = {}
        for _position, _site, _t, wave, _z, tiff in found:
            if wave not in per_wave:
                per_wave[wave] = _plane_metadata(tiff)
            if len(per_wave) == n_waves:
                break
        reference = per_wave[min(per_wave)]
        plane_shape: tuple[int, int] = reference["shape"]
        dtype: np.dtype[Any] = reference["dtype"]

        channels = []
        for i, name in enumerate(wave_names, start=1):
            info = per_wave.get(i, {})
            exposure = info.get("Exposure Time") or info.get("camera-exposure-time")
            if isinstance(exposure, str):
                number = re.match(r"[\d.]+", exposure)
                exposure = float(number.group()) if number else None
            wavelength = info.get("wavelength")
            channels.append(
                ChannelInfo(
                    name=name,
                    emission_nm=float(wavelength) if wavelength else None,
                    exposure_ms=float(exposure) if exposure is not None else None,
                )
            )

        pixel_size = reference.get("spatial-calibration-x")
        if not reference.get("spatial-calibration-state", True):
            pixel_size = None
        if pixel_size is None:
            notes.append("No spatial calibration in the TIFF tags; set the pixel size manually.")
        z_step = None
        if len(z_index) > 1:
            z_step = reference.get("z-step") or reference.get("_Z_Step_")
            if z_step is None:
                notes.append("Z step not found in the TIFF tags; set it manually.")
        magnification = reference.get("_MagSetting_")
        if isinstance(magnification, str):
            number = re.match(r"[\d.]+", magnification)
            magnification = float(number.group()) if number else None
        bits = reference.get("bits-per-pixel")

        metadata = ImageMetadata(
            channels=tuple(channels),
            pixel_size_um=float(pixel_size) if pixel_size else None,
            z_step_um=float(z_step) if z_step else None,
            objective_na=float(reference["_MagNA_"]) if reference.get("_MagNA_") else None,
            magnification=float(magnification) if magnification else None,
            immersion_ri=float(reference["_MagRI_"]) if reference.get("_MagRI_") else None,
            modality="unknown",
            bit_depth=int(bits) if bits else int(dtype.itemsize * 8),
        )
        notes.append("Confocal vs widefield is not recorded by MetaXpress; set the modality.")

        # Nominal site positions from the site grid (true stage positions are per-TIFF).
        x_sites = _int(entries, "XSites", 1) if _flag(entries, "Sites") else 1
        step_x = plane_shape[1] * metadata.pixel_size_um if metadata.pixel_size_um else None
        step_y = plane_shape[0] * metadata.pixel_size_um if metadata.pixel_size_um else None

        wells: dict[tuple[int, int], Well] = {}
        for position, by_site in sorted(files.items()):
            fields = []
            for site, planes in sorted(by_site.items()):

                def load(planes: dict[tuple[int, int, int], Path] = planes) -> Any:
                    return lazy_tiff_stack(planes, shape_tcz, plane_shape, dtype)

                gx, gy = (site - 1) % x_sites, (site - 1) // x_sites
                fields.append(
                    Field(
                        index=site,
                        load=load,
                        x_um=gx * step_x if step_x is not None else None,
                        y_um=gy * step_y if step_y is not None else None,
                    )
                )
            wells[position] = Well(row=position[0], col=position[1], fields=tuple(fields))

        return Plate(
            name=htd.stem,
            path=path,
            vendor=cls.vendor,
            rows=rows,
            cols=cols,
            wells=wells,
            metadata=metadata,
            barcode=(entries.get("Description") or [None])[0] or None,  # type: ignore[list-item]
            shape_tczyx=(*shape_tcz, *plane_shape),
            notes=tuple(notes),
        )

    @staticmethod
    def _scan(root: Path) -> list[tuple[Path, int, int]]:
        """All TIFFs as (path, timepoint id, z-step folder id), walking TimePoint_/ZStep_ dirs."""
        results: list[tuple[Path, int, int]] = []

        def walk(folder: Path, time_id: int, z_id: int) -> None:
            for entry in sorted(folder.iterdir()):
                if entry.is_dir():
                    match = _SUBDIR_RE.match(entry.name)
                    if match is None:
                        continue
                    number = int(match.group(2))
                    if match.group(1).lower() == "timepoint":
                        walk(entry, number, z_id)
                    else:
                        walk(entry, time_id, number)
                elif entry.suffix.lower() in (".tif", ".tiff"):
                    results.append((entry, time_id, z_id))

        walk(root, 1, 0)
        return results
