"""Is a plate completely written? One checker per export format (Phase 19).

Supporting a new imager means adding a reader (``io/``) and a checker here (register it in
:data:`CHECKERS` under the reader's name); nothing else. Each checker answers
:class:`Completeness` and never modifies the data:

- **ImageXpress** (MetaXpress export): the number of image TIFFs the ``.HTD`` file implies
  (selected wells × selected sites × wavelengths × time points × Z planes, plus the projection
  MetaXpress saves with a Z series) is present, and every TIFF can be opened;
- **Operetta** (Harmony export): the index XML is present and parses, and every image it
  references exists and can be opened;
- **Nikon** ``.nd2``: every file opens and its last frame can be read;
- **OME-Zarr** and other formats: the plate opens with its reader.

The watcher (:mod:`hcs_studio.automation.watcher`) also waits until nothing changed for a quiet
period, so a plate still being copied is never started.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcs_studio.io.base import READERS, ReaderError, detect_reader


@dataclass(frozen=True)
class Completeness:
    complete: bool
    reason: str = ""
    expected: int | None = None
    found: int | None = None
    reader: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def _tiff_readable(path: Path) -> bool:
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            page: Any = tif.pages[0]
            end = int(page.dataoffsets[0]) + int(page.databytecounts[0])
            return bool(page.shape) and path.stat().st_size >= end  # pixel data fully written
    except Exception:
        return False


def _selected(values: list[str]) -> int:
    return sum(1 for v in values if v.strip().upper() == "TRUE")


def imagexpress(path: Path) -> Completeness:
    from hcs_studio.io.imagexpress import _FILE_RE, ImageXpressReader, find_htd, parse_htd

    htd = find_htd(path)
    if htd is None:
        return Completeness(False, "no .HTD file yet", reader="imagexpress")
    try:
        entries = parse_htd(htd)
    except (ReaderError, OSError) as exc:
        return Completeness(False, f"HTD not readable yet: {exc}", reader="imagexpress")

    def number(key: str, default: int) -> int:
        try:
            return int(entries[key][0])
        except (KeyError, IndexError, ValueError):
            return default

    def flag(key: str) -> bool:
        return (entries.get(key) or ["FALSE"])[0].strip().upper() == "TRUE"

    notes = []
    rows = [k for k in entries if re.fullmatch(r"WellsSelection\d+", k)]
    if rows:
        wells = sum(_selected(entries[k]) for k in rows)
    else:
        wells = number("XWells", 0) * number("YWells", 0)
        notes.append("the HTD selects no wells: every well of the plate is expected")
    sites = 1
    if flag("Sites"):
        selection = [k for k in entries if re.fullmatch(r"SiteSelection\d+", k)]
        sites = (
            sum(_selected(entries[k]) for k in selection)
            if selection
            else number("XSites", 1) * number("YSites", 1)
        )
    waves = number("NWavelengths", 1) if flag("Waves") else 1
    times = max(1, number("TimePoints", 1))
    z = number("ZSteps", 1) if flag("ZSeries") else 1
    planes = z + 1 if z > 1 else 1  # MetaXpress also saves a projection with a Z series
    expected = wells * sites * waves * times * planes
    tiffs = [
        p
        for p, _, _ in ImageXpressReader._scan(htd.parent)
        if _FILE_RE.match(p.name) and "_thumb" not in p.name.lower()
    ]
    if len(tiffs) < expected:
        return Completeness(
            False, f"{len(tiffs)} of {expected} images written", expected, len(tiffs),
            "imagexpress", tuple(notes),
        )  # fmt: skip
    unreadable = [p.name for p in tiffs if not _tiff_readable(p)]
    if unreadable:
        return Completeness(
            False, f"{len(unreadable)} image(s) not readable yet, e.g. {unreadable[0]}",
            expected, len(tiffs), "imagexpress", tuple(notes),
        )  # fmt: skip
    return Completeness(True, "", expected, len(tiffs), "imagexpress", tuple(notes))


def operetta(path: Path) -> Completeness:
    import xml.etree.ElementTree as ET

    from hcs_studio.io.operetta import find_index

    index = find_index(path)
    if index is None:
        return Completeness(False, "no index XML yet", reader="operetta")
    try:
        tree = ET.parse(index)
    except (ET.ParseError, OSError) as exc:
        return Completeness(False, f"index XML not complete yet: {exc}", reader="operetta")
    urls = [
        (element.text or "").strip()
        for element in tree.iter()
        if element.tag.rsplit("}", 1)[-1] == "URL" and (element.text or "").strip()
    ]
    if not urls:
        return Completeness(False, "the index lists no images", reader="operetta")
    missing = [u for u in urls if not (index.parent / u).exists()]
    if missing:
        return Completeness(
            False, f"{len(missing)} of {len(urls)} images missing, e.g. {missing[0]}",
            len(urls), len(urls) - len(missing), "operetta",
        )  # fmt: skip
    unreadable = [u for u in urls if not _tiff_readable(index.parent / u)]
    if unreadable:
        return Completeness(
            False, f"{len(unreadable)} image(s) not readable yet, e.g. {unreadable[0]}",
            len(urls), len(urls), "operetta",
        )  # fmt: skip
    return Completeness(True, "", len(urls), len(urls), "operetta")


def nikon(path: Path) -> Completeness:
    from hcs_studio.io.nikon import NikonReader, _open_nd2

    files = NikonReader._files(path)
    if not files:
        return Completeness(False, "no .nd2 file yet", reader="nikon")
    for file in files:
        try:
            handle = _open_nd2(file)
            try:
                count = int(getattr(handle.attributes, "sequenceCount", 0) or 0)
                if count < 1:
                    return Completeness(False, f"{file.name}: no frames yet", reader="nikon")
                handle.read_frame(count - 1)
            finally:
                handle.close()
        except Exception as exc:
            return Completeness(
                False, f"{file.name} is still being written or not readable ({exc})", reader="nikon"
            )
    return Completeness(True, "", len(files), len(files), "nikon")


def generic(path: Path, reader: str) -> Completeness:
    from hcs_studio.io.base import open_plate

    try:
        plate = open_plate(path)
    except Exception as exc:
        return Completeness(False, f"does not open yet: {exc}", reader=reader)
    return Completeness(True, "", plate.n_fields, plate.n_fields, reader)


CHECKERS: dict[str, Callable[[Path], Completeness]] = {
    "imagexpress": imagexpress,
    "operetta": operetta,
    "nikon": nikon,
}


def check(path: Path) -> Completeness:
    """Completeness of a plate export, by the reader that recognises it."""
    path = Path(path)
    if not path.exists():
        return Completeness(False, "does not exist")
    reader = detect_reader(path)
    if reader is None:
        return Completeness(False, "not a plate export (no reader recognises it yet)")
    name = next((n for n in READERS if READERS.get(n) is reader), reader.__name__)
    checker = CHECKERS.get(name)
    return checker(path) if checker is not None else generic(path, name)
