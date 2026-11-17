"""Plain-text description of how a path is read: for checking readers against real data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcs_studio.io.base import ReaderError, detect_reader
from hcs_studio.model.plate import Plate


def _listing(path: Path, limit: int = 25) -> list[str]:
    if not path.is_dir():
        return [f"  {path.name}  ({path.stat().st_size:,} bytes)"]
    entries = sorted(path.iterdir())
    lines = [
        f"  {e.name}{'/' if e.is_dir() else f'  ({e.stat().st_size:,} bytes)'}"
        for e in entries[:limit]
    ]
    if len(entries) > limit:
        lines.append(f"  … and {len(entries) - limit} more")
    return lines


def _plate_summary(plate: Plate) -> list[str]:
    meta = plate.metadata
    lines = [
        f"Plate: {plate.name}  (barcode: {plate.barcode or 'none'})",
        f"Layout: {plate.rows} rows × {plate.cols} columns; "
        f"{len(plate.wells)} wells with images, {plate.n_fields} fields",
        f"Field dimensions (T, C, Z, Y, X): {plate.shape_tczyx}",
        f"Pixel size: {meta.pixel_size_um} µm; Z step: {meta.z_step_um} µm",
        f"Objective: {meta.objective_name or 'unknown'}; magnification {meta.magnification}; "
        f"NA {meta.objective_na}; immersion RI {meta.immersion_ri}",
        f"Modality: {meta.modality}; pinhole: {meta.pinhole_um} µm; bit depth: {meta.bit_depth}",
        "Channels:",
    ]
    lines += [
        f"  {i + 1}. {c.name}  (ex {c.excitation_nm} nm, em {c.emission_nm} nm, "
        f"exposure {c.exposure_ms} ms)"
        for i, c in enumerate(meta.channels)
    ]
    lines.append("Wells:")
    for well in list(plate.wells.values())[:12]:
        label = f"  [{well.label}]" if well.label else ""
        first = well.fields[0]
        lines.append(
            f"  {well.name}: {len(well.fields)} field(s); first at "
            f"x={first.x_um}, y={first.y_um} µm{label}"
        )
    if len(plate.wells) > 12:
        lines.append(f"  … and {len(plate.wells) - 12} more wells")
    lines += [f"NOTE: {n}" for n in plate.notes]
    return lines


def _nd2_raw(path: Path) -> list[str]:
    """Raw structure of the first .nd2 file, so naming conventions can be checked."""
    import nd2

    file = path if path.is_file() else sorted(path.glob("*.nd2"))[0]
    lines = [f"Raw structure of {file.name}:"]
    with nd2.ND2File(file) as handle:
        lines.append(f"  sizes: {dict(handle.sizes)}")
        loops: list[Any] = list(handle.experiment or [])
        lines.append(f"  experiment loops: {[getattr(loop, 'type', '?') for loop in loops]}")
        for loop in loops:
            if getattr(loop, "type", "") == "XYPosLoop":
                names = [p.name for p in loop.parameters.points]
                lines.append(f"  position names ({len(names)}): {names[:12]}")
        try:
            jobs = handle.jobs()
        except Exception as exc:
            lines.append(f"  JOBS metadata could not be read: {exc}")
        else:
            if jobs is None:
                lines.append("  JOBS metadata: none")
            else:
                job: dict[str, Any] = dict(jobs.get("Job") or {})
                lines.append(f"  JOBS metadata: present; ProgramDesc={jobs.get('ProgramDesc')}")
                lines.append(f"  JOBS task keys: {list((job.get('Tasks') or {}).keys())[:20]}")
        text = handle.text_info or {}
        for key in ("description", "capturing", "optics"):
            if key in text:
                lines.append(f"  text_info[{key}]: {str(text[key])[:400]!r}")
    return lines


def describe_path(path: Path) -> str:
    """Describe a plate folder or image file. Reads metadata only; never writes."""
    path = Path(path)
    if not path.exists():
        return f"{path}: not found"
    lines = [f"Path: {path}", "Contents:", *_listing(path)]
    reader = detect_reader(path)
    if reader is None:
        lines.append("Format: not recognised")
        return "\n".join(lines)
    lines.append(f"Format: {reader.vendor}")
    try:
        lines += _plate_summary(reader.read(path))
    except (ReaderError, OSError, ValueError) as exc:
        lines.append(f"READ FAILED: {type(exc).__name__}: {exc}")
    if reader.vendor == "Nikon":
        try:
            lines += _nd2_raw(path)
        except Exception as exc:
            lines.append(f"Raw nd2 structure could not be read: {type(exc).__name__}: {exc}")
    return "\n".join(lines)
