"""Exports of a comparison: figure montages, panel data, movies, presets, batch figures.

Every export writes ``<file>.json`` next to it: the full comparison (panels, display,
settings), the source files with SHA-256, the processing steps, the panel geometry and
the software version. ``regenerate(sidecar)`` re-renders the figure from it alone.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.compare.render import (
    auto_contrast,
    compose,
    crop_slices,
    default_channels,
    render,
)
from hcs_studio.compare.sources import PanelImage, file_sha256, load_panel, plane
from hcs_studio.compare.spec import Comparison, ExportSettings
from hcs_studio.io.sidecar import config_dir

SIDECAR_FORMAT = "hcs-studio-figure/1"
#: Raster figures above this are refused (an RGBA buffer of 250 MP is 1 GB).
MAX_MEGAPIXELS = 250.0


def fingerprint(path: Path) -> str:
    """SHA-256 of a file; for a folder (OME-Zarr, vendor exports) a ``tree:`` digest of
    every file's relative path, size and modification time (contents are not read)."""
    if path.is_file():
        return file_sha256(path)
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        digest.update(f"{item.relative_to(path)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return "tree:" + digest.hexdigest()


def _sidecar(path: Path, comparison: Comparison, kind: str, extra: dict[str, Any]) -> Path:
    from hcs_studio import __version__

    sources = extra.pop("sources", [])
    seen: dict[str, str] = {}
    for source in sources:
        for f in source.get("files", []):
            if f not in seen and Path(f).exists():
                seen[f] = fingerprint(Path(f))
    data = {
        "format": SIDECAR_FORMAT,
        "kind": kind,
        "software": "HCS Studio",
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "output": path.name,
        "comparison": comparison.model_dump(mode="json"),
        "sources": sources,
        "checksums": seen,
        **extra,
    }
    sidecar = path.with_name(path.name + ".json")
    sidecar.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def _filled(comparison: Comparison, images: dict[int, PanelImage]) -> Comparison:
    """Freeze what was chosen automatically (fields, channels) so the sidecar is exact."""
    panels = [
        p.model_copy(update={"field": images[k].field}) for k, p in enumerate(comparison.panels)
    ]
    channels = comparison.channels or auto_contrast(
        [plane(images[k].image, p.z) for k, p in enumerate(comparison.panels)],
        default_channels(images[0].image.metadata),
    )
    return comparison.model_copy(update={"panels": panels, "channels": channels})


def load_images(comparison: Comparison) -> dict[int, PanelImage]:
    return {
        k: load_panel(p, comparison.representative_readout) for k, p in enumerate(comparison.panels)
    }


def export_figure(
    comparison: Comparison, path: Path, images: dict[int, PanelImage] | None = None
) -> Path:
    """Write the montage (PNG/TIFF at the settings' dpi, or SVG/PDF) and its sidecar."""
    path = Path(path)
    images = images or load_images(comparison)
    comparison = _filled(comparison, images)
    fmt = comparison.settings.format
    if path.suffix.lower().lstrip(".") not in (fmt, "tif" if fmt == "tiff" else fmt):
        path = path.with_suffix("." + fmt)
    fig, rendered = render(comparison, images)
    if fmt in ("png", "tiff"):
        dpi = comparison.settings.dpi
        megapixels = rendered.width_in * dpi * rendered.height_in * dpi / 1e6
        if megapixels > MAX_MEGAPIXELS:
            raise ValueError(
                f"The figure would be {megapixels:.0f} megapixels at {dpi} dpi (limit "
                f"{MAX_MEGAPIXELS:.0f}): lower the dpi, use fewer panels or export SVG/PDF"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"facecolor": comparison.settings.background}
    if fmt == "png":
        options["metadata"] = {"Software": None}
    elif fmt == "svg":
        options["metadata"] = {"Date": None}
    elif fmt == "pdf":
        options["metadata"] = {"CreationDate": None}
    import matplotlib

    # text stays text (editable in Illustrator / Inkscape), fonts embedded as TrueType
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(path, dpi=comparison.settings.dpi, format=fmt, **options)
    _sidecar(
        path,
        comparison,
        "figure",
        {
            "sources": rendered.sources,
            "geometry": {
                "width_in": rendered.width_in,
                "height_in": rendered.height_in,
                "dpi": comparison.settings.dpi,
                "panels": rendered.panels,
            },
        },
    )
    return path


def regenerate(sidecar: Path, output: Path | None = None) -> Path:
    """Re-render a figure from its sidecar (same panels, display and settings)."""
    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    if data.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"{sidecar} is not an HCS Studio figure sidecar")
    comparison = Comparison.model_validate(data["comparison"])
    changed = [
        f
        for f, digest in data.get("checksums", {}).items()
        if Path(f).exists() and fingerprint(Path(f)) != digest
    ]
    if changed:
        raise ValueError(f"source files changed since the export: {', '.join(changed[:3])}")
    target = Path(output) if output else Path(sidecar).with_name("regenerated_" + data["output"])
    return export_figure(comparison, target)


# ------------------------------------------------------------------- panel data


def export_data(
    comparison: Comparison,
    folder: Path,
    dtype: str = "float32",
    images: dict[int, PanelImage] | None = None,
) -> list[Path]:
    """One OME-TIFF per panel (the shown plane, cropped; float32 or uint16) and a CSV of
    the shown wells' readouts, with a sidecar."""
    from hcs_studio.output.writers import write_ome_tiff

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    images = images or load_images(comparison)
    comparison = _filled(comparison, images)
    written: list[Path] = []
    rows = []
    for k, panel in enumerate(comparison.panels):
        img = images[k]
        data = plane(img.image, panel.z)
        ys, xs = crop_slices(comparison, data.shape[1:], img.image.metadata.pixel_size_um)
        data = data[:, ys, xs]
        if dtype == "uint16":
            data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
        elif dtype != "float32":
            raise ValueError("dtype must be float32 or uint16")
        name = f"panel_r{panel.row + 1}c{panel.col + 1}_{panel.well}_f{img.field:03d}.ome.tif"
        write_ome_tiff(folder / name, data[None, :, None], img.image.metadata)
        written.append(folder / name)
        summary = Path(panel.run) / "tables" / "per_well_summary.csv"
        row: dict[str, Any] = {
            "panel": name,
            "run": Path(panel.run).name,
            "well": panel.well,
            "field": img.field,
            "source": img.source,
        }
        if summary.exists():
            table = pd.read_csv(summary)
            match = table[table["well"] == panel.well]
            if len(match):
                row.update({str(k): v for k, v in match.iloc[0].to_dict().items()})
        rows.append(row)
    csv = folder / "panels_readouts.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    written.append(csv)
    sources = [
        {
            "panel": k,
            "run": p.run,
            "well": p.well,
            "field": images[k].field,
            "source": images[k].source,
            "files": list(images[k].files),
            "steps": list(images[k].steps),
        }
        for k, p in enumerate(comparison.panels)
    ]
    _sidecar(
        folder / "panels",
        comparison,
        "data",
        {"sources": sources, "dtype": dtype, "files": [p.name for p in written]},
    )
    return written


# ----------------------------------------------------------------------- movies


def frame_count(comparison: Comparison, images: dict[int, PanelImage], axis: str) -> int:
    """Frames of a movie: Z planes of the loaded panels, or timepoints of the raw data."""
    if axis == "z":
        return int(min(img.image.data.shape[1] for img in images.values()))
    if axis == "t":
        sources = {p.source for p in comparison.panels}
        if not all(s == "raw" or s.startswith("step:") for s in sources):
            raise ValueError(
                "Time movies need the raw data (image source 'raw' or 'step:<n>'): run "
                "outputs keep the first timepoint only"
            )
        return min(timepoint_count(Path(p.run)) for p in comparison.panels)
    raise ValueError(f"Unknown movie axis {axis!r} (z or t)")


def timepoint_count(run: Path) -> int:
    """Timepoints in a run's raw data (reopened from the manifest)."""
    from hcs_studio.io.base import reopen_plate

    manifest = json.loads((Path(run) / "run_manifest.json").read_text(encoding="utf-8"))
    plate = reopen_plate(manifest["plate"]["source"])
    shape = plate.shape_tczyx
    if shape is None:
        first = next(iter(plate.wells.values())).fields[0]
        return int(first.load().shape[0])
    return int(shape[0])


def export_movie(
    comparison: Comparison,
    path: Path,
    axis: str = "z",
    fps: float = 4.0,
    dpi: int = 120,
    images: dict[int, PanelImage] | None = None,
) -> Path:
    """A GIF or MP4 with all panels in step: through Z (``axis="z"``) or through time
    (``"t"``, from the raw data: one load per panel and timepoint)."""
    path = Path(path)
    if path.suffix.lower() not in (".gif", ".mp4"):
        raise ValueError("Movies are written as .gif or .mp4")
    if axis == "t":
        n = frame_count(comparison, {}, axis)
        images = load_images(comparison)
    else:
        images = images or load_images(comparison)
        n = frame_count(comparison, images, axis)
    comparison = _filled(comparison, images)
    if n < 2:
        what = "Z plane" if axis == "z" else "timepoint"
        raise ValueError(f"The panels have a single {what}: nothing to animate")
    frames = []
    for k in range(n):
        if axis == "z":
            step = comparison.model_copy(
                update={
                    "panels": [p.model_copy(update={"z": k}) for p in comparison.panels],
                    "settings": comparison.settings.model_copy(update={"stamp": True}),
                }
            )
            frame_images = images
        else:
            step = comparison.model_copy(
                update={
                    "panels": [p.model_copy(update={"timepoint": k}) for p in comparison.panels],
                    "settings": comparison.settings.model_copy(
                        update={"stamp": True, "stamp_time": True}
                    ),
                }
            )
            frame_images = load_images(step)
        fig, _ = render(step, frame_images, dpi=dpi)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    if path.suffix.lower() == ".gif":
        from PIL import Image

        pictures = [Image.fromarray(f) for f in frames]
        pictures[0].save(
            path, save_all=True, append_images=pictures[1:], duration=int(1000 / fps), loop=0
        )
    elif path.suffix.lower() == ".mp4":
        import imageio_ffmpeg

        h, w = frames[0].shape[:2]
        H, W = h + h % 2, w + w % 2  # yuv420p needs even sizes
        writer = imageio_ffmpeg.write_frames(
            str(path), (W, H), fps=fps, quality=8, macro_block_size=1, pix_fmt_out="yuv420p"
        )
        writer.send(None)
        for f in frames:
            padded = np.full((H, W, 3), 255, np.uint8)
            padded[:h, :w] = f
            writer.send(np.ascontiguousarray(padded))
        writer.close()
    else:
        raise ValueError("Movies are written as .gif or .mp4")
    sources = [
        {
            "panel": k,
            "run": p.run,
            "well": p.well,
            "field": images[k].field,
            "source": images[k].source,
            "files": list(images[k].files),
        }
        for k, p in enumerate(comparison.panels)
    ]
    _sidecar(path, comparison, "movie", {"sources": sources, "axis": axis, "frames": n, "fps": fps})
    return path


# ---------------------------------------------------------------------- presets


def presets_folder() -> Path:
    return config_dir() / "export_presets"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip() or "preset"


def save_preset(name: str, settings: ExportSettings) -> Path:
    folder = presets_folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_safe(name)}.json"
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_presets() -> list[str]:
    folder = presets_folder()
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def load_preset(name: str) -> ExportSettings:
    return ExportSettings.model_validate_json(
        (presets_folder() / f"{_safe(name)}.json").read_text(encoding="utf-8")
    )


# ------------------------------------------------------------------------ batch


def batch_compound_figures(
    run: Path, folder: Path, base: Comparison, compounds: list[str] | None = None
) -> list[Path]:
    """One figure per compound of the plate layout: concentrations as columns (vehicle
    first), cell lines as rows."""
    from hcs_studio.compare.arrange import compound_grid, layout_of

    layout = layout_of(run)
    names = compounds or sorted(
        {
            t.compound
            for a in layout.wells.values()
            for t in a.active_treatments
            if t.kind != "percent"
        }
    )
    out = []
    for compound in names:
        grid = compound_grid(run, compound, base)
        if not grid.panels:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", compound)
        out.append(export_figure(grid, Path(folder) / f"{safe}.{base.settings.format}"))
    return out


def preview_rgb(comparison: Comparison, k: int, images: dict[int, PanelImage]) -> np.ndarray:
    """The composed RGB of panel ``k`` (as shown), for interactive views."""
    panel = comparison.panels[k]
    data = plane(images[k].image, panel.z)
    channels = comparison.display_of(panel) or default_channels(images[k].image.metadata)
    return compose(data, channels)
