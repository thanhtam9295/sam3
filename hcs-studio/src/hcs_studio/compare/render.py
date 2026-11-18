"""Compositing and figure montages of a comparison.

Pixels: each visible channel is mapped linearly low → high, clipped, raised to gamma and
added in its colour (additive RGB). Figures are drawn with matplotlib: raster images
(``interpolation='none'``, so vector formats embed them unresampled) with vector labels,
headers, legend and scale bars. Panel geometry is computed in millimetres from the export
settings, so the same comparison always gives the same figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.compare.sources import PanelImage, load_panel, plane
from hcs_studio.compare.spec import ChannelDisplay, Comparison, PanelSpec
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.utils.display import _RGB, channel_colormaps

MM = 1 / 25.4  # inch per mm
PT_MM = 25.4 / 72  # mm per point
OVERLAY_COLOURS = ("#ffff00", "#00ffff", "#ff00ff", "#ff8000", "#80ff00")
CATEGORY_COLOURS = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
)


def hex_rgb(colour: str) -> tuple[float, float, float]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def default_channels(metadata: ImageMetadata) -> list[ChannelDisplay]:
    names = channel_colormaps(metadata.channels)
    out = []
    for i, (channel, name) in enumerate(zip(metadata.channels, names, strict=True)):
        r, g, b = _RGB.get(name, _RGB["gray"])
        colour = f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
        out.append(ChannelDisplay(index=i, name=channel.name, color=colour))
    return out


def auto_contrast(
    planes: list[NDArray[Any]],
    channels: list[ChannelDisplay],
    low_pct: float = 0.1,
    high_pct: float = 99.9,
) -> list[ChannelDisplay]:
    """Linked contrast: the same low/high for every panel, from the pooled pixels."""
    out = []
    for ch in channels:
        values = np.concatenate(
            [
                p[ch.index].ravel()[:: max(1, p[ch.index].size // 200000)]
                for p in planes
                if ch.index < p.shape[0]
            ]
        )
        low, high = (float(v) for v in np.percentile(values, (low_pct, high_pct)))
        if high <= low:
            high = low + 1.0
        out.append(ch.model_copy(update={"low": round(low, 4), "high": round(high, 4)}))
    return out


def compose(planes: NDArray[Any], display: list[ChannelDisplay]) -> NDArray[np.float32]:
    """(C, Y, X) → RGB (Y, X, 3) in [0, 1]."""
    rgb = np.zeros((*planes.shape[1:], 3), dtype=np.float32)
    for ch in display:
        if not ch.visible or ch.index >= planes.shape[0]:
            continue
        rgb += gray(planes[ch.index], ch)[..., None] * np.asarray(hex_rgb(ch.color), np.float32)
    return np.clip(rgb, 0, 1)


def gray(values: NDArray[Any], ch: ChannelDisplay) -> NDArray[np.float32]:
    span = (ch.high - ch.low) or 1.0
    scaled = np.clip((values.astype(np.float32) - ch.low) / span, 0, 1)
    return np.asarray(scaled ** (1 / ch.gamma) if ch.gamma != 1 else scaled, dtype=np.float32)


def outline(labels: NDArray[Any]) -> NDArray[np.bool_]:
    """Pixels on the boundary of labelled objects."""
    labels = np.asarray(labels)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def category_colours(panel: PanelSpec, labels_name: str, column: str, field: int) -> dict[int, str]:
    """Object id → colour for a categorical column (class_, cluster_, gate_ …)."""
    from pathlib import Path

    import pandas as pd

    well = Path(panel.run) / "wells" / panel.well
    frame = (
        pd.read_parquet(well / "objects.parquet") if (well / "objects.parquet").exists() else None
    )
    extra = well / "objects_annotations.parquet"
    if frame is None:
        return {}
    if extra.exists():
        notes = pd.read_parquet(extra)
        keys = [k for k in ("field", "object_type", "object_id") if k in notes and k in frame]
        frame = frame.merge(notes, on=keys, how="left", suffixes=("", "_ann"))
    frame = frame[
        (frame["field"] == field) & (frame.get("object_type", labels_name) == labels_name)
    ]
    if column not in frame:
        return {}
    categories = sorted(str(v) for v in frame[column].dropna().unique())
    lookup = {c: CATEGORY_COLOURS[k % len(CATEGORY_COLOURS)] for k, c in enumerate(categories)}
    return {
        int(i): lookup[str(v)]
        for i, v in zip(frame["object_id"], frame[column], strict=True)
        if pd.notna(v)
    }


def overlay(
    rgb: NDArray[np.float32], loaded: PanelImage, panel: PanelSpec, z: int | str
) -> NDArray[np.float32]:
    out = rgb.copy()
    for k, item in enumerate(panel.overlays):
        name, _, column = item.partition(":")
        try:
            ls = loaded.image.label_set(name)
        except (KeyError, ValueError, StopIteration):
            continue
        data = ls.data.max(axis=0) if (z == "max" or ls.data.shape[0] == 1) else ls.data[int(z)]
        edge = outline(data)
        if column:
            colours = category_colours(panel, name, column, loaded.field)
            ids = data[edge]
            colour = np.array([hex_rgb(colours.get(int(i), "#808080")) for i in ids], np.float32)
            out[edge] = colour if len(colour) else out[edge]
        else:
            out[edge] = hex_rgb(OVERLAY_COLOURS[k % len(OVERLAY_COLOURS)])
    return out


# ------------------------------------------------------------------ crop, scale


def crop_slices(
    comparison: Comparison, shape: tuple[int, int], pixel_um: float | None
) -> tuple[slice, slice]:
    if comparison.crop is None:
        return slice(0, shape[0]), slice(0, shape[1])
    scale = pixel_um or 1.0
    y0, x0, y1, x1 = (v / scale for v in comparison.crop)
    ys = slice(max(0, round(min(y0, y1))), min(shape[0], round(max(y0, y1))))
    xs = slice(max(0, round(min(x0, x1))), min(shape[1], round(max(x0, x1))))
    if ys.stop - ys.start < 2 or xs.stop - xs.start < 2:
        raise ValueError("The crop is outside the image or smaller than 2 pixels")
    return ys, xs


def wrap_label(text: str, width_mm: float, font_mm: float, lines: int = 2) -> str:
    """Wrap a label to the panel width (about 0.55 em per character), at most ``lines``."""
    import textwrap

    chars = max(4, int(width_mm / (font_mm * 0.55)))
    wrapped = textwrap.wrap(text, chars) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1][: max(1, chars - 1)].rstrip() + "…"
    return "\n".join(wrapped)


def nice_length(width_um: float) -> float:
    """A round scale-bar length (1, 2 or 5 × 10^k) near 20 % of the width."""
    target = width_um * 0.2
    power = 10 ** math.floor(math.log10(target)) if target > 0 else 1
    for step in (5, 2, 1):
        if step * power <= target:
            return float(step * power)
    return float(power)


# ------------------------------------------------------------------- the figure


@dataclass
class Rendered:
    panels: list[dict[str, Any]] = field(default_factory=list)
    width_in: float = 0.0
    height_in: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)


def annotation_text(panel: PanelSpec, template: str, field_number: int) -> str:
    from pathlib import Path

    from hcs_studio.model.layout import run_layout_file
    from hcs_studio.model.plate import parse_well_name

    values: dict[str, Any] = {
        "well": panel.well,
        "field": field_number,
        "cell_line": "",
        "treatment": "",
        "role": "",
        "run": Path(panel.run).name,
    }
    layout_file = run_layout_file(Path(panel.run))
    if layout_file is not None:
        a = _layout(str(layout_file)).get(parse_well_name(panel.well))
        values.update(
            cell_line=a.cell_line,
            role=a.role.replace("_", " "),
            treatment=" + ".join(t.label() for t in a.active_treatments)
            or (a.role.replace("_", " ") if a.role != "sample" else ""),
        )
        values.update(dict(a.custom))
    try:
        text = template.format(**values)
    except (KeyError, IndexError, ValueError):
        text = panel.well
    return " · ".join(p for p in (s.strip() for s in text.split("·")) if p)


_LAYOUTS: dict[str, Any] = {}


def _layout(path: str) -> Any:
    from hcs_studio.model.layout import load_plate_layout

    if path not in _LAYOUTS:
        _LAYOUTS[path] = load_plate_layout(__import__("pathlib").Path(path))
    return _LAYOUTS[path]


def render(
    comparison: Comparison, images: dict[int, PanelImage] | None = None, dpi: int | None = None
) -> tuple[Any, Rendered]:
    """The montage as a matplotlib Figure and the geometry of every panel."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    s = comparison.settings
    dpi = dpi or s.dpi
    if not comparison.panels:
        raise ValueError("The comparison has no panels")
    loaded = images or {
        k: load_panel(p, comparison.representative_readout) for k, p in enumerate(comparison.panels)
    }
    first = loaded[0].image
    channels = comparison.channels or default_channels(first.metadata)
    visible = [c for c in channels if c.visible]
    split = s.split_channels
    per_panel = (len(visible) + 1) if split else 1
    rows, cols = comparison.rows, comparison.cols * per_panel
    planes = {
        k: plane(img.image, p.z)
        for k, (p, img) in enumerate(zip(comparison.panels, loaded.values(), strict=True))
    }
    pixel_um = first.metadata.pixel_size_um
    ys, xs = crop_slices(comparison, planes[0].shape[1:], pixel_um)
    crop_h, crop_w = ys.stop - ys.start, xs.stop - xs.start

    font_mm = s.font_size_pt * PT_MM
    left = font_mm * 2.2 if (s.headers and comparison.row_headers) else 0.5
    top = font_mm * 2.0 if (s.headers and (comparison.col_headers or split)) else 0.5
    label_h = font_mm * 3.0 if s.panel_labels else 0.0  # up to two lines
    legend_h = font_mm * 2.2 if (s.legend and not split) else 0.0
    gap = s.spacing_mm
    panel_w = (s.width_mm - left - 0.5 - gap * (cols - 1)) / cols
    panel_h = panel_w * crop_h / crop_w
    height_mm = top + rows * (panel_h + label_h) + gap * (rows - 1) + legend_h + 0.5
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(s.width_mm * MM, height_mm * MM), dpi=dpi, facecolor=s.background)
    FigureCanvasAgg(fig)
    font: dict[str, Any] = {"family": s.font, "size": s.font_size_pt, "color": s.text_color}
    result = Rendered(width_in=s.width_mm * MM, height_in=height_mm * MM)

    def box(r: int, c: int) -> tuple[float, float, float, float]:
        x = left + c * (panel_w + gap)
        y_top = top + r * (panel_h + label_h + gap)
        return x, y_top, panel_w, panel_h

    bar = s.scale_bar
    bar_um = bar.length_um or (nice_length(crop_w * pixel_um) if pixel_um else None)
    for k, panel in enumerate(comparison.panels):
        img = loaded[k]
        data = planes[k][:, ys, xs]
        display = comparison.display_of(panel)
        views: list[tuple[int | None, NDArray[np.float32]]] = []
        if split:
            for ch in [c for c in display if c.visible]:
                g = gray(data[ch.index], ch)
                views.append((ch.index, np.repeat(g[..., None], 3, axis=2)))
        rgb = compose(data, display)
        if panel.overlays:
            full = compose(planes[k], display)
            full = overlay(full, img, panel, panel.z)
            rgb = full[ys, xs]
        views.append((None, rgb))
        for v, (channel, pixels) in enumerate(views):
            c = panel.col * per_panel + v
            x, y_top, w, h = box(panel.row, c)
            ax = fig.add_axes(
                (x / s.width_mm, 1 - (y_top + h) / height_mm, w / s.width_mm, h / height_mm)
            )
            ax.imshow(pixels, interpolation="none", extent=(0, crop_w, crop_h, 0), aspect="auto")
            ax.set_xlim(0, crop_w)
            ax.set_ylim(crop_h, 0)
            ax.set_axis_off()
            last = (panel.row, c) == (comparison.rows - 1, cols - 1)
            info: dict[str, Any] = {
                "panel": k,
                "well": panel.well,
                "field": img.field,
                "row": panel.row,
                "col": c,
                "channel": channel,
                # output pixels (PNG/TIFF), from the top left of the image
                "bbox_px": [
                    x * MM * dpi,
                    y_top * MM * dpi,
                    (x + w) * MM * dpi,
                    (y_top + h) * MM * dpi,
                ],
                "image_px": [crop_w, crop_h],
                "output_px_per_image_px": w * MM * dpi / crop_w,
            }
            if bar.enabled and bar_um and pixel_um and (bar.panels == "all" or last):
                length = bar_um / pixel_um
                thick = bar.thickness_pt / 72 / (h * MM) * crop_h
                margin = 0.05 * crop_w
                bx = crop_w - margin - length if "right" in bar.position else margin
                by = crop_h - margin - thick if "lower" in bar.position else margin
                ax.add_patch(
                    Rectangle(
                        (bx, by),
                        length,
                        thick,
                        facecolor=bar.color,
                        edgecolor="none",
                        gid=f"scalebar-{k}-{v}",
                    )
                )
                if bar.show_label:
                    ty = by - thick if "lower" in bar.position else by + 2 * thick
                    right = "right" in bar.position
                    ax.text(
                        bx + length if right else bx,
                        ty,
                        f"{bar_um:g} µm",
                        ha="right" if right else "left",
                        va="bottom" if "lower" in bar.position else "top",
                        fontsize=s.font_size_pt,
                        color=bar.color,
                        family=s.font,
                    )
                info["scale_bar"] = {
                    "length_um": bar_um,
                    "length_image_px": length,
                    "length_output_px": length * info["output_px_per_image_px"],
                    "gid": f"scalebar-{k}-{v}",
                }
            if s.stamp:
                n_z = img.image.data.shape[1]
                stamp = (
                    "max Z"
                    if (panel.z == "max" and n_z > 1)
                    else (f"z {int(panel.z) + 1}/{n_z}" if n_z > 1 else "")
                )
                if panel.timepoint:
                    stamp = f"t {panel.timepoint} " + stamp
                if stamp:
                    ax.text(
                        0.03 * crop_w,
                        0.03 * crop_h,
                        stamp,
                        ha="left",
                        va="top",
                        fontsize=s.font_size_pt,
                        color="#ffffff",
                        family=s.font,
                    )
            if s.panel_labels and v == 0:
                text = (
                    panel.label
                    if panel.label is not None
                    else annotation_text(panel, s.label_template, img.field)
                )
                if panel.display is not None and comparison.linked:
                    text = "⚠ contrast unlinked · " + text
                strip_w = per_panel * w + (per_panel - 1) * gap
                fig.text(
                    x / s.width_mm,
                    1 - (y_top + h + font_mm * 0.25) / height_mm,
                    wrap_label(text, strip_w, font_mm),
                    ha="left",
                    va="top",
                    **font,
                )
            if split and panel.row == 0 and s.headers:
                name = (
                    "merge"
                    if channel is None
                    else next(ch.name for ch in display if ch.index == channel)
                )
                fig.text(
                    (x + w / 2) / s.width_mm,
                    1 - (top - font_mm * 0.4) / height_mm,
                    name,
                    ha="center",
                    va="bottom",
                    **font,
                )
            result.panels.append(info)
        result.sources.append(
            {
                "panel": k,
                "run": panel.run,
                "well": panel.well,
                "field": img.field,
                "source": img.source,
                "files": list(img.files),
                "steps": list(img.steps),
            }
        )
    if s.headers:
        for c, text in enumerate(comparison.col_headers[: comparison.cols]):
            if split:
                continue
            x, _, w, _ = box(0, c)
            fig.text(
                (x + w / 2) / s.width_mm,
                1 - (top - font_mm * 0.4) / height_mm,
                text,
                ha="center",
                va="bottom",
                **font,
            )
        for r, text in enumerate(comparison.row_headers[: comparison.rows]):
            _, y_top, _, h = box(r, 0)
            fig.text(
                (left - font_mm * 0.5) / s.width_mm,
                1 - (y_top + h / 2) / height_mm,
                text,
                ha="right",
                va="center",
                rotation=90,
                **font,
            )
    if legend_h:
        x = left
        for ch in visible:
            label = fig.text(
                x / s.width_mm,
                (legend_h * 0.45) / height_mm,
                f"■ {ch.name}",
                ha="left",
                va="center",
                family=s.font,
                size=s.font_size_pt,
                color=ch.color if ch.color.lower() != s.background.lower() else s.text_color,
            )
            x += font_mm * (len(ch.name) * 0.62 + 3)
            del label
    return fig, result
