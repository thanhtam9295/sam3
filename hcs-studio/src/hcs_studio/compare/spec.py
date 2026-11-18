"""A comparison: which fields, in which grid, shown how — and how to export it.

A ``Comparison`` is plain data (pydantic, JSON round-trip); the export sidecar stores it
in full, so a figure can be regenerated exactly from the sidecar alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SOURCES = ("raw", "corrected", "enhanced")
PRESETS_MM = {
    "single column (85 mm)": 85.0,
    "1.5 column (120 mm)": 120.0,
    "double column (180 mm)": 180.0,
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChannelDisplay(_Model):
    """Display of one channel: linear map low → high, then gamma, in a colour."""

    index: int  # 0-based channel of the image
    name: str = ""
    color: str = "#ffffff"
    low: float = 0.0
    high: float = 1.0
    gamma: float = Field(default=1.0, gt=0)
    visible: bool = True


class PanelSpec(_Model):
    run: str  # run folder
    well: str
    #: field position (1 = first); None = the first field, or the representative field
    #: when ``Comparison.representative_readout`` is set
    field: int | None = None
    #: raw | corrected | enhanced | step:<n> (the image after pipeline step n, recomputed)
    source: str = "corrected"
    #: Z plane (0-based) or "max" (maximum projection of what is loaded)
    z: int | Literal["max"] = "max"
    timepoint: int = 0
    #: label sets drawn as outlines; "<labels>:<column>" colours objects by a category
    overlays: list[str] = Field(default_factory=list)
    #: text under the panel; None = from the label template
    label: str | None = None
    #: per-panel display (contrast unlinked) — None follows the linked channels
    display: list[ChannelDisplay] | None = None
    row: int = 0
    col: int = 0


class ScaleBar(_Model):
    enabled: bool = True
    length_um: float | None = None  # None: a round length near 20 % of the panel width
    position: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right"
    color: str = "#ffffff"
    thickness_pt: float = 2.0
    show_label: bool = True
    panels: Literal["all", "last"] = "last"


class ExportSettings(_Model):
    format: Literal["png", "tiff", "svg", "pdf"] = "png"
    dpi: int = 300
    width_mm: float = 180.0
    spacing_mm: float = 1.0
    background: str = "#ffffff"
    text_color: str = "#000000"
    font: str = "DejaVu Sans"
    font_size_pt: float = 7.0
    label_template: str = "{well} · {cell_line} · {treatment}"
    panel_labels: bool = True
    headers: bool = True
    legend: bool = True
    split_channels: bool = False
    stamp: bool = False  # Z / time stamp in each panel
    stamp_time: bool = False  # always show the timepoint (time-lapse movies)
    scale_bar: ScaleBar = Field(default_factory=ScaleBar)


class Comparison(_Model):
    format_version: int = 1
    title: str = ""
    rows: int = 1
    cols: int = 1
    panels: list[PanelSpec] = Field(default_factory=list)
    #: linked display settings (default): identical for every panel
    channels: list[ChannelDisplay] = Field(default_factory=list)
    linked: bool = True
    row_headers: list[str] = Field(default_factory=list)
    col_headers: list[str] = Field(default_factory=list)
    #: synchronized crop (y0, x0, y1, x1) in µm of the field (pixels if uncalibrated)
    crop: tuple[float, float, float, float] | None = None
    representative_readout: str | None = None
    settings: ExportSettings = Field(default_factory=ExportSettings)

    def panel_at(self, row: int, col: int) -> PanelSpec | None:
        return next((p for p in self.panels if (p.row, p.col) == (row, col)), None)

    def display_of(self, panel: PanelSpec) -> list[ChannelDisplay]:
        return panel.display if (panel.display is not None) else self.channels

    def unlinked(self) -> list[PanelSpec]:
        return [p for p in self.panels if p.display is not None]
