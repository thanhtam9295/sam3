"""Compare tab: fields side by side with linked zoom, pan, Z and contrast; figure export."""

from __future__ import annotations

import contextlib
import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.compare import arrange
from hcs_studio.compare.export import (
    batch_compound_figures,
    export_data,
    export_figure,
    export_movie,
    list_presets,
    load_preset,
    save_preset,
)
from hcs_studio.compare.render import auto_contrast, compose, default_channels, overlay
from hcs_studio.compare.sources import PanelImage, load_panel, plane, step_names
from hcs_studio.compare.spec import PRESETS_MM, ChannelDisplay, Comparison, ExportSettings, ScaleBar
from hcs_studio.gui.annotation_filter import AnnotationFilterBar
from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.model.plate import well_name

ARRANGEMENTS = ("grid", "by concentration × cell line", "combination matrix", "one compound")


class ComparePanel(QWidget):
    """Shows ``Comparison`` panels; ``images_loaded`` after a (background) load."""

    images_loaded = Signal()

    def __init__(
        self,
        log: Callable[[str], None] | None = None,
        last_selection: Callable[[], tuple[Path, list[str]] | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.log = log or (lambda text: None)
        self.last_selection = last_selection or (lambda: None)
        self.run: Path | None = None
        self.wells: list[str] = []
        self.comparison = Comparison()
        self.images: dict[int, PanelImage] = {}
        self.current = 0  # the panel last clicked
        self.locked: set[int] = set()
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_loaded)
        self.task.failed.connect(self._on_failed)
        self._build()

    # ------------------------------------------------------------------- UI

    def _build(self) -> None:
        self.well_list = QListWidget()
        self.well_list.setMaximumHeight(90)
        add_last = QPushButton("Add last selection")
        add_last.setToolTip("Wells last selected in Results, Pharmacology or Profiles")
        add_last.clicked.connect(self._add_last)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.set_wells(self.run, []) if self.run else None)
        self.filter_bar = AnnotationFilterBar(self._layout)
        self.filter_bar.selected.connect(lambda keys: self.add_wells([well_name(*k) for k in keys]))

        self.arrangement = QComboBox()
        self.arrangement.addItems(list(ARRANGEMENTS))
        self.rows = QSpinBox()
        self.rows.setRange(0, 32)
        self.rows.setSpecialValueText("auto")
        self.cols = QSpinBox()
        self.cols.setRange(0, 48)
        self.cols.setSpecialValueText("auto")
        do_arrange = QPushButton("Arrange")
        do_arrange.clicked.connect(self.rearrange)

        self.source = QComboBox()
        self.source.addItems(["corrected", "enhanced", "raw"])
        self.source.activated.connect(lambda _: self.reload())
        self.representative = QCheckBox("Representative field by")
        self.readout = QLineEdit()
        self.readout.setPlaceholderText("e.g. n_nuclei (empty: object count)")
        self.representative.toggled.connect(lambda _: self.reload())
        self.z_max = QCheckBox("Max over Z")
        self.z_max.setChecked(True)
        self.z = QSlider(Qt.Orientation.Horizontal)
        self.z.setRange(0, 0)
        self.z.valueChanged.connect(lambda _: self._set_z())
        self.z_max.toggled.connect(lambda _: self._set_z())

        self.channel_table = QTableWidget(0, 5)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Colour", "Low", "High", "Gamma"])
        self.channel_table.setMaximumHeight(130)
        self.channel_table.itemChanged.connect(self._channel_edited)
        self.channel_table.cellDoubleClicked.connect(self._pick_colour)
        auto = QPushButton("Auto contrast")
        auto.clicked.connect(self.auto_contrast)
        self.linked = QCheckBox("Linked contrast (all panels identical)")
        self.linked.setChecked(True)
        self.linked.toggled.connect(self._toggle_linked)
        unlink = QPushButton("Unlink clicked panel")
        unlink.clicked.connect(self.unlink_current)
        lock = QPushButton("Lock / unlock zoom of clicked panel")
        lock.clicked.connect(self.toggle_lock)
        self.overlays = QListWidget()
        self.overlays.setMaximumHeight(60)
        self.overlays.itemChanged.connect(lambda _: self._set_overlays())
        self.template = QLineEdit(ExportSettings().label_template)
        self.template.editingFinished.connect(self._set_template)
        crop = QPushButton("Crop to the view")
        crop.setToolTip("Use the current (linked) zoom as the crop of every panel")
        crop.clicked.connect(self.crop_to_view)
        uncrop = QPushButton("No crop")
        uncrop.clicked.connect(lambda: self.set_crop(None))

        self.format = QComboBox()
        self.format.addItems(["png", "tiff", "svg", "pdf"])
        self.dpi = QComboBox()
        self.dpi.addItems(["300", "600"])
        self.width_preset = QComboBox()
        self.width_preset.addItems(list(PRESETS_MM))
        self.width_preset.setCurrentIndex(2)
        self.bar_length = QDoubleSpinBox()
        self.bar_length.setRange(0, 10000)
        self.bar_length.setSpecialValueText("auto")
        self.bar_length.setSuffix(" µm")
        self.bar_position = QComboBox()
        self.bar_position.addItems(["lower right", "lower left", "upper right", "upper left"])
        self.split = QCheckBox("Split channels (grayscale + merge)")
        self.headers = QCheckBox("Row / column headers")
        self.headers.setChecked(True)
        self.legend = QCheckBox("Channel legend")
        self.legend.setChecked(True)
        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(4, 24)
        self.font_size.setValue(7)
        self.background = QComboBox()
        self.background.addItems(["#ffffff", "#000000"])
        self.presets = QComboBox()
        self._refresh_presets()
        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_preset)
        save_button = QPushButton("Save as…")
        save_button.clicked.connect(self._save_preset)
        buttons = [
            ("Export figure…", self._export_figure),
            ("Export data…", self._export_data),
            ("Export movie (Z)…", self._export_movie),
            ("Batch: a figure per compound…", self._batch),
        ]
        self.status = QLabel("Choose wells: Results, Pharmacology, Profiles or the filter above.")
        self.status.setWordWrap(True)

        def row(*widgets: QWidget) -> QWidget:
            box = QWidget()
            line = QHBoxLayout(box)
            line.setContentsMargins(0, 0, 0, 0)
            for w in widgets:
                line.addWidget(w)
            return box

        controls = QWidget()
        form = QVBoxLayout(controls)
        wells_box = QGroupBox("Wells")
        wells_layout = QVBoxLayout(wells_box)
        wells_layout.addWidget(self.well_list)
        wells_layout.addWidget(row(add_last, clear))
        wells_layout.addWidget(self.filter_bar)
        form.addWidget(wells_box)
        view_box = QGroupBox("Panels")
        view = QFormLayout(view_box)
        view.addRow(
            "Arrange",
            row(self.arrangement, QLabel("rows"), self.rows, QLabel("cols"), self.cols, do_arrange),
        )
        view.addRow("Image", self.source)
        view.addRow(row(self.representative, self.readout))
        view.addRow("Z", row(self.z, self.z_max))
        view.addRow(self.channel_table)
        view.addRow(row(auto, self.linked))
        view.addRow(row(unlink, lock))
        view.addRow("Outlines", self.overlays)
        view.addRow("Labels", self.template)
        view.addRow(row(crop, uncrop))
        form.addWidget(view_box)
        export_box = QGroupBox("Export")
        export = QFormLayout(export_box)
        export.addRow("Format", row(self.format, QLabel("dpi"), self.dpi))
        export.addRow("Width", self.width_preset)
        export.addRow("Scale bar", row(self.bar_length, self.bar_position))
        export.addRow(self.split)
        export.addRow(row(self.headers, self.legend))
        export.addRow("Font size (pt)", row(self.font_size, QLabel("background"), self.background))
        export.addRow("Preset", row(self.presets, load_button, save_button))
        for text, slot in buttons:
            button = QPushButton(text)
            button.clicked.connect(slot)
            export.addRow(button)
        form.addWidget(export_box)
        form.addWidget(self.status)
        form.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(470)

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("#202020")
        self.views: list[pg.ViewBox] = []
        self.items: list[pg.ImageItem] = []
        self.labels: list[pg.LabelItem] = []
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([470, 1200])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # --------------------------------------------------------------- wells

    def _layout(self) -> Any:
        if self.run is None:
            return None
        try:
            return arrange.layout_of(self.run)
        except ValueError:
            return None

    def set_wells(self, run: Path | None, wells: list[str]) -> None:
        """Compare these wells of a run (replaces the current panels)."""
        if run is not None and Path(run) != self.run:
            self.run = Path(run)
            self.source.clear()
            self.source.addItems(["corrected", "enhanced", "raw"])
            with contextlib.suppress(OSError, ValueError):
                self.source.addItems([f"step:{n.split('.')[0]}" for n in step_names(self.run)])
            self.filter_bar.refresh()
            self.comparison = self.comparison.model_copy(update={"channels": [], "crop": None})
        self.wells = list(dict.fromkeys(wells))
        self.well_list.clear()
        self.well_list.addItems(self.wells)
        self.rearrange()

    def add_wells(self, wells: list[str]) -> None:
        if self.run is not None:
            self.set_wells(self.run, [*self.wells, *wells])

    def _add_last(self) -> None:
        selection = self.last_selection()
        if not selection:
            self.status.setText("Nothing selected yet in Results, Pharmacology or Profiles.")
            return
        run, wells = selection
        if self.run is not None and Path(run) == self.run:
            self.add_wells(wells)
        else:
            self.set_wells(Path(run), wells)

    def rearrange(self) -> None:
        if self.run is None or not self.wells:
            self.comparison = self.comparison.model_copy(update={"panels": []})
            self._build_views()
            return
        base = self.comparison.model_copy(update={"panels": []})
        mode = self.arrangement.currentText()
        try:
            if mode == "by concentration × cell line":
                comparison = arrange.by_annotation(self.run, self.wells, base)
            elif mode == "combination matrix":
                comparison = self._combination(base)
            elif mode == "one compound":
                compound, ok = QInputDialog.getText(self, "One compound", "Compound:")
                if not ok:
                    return
                comparison = arrange.compound_grid(self.run, compound.strip(), base)
            else:
                comparison = arrange.grid(
                    self.run, self.wells, base, self.rows.value() or None, self.cols.value() or None
                )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        source = self.source.currentText() or "corrected"
        panels = [p.model_copy(update={"source": source}) for p in comparison.panels]
        self.comparison = comparison.model_copy(update={"panels": panels})
        self.reload()

    def _combination(self, base: Comparison) -> Comparison:
        assert self.run is not None
        layout = arrange.layout_of(self.run)
        a = layout.get(next(iter(sorted(layout.wells))))  # any well for the cell line default
        text, ok = QInputDialog.getText(
            self, "Combination matrix", "cell line; drug A; drug B", text=f"{a.cell_line}; ; "
        )
        if not ok:
            raise ValueError("cancelled")
        parts = [p.strip() for p in text.split(";")]
        if len(parts) != 3 or not all(parts[1:]):
            raise ValueError("Give: cell line; drug A; drug B")
        return arrange.combination_grid(self.run, parts[0], parts[1], parts[2], base)

    # ------------------------------------------------------------- loading

    def reload(self) -> None:
        if not self.comparison.panels:
            self._build_views()
            return
        source = self.source.currentText() or "corrected"
        readout = (self.readout.text().strip() or None) if self.representative.isChecked() else None
        panels = [
            p.model_copy(update={"source": source, "field": None}) for p in self.comparison.panels
        ]
        comparison = self.comparison.model_copy(
            update={
                "panels": panels,
                "representative_readout": ""
                if self.representative.isChecked() and readout is None
                else readout,
            }
        )
        self.comparison = comparison
        self.status.setText(f"Loading {len(panels)} field(s)…")

        def work() -> dict[int, PanelImage]:
            return {
                k: load_panel(p, comparison.representative_readout)
                for k, p in enumerate(comparison.panels)
            }

        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Could not load: {message}")
        self.log(f"Compare: {message}")

    def _on_loaded(self, images: dict[int, PanelImage]) -> None:
        self.images = images
        panels = [
            p.model_copy(update={"field": images[k].field})
            for k, p in enumerate(self.comparison.panels)
        ]
        self.comparison = self.comparison.model_copy(update={"panels": panels})
        first = images[0].image
        if not self.comparison.channels or len(self.comparison.channels) != first.data.shape[0]:
            self.comparison = self.comparison.model_copy(
                update={"channels": default_channels(first.metadata)}
            )
            self.auto_contrast(redraw=False)
        n_z = min(img.image.data.shape[1] for img in images.values())
        self.z.setRange(0, max(0, n_z - 1))
        self.z.setEnabled(n_z > 1)
        names = sorted({ls.name for img in images.values() for ls in img.image.labels})
        current = {
            self.overlays.item(i).text()
            for i in range(self.overlays.count())
            if self.overlays.item(i).checkState() == Qt.CheckState.Checked
        }
        self.overlays.blockSignals(True)
        self.overlays.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in current else Qt.CheckState.Unchecked
            )
            self.overlays.addItem(item)
        self.overlays.blockSignals(False)
        self._fill_channels()
        self._build_views()
        self.status.setText(
            f"{len(images)} panel(s), {self.comparison.rows} × {self.comparison.cols}; "
            f"fields {', '.join(str(img.field) for img in list(images.values())[:8])}"
        )
        self.images_loaded.emit()

    # ------------------------------------------------------------- display

    def auto_contrast(self, redraw: bool = True) -> None:
        if not self.images:
            return
        planes = [
            plane(img.image, p.z)
            for p, img in zip(self.comparison.panels, self.images.values(), strict=True)
        ]
        channels = auto_contrast(
            planes, self.comparison.channels or default_channels(self.images[0].image.metadata)
        )
        self.comparison = self.comparison.model_copy(update={"channels": channels})
        if redraw:
            self._fill_channels()
            self.redraw()

    def _fill_channels(self) -> None:
        table = self.channel_table
        table.blockSignals(True)
        channels = self.comparison.channels
        table.setRowCount(len(channels))
        for r, ch in enumerate(channels):
            name = QTableWidgetItem(ch.name)
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if ch.visible else Qt.CheckState.Unchecked)
            table.setItem(r, 0, name)
            colour = QTableWidgetItem(ch.color)
            colour.setBackground(QColor(ch.color))
            colour.setFlags(colour.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(r, 1, colour)
            for c, value in ((2, ch.low), (3, ch.high), (4, ch.gamma)):
                table.setItem(r, c, QTableWidgetItem(f"{value:.6g}"))
        table.blockSignals(False)

    def _channel_edited(self, item: QTableWidgetItem) -> None:
        r = item.row()
        channels = list(self.comparison.channels)
        if not 0 <= r < len(channels):
            return
        cells = [self.channel_table.item(r, c) for c in range(5)]
        if any(cell is None for cell in cells):
            return
        name, _, low, high, gamma = cells
        assert name is not None and low is not None and high is not None and gamma is not None
        try:
            update: dict[str, Any] = {
                "visible": name.checkState() == Qt.CheckState.Checked,
                "low": float(low.text()),
                "high": float(high.text()),
                "gamma": max(0.05, float(gamma.text())),
            }
        except ValueError:
            return
        self.set_channel(r, **update)

    def set_channel(self, index: int, **update: Any) -> None:
        """Change a linked channel display (all panels that follow the link)."""
        channels = list(self.comparison.channels)
        channels[index] = channels[index].model_copy(update=update)
        self.comparison = self.comparison.model_copy(update={"channels": channels})
        self.redraw()

    def _pick_colour(self, row: int, column: int) -> None:
        if column != 1:
            return
        colour = QColorDialog.getColor(QColor(self.comparison.channels[row].color), self)
        if colour.isValid():
            self.set_channel(row, color=colour.name())
            self._fill_channels()

    def _toggle_linked(self, on: bool) -> None:
        if on:
            panels = [p.model_copy(update={"display": None}) for p in self.comparison.panels]
            self.comparison = self.comparison.model_copy(update={"panels": panels, "linked": True})
        else:
            self.comparison = self.comparison.model_copy(update={"linked": False})
        self.redraw()

    def unlink_current(self) -> None:
        """Give the clicked panel its own contrast (it gets a warning badge)."""
        if not self.comparison.panels:
            return
        panels = list(self.comparison.panels)
        k = self.current
        planes = plane(self.images[k].image, panels[k].z)
        own = auto_contrast([planes], self.comparison.channels)
        panels[k] = panels[k].model_copy(update={"display": own})
        self.comparison = self.comparison.model_copy(update={"panels": panels})
        self.redraw()

    def _set_z(self) -> None:
        z: int | str = "max" if self.z_max.isChecked() else self.z.value()
        panels = [p.model_copy(update={"z": z}) for p in self.comparison.panels]
        self.comparison = self.comparison.model_copy(update={"panels": panels})
        self.redraw()

    def _set_overlays(self) -> None:
        names = [
            self.overlays.item(i).text()
            for i in range(self.overlays.count())
            if self.overlays.item(i).checkState() == Qt.CheckState.Checked
        ]
        panels = [p.model_copy(update={"overlays": names}) for p in self.comparison.panels]
        self.comparison = self.comparison.model_copy(update={"panels": panels})
        self.redraw()

    def _set_template(self) -> None:
        settings = self.comparison.settings.model_copy(
            update={"label_template": self.template.text()}
        )
        self.comparison = self.comparison.model_copy(update={"settings": settings})
        self.redraw()

    # --------------------------------------------------------------- views

    def _build_views(self) -> None:
        self.canvas.clear()
        self.views, self.items, self.labels = [], [], []
        first: pg.ViewBox | None = None
        for k, panel in enumerate(self.comparison.panels):
            cell = self.canvas.addLayout(row=panel.row, col=panel.col)
            label = cell.addLabel("", row=1, col=0, color="#dddddd", size="7pt")
            label.setMaximumHeight(40)
            view = cell.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
            cell.layout.setRowStretchFactor(0, 20)  # the image, not the label, grows
            cell.layout.setRowStretchFactor(1, 1)
            item = pg.ImageItem()
            view.addItem(item)
            if first is None:
                first = view
            elif k not in self.locked:
                view.setXLink(first)
                view.setYLink(first)
            item.mouseClickEvent = self._clicker(k)
            self.views.append(view)
            self.items.append(item)
            self.labels.append(label)
        self.redraw()
        if self.views:
            self.views[0].autoRange(padding=0.02)

    def _clicker(self, k: int) -> Callable[[Any], None]:
        def clicked(event: Any) -> None:
            self.current = k
            panel = self.comparison.panels[k]
            self.status.setText(f"Panel {k + 1}: {panel.well}, field {panel.field}")
            event.accept()

        return clicked

    def panel_rgb(self, k: int) -> np.ndarray:
        panel = self.comparison.panels[k]
        img = self.images[k]
        display = self.comparison.display_of(panel)
        rgb = compose(plane(img.image, panel.z), display)
        if panel.overlays:
            rgb = overlay(rgb, img, panel, panel.z)
        return rgb

    def redraw(self) -> None:
        from hcs_studio.compare.render import annotation_text

        if not self.images or len(self.images) != len(self.items):
            return
        template = self.comparison.settings.label_template
        for k, (item, label) in enumerate(zip(self.items, self.labels, strict=True)):
            rgb = self.panel_rgb(k)
            item.setImage((rgb * 255).astype(np.uint8).transpose(1, 0, 2), levels=(0, 255))
            panel = self.comparison.panels[k]
            text = panel.label or annotation_text(panel, template, self.images[k].field)
            if panel.display is not None and self.comparison.linked:
                text = "⚠ contrast unlinked · " + text
            if k in self.locked:
                text = "🔒 " + text
            # one part per line, so long labels do not widen the columns
            label.setText("<br>".join(html.escape(part) for part in text.split(" · ")))

    def toggle_lock(self) -> None:
        k = self.current
        if k in self.locked:
            self.locked.discard(k)
        else:
            self.locked.add(k)
        self._build_views()

    def set_crop(self, crop: tuple[float, float, float, float] | None) -> None:
        self.comparison = self.comparison.model_copy(update={"crop": crop})
        self.status.setText(
            "No crop" if crop is None else f"Crop (µm): {tuple(round(v, 1) for v in crop)}"
        )

    def crop_to_view(self) -> None:
        if not self.views or not self.images:
            return
        (x0, x1), (y0, y1) = self.views[0].viewRange()
        pixel = self.images[0].image.metadata.pixel_size_um or 1.0
        h, w = self.images[0].image.data.shape[-2:]
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        self.set_crop((y0 * pixel, x0 * pixel, y1 * pixel, x1 * pixel))

    # -------------------------------------------------------------- export

    def settings(self) -> ExportSettings:
        return self.comparison.settings.model_copy(
            update={
                "format": self.format.currentText(),
                "dpi": int(self.dpi.currentText()),
                "width_mm": PRESETS_MM[self.width_preset.currentText()],
                "split_channels": self.split.isChecked(),
                "headers": self.headers.isChecked(),
                "legend": self.legend.isChecked(),
                "font_size_pt": self.font_size.value(),
                "background": self.background.currentText(),
                "label_template": self.template.text(),
                "scale_bar": ScaleBar.model_validate(
                    {
                        "length_um": self.bar_length.value() or None,
                        "position": self.bar_position.currentText(),
                        "color": "#ffffff",
                    }
                ),
            }
        )

    def _apply_settings(self, s: ExportSettings) -> None:
        self.format.setCurrentText(s.format)
        self.dpi.setCurrentText(str(s.dpi))
        name = next((n for n, mm in PRESETS_MM.items() if mm == s.width_mm), None)
        if name:
            self.width_preset.setCurrentText(name)
        self.split.setChecked(s.split_channels)
        self.headers.setChecked(s.headers)
        self.legend.setChecked(s.legend)
        self.font_size.setValue(s.font_size_pt)
        self.template.setText(s.label_template)
        self.bar_length.setValue(s.scale_bar.length_um or 0)
        self.bar_position.setCurrentText(s.scale_bar.position)

    def export_comparison(self) -> Comparison:
        return self.comparison.model_copy(update={"settings": self.settings()})

    def _refresh_presets(self) -> None:
        self.presets.clear()
        self.presets.addItems(list_presets())

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save export preset", "Name:")
        if ok and name.strip():
            save_preset(name.strip(), self.settings())
            self._refresh_presets()
            self.presets.setCurrentText(name.strip())

    def _load_preset(self) -> None:
        if self.presets.currentText():
            self._apply_settings(load_preset(self.presets.currentText()))

    def _ready(self) -> bool:
        if not self.images:
            self.status.setText("Load some panels first.")
            return False
        return True

    def export_to(self, path: Path) -> Path:
        out = export_figure(self.export_comparison(), path, self.images)
        self.status.setText(f"Wrote {out.name} and {out.name}.json")
        self.log(f"Compare: figure {out}")
        return out

    def _export_figure(self) -> None:
        if not self._ready():
            return
        fmt = self.format.currentText()
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", f"comparison.{fmt}")
        if path:
            self.export_to(Path(path))

    def _export_data(self) -> None:
        if not self._ready():
            return
        folder = QFileDialog.getExistingDirectory(self, "Export panel data to")
        if folder:
            dtype, ok = QInputDialog.getItem(
                self, "Data type", "Pixels:", ["float32", "uint16"], 0, False
            )
            if ok:
                files = export_data(self.export_comparison(), Path(folder), dtype, self.images)
                self.status.setText(f"Wrote {len(files)} file(s) to {folder}")

    def _export_movie(self) -> None:
        if not self._ready():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export movie", "comparison.mp4", "Movies (*.mp4 *.gif)"
        )
        if path:
            try:
                out = export_movie(self.export_comparison(), Path(path), images=self.images)
            except ValueError as exc:
                self.status.setText(str(exc))
                return
            self.status.setText(f"Wrote {out.name}")

    def _batch(self) -> None:
        if self.run is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Write one figure per compound to")
        if folder:
            try:
                files = batch_compound_figures(self.run, Path(folder), self.export_comparison())
            except ValueError as exc:
                self.status.setText(str(exc))
                return
            self.status.setText(f"Wrote {len(files)} figure(s) to {folder}")


__all__ = ["ARRANGEMENTS", "ChannelDisplay", "ComparePanel"]
