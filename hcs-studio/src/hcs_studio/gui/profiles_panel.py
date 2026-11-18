"""Profiles tab: PCA / UMAP of phenotypic profiles, linked to the results plate and the
viewer; scree and loadings; dose-response curves; figure and coordinate export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.analysis.profiling import (
    PALETTE,
    ProfileOptions,
    ProfileResult,
    colour_key,
    fit_dose_response,
    four_pl,
    load_profile,
    run_profiling,
    save_scatter,
)
from hcs_studio.gui.workers import BackgroundTask

GROUP = "group (controls / compound)"
LEVELS = {"objects": "cell", "wells": "well"}


class LassoViewBox(pg.ViewBox):  # type: ignore[misc]
    """A view box that, in lasso mode, draws a free-hand polygon with the left button."""

    lasso_finished = Signal(object)  # list of (x, y) in data coordinates

    def __init__(self) -> None:
        super().__init__()
        self.lasso = False
        self._path: list[tuple[float, float]] = []
        self._curve = pg.PlotCurveItem(pen=pg.mkPen("#e8910c", width=2))
        self._curve.setZValue(10)
        self._curve.setVisible(False)
        self.addItem(self._curve, ignoreBounds=True)

    def mouseDragEvent(self, ev: Any, axis: Any = None) -> None:
        if not self.lasso or ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        point = self.mapSceneToView(ev.scenePos())
        if ev.isStart():
            self._path = []
            self._curve.setVisible(True)
        self._path.append((point.x(), point.y()))
        xs, ys = zip(*self._path, strict=True)
        self._curve.setData([*list(xs), xs[0]], [*list(ys), ys[0]])
        if ev.isFinish():
            self.finish_lasso(self._path)

    def finish_lasso(self, path: list[tuple[float, float]]) -> None:
        self._curve.setData([], [])
        self._curve.setVisible(False)  # (an empty curve also breaks the SVG exporter)
        if len(path) >= 3:
            self.lasso_finished.emit(list(path))


def inside(points: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """Points (n, 2) inside a polygon (even-odd rule)."""
    from matplotlib.path import Path as MplPath

    if not len(points):
        return np.zeros(0, dtype=bool)
    return np.asarray(MplPath(np.asarray(polygon)).contains_points(points), dtype=bool)


def image_from_rgb(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ProfilesPanel(QWidget):
    """``wells_selected(folder, wells)``: wells picked in the plot (for the results plate).
    ``open_object(folder, well, field, object_id, object_type)``: show an object in the
    viewer."""

    wells_selected = Signal(object, list)
    open_object = Signal(object, str, int, int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runs: list[Path] = []
        self.result: ProfileResult | None = None
        self.highlight: set[str] = set()
        self.selection = np.zeros(0, dtype=bool)
        self._points = pd.DataFrame()
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_computed)
        self.task.failed.connect(self._on_failed)

        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(70)
        add_run = QPushButton("Add plate (run folder)…")
        add_run.clicked.connect(self._choose_run)
        self.object_type = QComboBox()
        self.level = QComboBox()
        self.level.addItems(list(LEVELS))
        self.normalisation = QComboBox()
        self.normalisation.addItems(["negative_controls", "all_wells", "none"])
        self.plate_correction = QCheckBox("Centre each plate")
        self.correlation = QDoubleSpinBox()
        self.correlation.setRange(0.5, 1.0)
        self.correlation.setSingleStep(0.01)
        self.correlation.setValue(0.95)
        self.max_cells = QSpinBox()
        self.max_cells.setRange(100, 1_000_000)
        self.max_cells.setSingleStep(5000)
        self.max_cells.setValue(50_000)
        self.components = QSpinBox()
        self.components.setRange(2, 100)
        self.components.setValue(10)
        self.use_umap = QCheckBox("UMAP")
        self.use_umap.setChecked(True)
        self.neighbors = QSpinBox()
        self.neighbors.setRange(2, 200)
        self.neighbors.setValue(15)
        self.min_dist = QDoubleSpinBox()
        self.min_dist.setRange(0.0, 1.0)
        self.min_dist.setSingleStep(0.05)
        self.min_dist.setValue(0.1)
        self.umap_metric = QComboBox()
        self.umap_metric.addItems(["euclidean", "cosine", "correlation", "manhattan"])
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.dose_readouts = QLineEdit("profile_distance_to_control")
        self.dose_readouts.setToolTip("Comma-separated per-well readouts for dose-response fits")
        self.compute_button = QPushButton("Compute profiles")
        self.compute_button.clicked.connect(self.compute)
        self.status = QLabel("Open a run in the Results tab.")
        self.status.setWordWrap(True)

        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow(self.run_list)
        form.addRow(add_run)
        form.addRow("Objects", self.object_type)
        form.addRow("Embed", self.level)
        form.addRow("Normalise to", self.normalisation)
        form.addRow(self.plate_correction)
        form.addRow("Drop |r| above", self.correlation)
        form.addRow("Objects sampled", self.max_cells)
        form.addRow("PCA components", self.components)
        form.addRow(self.use_umap)
        form.addRow("n_neighbors", self.neighbors)
        form.addRow("min_dist", self.min_dist)
        form.addRow("Metric", self.umap_metric)
        form.addRow("Seed", self.seed)
        form.addRow("Dose readouts", self.dose_readouts)
        form.addRow(self.compute_button)
        form.addRow(self.status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(280)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_embedding_tab(), "Embedding")
        self.tabs.addTab(self._build_scree_tab(), "Scree and loadings")
        self.tabs.addTab(self._build_dose_tab(), "Dose-response")
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 4)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ------------------------------------------------------------- embedding tab

    def _build_embedding_tab(self) -> QWidget:
        self.space = QComboBox()
        self.space.currentTextChanged.connect(self.redraw)
        self.colour_by = QComboBox()
        self.colour_by.currentTextChanged.connect(self.redraw)
        self.lasso_button = QPushButton("Lasso")
        self.lasso_button.setCheckable(True)
        self.lasso_button.toggled.connect(self._toggle_lasso)
        clear = QPushButton("Clear selection")
        clear.clicked.connect(self.clear_selection)
        self.view_box = LassoViewBox()
        self.view_box.lasso_finished.connect(self.select_polygon)
        self.plot = pg.PlotWidget(viewBox=self.view_box, background="w")
        self.scatter = pg.ScatterPlotItem(pxMode=True)
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.plot.addItem(self.scatter)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.selection_label = QLabel("Click a point to see the object; draw a lasso to pick.")
        self.selection_label.setWordWrap(True)
        self.crop = QLabel()
        self.crop.setFixedSize(150, 150)
        self.crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop.setStyleSheet("background: #111; color: #aaa;")
        show = QPushButton("Open selected object in viewer")
        show.clicked.connect(self._open_selected)
        export_png = QPushButton("Export PNG…")
        export_png.clicked.connect(lambda: self._export_figure("png"))
        export_svg = QPushButton("Export SVG…")
        export_svg.clicked.connect(lambda: self._export_figure("svg"))
        export_csv = QPushButton("Export coordinates…")
        export_csv.clicked.connect(self._export_coordinates)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        bar = QHBoxLayout()
        for label, control in (("Space", self.space), ("Colour by", self.colour_by)):
            bar.addWidget(QLabel(label))
            bar.addWidget(control)
        bar.addWidget(self.lasso_button)
        bar.addWidget(clear)
        bar.addStretch(1)
        layout.addLayout(bar)
        body = QHBoxLayout()
        body.addWidget(self.plot, 1)
        side = QVBoxLayout()
        side.addWidget(self.crop)
        side.addWidget(self.selection_label)
        side.addWidget(show)
        side.addStretch(1)
        body.addLayout(side)
        layout.addLayout(body, 1)
        layout.addWidget(self.legend)
        exports = QHBoxLayout()
        for button in (export_png, export_svg, export_csv):
            exports.addWidget(button)
        exports.addStretch(1)
        layout.addLayout(exports)
        return widget

    def _toggle_lasso(self, on: bool) -> None:
        self.view_box.lasso = on
        self.view_box.setMouseEnabled(not on, not on)

    def points(self) -> pd.DataFrame:
        """The plotted points (objects or wells) of the current result."""
        return self._points

    def _coordinates(self) -> tuple[np.ndarray, np.ndarray] | None:
        frame = self._points
        prefix = "UMAP" if self.space.currentText() == "UMAP" else "PC"
        if frame.empty or f"{prefix}1" not in frame or f"{prefix}2" not in frame:
            return None
        return frame[f"{prefix}1"].to_numpy(float), frame[f"{prefix}2"].to_numpy(float)

    def redraw(self) -> None:
        self.scatter.clear()
        coords = self._coordinates()
        if coords is None:
            self.legend.setText("")
            return
        x, y = coords
        frame = self._points
        column = self.colour_by.currentText()
        values = colour_key(frame) if column == GROUP or column not in frame else frame[column]
        numeric = column not in (GROUP, "well", "plate") and pd.api.types.is_float_dtype(values)
        brushes: list[Any]
        if numeric:
            v = values.to_numpy(float)
            finite = v[np.isfinite(v)]
            lo, hi = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
            cmap = pg.colormap.get("viridis")
            t = np.clip((v - lo) / ((hi - lo) or 1), 0, 1)
            colours = cmap.map(np.nan_to_num(t), mode="qcolor")
            brushes = [
                pg.mkBrush(c) if np.isfinite(val) else pg.mkBrush("#bbb")
                for c, val in zip(colours, v, strict=True)
            ]
            self.legend.setText(f"{column}: {lo:.3g} (dark) to {hi:.3g} (yellow)")
        else:
            labels = values.astype(str).where(values.notna(), "–")
            names = sorted(labels.unique(), key=lambda s: (not s.endswith("control"), s))
            index = {n: i for i, n in enumerate(names)}
            brushes = [pg.mkBrush(PALETTE[index[n] % len(PALETTE)]) for n in labels]
            self.legend.setText(
                "  ".join(
                    f'<span style="color:{PALETTE[i % len(PALETTE)]}">●</span> {n}'
                    for i, n in enumerate(names[:24])
                )
            )
        level = self.level_of_result()
        size = 5 if level == "cell" else 10
        marked = self.selection if len(self.selection) == len(frame) else np.zeros(len(frame), bool)
        lit = frame["well"].isin(self.highlight).to_numpy() | marked
        sizes = np.where(lit, size + 5, size)
        pens = [pg.mkPen("k", width=2) if hit else pg.mkPen(None) for hit in lit]
        self.scatter.setData(
            x=x, y=y, brush=brushes, size=sizes, pen=pens, data=np.arange(len(frame))
        )
        prefix = self.space.currentText()
        variance = self.result.variance["explained_variance_ratio"].tolist() if self.result else []
        if prefix == "PCA" and len(variance) > 1:
            self.plot.setLabel("bottom", f"PC1 ({100 * variance[0]:.0f} %)")
            self.plot.setLabel("left", f"PC2 ({100 * variance[1]:.0f} %)")
        else:
            self.plot.setLabel("bottom", f"{'UMAP' if prefix == 'UMAP' else 'PC'}1")
            self.plot.setLabel("left", f"{'UMAP' if prefix == 'UMAP' else 'PC'}2")

    def level_of_result(self) -> str:
        if self.result is None or self._points.empty:
            return "cell"
        return "cell" if "object_id" in self._points else "well"

    def select_polygon(self, polygon: list[tuple[float, float]]) -> None:
        coords = self._coordinates()
        if coords is None:
            return
        self.selection = inside(np.column_stack(coords), polygon)
        self._report_selection()

    def select_indices(self, indices: list[int]) -> None:
        self.selection = np.zeros(len(self._points), dtype=bool)
        self.selection[indices] = True
        self._report_selection()

    def clear_selection(self) -> None:
        self.selection = np.zeros(len(self._points), dtype=bool)
        self._report_selection()

    def selected_points(self) -> pd.DataFrame:
        if len(self.selection) != len(self._points):
            return self._points.iloc[0:0]
        return self._points[self.selection]

    def _report_selection(self) -> None:
        picked = self.selected_points()
        wells = sorted(picked["well"].unique()) if not picked.empty else []
        what = "objects" if self.level_of_result() == "cell" else "wells"
        self.selection_label.setText(
            f"{len(picked)} {what} selected in {len(wells)} well(s)"
            + (f": {', '.join(wells[:12])}{'…' if len(wells) > 12 else ''}" if wells else "")
        )
        if self.runs:
            for run in self._runs_of(picked):
                mine = picked[picked["run"] == run.name] if "run" in picked else picked
                self.wells_selected.emit(run, sorted(mine["well"].unique()))
            if picked.empty:
                self.wells_selected.emit(self.runs[0], [])
        self.redraw()

    def _runs_of(self, frame: pd.DataFrame) -> list[Path]:
        if "run" not in frame:
            return self.runs[:1]
        names = set(frame["run"])
        return [r for r in self.runs if r.name in names]

    def _on_point_clicked(self, _item: Any, points: Any, *args: Any) -> None:
        if not len(points):
            return
        index = int(points[0].data())
        self.select_indices([index])
        self.show_point(index)

    def show_point(self, index: int) -> None:
        """Show the object (crop of its field preview) or the well behind a point."""
        record = self._points.iloc[index].to_dict()
        run = next((r for r in self.runs if r.name == record.get("run")), None) or (
            self.runs[0] if self.runs else None
        )
        self.crop.clear()
        if run is None:
            return
        if "object_id" in record and "centroid_y_um" in record:
            from hcs_studio.output.results import RunResults

            crop = RunResults(run).object_crop(
                str(record["well"]),
                int(record["field"]),
                float(record["centroid_y_um"]),
                float(record["centroid_x_um"]),
            )
            if crop is not None:
                pixmap = QPixmap.fromImage(image_from_rgb(crop))
                self.crop.setPixmap(pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                self.crop.setText("no preview")
            self.selection_label.setText(
                f"{record['well']} field {int(record['field'])} object "
                f"{int(record['object_id'])}"
                + (f" · {record['compound']}" if isinstance(record.get("compound"), str) else "")
            )
        else:
            self.selection_label.setText(
                f"Well {record['well']}"
                + (f" · {record['compound']}" if isinstance(record.get("compound"), str) else "")
            )

    def _open_selected(self) -> None:
        picked = self.selected_points()
        if picked.empty or "object_id" not in picked or not self.runs:
            self.selection_label.setText("Select an object (embed objects, then click a point).")
            return
        record = picked.iloc[0]
        run = next((r for r in self.runs if r.name == record.get("run")), self.runs[0])
        self.open_object.emit(
            run,
            str(record["well"]),
            int(record["field"]),
            int(record["object_id"]),
            self.object_type.currentText(),
        )

    def highlight_wells(self, wells: list[str]) -> None:
        """Plate → plot: enlarge the points of these wells."""
        self.highlight = set(wells)
        self.redraw()

    def _export_figure(self, kind: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", f"profiles.{kind}", f"{kind.upper()} (*.{kind})"
        )
        if path:
            self.export_figure(Path(path))

    def export_figure(self, path: Path) -> None:
        """The current embedding as a figure (SVG or PNG), coloured as on screen."""
        coords = self._coordinates()
        if coords is None:
            return
        frame = self._points
        column = self.colour_by.currentText()
        values = colour_key(frame) if column == GROUP or column not in frame else frame[column]
        prefix = "UMAP" if self.space.currentText() == "UMAP" else "PC"
        marked = self.selection if len(self.selection) == len(frame) else None
        save_scatter(
            path,
            coords[0],
            coords[1],
            values.rename(column),
            title=f"{self.space.currentText()} of {self.object_type.currentText()}",
            xlabel=f"{prefix}1",
            ylabel=f"{prefix}2",
            highlight=marked,
        )

    def _export_coordinates(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export coordinates", "profile_coordinates.csv", "CSV (*.csv)"
        )
        if path:
            self._points.to_csv(path, index=False)

    # --------------------------------------------------------------- scree tab

    def _build_scree_tab(self) -> QWidget:
        self.scree = pg.PlotWidget(background="w")
        self.scree.setLabel("bottom", "Principal component")
        self.scree.setLabel("left", "Variance explained (%)")
        self.component = QComboBox()
        self.component.currentTextChanged.connect(self._fill_loadings)
        self.loadings = QTableWidget(0, 2)
        self.loadings.setHorizontalHeaderLabels(["Feature", "Loading"])
        self.loadings.horizontalHeader().setStretchLastSection(True)
        self.dropped = QLabel("")
        self.dropped.setWordWrap(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.scree, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("Loadings of"))
        row.addWidget(self.component)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.loadings, 2)
        layout.addWidget(self.dropped)
        return widget

    def _fill_scree(self) -> None:
        self.scree.clear()
        if self.result is None:
            return
        variance = self.result.variance
        x = np.arange(1, len(variance) + 1)
        self.scree.addItem(
            pg.BarGraphItem(
                x=x, height=variance["explained_variance_ratio"] * 100, width=0.7, brush="#4477aa"
            )
        )
        self.scree.plot(
            x, variance["cumulative"] * 100, pen=pg.mkPen("#222"), symbol="o", symbolSize=5
        )
        self.component.blockSignals(True)
        self.component.clear()
        self.component.addItems(list(self.result.loadings.columns))
        self.component.blockSignals(False)
        self._fill_loadings()
        dropped = self.result.dropped
        self.dropped.setText(
            f"{len(self.result.features)} features used; {len(dropped)} dropped"
            + (
                ": " + "; ".join(f"{k} ({v})" for k, v in list(dropped.items())[:8])
                if dropped
                else ""
            )
            + ("…" if len(dropped) > 8 else "")
        )

    def _fill_loadings(self) -> None:
        if self.result is None or not self.component.currentText():
            return
        column = self.result.loadings[self.component.currentText()]
        top = column.reindex(column.abs().sort_values(ascending=False).index)[:30]
        self.loadings.setRowCount(len(top))
        for row, (name, value) in enumerate(top.items()):
            self.loadings.setItem(row, 0, QTableWidgetItem(str(name)))
            item = QTableWidgetItem(f"{value:+.3f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.loadings.setItem(row, 1, item)

    # ---------------------------------------------------------------- dose tab

    def _build_dose_tab(self) -> QWidget:
        self.compound = QComboBox()
        self.compound.currentTextChanged.connect(self.draw_dose)
        self.readout = QComboBox()
        self.readout.currentTextChanged.connect(self.draw_dose)
        self.dose_plot = pg.PlotWidget(background="w")
        self.dose_plot.setLogMode(x=True, y=False)
        self.dose_plot.setLabel("bottom", "Concentration")
        self.dose_label = QLabel("")
        self.dose_label.setWordWrap(True)
        export = QPushButton("Export PNG…")
        export.clicked.connect(self._export_dose)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        for label, control in (("Compound", self.compound), ("Readout", self.readout)):
            row.addWidget(QLabel(label))
            row.addWidget(control, 1)
        layout.addLayout(row)
        layout.addWidget(self.dose_plot, 1)
        layout.addWidget(self.dose_label)
        layout.addWidget(export)
        return widget

    def _fill_dose_choices(self) -> None:
        if self.result is None:
            return
        wells = self.result.wells
        treated = wells[wells["concentration"].notna() & wells["control_type"].isna()]
        compounds = sorted(treated["compound"].dropna().unique())
        numeric = [
            c
            for c in wells.select_dtypes("number").columns
            if c not in ("concentration", "n_objects")
        ]
        preferred = [c for c in numeric if c.startswith("profile_")] + [
            c for c in numeric if not c.startswith("profile_")
        ]
        for combo, items in ((self.compound, compounds), (self.readout, preferred)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(i) for i in items])
            if keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        self.draw_dose()

    def draw_dose(self) -> dict[str, Any] | None:
        """Plot the chosen compound and readout with its 4-parameter logistic fit."""
        self.dose_plot.clear()
        if self.result is None or not self.compound.currentText() or not self.readout.currentText():
            return None
        wells = self.result.wells
        readout = self.readout.currentText()
        group = wells[
            (wells["compound"] == self.compound.currentText())
            & (wells["concentration"] > 0)
            & wells["control_type"].isna()
        ]
        if readout not in group or group.empty:
            return None
        stats = group.groupby("concentration")[readout].agg(["mean", "std"]).reset_index()
        x = np.log10(stats["concentration"].to_numpy(float))
        self.dose_plot.addItem(
            pg.ErrorBarItem(
                x=x,
                y=stats["mean"].to_numpy(float),
                height=2 * stats["std"].fillna(0).to_numpy(float),
                beam=0.05,
            )
        )
        self.dose_plot.plot(
            stats["concentration"].to_numpy(float),
            stats["mean"].to_numpy(float),
            pen=None,
            symbol="o",
            symbolBrush="#4477aa",
            symbolSize=8,
        )
        controls = wells[wells["control_type"] == "negative"][readout].dropna()
        if len(controls):
            self.dose_plot.addItem(
                pg.InfiniteLine(
                    float(controls.median()),
                    angle=0,
                    pen=pg.mkPen("#999", style=Qt.PenStyle.DashLine),
                )
            )
        fit = fit_dose_response(group["concentration"].to_numpy(), group[readout].to_numpy())
        if fit.get("fit"):
            grid = np.logspace(x.min(), x.max(), 100)
            curve = four_pl(grid, fit["bottom"], fit["top"], math.log10(fit["ec50"]), fit["hill"])
            self.dose_plot.plot(grid, curve, pen=pg.mkPen("#ee6677", width=2))
            self.dose_label.setText(
                f"EC50 {fit['ec50']:.3g} · Hill {fit['hill']:.2f} · bottom {fit['bottom']:.3g} · "
                f"top {fit['top']:.3g} · R² {fit['r2']:.3f} · {fit['n']} wells"
                + (f" · {fit['note']}" if fit.get("note") else "")
                + " (dashed: median of negative controls)"
            )
        else:
            self.dose_label.setText(f"No fit: {fit.get('note', '')}")
        self.dose_plot.setLabel("left", readout)
        return fit

    def _export_dose(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", "dose.png", "PNG (*.png)")
        if path:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(self.dose_plot.getPlotItem())
            exporter.parameters()["width"] = 1600
            exporter.export(path)

    # ------------------------------------------------------------------ compute

    def set_run(self, folder: Path) -> None:
        """Use a run (from the Results tab); shows its saved profiles, if any."""
        from hcs_studio.output.results import RunResults

        folder = Path(folder)
        self.runs = [folder]
        self.run_list.clear()
        self.run_list.addItem(folder.name)
        try:
            kinds = list(RunResults(folder).object_features())
        except ValueError:
            kinds = []
        saved = folder / "advanced" / "profiling.json"
        if saved.exists():  # the object type profiled before, even without a plate table
            import json

            try:
                kind = json.loads(saved.read_text(encoding="utf-8"))["options"]["object_type"]
            except (OSError, ValueError, KeyError):
                kind = None
            if kind and kind not in kinds:
                kinds.append(kind)
        keep = self.object_type.currentText()
        self.object_type.clear()
        self.object_type.addItems(kinds)
        if keep in kinds:
            self.object_type.setCurrentText(keep)
        self.highlight = set()
        result = load_profile(folder / "advanced")
        if result is not None:
            self.show_result(result)
            self.status.setText(f"Saved profiles of {folder.name}. Compute again to change them.")
        else:
            self.result = None
            self._points = pd.DataFrame()
            self.redraw()
            self.status.setText("No profiles yet: choose options and Compute profiles.")

    def _choose_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder of another plate")
        if folder:
            self.add_run(Path(folder))

    def add_run(self, folder: Path) -> None:
        if folder not in self.runs:
            self.runs.append(folder)
            self.run_list.addItem(folder.name)

    def options(self) -> ProfileOptions:
        return ProfileOptions(
            object_type=self.object_type.currentText() or "nuclei",
            level=LEVELS[self.level.currentText()],  # type: ignore[arg-type]
            normalisation=self.normalisation.currentText(),  # type: ignore[arg-type]
            plate_correction=self.plate_correction.isChecked(),
            correlation_threshold=self.correlation.value(),
            max_cells=self.max_cells.value(),
            pca_components=self.components.value(),
            umap=self.use_umap.isChecked(),
            n_neighbors=self.neighbors.value(),
            min_dist=self.min_dist.value(),
            metric=self.umap_metric.currentText(),
            seed=self.seed.value(),
            dose_readouts=[r.strip() for r in self.dose_readouts.text().split(",") if r.strip()],
        )

    def compute(self) -> None:
        if not self.runs:
            self.status.setText("Open a run in the Results tab first.")
            return
        options, runs = self.options(), list(self.runs)
        self.compute_button.setEnabled(False)
        self.status.setText(f"Computing profiles of {len(runs)} plate(s)…")
        self.task.request(lambda: run_profiling(runs, options))

    def _on_computed(self, result: ProfileResult) -> None:
        self.compute_button.setEnabled(True)
        notes = " ".join(result.notes)
        self.status.setText(
            f"{len(result.cells)} points, {len(result.features)} features "
            f"({len(result.dropped)} dropped). Saved in {result.folder}. {notes}"
        )
        self.show_result(result)

    def _on_failed(self, message: str) -> None:
        self.compute_button.setEnabled(True)
        self.status.setText(f"Profiling failed: {message}")

    def show_result(self, result: ProfileResult) -> None:
        self.result = result
        self._points = result.cells.reset_index(drop=True)
        self.selection = np.zeros(len(self._points), dtype=bool)
        spaces = ["PCA"] + (["UMAP"] if "UMAP1" in self._points else [])
        keep = self.space.currentText()
        self.space.blockSignals(True)
        self.space.clear()
        self.space.addItems(spaces)
        if keep in spaces:
            self.space.setCurrentText(keep)
        self.space.blockSignals(False)
        colour_options = (
            [GROUP]
            + [
                c
                for c in (
                    "cell_line",
                    "compound",
                    "concentration",
                    "role",
                    "replicate_group",
                    "treatment",
                    "control_type",
                    "well",
                    "plate",
                )
                if c in self._points
            ]
            + [
                c
                for c in self._points.columns
                if str(c).startswith(("class_", "gate_", "cluster_", "spatial_"))
            ]
        )
        keep = self.colour_by.currentText()
        self.colour_by.blockSignals(True)
        self.colour_by.clear()
        self.colour_by.addItems(colour_options)
        if keep in colour_options:
            self.colour_by.setCurrentText(keep)
        self.colour_by.blockSignals(False)
        self.redraw()
        self.plot.getPlotItem().enableAutoRange()
        self._fill_scree()
        self._fill_dose_choices()

    # convenience for tests and callers
    def point_position(self, index: int) -> QPointF:
        coords = self._coordinates()
        assert coords is not None
        return QPointF(float(coords[0][index]), float(coords[1][index]))
