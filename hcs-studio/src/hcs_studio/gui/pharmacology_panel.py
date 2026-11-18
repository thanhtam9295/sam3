"""Pharmacology tab: dose–response curves, combination heat maps, synergy and QC tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.model import units
from hcs_studio.pharmacology.analysis import (
    CurveResult,
    PharmOptions,
    PharmResult,
    curves_table,
)
from hcs_studio.pharmacology.combination import MODELS, CombinationResult
from hcs_studio.pharmacology.normalize import LABELS, NORMALIZATIONS

COLOURS = ("#2a6fdb", "#e4572e", "#29a36a", "#8e44ad", "#f2a541", "#17becf", "#7f7f7f")
Selection = list[tuple[str, str, str]]  # (run, plate, well)


def _conc(value: float, unit: str) -> str:
    try:
        return units.format_concentration(value, unit)
    except (ValueError, KeyError):
        return f"{value:.3g}"


class PharmacologyPanel(QWidget):
    """``wells_selected(run folder name, wells)``: points or matrix cells clicked."""

    wells_selected = Signal(str, list)
    show_well = Signal(str, str)

    def __init__(self, log: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.log = log or (lambda text: None)
        self.runs: list[Path] = []
        self.result: PharmResult | None = None
        self.excluded: set[tuple[str, str]] = set()
        self.included: set[tuple[str, str]] = set()
        self.selection: Selection = []
        self._write_next = True
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_finished)
        self.task.failed.connect(self._on_failed)

        # --- options
        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(70)
        add = QPushButton("Add run…")
        add.clicked.connect(self._choose_run)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_run)
        self.readouts = QListWidget()
        self.readouts.setMaximumHeight(120)
        self.normalization = QComboBox()
        for name in NORMALIZATIONS:
            self.normalization.addItem(LABELS[name], name)
        self.normalization.setCurrentIndex(1)
        self.curve_model = QComboBox()
        self.curve_model.addItems(["4pl", "5pl", "3pl_bottom", "3pl_top"])
        self.robust = QCheckBox("Robust fit")
        self.model_boxes = {m: QCheckBox(m.upper()) for m in MODELS}
        for box in self.model_boxes.values():
            box.setChecked(True)
        self.outliers = QDoubleSpinBox()
        self.outliers.setRange(0, 20)
        self.outliers.setDecimals(1)
        self.outliers.setSpecialValueText("off")
        self.outliers.setToolTip("Exclude wells beyond median ± k·MAD of their replicates")
        self.z_threshold = QDoubleSpinBox()
        self.z_threshold.setRange(-5, 1)
        self.z_threshold.setValue(0.5)
        self.doubling = QDoubleSpinBox()
        self.doubling.setRange(0, 1000)
        self.doubling.setSuffix(" h")
        self.doubling.setSpecialValueText("—")
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0, 1000)
        self.duration.setSuffix(" h")
        self.duration.setSpecialValueText("—")
        self.analyse_button = QPushButton("Analyse")
        self.analyse_button.clicked.connect(lambda: self.analyse(write=True))
        self.status = QLabel("Open a run in the Results tab, or add runs, then choose readouts.")
        self.status.setWordWrap(True)

        form = QFormLayout()
        runs_row = QHBoxLayout()
        runs_row.addWidget(add)
        runs_row.addWidget(remove)
        form.addRow("Runs (plates merged)", self.run_list)
        form.addRow("", runs_row)
        form.addRow("Readouts", self.readouts)
        form.addRow("Normalization", self.normalization)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.curve_model)
        fit_row.addWidget(self.robust)
        form.addRow("Curve model", fit_row)
        models_row = QHBoxLayout()
        for box in self.model_boxes.values():
            models_row.addWidget(box)
        form.addRow("Synergy models", models_row)
        form.addRow("Outliers (k × MAD)", self.outliers)
        form.addRow("Flag Z′ below", self.z_threshold)
        gr_row = QHBoxLayout()
        gr_row.addWidget(self.doubling)
        gr_row.addWidget(QLabel("over"))
        gr_row.addWidget(self.duration)
        form.addRow("GR: doubling time", gr_row)
        form.addRow(self.analyse_button)
        form.addRow(self.status)
        options = QWidget()
        options.setLayout(form)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Results"])
        self.tree.currentItemChanged.connect(lambda item, _: self._show_item(item))
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(options)
        left.addWidget(self.tree)

        # --- views
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.addLegend(offset=(10, 10))
        self.curve_plot.setLabel("bottom", "log10 concentration")
        self.curve_info = QLabel("")
        self.curve_info.setWordWrap(True)
        curve_page = QWidget()
        curve_layout = QVBoxLayout(curve_page)
        curve_layout.addWidget(self.curve_plot, 1)
        curve_layout.addWidget(self.curve_info)

        self.matrix_model = QComboBox()
        self.matrix_model.currentIndexChanged.connect(lambda _: self._draw_matrix())
        self.heat = pg.GraphicsLayoutWidget()
        self.response_plot = self.heat.addPlot(title="response (% inhibition)")
        self.synergy_plot = self.heat.addPlot(title="synergy")
        self.response_image = pg.ImageItem()
        self.synergy_image = pg.ImageItem()
        self.response_plot.addItem(self.response_image)
        self.synergy_plot.addItem(self.synergy_image)
        for plot, image in (
            (self.response_plot, self.response_image),
            (self.synergy_plot, self.synergy_image),
        ):
            plot.setAspectLocked(False)
            plot.invertY(False)
            image.mouseClickEvent = self._matrix_clicker(image)
        self.matrix_info = QLabel("")
        self.matrix_info.setWordWrap(True)
        matrix_page = QWidget()
        matrix_layout = QVBoxLayout(matrix_page)
        top = QHBoxLayout()
        top.addWidget(QLabel("Synergy model"))
        top.addWidget(self.matrix_model)
        top.addStretch(1)
        matrix_layout.addLayout(top)
        matrix_layout.addWidget(self.heat, 1)
        matrix_layout.addWidget(self.matrix_info)

        self.views = QStackedWidget()
        self.views.addWidget(QLabel("Choose a curve or a combination in the list."))
        self.views.addWidget(curve_page)
        self.views.addWidget(matrix_page)

        self.selection_label = QLabel("Click a point or a matrix cell to select its wells.")
        self.selection_label.setWordWrap(True)
        toggle = QPushButton("Exclude / include selected wells")
        toggle.clicked.connect(self.toggle_selected)
        show = QPushButton("Show in Results")
        show.clicked.connect(self._show_selected)
        save = QPushButton("Write report")
        save.setToolTip(
            "Write pharmacology/ (tables, figures, HTML and PDF) with the current exclusions"
        )
        save.clicked.connect(lambda: self.analyse(write=True))
        actions = QHBoxLayout()
        for widget in (self.selection_label, toggle, show, save):
            actions.addWidget(widget)

        self.fits_table = QTableWidget()
        self.synergy_table = QTableWidget()
        self.qc_table = QTableWidget()
        for table in (self.fits_table, self.synergy_table, self.qc_table):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSortingEnabled(True)
        self.tables = QTabWidget()
        self.tables.addTab(self.fits_table, "IC50 / AUC")
        self.tables.addTab(self.synergy_table, "Synergy ranking")
        self.tables.addTab(self.qc_table, "Plate QC")

        right = QSplitter(Qt.Orientation.Vertical)
        view_box = QWidget()
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(self.views, 1)
        view_layout.addLayout(actions)
        right.addWidget(view_box)
        right.addWidget(self.tables)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._current: CurveResult | CombinationResult | None = None
        self._points: list[tuple[Any, CurveResult]] = []

    # ------------------------------------------------------------------- runs

    def set_run(self, folder: Path) -> None:
        """Use a run (from the Results tab)."""
        self.runs = [Path(folder)]
        self.excluded.clear()
        self.included.clear()
        self._refresh_runs()

    def add_run(self, folder: Path) -> None:
        if Path(folder) not in self.runs:
            self.runs.append(Path(folder))
            self._refresh_runs()

    def _choose_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder")
        if folder:
            self.add_run(Path(folder))

    def _remove_run(self) -> None:
        row = self.run_list.currentRow()
        if 0 <= row < len(self.runs):
            del self.runs[row]
            self._refresh_runs()

    def _refresh_runs(self) -> None:
        from hcs_studio.pharmacology.normalize import load_wells, readout_choices

        self.run_list.clear()
        self.run_list.addItems([p.name for p in self.runs])
        checked = {
            self.readouts.item(i).text()
            for i in range(self.readouts.count())
            if self.readouts.item(i).checkState() == Qt.CheckState.Checked
        }
        self.readouts.clear()
        try:
            choices = readout_choices(load_wells(self.runs)) if self.runs else []
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Could not read the runs: {exc}")
            return
        preferred = next(
            (c for c in choices if c.endswith("_count")), choices[0] if choices else ""
        )
        for name in choices:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = name in checked or (not checked and name == preferred)
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.readouts.addItem(item)
        self.status.setText(
            f"{len(self.runs)} run(s), {len(choices)} readouts. Choose and Analyse."
        )

    def set_readouts(self, names: list[str]) -> None:
        for i in range(self.readouts.count()):
            item = self.readouts.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in names else Qt.CheckState.Unchecked
            )

    # ---------------------------------------------------------------- analyse

    def options(self) -> PharmOptions:
        readouts = [
            self.readouts.item(i).text()
            for i in range(self.readouts.count())
            if self.readouts.item(i).checkState() == Qt.CheckState.Checked
        ]
        return PharmOptions(
            readouts=readouts,
            normalization=str(self.normalization.currentData()),
            models=tuple(m for m, box in self.model_boxes.items() if box.isChecked()),
            curve_model=self.curve_model.currentText(),
            robust=self.robust.isChecked(),
            outlier_mad=self.outliers.value() or None,
            z_threshold=self.z_threshold.value(),
            doubling_time_h=self.doubling.value() or None,
            assay_duration_h=self.duration.value() or None,
            excluded_wells=sorted(self.excluded),
            included_wells=sorted(self.included),
        )

    def analyse(self, write: bool = True) -> None:
        if not self.runs:
            self.status.setText("Add a run first.")
            return
        options = self.options()
        if not options.readouts:
            self.status.setText("Tick at least one readout.")
            return
        runs = list(self.runs)
        self._write_next = write
        self.analyse_button.setEnabled(False)
        self.status.setText("Analysing…" + (" and writing the report…" if write else ""))

        def work() -> PharmResult:
            from hcs_studio.pharmacology.analysis import run_pharmacology

            return run_pharmacology(runs, options, write=write)

        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self.analyse_button.setEnabled(True)
        self.status.setText(f"Analysis failed: {message}")
        self.log(f"Pharmacology failed: {message}")

    def _on_finished(self, result: PharmResult) -> None:
        self.analyse_button.setEnabled(True)
        self.set_result(result)
        where = f" — wrote {result.folder}" if result.folder else ""
        self.status.setText(f"Done in {result.seconds:.1f} s{where}")
        self.log(f"Pharmacology: {len(result.readouts)} readout(s){where}")

    # ---------------------------------------------------------------- results

    def set_result(self, result: PharmResult) -> None:
        keep = self._key(self._current) if self._current is not None else None
        self.result = result
        self.tree.clear()
        self.matrix_model.blockSignals(True)
        self.matrix_model.clear()
        self.matrix_model.addItems([m for m in result.options.models])
        self.matrix_model.blockSignals(False)
        restore = None
        for readout, rr in result.readouts.items():
            top = QTreeWidgetItem([readout])
            self.tree.addTopLevelItem(top)
            lines: dict[str, QTreeWidgetItem] = {}

            def line_item(
                name: str,
                top: QTreeWidgetItem = top,
                lines: dict[str, QTreeWidgetItem] = lines,
            ) -> QTreeWidgetItem:
                if name not in lines:
                    lines[name] = QTreeWidgetItem(top, [name or "no cell line"])
                return lines[name]

            for c in rr.curves:
                line = line_item(c.cell_line)
                item = QTreeWidgetItem(line, [f"{c.compound}  {self._ec50_text(c)}"])
                item.setData(0, Qt.ItemDataRole.UserRole, c)
                if self._key(c) == keep:
                    restore = item
            for combo in rr.combinations:
                m = combo.matrix
                line = line_item(m.cell_line)
                first = next(iter(combo.models.values()), None)
                label = f"{m.drug_a} × {m.drug_b}" + (f"  ({first.label})" if first else "")
                item = QTreeWidgetItem(line, [label])
                item.setData(0, Qt.ItemDataRole.UserRole, combo)
                if self._key(combo) == keep:
                    restore = item
            for note in rr.notes:
                QTreeWidgetItem(top, [f"⚠ {note}"])
        self.tree.expandAll()
        self._fill_tables(result)
        if restore is not None:
            self.tree.setCurrentItem(restore)

    @staticmethod
    def _key(item: Any) -> tuple[str, ...]:
        if isinstance(item, CurveResult):
            return ("curve", *item.key)
        if isinstance(item, CombinationResult):
            m = item.matrix
            return ("combo", m.cell_line, m.drug_a, m.drug_b)
        return ()

    @staticmethod
    def _ec50_text(c: CurveResult) -> str:
        return f"EC50 {_conc(c.fit.ec50, c.unit)}" if math.isfinite(c.fit.log_ec50) else "no fit"

    def _show_item(self, item: QTreeWidgetItem | None) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        self._current = data
        if isinstance(data, CurveResult):
            self.views.setCurrentIndex(1)
            self._draw_curve(data)
        elif isinstance(data, CombinationResult):
            self.views.setCurrentIndex(2)
            self._draw_matrix()
        else:
            self.views.setCurrentIndex(0)

    # ----------------------------------------------------------------- curves

    def _draw_curve(self, c: CurveResult) -> None:
        plot = self.curve_plot
        plot.clear()
        self._points = []
        method = self.result.options.normalization if self.result else ""
        plot.setLabel("left", LABELS.get(method, method))
        # x: log10 of the concentration in the compound's display unit
        try:
            shift = math.log10(units.from_base(1.0, c.unit))
        except (ValueError, KeyError):
            shift = 0.0
        plot.setLabel("bottom", f"log10 concentration ({c.unit or 'base unit'})")
        colour = QColor(COLOURS[0])
        points = c.points
        spots = []
        conc = points["conc"].to_numpy(float)
        values = points["normalized"].to_numpy(float)
        excluded = points["excluded"].to_numpy(bool)
        for k in range(len(points)):
            spots.append(
                {
                    "pos": (math.log10(conc[k]) + shift, values[k]),
                    "data": k,
                    "brush": QBrush(Qt.BrushStyle.NoBrush) if excluded[k] else pg.mkBrush(colour),
                    "pen": pg.mkPen(colour, width=1.5),
                    "size": 9,
                }
            )
        scatter = pg.ScatterPlotItem(spots=spots, name=f"{c.compound} ({c.cell_line})")
        scatter.sigClicked.connect(lambda item, hits, event: self._points_clicked(c, hits, event))
        plot.addItem(scatter)
        self._points.append((scatter, c))
        used = points[~points["excluded"]]
        means = used.groupby("conc")["normalized"].agg(["mean", "std"]).reset_index()
        x = np.log10(means["conc"].to_numpy(float))  # base unit, as the fit
        plot.addItem(
            pg.ErrorBarItem(
                x=x + shift,
                y=means["mean"].to_numpy(float),
                height=2 * means["std"].fillna(0).to_numpy(float),
                pen=pg.mkPen(colour),
            )
        )
        if math.isfinite(c.fit.top):
            xs = np.linspace(float(np.min(x)) - 0.3, float(np.max(x)) + 0.3, 200)
            plot.plot(xs + shift, c.fit.predict(xs), pen=pg.mkPen(colour, width=2))
        f = c.fit
        extra = "".join(f", {k} {v:.3g}" for k, v in c.gr.items() if math.isfinite(v))
        self.curve_info.setText(
            f"{c.compound} · {c.cell_line}: EC50 {_conc(f.ec50, c.unit)} "
            f"[{self._log_conc(f.log_ec50_ci[0], c.unit)}, "
            f"{self._log_conc(f.log_ec50_ci[1], c.unit)}], "
            f"absolute IC50 {self._log_conc(f.log_abs_ic50, c.unit)}, "
            f"Hill {f.hill:.2f}, top {f.top:.1f}, bottom {f.bottom:.1f}, R² {f.r2:.3f}, "
            f"AUC {c.auc:.1f}, Emax {c.emax_observed:.1f} %{extra}. "
            + ("; ".join(f.flags) if f.flags else "")
            + "  Click a point to select its well; Ctrl+click to exclude or include it."
        )

    @staticmethod
    def _log_conc(log_value: float, unit: str) -> str:
        return _conc(10**log_value, unit) if math.isfinite(log_value) else "—"

    def _points_clicked(self, c: CurveResult, hits: Any, event: Any) -> None:
        if not len(hits):
            return
        k = int(hits[0].data())
        plate = str(c.points["plate"].iloc[k])
        well = str(c.points["well"].iloc[k])
        self.select([(self._run_of(plate), plate, well)])
        modifiers = event.modifiers() if event is not None else Qt.KeyboardModifier.NoModifier
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.toggle_selected()

    # ---------------------------------------------------------------- matrices

    def _draw_matrix(self) -> None:
        combo = self._current
        if not isinstance(combo, CombinationResult):
            return
        m = combo.matrix
        model = self.matrix_model.currentText() or next(iter(combo.models), "")
        res = combo.models.get(model)
        self.response_image.setImage(np.nan_to_num(m.response.T, nan=-1), levels=(0, 100))
        self.response_image.setColorMap(pg.colormap.get("viridis"))
        if res is not None:
            bound = max(
                10.0,
                float(np.nanmax(np.abs(res.synergy))) if np.isfinite(res.synergy).any() else 10,
            )
            self.synergy_image.setImage(np.nan_to_num(res.synergy.T), levels=(-bound, bound))
            self.synergy_image.setColorMap(pg.colormap.get("CET-D1"))
            self.synergy_plot.setTitle(
                f"{model.upper()} synergy: mean {res.mean:.1f} ({res.label})"
            )
        for plot, values in (
            (self.response_plot, m.response),
            (self.synergy_plot, res.synergy if res else m.response),
        ):
            for item in [i for i in plot.items if isinstance(i, pg.TextItem)]:
                plot.removeItem(item)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    v = values[i, j]
                    dark = plot is self.response_plot and math.isfinite(v) and v < 50
                    text = pg.TextItem(
                        "×" if not math.isfinite(v) else f"{v:.0f}",
                        color="#999" if not math.isfinite(v) else ("w" if dark else "k"),
                        anchor=(0.5, 0.5),
                    )
                    text.setPos(j + 0.5, i + 0.5)
                    plot.addItem(text)
            plot.getAxis("bottom").setTicks(
                [
                    [
                        (j + 0.5, f"{v:.3g}")
                        for j, v in enumerate([units.from_base(c, m.unit_b) for c in m.conc_b])
                    ]
                ]
            )
            plot.getAxis("left").setTicks(
                [
                    [
                        (i + 0.5, f"{v:.3g}")
                        for i, v in enumerate([units.from_base(c, m.unit_a) for c in m.conc_a])
                    ]
                ]
            )
            plot.setLabel("bottom", f"{m.drug_b} ({m.unit_b})")
            plot.setLabel("left", f"{m.drug_a} ({m.unit_a})")
        notes = "; ".join(n for r in combo.models.values() for n in r.notes)
        scores = ", ".join(
            f"{k.upper()} {r.mean:+.1f} (3×3 max {r.window_mean:+.1f}, p {r.p_value:.2g})"
            for k, r in combo.models.items()
        )
        self.matrix_info.setText(
            f"{m.name}: {scores}. {m.missing} missing cell(s) marked ×. {notes} "
            "Labels follow SynergyFinder conventions: a guide, not proof. Click a cell to "
            "select its wells."
        )

    def _matrix_clicker(self, image: pg.ImageItem) -> Any:
        def clicked(event: Any) -> None:
            combo = self._current
            if not isinstance(combo, CombinationResult):
                return
            pos = event.pos()
            j, i = int(pos.x()), int(pos.y())
            self.select_cell(i, j)
            event.accept()

        return clicked

    def select_cell(self, i: int, j: int) -> None:
        combo = self._current
        if not isinstance(combo, CombinationResult):
            return
        wells = combo.matrix.wells.get((i, j), [])
        self.select([(self._run_of(p), p, w) for p, w in wells])

    # -------------------------------------------------------------- selection

    def _run_of(self, plate: str) -> str:
        if self.result is None:
            return ""
        for rr in self.result.readouts.values():
            hit = rr.data.wells[rr.data.wells["plate"] == plate]
            if len(hit):
                return str(hit["run"].iloc[0])
        return ""

    def select(self, wells: Selection) -> None:
        self.selection = wells
        names = ", ".join(f"{w}" for _, _, w in wells[:8]) + (" …" if len(wells) > 8 else "")
        self.selection_label.setText(f"Selected: {names}" if wells else "Nothing selected.")
        by_run: dict[str, list[str]] = {}
        for run, _, well in wells:
            by_run.setdefault(run, []).append(well)
        for run, names_ in by_run.items():
            self.wells_selected.emit(run, names_)

    def toggle_selected(self) -> None:
        """Exclude the selected wells (or include them again) and re-analyse."""
        if not self.selection or self.result is None:
            return
        excluded_now = set()
        for rr in self.result.readouts.values():
            frame = rr.data.wells
            excluded_now |= set(
                map(tuple, frame.loc[frame["excluded"], ["plate", "well"]].to_numpy())
            )
        for _, plate, well in self.selection:
            key = (plate, well)
            if key in excluded_now:
                self.excluded.discard(key)
                self.included.add(key)
            else:
                self.included.discard(key)
                self.excluded.add(key)
        self.analyse(write=False)

    def _show_selected(self) -> None:
        if self.selection:
            run, _, well = self.selection[0]
            self.show_well.emit(run, well)

    # ----------------------------------------------------------------- tables

    def _fill_tables(self, result: PharmResult) -> None:
        import pandas as pd

        fits = pd.concat(
            [curves_table(rr.curves) for rr in result.readouts.values()], ignore_index=True
        )
        self._fill(
            self.fits_table,
            fits,
            [
                "readout",
                "cell_line",
                "compound",
                "ec50",
                "abs_ic50",
                "hill",
                "emax_observed",
                "auc",
                "r2",
                "flags",
            ],
            heat="ec50",
        )
        from hcs_studio.pharmacology.analysis import combination_tables

        scores = pd.concat(
            [combination_tables(rr) for rr in result.readouts.values()], ignore_index=True
        )
        first = next(iter(result.options.models), None)
        if first and not scores.empty:
            scores = scores.sort_values(f"{first}_mean", ascending=False)
        cols = ["readout", "cell_line", "drug_a", "drug_b"]
        for model in result.options.models:
            cols += [f"{model}_mean", f"{model}_max_3x3", f"{model}_p", f"{model}_label"]
        self._fill(self.synergy_table, scores, cols, heat=f"{first}_mean" if first else None)
        qc = pd.concat([rr.qc for rr in result.readouts.values()], ignore_index=True)
        self._fill(
            self.qc_table,
            qc,
            [
                "readout",
                "plate",
                "z_prime",
                "ssmd",
                "signal_window",
                "cv_vehicle_pct",
                "cv_positive_pct",
                "flag",
            ],
        )

    @staticmethod
    def _fill(table: QTableWidget, frame: Any, columns: list[str], heat: str | None = None) -> None:
        table.setSortingEnabled(False)
        cols = [c for c in columns if c in frame]
        table.clear()
        table.setColumnCount(len(cols))
        table.setRowCount(len(frame))
        table.setHorizontalHeaderLabels(cols)
        values = frame[heat].to_numpy(float) if heat and heat in frame else None
        if values is not None and np.isfinite(values).any():
            scale = np.log10(values) if heat == "ec50" else values
            lo, hi = np.nanmin(scale), np.nanmax(scale)
        for r, row in enumerate(frame[cols].itertuples(index=False)):
            for c, value in enumerate(row):
                if isinstance(value, float):
                    item = QTableWidgetItem()
                    item.setData(
                        Qt.ItemDataRole.DisplayRole,
                        None if math.isnan(value) else float(f"{value:.4g}"),
                    )
                else:
                    item = QTableWidgetItem(str(value))
                if values is not None and cols[c] == heat and math.isfinite(values[r]):
                    v = math.log10(values[r]) if heat == "ec50" else values[r]
                    t = (v - lo) / ((hi - lo) or 1)
                    item.setBackground(QColor.fromHsvF(0.6 - 0.6 * t, 0.35, 1.0))
                table.setItem(r, c, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
