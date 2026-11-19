"""Experiments tab (Phase 17a): the registry of plates, runs and results.

Search plates (saved searches too), open a plate's page (acquisitions, runs, QC), compare
potency across runs, plot a readout over time, browse compounds and cell lines, export
tables, and index run folders (Scan folder…). Every query runs in a background thread.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask


def fill_table(table: QTableWidget, frame: pd.DataFrame, hidden: tuple[str, ...] = ()) -> None:
    columns = [c for c in frame.columns if c not in hidden]
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(frame))
    table.setHorizontalHeaderLabels(columns)
    for row, record in enumerate(frame.to_dict("records")):
        for col, name in enumerate(columns):
            table.setItem(row, col, QTableWidgetItem(_text(record[name])))
    table.resizeColumnsToContents()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    if isinstance(value, datetime | pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


class ExperimentsPanel(QWidget):
    """``open_run(Path)`` asks the main window to show a run in the Results tab."""

    open_run = Signal(object)
    log = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._pending: Callable[[Any], None] | None = None
        self.plates = pd.DataFrame()
        self.potency = pd.DataFrame()
        self.trend = pd.DataFrame()
        self.page: dict[str, Any] = {}

        self.text = QLineEdit()
        self.text.setPlaceholderText("barcode, plate, experiment or assay")
        self.text.returnPressed.connect(self.search)
        self.cell_line = QLineEdit()
        self.cell_line.setPlaceholderText("cell line")
        self.compound = QLineEdit()
        self.compound.setPlaceholderText("compound")
        self.saved = QComboBox()
        self.saved.setPlaceholderText("Saved searches")
        self.saved.activated.connect(self._apply_saved)
        search = QPushButton("Search")
        search.clicked.connect(self.search)
        save = QPushButton("Save search…")
        save.clicked.connect(self._save_search)
        scan = QPushButton("Scan folder…")
        scan.setToolTip("Index run folders (and the folders below) into the registry")
        scan.clicked.connect(self._choose_scan)
        bar = QHBoxLayout()
        for widget in (self.text, self.cell_line, self.compound, search, self.saved, save, scan):
            bar.addWidget(widget)
        self.status = QLabel("")

        # plates
        self.plate_table = QTableWidget()
        self.plate_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.plate_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plate_table.itemSelectionChanged.connect(self._plate_selected)
        self.plate_details = QTextBrowser()
        self.open_button = QPushButton("Open the latest run in Results")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_latest)
        right = QVBoxLayout()
        right.addWidget(self.plate_details)
        right.addWidget(self.open_button)
        right_widget = QWidget()
        right_widget.setLayout(right)
        plates = QSplitter()
        plates.addWidget(self.plate_table)
        plates.addWidget(right_widget)
        plates.setStretchFactor(0, 3)

        # potency
        self.pot_compound = QLineEdit()
        self.pot_compound.setPlaceholderText("all compounds")
        self.pot_line = QLineEdit()
        self.pot_line.setPlaceholderText("all cell lines")
        self.pot_readout = QLineEdit()
        self.pot_readout.setPlaceholderText("all readouts")
        show = QPushButton("Show potency")
        show.clicked.connect(self.show_potency)
        export = QPushButton("Export…")
        export.clicked.connect(lambda: self._export(self.potency))
        pot_bar = QHBoxLayout()
        for widget in (self.pot_compound, self.pot_line, self.pot_readout, show, export):
            pot_bar.addWidget(widget)
        self.potency_table = QTableWidget()
        self.potency_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        potency = QWidget()
        pot_layout = QVBoxLayout(potency)
        pot_layout.addLayout(pot_bar)
        pot_layout.addWidget(self.potency_table)

        # readout over time
        self.trend_readout = QComboBox()
        self.trend_readout.setEditable(True)
        self.trend_role = QComboBox()
        self.trend_role.addItems(["", "negative_control", "positive_control", "sample"])
        self.trend_compound = QLineEdit()
        self.trend_compound.setPlaceholderText("compound (any dose)")
        self.trend_line = QLineEdit()
        self.trend_line.setPlaceholderText("cell line")
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(self.show_trend)
        trend_form = QFormLayout()
        trend_form.addRow("Readout", self.trend_readout)
        trend_form.addRow("Wells: role", self.trend_role)
        trend_form.addRow("Compound", self.trend_compound)
        trend_form.addRow("Cell line", self.trend_line)
        trend_form.addRow("", plot_button)
        self.trend_plot = pg.PlotWidget()
        self.trend_plot.setLabel("bottom", "run (oldest → newest)")
        trend = QWidget()
        trend_layout = QHBoxLayout(trend)
        form_widget = QWidget()
        form_widget.setLayout(trend_form)
        trend_layout.addWidget(form_widget, 1)
        trend_layout.addWidget(self.trend_plot, 3)

        # compounds and cell lines
        self.compound_list = QListWidget()
        self.compound_list.currentTextChanged.connect(lambda n: self._show_reference("compound", n))
        self.line_list = QListWidget()
        self.line_list.currentTextChanged.connect(lambda n: self._show_reference("cell_line", n))
        self.reference = QTextBrowser()
        lists = QTabWidget()
        lists.addTab(self.compound_list, "Compounds")
        lists.addTab(self.line_list, "Cell lines")
        reference = QSplitter()
        reference.addWidget(lists)
        reference.addWidget(self.reference)
        reference.setStretchFactor(1, 3)

        self.views = QTabWidget()
        self.views.addTab(plates, "Plates")
        self.views.addTab(potency, "Potency")
        self.views.addTab(trend, "Readout over time")
        self.views.addTab(reference, "Compounds and cell lines")
        self.views.addTab(self._build_normalization(), "Normalization")
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.views, 1)
        layout.addWidget(self.status)

    # --------------------------------------------------------------- running

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], message: str) -> None:
        self._pending = done
        self.status.setText(message)
        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.status.setText(f"Failed: {message}")

    def _on_done(self, result: Any) -> None:
        done, self._pending = self._pending, None
        if done is not None:
            done(result)

    def _query(self) -> Any:
        from hcs_studio.db.queries import PlateQuery

        return PlateQuery(
            text=self.text.text().strip(),
            cell_line=self.cell_line.text().strip(),
            compound=self.compound.text().strip(),
        )

    # -------------------------------------------------------------- actions

    def refresh(self) -> None:
        """Plates, saved searches, compounds, cell lines and readouts (e.g. after a scan)."""
        query = self._query()

        def work() -> dict[str, Any]:
            from hcs_studio.db import normalization as dbn
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return {
                    "plates": queries.search_plates(s, query),
                    "saved": list(queries.saved_searches(s)),
                    "compounds": queries.names(s, "compound"),
                    "cell_lines": queries.names(s, "cell_line"),
                    "readouts": sorted(queries.readouts_in_registry(s)),
                    "runs": queries.run_names(s),
                    "maps": list(dbn.uniformity_maps(s).get("name", [])),
                }

        self._run(work, self._show_all, "Reading the registry…")

    def _show_all(self, data: dict[str, Any]) -> None:
        self._show_plates(data["plates"])
        self.saved.clear()
        self.saved.addItems(data["saved"])
        self.compound_list.clear()
        self.compound_list.addItems(data["compounds"])
        self.line_list.clear()
        self.line_list.addItems(data["cell_lines"])
        for combo, items in (
            (self.trend_readout, data["readouts"]),
            (self.norm_readout, data["readouts"]),
            (self.norm_run, data["runs"]),
            (self.norm_run_b, data["runs"]),
            (self.norm_map, data["maps"]),
        ):
            keep = combo.currentText()
            combo.clear()
            combo.addItems(items)
            if keep:
                combo.setCurrentText(keep)

    def search(self) -> None:
        query = self._query()

        def work() -> pd.DataFrame:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return queries.search_plates(s, query)

        self._run(work, self._show_plates, "Searching…")

    def _show_plates(self, frame: pd.DataFrame) -> None:
        self.plates = frame
        fill_table(self.plate_table, frame, hidden=("plate_id",))
        self.status.setText(f"{len(frame)} plate(s)")

    def _plate_selected(self) -> None:
        rows = {i.row() for i in self.plate_table.selectedItems()}
        if not rows or self.plates.empty:
            return
        plate_id = int(self.plates.plate_id.iloc[min(rows)])

        def work() -> dict[str, Any]:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return queries.plate_page(s, plate_id)

        self._run(work, self._show_page, "Reading the plate…")

    def _show_page(self, page: dict[str, Any]) -> None:
        self.page = page
        info = page["plate"]
        parts = [
            f"<h3>{info['barcode'] or info['name']}</h3>",
            f"<p>{info['plate_type']} wells · assay {info['assay'] or '–'} · experiment "
            f"{info['experiment'] or '–'} · cell lines {', '.join(info['cell_lines']) or '–'}"
            f" · layout version {info['layout_version'] or '–'}</p>",
            "<p>Layout: "
            + ", ".join(f"{role} {n}" for role, n in sorted(page["layout_roles"].items()))
            + "</p>",
            "<h4>Acquisitions</h4><ul>",
        ]
        parts += [
            f"<li>{a['instrument'] or 'imager'} {_text(a['started'])} — {a['raw_path']}</li>"
            for a in page["acquisitions"]
        ]
        parts.append("</ul><h4>Runs</h4><ul>")
        parts += [
            f"<li>{r['name']}: {r['status']} ({r['pipeline']}, {_text(r['started'])})</li>"
            for r in page["runs"]
        ]
        parts.append("</ul>")
        if page["qc"]:
            parts.append("<h4>Plate QC</h4><ul>")
            parts += [
                f"<li>{q['readout']}: Z′ {_text(q['z_prime'])}, CV vehicle "
                f"{_text(q['cv_vehicle_pct'])} % {q['flag'] or ''}</li>"
                for q in page["qc"]
            ]
            parts.append("</ul>")
        self.plate_details.setHtml("".join(parts))
        self.open_button.setEnabled(bool(page["runs"]))
        self.status.setText("")

    def _open_latest(self) -> None:
        from hcs_studio.paths import resolve

        if not self.page.get("runs"):
            return
        try:
            folder = resolve(self.page["runs"][0]["folder"])
        except KeyError as exc:
            self.status.setText(str(exc))
            return
        self.open_run.emit(folder)

    def show_potency(self) -> None:
        compound, line = self.pot_compound.text().strip(), self.pot_line.text().strip()
        readout = self.pot_readout.text().strip()

        def work() -> pd.DataFrame:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return queries.potency_table(s, compound, line, readout)

        def done(frame: pd.DataFrame) -> None:
            self.potency = frame
            fill_table(self.potency_table, frame)
            self.status.setText(f"{len(frame)} compound × cell line × readout")

        self._run(work, done, "Collecting EC50s…")

    def show_trend(self) -> None:
        readout = self.trend_readout.currentText().strip()
        if not readout:
            self.status.setText("Choose a readout")
            return
        role, compound = self.trend_role.currentText(), self.trend_compound.text().strip()
        line = self.trend_line.text().strip()

        def work() -> pd.DataFrame:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return queries.readout_over_time(
                    s, readout, compound=compound, cell_line=line, role=role
                )

        def done(frame: pd.DataFrame) -> None:
            self.trend = frame
            self.trend_plot.clear()
            if frame.empty:
                self.status.setText("No wells match")
                return
            x = list(range(len(frame)))
            mean = frame["mean"].to_numpy(float)
            sd = frame["sd"].fillna(0).to_numpy(float)
            self.trend_plot.addItem(
                pg.ErrorBarItem(x=pd.Series(x).to_numpy(), y=mean, height=2 * sd)
            )
            self.trend_plot.plot(x, mean, pen=None, symbol="o")
            axis = self.trend_plot.getAxis("bottom")
            axis.setTicks(
                [
                    [
                        (i, str(b or r))
                        for i, (b, r) in enumerate(zip(frame.barcode, frame.run, strict=True))
                    ]
                ]
            )
            from hcs_studio.readouts import label

            self.trend_plot.setLabel("left", label(readout))
            self.status.setText(f"{len(frame)} run(s)")

        self._run(work, done, "Collecting the readout…")

    def _show_reference(self, kind: str, name: str) -> None:
        if not name:
            return

        def work() -> dict[str, Any]:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                if kind == "compound":
                    return queries.compound_page(s, name)
                return queries.cell_line_page(s, name)

        def done(page: dict[str, Any]) -> None:
            info = page.get("compound") or page.get("cell_line") or {}
            rows = "".join(
                f"<tr><td>{k}</td><td>{_text(v)}</td></tr>" for k, v in info.items() if v
            )
            fits = page["fits"]
            text = f"<h3>{name}</h3><table>{rows}</table><p>Plates: {', '.join(page['plates'])}</p>"
            if not fits.empty:
                text += (
                    "<h4>Dose–response fits</h4><table><tr><th>compound</th>"
                    "<th>cell line</th><th>readout</th><th>EC50 (M)</th><th>run</th></tr>"
                )
                text += "".join(
                    f"<tr><td>{f.compound}</td><td>{f.cell_line}</td><td>{f.readout}</td>"
                    f"<td>{_text(f.ec50_m)}</td><td>{f.barcode or f.run}</td></tr>"
                    for f in fits.itertuples()
                )
                text += "</table>"
            self.reference.setHtml(text)
            self.status.setText("")

        self._run(work, done, f"Reading {name}…")

    # --------------------------------------------------------------- normalization

    def _build_normalization(self) -> QWidget:
        self.norm_run = QComboBox()
        self.norm_run.setEditable(True)
        self.norm_readout = QComboBox()
        self.norm_readout.setEditable(True)
        self.norm_method = QComboBox()
        self.norm_method.addItems(
            ["robust_z", "b_score", "percent_of_control", "percent_inhibition", "uniformity"]
        )
        self.norm_map = QComboBox()
        self.norm_map.setToolTip("Uniformity map (method uniformity)")
        self.norm_run_b = QComboBox()
        self.norm_run_b.setEditable(True)
        apply = QPushButton("Normalize")
        apply.clicked.connect(self.normalize)
        save_map = QPushButton("Save as uniformity map…")
        save_map.setToolTip("Fit a correction map on this run (a vehicle-only plate)")
        save_map.clicked.connect(self._save_map)
        compare = QPushButton("Compare replicates")
        compare.clicked.connect(self.compare_replicates)
        days = QPushButton("Day and plate effects of the plates listed")
        days.clicked.connect(self.day_effects)
        form = QFormLayout()
        form.addRow("Run", self.norm_run)
        form.addRow("Readout", self.norm_readout)
        form.addRow("Method", self.norm_method)
        form.addRow("Uniformity map", self.norm_map)
        form.addRow("", apply)
        form.addRow("", save_map)
        form.addRow("Replicate run", self.norm_run_b)
        form.addRow("", compare)
        form.addRow("", days)
        controls = QWidget()
        controls.setLayout(form)
        self.raw_plot = pg.PlotWidget(title="raw")
        self.norm_plot = pg.PlotWidget(title="normalized")
        self.scatter_plot = pg.PlotWidget(title="replicates")
        for plot in (self.raw_plot, self.norm_plot):
            plot.invertY(True)
            plot.setAspectLocked(True)
        self.norm_text = QTextBrowser()
        self.normalized = pd.DataFrame()
        self.agreement: Any = None
        plots = QHBoxLayout()
        for plot in (self.raw_plot, self.norm_plot, self.scatter_plot):
            plots.addWidget(plot)
        right = QVBoxLayout()
        right.addLayout(plots, 3)
        right.addWidget(self.norm_text, 2)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(controls, 1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        layout.addWidget(right_widget, 4)
        return widget

    def _heatmap(self, plot: pg.PlotWidget, frame: pd.DataFrame, column: str) -> None:
        from hcs_studio.model.plate import parse_well_name

        plot.clear()
        positions = [parse_well_name(w) for w in frame["well"]]
        rows = max(r for r, _ in positions) + 1
        cols = max(c for _, c in positions) + 1
        grid = np.full((rows, cols), np.nan)
        for (r, c), v in zip(positions, frame[column], strict=True):
            grid[r, c] = v
        finite = grid[np.isfinite(grid)]
        if not finite.size:
            return
        image = pg.ImageItem(np.nan_to_num(grid.T, nan=float(np.median(finite))))
        image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        low, high = np.percentile(finite, [2, 98])
        image.setLevels((float(low), float(high) if high > low else float(low) + 1))
        plot.addItem(image)

    def normalize(self) -> None:
        run, readout = self.norm_run.currentText().strip(), self.norm_readout.currentText().strip()
        method, map_name = self.norm_method.currentText(), self.norm_map.currentText()
        if not run or not readout:
            self.status.setText("Choose a run and a readout")
            return

        def work() -> Any:
            from hcs_studio.db import normalization as dbn
            from hcs_studio.db import session_scope
            from hcs_studio.normalization.plate import PositionalCorrectionBlocked

            with session_scope() as s:
                try:
                    return dbn.normalize_run(s, run, readout, method, map_name)
                except PositionalCorrectionBlocked as exc:
                    return str(exc)

        def done(result: Any) -> None:
            if isinstance(result, str):  # blocked by the guard rail
                self.norm_text.setHtml(f"<p><b>Not applied.</b> {result}</p>")
                self.status.setText("Row/column correction blocked on this layout")
                return
            data, normalized = result
            self.normalized = normalized.frame
            self._heatmap(self.raw_plot, normalized.frame, "raw")
            self._heatmap(self.norm_plot, normalized.frame, "normalized")
            notes = "".join(f"<li>{n}</li>" for n in normalized.notes)
            guard = "" if normalized.guard.allowed else f"<p>Layout: {normalized.guard.reason}</p>"
            self.norm_text.setHtml(
                f"<p>{data.barcode} ({data.run}), {readout}: {method}</p><ul>{notes}</ul>{guard}"
            )
            self.status.setText(f"Normalized {len(normalized.frame)} wells")

        self._run(work, done, "Normalizing…")

    def _save_map(self) -> None:
        name, ok = QInputDialog.getText(self, "Save uniformity map", "Name of the map:")
        if ok and name.strip():
            self.save_map(name.strip())

    def save_map(self, name: str, reason: str = "") -> None:
        from hcs_studio.services import context
        from hcs_studio.services.registry import save_uniformity_map

        run, readout = self.norm_run.currentText().strip(), self.norm_readout.currentText().strip()
        ctx = context(reason)

        def done(stats: dict[str, Any]) -> None:
            edge = (
                f"{stats['before_edge_effect_pct']:.1f} % → {stats['after_edge_effect_pct']:.1f} %"
            )
            cv = f"{stats['before_cv_pct']:.1f} % → {stats['after_cv_pct']:.1f} %"
            self.norm_text.setHtml(
                f"<p>Uniformity map <b>{name}</b> saved. Edge effect {edge}, CV {cv}.</p>"
            )
            self.refresh()

        self._run(lambda: save_uniformity_map(ctx, name, run, readout), done, "Fitting the map…")

    def compare_replicates(self) -> None:
        run_a, run_b = self.norm_run.currentText().strip(), self.norm_run_b.currentText().strip()
        readout = self.norm_readout.currentText().strip()

        def work() -> Any:
            from hcs_studio.db import normalization as dbn
            from hcs_studio.db import session_scope

            with session_scope() as s:
                return dbn.concordance(s, run_a, run_b, readout)

        def done(result: Any) -> None:
            a, b, agreement = result
            self.agreement = agreement
            self.scatter_plot.clear()
            wells = [w for w in a.values if w in b.values]
            self.scatter_plot.plot(
                [a.values[w] for w in wells],
                [b.values[w] for w in wells],
                pen=None,
                symbol="o",
                symbolSize=4,
            )
            verdict = "; ".join(agreement.flags) if agreement.flags else "concordant"
            self.norm_text.setHtml(
                f"<p>{a.barcode} vs {b.barcode}: n = {agreement.n}, Pearson r "
                f"{agreement.pearson:.3f}, Spearman ρ {agreement.spearman:.3f}; Bland–Altman bias "
                f"{agreement.bias:.4g} (95 % limits {agreement.loa_low:.4g} … "
                f"{agreement.loa_high:.4g}). <b>{verdict}</b></p>"
            )
            self.status.setText("Replicates compared")

        self._run(work, done, "Comparing replicates…")

    def day_effects(self) -> None:
        readout = self.norm_readout.currentText().strip()
        runs = [r for r in self.plates.get("latest_run", pd.Series(dtype=str)) if r]
        if not readout or len(runs) < 2:
            self.status.setText("Choose a readout and list at least two plates with runs")
            return

        def work() -> Any:
            from hcs_studio.db import normalization as dbn
            from hcs_studio.db import session_scope

            with session_scope() as s:
                return dbn.day_effects(s, readout, runs=runs)

        def done(effects: Any) -> None:
            rows = "".join(
                f"<tr><td>{r['normalization']}</td><td>{_text(r['r2_day'])}</td>"
                f"<td>{_text(r['r2_plate'])}</td><td>{_text(r['vehicle_cv_pct'])}</td></tr>"
                for r in effects.summary.to_dict("records")
            )
            notes = "".join(f"<li>{n}</li>" for n in effects.notes)
            self.norm_text.setHtml(
                "<p>Share of the vehicle wells' variance explained by day and by plate</p>"
                "<table><tr><th>normalization</th><th>R² day</th><th>R² plate</th>"
                f"<th>vehicle CV %</th></tr>{rows}</table><ul>{notes}</ul>"
            )
            self.status.setText(f"{len(runs)} plates")

        self._run(work, done, "Collecting day and plate effects…")

    # ----------------------------------------------------------- searches, files

    def _apply_saved(self) -> None:
        name = self.saved.currentText()

        def work() -> Any:
            from hcs_studio.db import queries, session_scope

            with session_scope() as s:
                return queries.saved_searches(s).get(name)

        def done(query: Any) -> None:
            if query is None:
                return
            self.text.setText(query.text)
            self.cell_line.setText(query.cell_line)
            self.compound.setText(query.compound)
            self.search()

        self._run(work, done, f"Loading {name}…")

    def _save_search(self) -> None:
        name, ok = QInputDialog.getText(self, "Save search", "Name:")
        if ok and name.strip():
            self.save_search(name.strip())

    def save_search(self, name: str) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.registry import save_search

        query, ctx = self._query(), context()
        self._run(lambda: save_search(ctx, name, query), lambda _: self.refresh(), "Saving…")

    def _choose_scan(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Index run folders below")
        if folder:
            self.scan([Path(folder)])

    def scan(self, paths: list[Path]) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.registry import scan_folders

        ctx = context()

        def done(report: Any) -> None:
            self.log.emit(report.text())
            self.status.setText(report.text().splitlines()[0])
            self.refresh()

        self._run(lambda: scan_folders(ctx, paths), done, "Indexing run folders…")

    def _export(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            self.status.setText("Nothing to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "Excel (*.xlsx);;CSV (*.csv)")
        if path:
            from hcs_studio.db.queries import export_table

            try:
                self.status.setText(f"Wrote {export_table(frame, Path(path))}")
            except (OSError, ValueError) as exc:
                self.status.setText(f"Not written: {exc}")
