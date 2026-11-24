"""The Screen tab (Phase 23b): review the hits of a primary screen.

Open a screen folder (written by ``hcs-studio screen score``): a plate heat map of the chosen
score (artefact wells hatched, hits and artefacts in the tooltips), the ranked hit list, and
include / exclude buttons for the selected compound — each needs a reason and goes through the
``screen.review`` service, which rewrites the hit list. The screen report and the hit gallery
open in the browser.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.plate_map import PlateMapView
from hcs_studio.gui.results_panel import DIVERGING, colour_at
from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.model.plate import parse_well_name

HIT_COLUMNS = ("rank", "compound", "concentration", "n_hit", "n", "override")


class ScreenPanel(QWidget):
    log = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.folder: Path | None = None
        self.summary_data: dict[str, Any] = {}
        self.wells = pd.DataFrame()
        self.hits = pd.DataFrame()
        self.plates = pd.DataFrame()
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._pending: Callable[[Any], None] | None = None

        self.summary = QLabel("Open a screen folder (hcs-studio screen score …).")
        self.summary.setWordWrap(True)
        self.open_button = QPushButton("Open screen…")
        self.open_button.clicked.connect(self._choose)
        self.report_button = QPushButton("Open report")
        self.report_button.clicked.connect(lambda: self._open_page("report.html"))
        self.gallery_button = QPushButton("Open hit gallery")
        self.gallery_button.clicked.connect(lambda: self._open_page("gallery/index.html"))
        self.plate_combo = QComboBox()
        self.plate_combo.currentTextChanged.connect(self._draw_map)
        self.score_combo = QComboBox()
        self.score_combo.currentTextChanged.connect(self._draw_map)
        self.map = PlateMapView()
        self.map.setDragMode(PlateMapView.DragMode.NoDrag)
        self.map.well_clicked.connect(self._well_clicked)
        self.hit_table = QTableWidget(0, len(HIT_COLUMNS) + 1)
        self.hit_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.hit_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hit_table.itemSelectionChanged.connect(self._hit_selected)
        self.compound = QLineEdit()
        self.compound.setPlaceholderText("Compound (select a hit or click a well)")
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Reason (required)")
        self.include_button = QPushButton("Include as hit")
        self.exclude_button = QPushButton("Exclude from hits")
        self.include_button.clicked.connect(lambda: self.review("include"))
        self.exclude_button.clicked.connect(lambda: self.review("exclude"))
        self.status = QLabel("")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        top = QHBoxLayout()
        top.addWidget(self.open_button)
        top.addWidget(self.report_button)
        top.addWidget(self.gallery_button)
        top.addStretch(1)
        left_layout.addLayout(top)
        left_layout.addWidget(self.summary)
        pickers = QHBoxLayout()
        pickers.addWidget(QLabel("Plate"))
        pickers.addWidget(self.plate_combo, 1)
        pickers.addWidget(QLabel("Score"))
        pickers.addWidget(self.score_combo, 1)
        left_layout.addLayout(pickers)
        left_layout.addWidget(self.map, 1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Hits (ranked):"))
        right_layout.addWidget(self.hit_table, 1)
        right_layout.addWidget(self.compound)
        right_layout.addWidget(self.reason)
        buttons = QHBoxLayout()
        buttons.addWidget(self.include_button)
        buttons.addWidget(self.exclude_button)
        right_layout.addLayout(buttons)
        right_layout.addWidget(self.status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([550, 550])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._set_enabled(False)

    # ---------------------------------------------------------------- loading

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Screen folder")
        if folder:
            self.open(Path(folder))

    def open(self, folder: Path) -> None:
        folder = Path(folder)
        if not (folder / "screen.json").exists():
            self.summary.setText(f"{folder.name}: not a screen folder (no screen.json).")
            return
        self.folder = folder
        self._run(lambda: self._load(folder), self._loaded, f"Loading {folder.name}…")

    @staticmethod
    def _load(folder: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary = json.loads((folder / "screen.json").read_text(encoding="utf-8"))
        wells = pd.read_csv(folder / "wells.csv", keep_default_na=False, na_values=[""])
        hits = pd.read_csv(folder / "hits.csv", keep_default_na=False, na_values=[""])
        plates = pd.read_csv(folder / "plates.csv")
        return summary, wells, hits, plates

    def _loaded(
        self, data: tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        self.summary_data, self.wells, self.hits, self.plates = data
        s = self.summary_data
        self.summary.setText(
            f"{s.get('name', '')}: {s.get('plates', 0)} plates, {s.get('compounds', 0)} compounds, "
            f"{s.get('hits', 0)} hits ({float(s.get('hit_rate_pct', 0)):.2f} %), "
            f"{s.get('artifact_wells', 0)} artefact wells · rule: "
            f"{s.get('config', {}).get('hit_rule', '')}"
        )
        aliases = [r["alias"] for r in s.get("config", {}).get("readouts", [])]
        scores = [f"{a}_{k}" for a in aliases for k in ("score", "z", "b", "pct")]
        scores = [c for c in scores if c in self.wells and self.wells[c].notna().any()]
        for combo, items in (
            (self.plate_combo, list(self.plates["plate"])),
            (self.score_combo, scores),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(i) for i in items])
            combo.blockSignals(False)
        self._fill_hits()
        self._draw_map()
        self._set_enabled(True)
        self.gallery_button.setEnabled(
            bool(self.folder and (self.folder / "gallery" / "index.html").exists())
        )

    def _fill_hits(self) -> None:
        columns = [*HIT_COLUMNS, self.score_combo.itemText(0) or "score"]
        self.hit_table.setColumnCount(len(columns))
        self.hit_table.setHorizontalHeaderLabels(columns)
        self.hit_table.setRowCount(len(self.hits))
        for r, row in enumerate(self.hits.to_dict("records")):
            for c, column in enumerate(columns):
                value = row.get(column, "")
                text = (
                    f"{value:.3g}"
                    if isinstance(value, float) and math.isfinite(value)
                    else ("" if isinstance(value, float) else str(value))
                )
                self.hit_table.setItem(r, c, QTableWidgetItem(text))
        self.hit_table.resizeColumnsToContents()

    def _draw_map(self) -> None:
        plate, score = self.plate_combo.currentText(), self.score_combo.currentText()
        if self.wells.empty or not plate or score not in self.wells:
            return
        wells = self.wells[self.wells["plate"] == plate]
        rows = 16 if wells["row"].max() >= 8 else 8
        cols = 24 if wells["col"].max() >= 12 else 12
        values = wells[score].astype(float)
        finite = [abs(v) for v in values if math.isfinite(v)]
        if score.endswith("_pct"):
            lo, hi = 0.0, 100.0
        else:
            span = max(max(finite, default=1.0), 3.0)
            lo, hi = -span, span
        fills, tips = {}, {}
        for row in wells.to_dict("records"):
            key = (int(row["row"]), int(row["col"]))
            value = row[score]
            if isinstance(value, float) and math.isfinite(value):
                fills[key] = colour_at(DIVERGING, (value - lo) / (hi - lo))
            tip = (
                f"{row['well']} {row.get('compound') or row['role']}: {score} {value:.3g}"
                if isinstance(value, float) and math.isfinite(value)
                else f"{row['well']} {row.get('compound') or row['role']}"
            )
            if row.get("well_hit"):
                tip += "\nHIT"
            if isinstance(row.get("artifact"), str) and row["artifact"]:
                tip += f"\nartefact: {row['artifact']}"
            tips[key] = tip
        present = {(int(r), int(c)) for r, c in zip(wells["row"], wells["col"], strict=True)}
        self.map.set_grid(rows, cols, present, tips)
        self.map.set_fills(fills, tips)
        artefacts = wells[wells["artifact"].fillna("").astype(str) != ""]
        self.map.set_hatched(
            {(int(r), int(c)) for r, c in zip(artefacts["row"], artefacts["col"], strict=True)}
        )

    def _well_clicked(self, row: int, col: int) -> None:
        plate = self.plate_combo.currentText()
        hit = self.wells[
            (self.wells["plate"] == plate) & (self.wells["row"] == row) & (self.wells["col"] == col)
        ]
        if not hit.empty and str(hit["compound"].iloc[0] or ""):
            self.compound.setText(str(hit["compound"].iloc[0]))

    def _hit_selected(self) -> None:
        rows = {i.row() for i in self.hit_table.selectedIndexes()}
        if rows:
            self.compound.setText(str(self.hits["compound"].iloc[min(rows)]))
            wells = str(self.hits["wells"].iloc[min(rows)]).split(",")
            plate, well = wells[0].split(":")
            self.plate_combo.setCurrentText(plate)
            self.map.select_well(*parse_well_name(well))

    # ---------------------------------------------------------------- reviews

    def review(self, action: str) -> None:
        """Include or exclude the compound typed in, with the reason typed in."""
        from hcs_studio.services import context
        from hcs_studio.services.operations import review_hit

        reason, compound = self.reason.text().strip(), self.compound.text().strip()
        if not reason:
            self.status.setText("Give a reason for the decision.")
            return
        if not compound or self.folder is None:
            self.status.setText("Choose a compound first.")
            return
        ctx = context(reason)
        folder = self.folder

        def done(_: Any) -> None:
            self.reason.clear()
            self.log.emit(f"Screen review: {action} {compound} ({reason})")
            self.open(folder)

        self._run(
            lambda: review_hit(ctx, folder, compound, action),  # type: ignore[arg-type]
            done,
            f"Recording: {action} {compound}…",
        )

    def _open_page(self, name: str) -> None:
        if self.folder is not None and (self.folder / name).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder / name)))

    # ---------------------------------------------------------------- plumbing

    def _set_enabled(self, on: bool) -> None:
        for widget in (
            self.report_button,
            self.gallery_button,
            self.include_button,
            self.exclude_button,
            self.reason,
            self.compound,
        ):
            widget.setEnabled(on)

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], message: str) -> None:
        self._pending = done
        self.status.setText(message)
        self.task.request(work)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.status.setText(f"Failed: {message}")

    def _on_done(self, result: Any) -> None:
        done, self._pending = self._pending, None
        self.status.setText("")
        if done is not None:
            done(result)
