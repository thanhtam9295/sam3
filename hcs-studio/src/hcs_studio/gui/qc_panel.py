"""The QC tab (Phase 18): review the image QC of a run.

A plate heat map per QC metric (failed wells hatched; clicking a well lists its fields), a
gallery of flagged fields sorted by severity (thumbnails from ``qc/thumbs/``), and buttons
to confirm a flag, include or exclude a field, or pass / fail a well. Every review needs a
reason and goes through the ``qc.review`` service; the run's tables and report follow.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.plate_map import PlateMapView
from hcs_studio.gui.results_panel import VIRIDIS, colour_at
from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.model.plate import parse_well_name, well_name


class QcPanel(QWidget):
    log = Signal(str)
    #: A review changed the run's exclusions (its tables and report were rebuilt).
    reviewed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.folder: Path | None = None
        self.fields = pd.DataFrame()
        self.wells = pd.DataFrame()
        self.plate: dict[str, Any] = {}
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self._pending: Callable[[Any], None] | None = None

        self.summary = QLabel("Open a run with image QC.")
        self.summary.setWordWrap(True)
        self.metric_combo = QComboBox()
        self.metric_combo.currentTextChanged.connect(self._draw_map)
        self.report_button = QPushButton("Open QC report")
        self.report_button.clicked.connect(self._open_report)
        self.map = PlateMapView()
        self.map.setDragMode(PlateMapView.DragMode.NoDrag)
        self.map.well_clicked.connect(self._well_clicked)
        self.flag_filter = QComboBox()
        self.flag_filter.currentTextChanged.connect(self._fill_gallery)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(128, 128))
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setSpacing(4)
        self.gallery.currentItemChanged.connect(self._show_field)
        self.details = QTextBrowser()
        self.details.setMaximumHeight(130)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Reason (required for every review)")
        self.confirm_button = QPushButton("Confirm flag")
        self.include_button = QPushButton("Include field")
        self.exclude_button = QPushButton("Exclude field")
        self.pass_button = QPushButton("Pass well")
        self.fail_button = QPushButton("Fail well")
        for button, level, action in (
            (self.confirm_button, "field", "confirm"),
            (self.include_button, "field", "include"),
            (self.exclude_button, "field", "exclude"),
            (self.pass_button, "well", "include"),
            (self.fail_button, "well", "exclude"),
        ):
            button.clicked.connect(lambda _=False, lv=level, ac=action: self.review(lv, ac))
        self.status = QLabel("")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        top = QHBoxLayout()
        top.addWidget(QLabel("Metric"))
        top.addWidget(self.metric_combo, 1)
        top.addWidget(self.report_button)
        left_layout.addWidget(self.summary)
        left_layout.addLayout(top)
        left_layout.addWidget(self.map, 1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Flagged fields, most severe first:"))
        filters.addWidget(self.flag_filter)
        right_layout.addLayout(filters)
        right_layout.addWidget(self.gallery, 1)
        right_layout.addWidget(self.details)
        right_layout.addWidget(self.reason)
        buttons = QHBoxLayout()
        for button in (self.confirm_button, self.include_button, self.exclude_button):
            buttons.addWidget(button)
        buttons.addSpacing(12)
        buttons.addWidget(self.pass_button)
        buttons.addWidget(self.fail_button)
        right_layout.addLayout(buttons)
        right_layout.addWidget(self.status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([500, 600])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.selected_well: str | None = None
        self._set_enabled(False)

    # ---------------------------------------------------------------- loading

    def open(self, folder: Path) -> None:
        """Show the image QC of a run folder (or QC output folder); hides when it has none."""
        self.folder = Path(folder)
        if not (self.folder / "qc" / "field_qc.parquet").exists():
            self.gallery.clear()
            self.fields, self.wells, self.plate = pd.DataFrame(), pd.DataFrame(), {}
            self.summary.setText(f"{self.folder.name}: no image QC (add QC rules to the pipeline).")
            self.map.set_grid(0, 0, set(), {})
            self.gallery.clear()
            self._set_enabled(False)
            return
        folder_now = self.folder
        self._run(lambda: self._load(folder_now), self._loaded, "Loading QC…")

    @staticmethod
    def _load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        import json

        from hcs_studio.qc.review import load_fields

        fields = load_fields(folder)
        wells = pd.read_csv(folder / "qc" / "well_qc.csv", keep_default_na=False, na_values=[""])
        plate = json.loads((folder / "qc" / "plate_qc.json").read_text(encoding="utf-8"))
        return fields, wells, plate

    def _loaded(self, data: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]) -> None:
        self.fields, self.wells, self.plate = data
        self.wells["well"] = self.wells["well"].astype(str)
        plate = self.plate
        verdict = (
            "passed" if plate.get("passed") else "FAILED: " + "; ".join(plate.get("reasons", []))
        )
        rules = plate.get("rules", {})
        self.summary.setText(
            f"<b>{self.folder.name if self.folder else ''}</b> — plate {verdict}. "
            f"{plate.get('excluded_fields', 0)} of {plate.get('fields', 0)} fields excluded, "
            f"{plate.get('failed_wells', 0)} wells failed; rules {rules.get('name', '')} "
            f"v{rules.get('version', '')}; {plate.get('reviews', 0)} review(s)."
        )
        numeric = sorted(
            str(c)
            for c in self.fields.columns
            if pd.api.types.is_numeric_dtype(self.fields[c])
            and not pd.api.types.is_bool_dtype(self.fields[c])
            and not str(c).startswith(("emb_", "flag_"))
            and c not in ("row", "col", "field", "x_um", "y_um", "severity")
        )
        metrics = ["severity", "excluded fields", *numeric]
        keep = self.metric_combo.currentText()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(metrics)
        self.metric_combo.setCurrentText(keep if keep in metrics else "severity")
        self.metric_combo.blockSignals(False)
        flags = sorted({f for text in self.fields["flags"] for f in str(text).split(";") if f})
        keep = self.flag_filter.currentText()
        self.flag_filter.blockSignals(True)
        self.flag_filter.clear()
        self.flag_filter.addItems(["all flags", *flags])
        self.flag_filter.setCurrentText(keep if keep in flags else "all flags")
        self.flag_filter.blockSignals(False)
        self._draw_map()
        self._fill_gallery()
        self._set_enabled(True)
        self.status.setText("")

    # ---------------------------------------------------------------- display

    def _draw_map(self) -> None:
        if self.fields.empty:
            return
        rows = int(max(16 if self.fields["row"].max() >= 8 else 8, self.fields["row"].max() + 1))
        cols = int(max(24 if self.fields["col"].max() >= 12 else 12, self.fields["col"].max() + 1))
        present = {parse_well_name(w) for w in self.fields["well"]}
        metric = self.metric_combo.currentText() or "severity"
        if metric == "excluded fields":
            values = self.fields.groupby("well")["excluded"].sum().astype(float)
        elif metric in self.fields:
            agg = "max" if metric == "severity" else "median"
            values = self.fields.groupby("well")[metric].agg(agg).astype(float)
        else:
            return
        finite = [v for v in values if math.isfinite(v)]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        span = hi - lo or 1.0
        bad = self.wells[self.wells["failed"].astype(bool)]
        failed = dict(zip(bad["well"], bad["reasons"].fillna(""), strict=True))
        fills, tips = {}, {}
        for well, value in values.items():
            key = parse_well_name(str(well))
            if math.isfinite(value):
                fills[key] = colour_at(VIRIDIS, (value - lo) / span)
            tip = f"{well}: {metric} {value:.4g}"
            if well in failed:
                tip += f"\nfailed QC: {failed[str(well)]}"
            tips[key] = tip
        self.map.set_grid(rows, cols, present, tips)
        self.map.set_fills(fills, tips)
        self.map.set_hatched({parse_well_name(w) for w in failed})

    def _fill_gallery(self) -> None:
        from hcs_studio.qc.thumbs import THUMBS, thumb_name

        self.gallery.clear()
        if self.fields.empty or self.folder is None:
            return
        frame = self.fields
        if self.selected_well:
            frame = frame[frame["well"] == self.selected_well]
        else:
            frame = frame[frame["flags"].fillna("") != ""]
        wanted = self.flag_filter.currentText()
        if wanted and wanted != "all flags":
            frame = frame[frame["flags"].str.split(";").map(lambda f, w=wanted: w in f)]
        frame = frame.sort_values("severity", ascending=False).head(500)
        records = frame.to_dict("records")
        for index, row in zip(frame.index.tolist(), records, strict=True):
            state = "excluded" if row["excluded"] else "kept"
            review = f" ✓{row['review']}" if row.get("review") else ""
            text = (
                f"{row['well']} f{int(row['field'])}\n{row['flags'] or 'no flag'}\n{state}{review}"
            )
            item = QListWidgetItem(text)
            path = self.folder / "qc" / THUMBS / thumb_name(str(row["well"]), int(row["field"]))
            if path.exists():
                item.setIcon(QIcon(QPixmap(str(path))))
            item.setData(Qt.ItemDataRole.UserRole, int(index))
            self.gallery.addItem(item)
        if self.gallery.count():
            self.gallery.setCurrentRow(0)

    def _well_clicked(self, row: int, col: int) -> None:
        name = well_name(row, col)
        self.selected_well = None if self.selected_well == name else name
        self._fill_gallery()

    def _show_field(self, item: QListWidgetItem | None, _previous: Any = None) -> None:
        index = int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else -1
        if item is None or index not in self.fields.index:  # the gallery is being cleared
            self.details.clear()
            return
        row = self.fields.loc[index].to_dict()
        well = self.wells[self.wells["well"] == row["well"]]
        lines = [
            f"<b>{row['well']} field {int(row['field'])}</b> — "
            + ("<span style='color:#b00020'>excluded</span>" if row["excluded"] else "kept"),
            f"Flags: {row['flags'] or 'none'} (severity {float(row['severity']):.2f})",
            f"Reasons: {row['reasons'] or '–'}",
        ]
        if row.get("review"):
            lines.append(f"Reviewed: {row['review']} — {row.get('review_reason', '')}")
        if not well.empty and bool(well["failed"].iloc[0]):
            lines.append(f"Well failed: {well['reasons'].iloc[0]}")
        self.details.setHtml("<br>".join(lines))

    def current_field(self) -> tuple[str, int] | None:
        item = self.gallery.currentItem()
        if item is None or int(item.data(Qt.ItemDataRole.UserRole)) not in self.fields.index:
            return None
        row = self.fields.loc[int(item.data(Qt.ItemDataRole.UserRole))].to_dict()
        return str(row["well"]), int(row["field"])

    # ---------------------------------------------------------------- reviews

    def review(self, level: str, action: str) -> None:
        """Review the selected field (or its well) with the reason typed in."""
        from hcs_studio.services import context
        from hcs_studio.services.operations import review_qc

        reason = self.reason.text().strip()
        if not reason:
            self.status.setText("Give a reason for the review.")
            return
        target = self.current_field()
        if target is None or self.folder is None:
            self.status.setText("Choose a field first.")
            return
        well, field = target
        ctx = context(reason)
        folder = self.folder
        what = f"{well} field {field}" if level == "field" else f"well {well}"

        def work() -> Any:
            return review_qc(
                ctx, folder, level, well, action, field=field if level == "field" else None
            )

        def done(_: Any) -> None:
            self.reason.clear()
            self.log.emit(f"QC review: {action} {what} ({reason})")
            self.open(folder)
            self.reviewed.emit(folder)

        self._run(work, done, f"Recording review of {what}…")

    def _open_report(self) -> None:
        if self.folder is not None:
            path = self.folder / "qc" / "qc_report.html"
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ---------------------------------------------------------------- plumbing

    def _set_enabled(self, on: bool) -> None:
        widgets: tuple[QWidget, ...] = (
            self.confirm_button,
            self.include_button,
            self.exclude_button,
            self.pass_button,
            self.fail_button,
            self.report_button,
            self.reason,
        )
        for widget in widgets:
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
        if done is not None:
            done(result)
