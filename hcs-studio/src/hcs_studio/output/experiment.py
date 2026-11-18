"""The experiment report: one HTML (self-contained) and one PDF per run.

Sections: summary and provenance, plate layout figures, QC (field QC flags, plate QC),
dose–response table and curves, combination heat maps and scores, representative image
montages per compound, and methods text generated from the manifest. Written to
``<run>/experiment_report/``.
"""

from __future__ import annotations

import base64
import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hcs_studio.output.methods import methods_paragraphs

CSS = (
    "body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1150px;color:#222;"
    "line-height:1.45}table{border-collapse:collapse;font-size:12px;margin:8px 0}"
    "td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}th{background:#f3f3f3}"
    "td:first-child,th:first-child{text-align:left}img{max-width:100%}h2{margin-top:34px;"
    "border-bottom:1px solid #ddd}.note{color:#a05a00}.methods p{margin:4px 0}"
    ".grid{display:flex;flex-wrap:wrap;gap:12px}.grid figure{margin:0;flex:1 1 320px}"
    "figcaption{font-size:12px;color:#555}"
)


def _img(path: Path, alt: str = "") -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame is None or frame.empty:
        return "<p><i>none</i></p>"
    cols = [c for c in (columns or list(frame.columns)) if c in frame]

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.4g}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>"
        for r in frame[cols].itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _layout_has_drugs(folder: Path) -> bool:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    path = run_layout_file(folder)
    if path is None:
        return False
    layout = load_plate_layout(path)
    return any(t.kind != "percent" for a in layout.wells.values() for t in a.active_treatments)


def write_experiment_report(
    folder: Path,
    readouts: list[str] | None = None,
    montages: bool = True,
    montage_source: str = "corrected",
    representative_readout: str = "",
    pdf: bool = True,
) -> list[Path]:
    """Write ``experiment_report/experiment_report.html`` (+ ``.pdf``); runs pharmacology
    first when the run has drugs in its layout and no ``pharmacology/`` yet."""
    import matplotlib

    matplotlib.use("Agg")
    from hcs_studio import __version__
    from hcs_studio.io.layout_io import write_layout_figure
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    folder = Path(folder)
    out = folder / "experiment_report"
    out.mkdir(exist_ok=True)
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    written: list[Path] = []
    notes: list[str] = []

    # pharmacology (reuse, or run now)
    pharm_dir = folder / "pharmacology"
    if not (pharm_dir / "pharmacology.json").exists() and _layout_has_drugs(folder):
        from hcs_studio.analysis.run_steps import default_readouts
        from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology

        try:
            run_pharmacology([folder], PharmOptions(readouts=readouts or default_readouts(folder)))
        except (ValueError, KeyError, OSError) as exc:
            notes.append(f"Pharmacology could not run: {exc}")
    pharm_info = (
        json.loads((pharm_dir / "pharmacology.json").read_text("utf-8"))
        if (pharm_dir / "pharmacology.json").exists()
        else None
    )

    # plate layout figures
    layout_figures: list[tuple[Path, str]] = []
    layout_file = run_layout_file(folder)
    if layout_file is not None:
        layout = load_plate_layout(layout_file)
        for layer in ("compound", "concentration", "cell_line", "role"):
            path = out / f"layout_{layer}.png"
            write_layout_figure(layout, path, layer)
            layout_figures.append((path, f"plate layout: {layer.replace('_', ' ')}"))
            written.append(path)
    else:
        notes.append("The run has no plate layout.")

    # QC
    flags_file = folder / "tables" / "qc_flags.csv"
    flags = pd.read_csv(flags_file) if flags_file.exists() else pd.DataFrame()
    flag_summary = pd.DataFrame()
    if not flags.empty and "flag" in flags:
        flag_summary = flags.groupby("flag").size().rename("fields").reset_index()
    plate_qc = (
        pd.read_csv(pharm_dir / "plate_qc.csv")
        if (pharm_dir / "plate_qc.csv").exists()
        else pd.DataFrame()
    )
    fits = (
        pd.read_csv(pharm_dir / "dose_response_fits.csv")
        if (pharm_dir / "dose_response_fits.csv").exists()
        else pd.DataFrame()
    )
    scores = (
        pd.read_csv(pharm_dir / "combination_scores.csv")
        if (pharm_dir / "combination_scores.csv").exists()
        else pd.DataFrame()
    )
    figures = pharm_dir / "figures"
    curve_pngs = sorted(figures.glob("curves_*.png")) if figures.exists() else []
    combo_pngs = sorted(figures.glob("combination_*.png")) if figures.exists() else []
    qc_pngs = sorted(figures.glob("qc_*.png")) if figures.exists() else []

    # representative montages
    montage_files: list[Path] = []
    if montages and _layout_has_drugs(folder) and (folder / "wells").exists():
        from hcs_studio.compare.export import batch_compound_figures
        from hcs_studio.compare.spec import Comparison, ExportSettings, PanelSpec

        base = Comparison(
            panels=[PanelSpec(run=str(folder), well="A01", source=montage_source)],
            representative_readout=representative_readout,
            settings=ExportSettings(dpi=150, width_mm=180, label_template="{well}"),
        )
        try:
            montage_files = batch_compound_figures(folder, out / "montages", base)
            written += montage_files
        except (ValueError, KeyError, OSError) as exc:
            notes.append(f"Montages could not be made: {exc}")

    methods = methods_paragraphs(manifest, pharm_info)
    plate = manifest.get("plate", {})
    sw = manifest.get("software", {})
    summary = pd.DataFrame(
        [
            ("Plate", f"{plate.get('name', '')} ({plate.get('vendor', '')})"),
            ("Run", folder.name),
            ("Status", manifest.get("status", "")),
            ("Started", manifest.get("started", "")),
            ("Finished", manifest.get("finished", "") or ""),
            ("Pipeline", manifest.get("pipeline", {}).get("name", "")),
            ("Wells", str(len(manifest.get("wells", [])))),
            ("Failed wells", str(len(manifest.get("failed_wells", {})))),
            ("Software", f"HCS Studio {sw.get('version', '')} (report {__version__})"),
            ("Git commit", str(sw.get("git_commit") or "—")),
            ("Input checksums", str(manifest.get("input_checksums", ""))),
        ],
        columns=["", "value"],
    )

    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>Experiment report</title>"
        f"<style>{CSS}</style></head><body>",
        f"<h1>Experiment report — {html.escape(str(plate.get('name', '')))}</h1>",
        f"<p>Generated {datetime.now(UTC).isoformat(timespec='seconds')} by HCS Studio "
        f"{html.escape(__version__)} from <code>{html.escape(folder.name)}</code>.</p>",
    ]
    parts += [f"<p class='note'>{html.escape(n)}</p>" for n in notes]
    parts += ["<h2>Summary and provenance</h2>", _table(summary)]
    if layout_figures:
        parts.append("<h2>Plate layout</h2><div class='grid'>")
        parts += [
            f"<figure>{_img(p, c)}<figcaption>{html.escape(c)}</figcaption></figure>"
            for p, c in layout_figures
        ]
        parts.append("</div>")
    parts += ["<h2>Quality control</h2>", "<h3>Field QC flags</h3>", _table(flag_summary)]
    if not plate_qc.empty:
        parts += [
            "<h3>Plate QC</h3>",
            _table(
                plate_qc,
                [
                    "readout",
                    "plate",
                    "n_vehicle",
                    "n_positive",
                    "z_prime",
                    "ssmd",
                    "signal_window",
                    "cv_vehicle_pct",
                    "cv_positive_pct",
                    "flag",
                ],
            ),
        ]
    parts += [_img(p, p.stem) for p in qc_pngs]
    if not fits.empty:
        parts += [
            "<h2>Dose–response</h2>",
            _table(
                fits,
                [
                    "readout",
                    "cell_line",
                    "compound",
                    "ec50",
                    "ec50_ci_low",
                    "ec50_ci_high",
                    "abs_ic50",
                    "hill",
                    "emax_observed",
                    "auc",
                    "r2",
                    "flags",
                ],
            ),
        ]
        parts += [f"<figure>{_img(p, p.stem)}</figure>" for p in curve_pngs]
    if not scores.empty:
        models = [m for m in ("bliss", "hsa", "loewe", "zip") if f"{m}_mean" in scores]
        cols = ["readout", "cell_line", "drug_a", "drug_b"] + [
            f"{m}_{k}" for m in models for k in ("mean", "max_3x3", "p", "label")
        ]
        parts += [
            "<h2>Combinations</h2>",
            "<p><i>Synergy labels follow SynergyFinder conventions: a guide, not proof.</i></p>",
            _table(scores, cols),
        ]
        parts += [f"<figure>{_img(p, p.stem)}</figure>" for p in combo_pngs]
    if montage_files:
        parts.append(
            "<h2>Representative images</h2><p>For each compound: concentrations as columns, "
            "cell lines as rows; the field closest to each well's median "
            + html.escape(representative_readout or "object count")
            + "; identical contrast in all panels.</p>"
        )
        parts += [
            f"<figure>{_img(p, p.stem)}<figcaption>{html.escape(p.stem)}</figcaption></figure>"
            for p in montage_files
            if p.suffix == ".png"
        ]
    parts.append(
        "<h2>Methods</h2><div class='methods'>"
        + "".join(f"<p>{html.escape(p)}</p>" for p in methods)
        + "</div>"
    )
    parts.append("</body></html>")
    report = out / "experiment_report.html"
    report.write_text("\n".join(parts), encoding="utf-8")
    written.append(report)

    if pdf:
        pdf_path = out / "experiment_report.pdf"
        _pdf(
            pdf_path,
            plate,
            summary,
            layout_figures,
            flag_summary,
            plate_qc,
            qc_pngs,
            fits,
            curve_pngs,
            scores,
            combo_pngs,
            montage_files,
            methods,
            notes,
        )
        written.append(pdf_path)
    info = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "files": [str(p.relative_to(folder)) for p in written],
        "notes": notes,
        "methods": methods,
    }
    (out / "experiment_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return written


def _pdf(
    path: Path,
    plate: dict[str, Any],
    summary: pd.DataFrame,
    layout_figures: list[tuple[Path, str]],
    flags: pd.DataFrame,
    plate_qc: pd.DataFrame,
    qc_pngs: list[Path],
    fits: pd.DataFrame,
    curve_pngs: list[Path],
    scores: pd.DataFrame,
    combo_pngs: list[Path],
    montages: list[Path],
    methods: list[str],
    notes: list[str],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small", fontSize=8, leading=10)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Experiment report",
        author="HCS Studio",
    )
    width = A4[0] - 3 * cm

    def table(frame: pd.DataFrame, columns: list[str] | None = None) -> Any:
        if frame is None or frame.empty:
            return Paragraph("none", small)
        cols = [c for c in (columns or list(frame.columns)) if c in frame]

        def cell(v: Any) -> str:
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.3g}"
            return str(v)[:36]

        t = Table(
            [cols] + [[cell(v) for v in r] for r in frame[cols].itertuples(index=False)],
            repeatRows=1,
        )
        t.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ]
            )
        )
        return t

    def image(p: Path, max_width: float = width) -> Any:
        w, h = ImageReader(str(p)).getSize()
        scale = min(max_width / w, (A4[1] - 5 * cm) / h)
        return Image(str(p), width=w * scale, height=h * scale)

    story: list[Any] = [Paragraph(f"Experiment report — {plate.get('name', '')}", styles["Title"])]
    story += [Paragraph(n, small) for n in notes]
    story += [table(summary), Spacer(1, 10)]
    if layout_figures:
        story.append(Paragraph("Plate layout", styles["Heading2"]))
        story += [image(p, width * 0.8) for p, _ in layout_figures[:2]]
    story += [PageBreak(), Paragraph("Quality control", styles["Heading2"]), table(flags)]
    if not plate_qc.empty:
        story.append(
            table(
                plate_qc,
                ["readout", "plate", "z_prime", "ssmd", "signal_window", "cv_vehicle_pct", "flag"],
            )
        )
    story += [image(p) for p in qc_pngs]
    if not fits.empty:
        story += [
            PageBreak(),
            Paragraph("Dose–response", styles["Heading2"]),
            table(
                fits,
                [
                    "readout",
                    "cell_line",
                    "compound",
                    "ec50",
                    "ec50_ci_low",
                    "ec50_ci_high",
                    "hill",
                    "auc",
                    "r2",
                    "flags",
                ],
            ),
        ]
        story += [image(p, width * 0.75) for p in curve_pngs]
    if not scores.empty:
        models = [m for m in ("bliss", "hsa", "loewe", "zip") if f"{m}_mean" in scores]
        story += [
            PageBreak(),
            Paragraph("Combinations", styles["Heading2"]),
            table(
                scores,
                ["cell_line", "drug_a", "drug_b"]
                + [f"{m}_mean" for m in models]
                + [f"{m}_p" for m in models],
            ),
            Paragraph(
                "Synergy labels follow SynergyFinder conventions: a guide, not proof.", small
            ),
        ]
        story += [image(p) for p in combo_pngs]
    pngs = [p for p in montages if p.suffix == ".png"]
    if pngs:
        story += [PageBreak(), Paragraph("Representative images", styles["Heading2"])]
        story += [image(p) for p in pngs]
    story += [PageBreak(), Paragraph("Methods", styles["Heading2"])]
    story += [Paragraph(html.escape(p), small) for p in methods]
    doc.build(story)
