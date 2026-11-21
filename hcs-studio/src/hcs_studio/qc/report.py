"""``qc/qc_report.html``: the image QC of a plate on one self-contained page (Phase 18).

Plate decision and counts, plate heat maps of key metrics (failed wells hatched), failed
wells and excluded fields with their reasons, a gallery of flagged fields sorted by severity
(thumbnails embedded), compound artifacts, and the review history. Everything is inline
(images as data URIs), so the file can be mailed or archived on its own.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio import __version__

#: Heat maps: (column, title); per-channel metrics use the content channel.
HEATMAPS = (
    ("severity", "Worst severity per well"),
    ("{c}_laplacian_variance", "Fine detail (focus)"),
    ("{c}_saturated_fraction", "Saturated pixels"),
    ("{c}_background", "Background"),
    ("object_count", "Objects per field"),
)
MAX_GALLERY = 300


def _png(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _heatmap(values: pd.Series, failed: set[str], rows: int, cols: int, title: str) -> str:
    # Figure + Agg canvas directly: never matplotlib.use() here, which switches pyplot's
    # backend and can deadlock with Qt when called from a worker thread.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from hcs_studio.model.plate import parse_well_name, row_label

    grid = np.full((rows, cols), np.nan)
    for well, value in values.items():
        r, c = parse_well_name(str(well))
        if r < rows and c < cols:
            grid[r, c] = float(value)
    fig = Figure(figsize=(max(3.2, cols * 0.28 + 1.2), max(2.4, rows * 0.28 + 0.8)), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    image = ax.imshow(grid, cmap="viridis", aspect="equal")
    for well in failed:
        r, c = parse_well_name(well)
        ax.add_patch(
            Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////", edgecolor="#b00020", lw=0)
        )
    step_c = 1 if cols <= 24 else 2
    ax.set_xticks(range(0, cols, step_c), [str(c + 1) for c in range(0, cols, step_c)], fontsize=6)
    ax.set_yticks(range(rows), [row_label(r) for r in range(rows)], fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _png(fig)


def _thumb(qc: Path, well: str, field: int) -> str:
    from hcs_studio.qc.thumbs import THUMBS, thumb_name

    path = qc / THUMBS / thumb_name(well, field)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _rows(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "<p>(none)</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>"
        for row in frame.to_dict("records")
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def write_qc_report(folder: Path) -> Path:
    from hcs_studio.qc.review import load_overrides

    qc = Path(folder) / "qc"
    fields = pd.read_parquet(qc / "field_qc.parquet")
    wells = pd.read_csv(qc / "well_qc.csv", keep_default_na=False, na_values=[""])
    plate = json.loads((qc / "plate_qc.json").read_text(encoding="utf-8"))
    rules = plate.get("rules", {})
    content = 1
    try:
        from hcs_studio.qc.rules import load_rules

        content = load_rules(qc / "rules.yaml").settings.content_channel
    except (OSError, ValueError):
        pass
    rows = int(max(16 if fields["row"].max() >= 8 else 8, fields["row"].max() + 1))
    cols = int(max(24 if fields["col"].max() >= 12 else 12, fields["col"].max() + 1))
    failed = set(wells.loc[wells["failed"].astype(bool), "well"].astype(str))
    maps = []
    for column, title in HEATMAPS:
        name = column.format(c=f"c{content}")
        if name not in fields:
            continue
        per_well = fields.groupby("well")[name].agg("max" if name == "severity" else "median")
        maps.append(f'<img alt="{html.escape(title)}" src="data:image/png;base64,'
                    f'{_heatmap(per_well, failed, rows, cols, title)}">')  # fmt: skip
    flagged = fields[fields["flags"].fillna("") != ""].sort_values("severity", ascending=False)
    cards = []
    for row in flagged.head(MAX_GALLERY).to_dict("records"):
        image = _thumb(qc, str(row["well"]), int(row["field"]))
        tag = "excluded" if row.get("excluded") else "kept"
        review = f" · reviewed: {row['review']}" if row.get("review") else ""
        cards.append(
            f'<figure class="{tag}">'
            + (f'<img alt="" src="data:image/png;base64,{image}">' if image else "")
            + f"<figcaption><b>{html.escape(str(row['well']))} f{int(row['field'])}</b> · "
            f"{html.escape(str(row['flags']).replace(';', ', '))} · severity "
            f"{float(row['severity']):.1f} · {tag}{html.escape(review)}<br>"
            f"<small>{html.escape(str(row.get('reasons', '')))}</small></figcaption></figure>"
        )
    excluded = fields[fields["excluded"].astype(bool)].sort_values(["well", "field"])
    compound = wells[
        (wells.get("autofluorescence", pd.Series("", index=wells.index)).fillna("") != "")
        | wells.get("precipitation", pd.Series(False, index=wells.index)).astype(bool)
    ]
    reviews = pd.DataFrame([o.model_dump() for o in load_overrides(Path(folder))])
    counts = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in (plate.get("flag_counts") or {}).items()
    )
    verdict = (
        '<p class="ok">Plate passed QC.</p>'
        if plate.get("passed")
        else '<p class="bad">Plate failed QC: '
        + html.escape("; ".join(plate.get("reasons") or []))
        + "</p>"
    )
    note = "" if len(flagged) <= MAX_GALLERY else f"<p>Showing the {MAX_GALLERY} most severe.</p>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image QC</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px;color:#1d2330}}
table{{border-collapse:collapse;font-size:12px;margin:6px 0}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:left}}
th{{background:#f3f3f3}}.ok{{color:#1b7f3b;font-weight:600}}.bad{{color:#b00020;font-weight:600}}
.maps img{{max-width:100%;margin:4px}}.gallery{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:200px;border:2px solid #ccc;border-radius:6px;padding:4px;font-size:11px}}
figure.excluded{{border-color:#b00020}}figure img{{width:100%;display:block;image-rendering:pixelated}}
@media (prefers-color-scheme: dark){{body{{background:#14171c;color:#e4e7ec}}th{{background:#262b33}}td,th{{border-color:#3a414c}}}}
</style></head><body>
<h1>Image QC: {html.escape(str((plate.get("plate") or {}).get("name", Path(folder).name)))}</h1>
<p>Rules {html.escape(str(rules.get("name", "")))} v{rules.get("version", "")} (SHA-256
{html.escape(str(rules.get("sha256", ""))[:12])}…) · {plate.get("fields", len(fields))} fields ·
{plate.get("excluded_fields", 0)} excluded · {plate.get("failed_wells", 0)} of
{plate.get("wells", len(wells))} wells failed · {plate.get("reviews", 0)} review(s) ·
hcs-studio {__version__}</p>
{verdict}
<h2>Flags</h2><table><tr><th>flag</th><th>fields</th></tr>{counts}</table>
<h2>Plate maps</h2><p>Failed wells are hatched.</p><div class="maps">{"".join(maps)}</div>
<h2>Failed wells</h2>{_rows(wells[wells["failed"].astype(bool)], ["well", "n_fields", "n_good_fields", "flags", "reasons"])}
<h2>Excluded fields</h2>{_rows(excluded, ["well", "field", "flags", "reasons"])}
<h2>Flagged fields, most severe first</h2>{note}<div class="gallery">{"".join(cards)}</div>
<h2>Compound artifacts</h2>{_rows(compound, ["well", "autofluorescence", "precipitation"])}
<h2>Reviews</h2>{_rows(reviews, ["time", "actor", "level", "well", "field", "action", "reason"]) if not reviews.empty else "<p>(none)</p>"}
</body></html>"""
    target = qc / "qc_report.html"
    part = target.with_name(target.name + ".part")
    part.write_text(page, encoding="utf-8")
    part.replace(target)
    return target
