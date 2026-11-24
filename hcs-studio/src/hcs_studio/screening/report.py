"""``report.html`` of a screen (Phase 23b): the views for reviewing hits.

A self-contained page (inline CSS/JS, data as embedded JSON, canvas drawing so that a screen of
hundreds of plates stays fast), written next to the screen's tables:

- **summary**: plates, compounds, hits, hit rate, artefact wells, notes; the rule and score;
- **plates**: Z′ and robust Z′ (red below 0.5), whether B-score was used, hit rate;
- **plate heat maps**: every plate's wells coloured by the chosen score (blue below 0, red
  above; % activity 0–100), hits outlined in black, artefact wells grey with a cross, controls
  marked with a dot; hovering shows the well;
- **score vs plate order**: every sample well (grey; hits red), the vehicle (blue) and positive
  control (green) medians per plate, the threshold — drift over the screen shows as a trend;
- **distribution**: histogram of the sample wells' scores with the threshold;
- **replicates**: each compound's first two replicates against each other, the hit region
  shaded (both past the threshold), hits red;
- **hits**: the ranked list with scores, SSMD, replicates, overrides and their reasons, and
  links to the hit gallery when it exists.

Thresholds are read from the hit rule (comparisons of a score with a number).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROLE_CODES = {"sample": 0, "negative_control": 1, "positive_control": 2}


def _num(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")


def report_data(result: Any) -> dict[str, Any]:
    """Everything the page draws, JSON-ready and column-oriented."""
    from hcs_studio.screening.rules import thresholds

    config = result.config
    wells: pd.DataFrame = result.wells
    plates: pd.DataFrame = result.plates.sort_values("plate_order")
    names = list(plates["plate"])
    columns = [
        c
        for r in config.readouts
        for c in (f"{r.alias}_score", f"{r.alias}_z", f"{r.alias}_b", f"{r.alias}_pct", r.alias)
    ]
    columns = [c for c in columns if c in wells and wells[c].notna().any()]
    shown = wells[wells["role"].isin(ROLE_CODES)]
    shape = {}
    for name in names:
        rows = shown.loc[shown["plate"] == name, "row"]
        cols = shown.loc[shown["plate"] == name, "col"]
        shape[name] = (
            (16 if rows.max() < 16 else 32) if rows.max() >= 8 else 8,
            (24 if cols.max() < 24 else 48) if cols.max() >= 12 else 12,
        )
    data_wells: dict[str, Any] = {
        "plate": [names.index(p) for p in shown["plate"]],
        "row": shown["row"].astype(int).tolist(),
        "col": shown["col"].astype(int).tolist(),
        "well": shown["well"].tolist(),
        "role": [ROLE_CODES[r] for r in shown["role"]],
        "compound": shown["compound"].fillna("").tolist(),
        "hit": shown["well_hit"].astype(bool).astype(int).tolist(),
        "artifact": shown["artifact"].fillna("").tolist(),
        "values": {c: [_num(v) for v in shown[c]] for c in columns},
    }
    compounds: pd.DataFrame = result.compounds
    samples = wells[wells["role"] == "sample"].copy()
    samples["key"] = [
        f"{c}|{conc:.6g}|{line}"
        for c, conc, line in zip(
            samples["compound"],
            samples["concentration"].fillna(0),
            samples["cell_line"],
            strict=True,
        )
    ]
    order = samples.sort_values("plate_order")
    replicate_values: dict[str, list[list[float | None]]] = {c: [] for c in columns}
    replicate_names, replicate_hits = [], []
    final = dict(zip(compounds["key"], compounds.get("final_hit", compounds["hit"]), strict=True))
    for key, group in order[order["artifact"] == ""].groupby("key", sort=False):
        if len(group) < 2:
            continue
        replicate_names.append(str(group["compound"].iloc[0]))
        replicate_hits.append(int(bool(final.get(key, False))))
        for c in columns:
            replicate_values[c].append([_num(group[c].iloc[0]), _num(group[c].iloc[1])])
    hit_columns = [
        "rank",
        "compound",
        "concentration",
        "cell_line",
        "n",
        "n_hit",
        *[c for c in columns if c in result.hits],
        *[c for c in result.hits.columns if c.endswith(("ssmd_mm", "ssmd_robust"))],
        "override",
        "override_reason",
        "wells",
        "artifacts",
    ]
    hit_columns = [c for c in hit_columns if c in result.hits]
    hits = result.hits[hit_columns]
    plate_columns = [
        c for c in plates.columns if c.endswith(("z_prime", "z_prime_robust", "b_score"))
    ]
    return {
        "name": config.name,
        "rule": config.hit_rule,
        "score": config.score,
        "replicate_rule": config.replicate_rule,
        "columns": columns,
        "default": f"{config.primary}_score",
        "thresholds": {
            k: [[op, v] for op, v in items] for k, items in thresholds(config.hit_rule).items()
        },
        "plates": [
            {
                "plate": p,
                "rows": shape[p][0],
                "cols": shape[p][1],
                "hit_rate": _num(r["hit_rate_pct"]),
                "samples": int(r["samples"] or 0),
                **{c: (bool(r[c]) if c.endswith("b_score") else _num(r[c])) for c in plate_columns},
            }
            for p, (_, r) in zip(names, plates.iterrows(), strict=True)
        ],
        "wells": data_wells,
        "replicates": {"names": replicate_names, "hit": replicate_hits, "values": replicate_values},
        "hits": json.loads(hits.to_json(orient="records")),
        "hit_columns": hit_columns,
        "summary": {
            "plates": len(plates),
            "compounds": len(compounds),
            "hits": len(result.hits),
            "hit_rate_pct": _num(100 * len(result.hits) / max(len(compounds), 1), 3),
            "artifact_wells": int((samples["artifact"] != "").sum()),
            "overrides": int(
                (compounds.get("override", pd.Series(dtype=str)).fillna("") != "").sum()
            ),
        },
        "notes": result.notes,
    }


STYLE = """
body{font-family:system-ui,sans-serif;margin:16px;color:#222}h1{margin:0 0 4px}h2{margin:20px 0 6px}
.muted{color:#666;font-size:13px}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{border:1px solid #ddd;border-radius:6px;padding:6px 10px;min-width:90px}
.card b{display:block;font-size:20px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #e3e3e3;padding:2px 6px;text-align:right}
th{background:#f3f3f3}td.l,th.l{text-align:left}.bad{color:#c62828;font-weight:600}#maps{display:flex;flex-wrap:wrap;gap:10px}
#maps figure{margin:0;font-size:11px}#hover{position:fixed;background:#fff;border:1px solid #999;padding:2px 5px;font-size:12px;pointer-events:none;display:none}
canvas{border:1px solid #ddd}.row{display:flex;gap:16px;flex-wrap:wrap}
"""

SCRIPT = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const W = D.wells, N = W.role.length;
const sel = document.getElementById('score');
sel.innerHTML = D.columns.map((c) => `<option>${c}</option>`).join('');
sel.value = D.columns.includes(D.default) ? D.default : D.columns[0];
const hover = document.getElementById('hover');
function isPct(c) { return c.endsWith('_pct'); }
function isRaw(c) { return !/_(score|z|b|pct)$/.test(c); }
function range(c) {
  const v = W.values[c].filter((x) => x !== null).sort((a, b) => a - b);
  if (!v.length) return [0, 1];
  if (isPct(c)) return [0, 100];
  const lo = v[Math.floor(v.length * 0.01)], hi = v[Math.floor(v.length * 0.99)];
  if (isRaw(c)) return [lo, hi];
  const m = Math.max(Math.abs(lo), Math.abs(hi), 3);
  return [-m, m];
}
function colour(c, x, lohi) {
  if (x === null) return '#eee';
  const [lo, hi] = lohi;
  let t = (x - lo) / ((hi - lo) || 1) * 2 - 1;
  if (isPct(c)) t = -t;  // activity: high = red would read as "more"; keep red = strong effect
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? `rgb(${Math.round(255 + 200 * t)},${Math.round(255 + 120 * t)},255)` : `rgb(255,${Math.round(255 - 150 * t)},${Math.round(255 - 200 * t)})`;
}
function maps() {
  const c = sel.value, lohi = range(c), out = document.getElementById('maps');
  out.innerHTML = '';
  D.plates.forEach((p, k) => {
    const cell = p.cols > 24 ? 5 : 8, cv = document.createElement('canvas');
    cv.width = p.cols * cell; cv.height = p.rows * cell; cv.dataset.plate = p.plate;
    const g = cv.getContext('2d'), index = {};
    let drawn = 0;
    for (let i = 0; i < N; i++) {
      if (W.plate[i] !== k) continue;
      const x = W.col[i] * cell, y = W.row[i] * cell;
      index[`${W.row[i]},${W.col[i]}`] = i;
      g.fillStyle = W.artifact[i] ? '#bbb' : colour(c, W.values[c][i], lohi);
      g.fillRect(x, y, cell - 1, cell - 1);
      drawn++;
      if (W.artifact[i]) { g.strokeStyle = '#555'; g.beginPath(); g.moveTo(x, y); g.lineTo(x + cell - 1, y + cell - 1); g.stroke(); }
      if (W.hit[i]) { g.strokeStyle = '#000'; g.lineWidth = 1.5; g.strokeRect(x + 0.75, y + 0.75, cell - 2.5, cell - 2.5); g.lineWidth = 1; }
      if (W.role[i]) { g.fillStyle = W.role[i] === 1 ? '#1565c0' : '#2e7d32'; g.fillRect(x + cell / 2 - 1, y + cell / 2 - 1, 2, 2); }
    }
    cv.dataset.wells = drawn;
    cv.onmousemove = (e) => {
      const r = Math.floor(e.offsetY / cell), col = Math.floor(e.offsetX / cell), i = index[`${r},${col}`];
      if (i === undefined) { hover.style.display = 'none'; return; }
      const v = W.values[c][i];
      hover.textContent = `${p.plate} ${W.well[i]} ${W.compound[i] || ['', 'vehicle', 'positive'][W.role[i]]} ${c}=${v === null ? '–' : v}${W.hit[i] ? ' HIT' : ''}${W.artifact[i] ? ' [' + W.artifact[i] + ']' : ''}`;
      hover.style.display = 'block'; hover.style.left = e.clientX + 12 + 'px'; hover.style.top = e.clientY + 12 + 'px';
    };
    cv.onmouseleave = () => { hover.style.display = 'none'; };
    const fig = document.createElement('figure');
    const zp = p[`${D.default.replace('_score', '')}_z_prime`];
    fig.innerHTML = `<figcaption>${p.plate} <span class="${zp !== undefined && zp !== null && zp < 0.5 ? 'bad' : 'muted'}">Z′ ${zp === null || zp === undefined ? '–' : zp.toFixed(2)}</span></figcaption>`;
    fig.prepend(cv);
    out.appendChild(fig);
  });
}
function axes(g, cv, x0, y0, x1, y1, lo, hi, label) {
  g.strokeStyle = '#999'; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  g.fillStyle = '#444'; g.font = '10px sans-serif';
  g.fillText(String(+hi.toPrecision(3)), 2, y0 + 8); g.fillText(String(+lo.toPrecision(3)), 2, y1);
  g.fillText(label, (x0 + x1) / 2 - 30, cv.height - 4);
}
function thresholdsFor(c) { return D.thresholds[c] || []; }
function order() {
  const c = sel.value, cv = document.getElementById('order'), g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const vals = W.values[c].filter((x) => x !== null);
  if (!vals.length) return;
  const [lo0, hi0] = range(c), lo = Math.min(lo0, ...thresholdsFor(c).map((t) => t[1])), hi = Math.max(hi0, ...thresholdsFor(c).map((t) => t[1]));
  const x0 = 50, x1 = cv.width - 10, y0 = 10, y1 = cv.height - 24, n = D.plates.length;
  const X = (k) => x0 + (k + 0.5) / n * (x1 - x0), Y = (v) => y1 - (Math.max(lo, Math.min(hi, v)) - lo) / ((hi - lo) || 1) * (y1 - y0);
  axes(g, cv, x0, y0, x1, y1, lo, hi, 'plate order');
  let points = 0;
  for (let i = 0; i < N; i++) {
    const v = W.values[c][i];
    if (v === null || W.role[i] !== 0) continue;
    const jitter = ((i * 9301 + 49297) % 233280) / 233280 - 0.5;
    g.fillStyle = W.hit[i] ? 'rgba(198,40,40,0.9)' : 'rgba(120,120,120,0.35)';
    g.fillRect(X(W.plate[i]) + jitter * (x1 - x0) / n * 0.7 - 1, Y(v) - 1, 2, 2);
    points++;
  }
  [[1, '#1565c0'], [2, '#2e7d32']].forEach(([role, col]) => {
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    D.plates.forEach((_, k) => {
      const v = W.values[c].filter((x, i) => x !== null && W.plate[i] === k && W.role[i] === role).sort((a, b) => a - b);
      if (!v.length) return;
      const m = v[Math.floor(v.length / 2)];
      k ? g.lineTo(X(k), Y(m)) : g.moveTo(X(k), Y(m));
    });
    g.stroke(); g.lineWidth = 1;
  });
  thresholdsFor(c).forEach(([op, t]) => { g.strokeStyle = '#c62828'; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x0, Y(t)); g.lineTo(x1, Y(t)); g.stroke(); g.setLineDash([]); });
  cv.dataset.points = points;
}
function histogram() {
  const c = sel.value, cv = document.getElementById('hist'), g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const v = W.values[c].filter((x, i) => x !== null && W.role[i] === 0);
  if (!v.length) return;
  const [lo0, hi0] = range(c), ts = thresholdsFor(c).map((t) => t[1]);
  const lo = Math.min(lo0, ...ts), hi = Math.max(hi0, ...ts), bins = 60, counts = new Array(bins).fill(0);
  v.forEach((x) => { counts[Math.max(0, Math.min(bins - 1, Math.floor((x - lo) / ((hi - lo) || 1) * bins)))]++; });
  const top = Math.max(...counts), x0 = 50, x1 = cv.width - 10, y0 = 10, y1 = cv.height - 24;
  axes(g, cv, x0, y0, x1, y1, 0, top, c);
  g.fillStyle = '#90a4ae';
  counts.forEach((n, b) => { const h = Math.sqrt(n / top) * (y1 - y0); g.fillRect(x0 + b / bins * (x1 - x0), y1 - h, (x1 - x0) / bins - 1, h); });
  g.fillStyle = '#444'; g.fillText(String(+lo.toPrecision(3)), x0, y1 + 12); g.fillText(String(+hi.toPrecision(3)), x1 - 30, y1 + 12);
  ts.forEach((t) => { const x = x0 + (t - lo) / ((hi - lo) || 1) * (x1 - x0); g.strokeStyle = '#c62828'; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke(); g.setLineDash([]); });
  cv.dataset.wells = v.length;
}
function replicates() {
  const c = sel.value, cv = document.getElementById('reps'), g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const pairs = D.replicates.values[c] || [];
  const ok = pairs.filter((p) => p[0] !== null && p[1] !== null);
  if (!ok.length) { cv.dataset.points = 0; return; }
  const ts = thresholdsFor(c), flat = ok.flat().concat(ts.map((t) => t[1])).sort((a, b) => a - b);
  const lo = flat[0], hi = flat[flat.length - 1], x0 = 50, x1 = cv.width - 10, y0 = 10, y1 = cv.height - 24;
  const X = (v) => x0 + (v - lo) / ((hi - lo) || 1) * (x1 - x0), Y = (v) => y1 - (v - lo) / ((hi - lo) || 1) * (y1 - y0);
  axes(g, cv, x0, y0, x1, y1, lo, hi, 'replicate 1 → ; replicate 2 ↑');
  ts.forEach(([op, t]) => {
    g.fillStyle = 'rgba(198,40,40,0.08)';
    if (op[0] === '<') g.fillRect(x0, Y(t), X(t) - x0, y1 - Y(t)); else g.fillRect(X(t), y0, x1 - X(t), Y(t) - y0);
  });
  let n = 0;
  pairs.forEach((p, k) => {
    if (p[0] === null || p[1] === null) return;
    g.fillStyle = D.replicates.hit[k] ? '#c62828' : 'rgba(90,90,90,0.4)';
    g.fillRect(X(p[0]) - 1.5, Y(p[1]) - 1.5, 3, 3);
    n++;
  });
  cv.dataset.points = n;
}
function draw() { maps(); order(); histogram(); replicates(); }
sel.onchange = draw;
draw();
"""


def _table(frame: pd.DataFrame, links: dict[str, str] | None = None, bad: str | None = None) -> str:
    if frame.empty:
        return '<p class="muted">None.</p>'
    head = "".join(
        f"<th{' class=l' if not pd.api.types.is_numeric_dtype(frame[c]) else ''}>"
        f"{html.escape(str(c))}</th>"
        for c in frame.columns
    )
    rows = []
    for record in frame.to_dict("records"):
        cells = []
        for column, value in record.items():
            numeric = isinstance(value, int | float) and not isinstance(value, bool)
            text = (
                f"{value:.3g}"
                if isinstance(value, float) and math.isfinite(value)
                else ""
                if isinstance(value, float)
                else html.escape(str(value))
            )
            if column == "compound" and links and str(value) in links:
                text = f'<a href="{html.escape(links[str(value)])}">{text}</a>'
            css = (
                " class=bad"
                if bad and str(column).endswith(bad) and numeric and value < 0.5
                else ""
            )
            cells.append(f"<td{css if numeric else ' class=l'}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def write_report(result: Any, out: Path) -> Path:
    data = report_data(result)
    s = data["summary"]
    cards = "".join(
        f'<div class="card"><b>{html.escape(str(v))}</b>{html.escape(k.replace("_", " "))}</div>'
        for k, v in s.items()
    )
    gallery = Path(out) / "gallery" / "index.html"
    links = {}
    if gallery.exists():
        links = {str(c): f"gallery/index.html#{_anchor(str(c))}" for c in result.hits["compound"]}
    plates = result.plates.drop(columns=[c for c in ("run", "started") if c in result.plates])
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in data["notes"])
    payload = json.dumps(data, allow_nan=False, default=str).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Screen {html.escape(data["name"])}</title>
<style>{STYLE}</style></head><body>
<h1>Screen: {html.escape(data["name"])}</h1>
<p class="muted">Hit rule <code>{html.escape(data["rule"])}</code> · score {html.escape(data["score"])} · replicates: {html.escape(str(data["replicate_rule"]))}
{' · <a href="gallery/index.html">hit gallery</a>' if gallery.exists() else ""}</p>
<div class="cards" id="summary">{cards}</div>
{f"<h2>Notes</h2><ul>{notes}</ul>" if notes else ""}
<h2>Plates</h2>{_table(plates, bad="z_prime")}
<h2>Views</h2><p>Score <select id="score"></select> <span class="muted">blue below, red above; hits outlined; artefact wells grey with a cross; dots: vehicle (blue) and positive control (green)</span></p>
<div id="maps"></div>
<div class="row"><div><h2>Score vs plate order</h2><canvas id="order" width="640" height="260"></canvas></div>
<div><h2>Distribution</h2><canvas id="hist" width="420" height="260"></canvas></div>
<div><h2>Replicates</h2><canvas id="reps" width="300" height="300"></canvas></div></div>
<h2>Hits ({s["hits"]})</h2>{_table(result.hits[data["hit_columns"]], links)}
<div id="hover"></div>
<script type="application/json" id="data">{payload}</script>
<script>{SCRIPT}</script></body></html>"""
    target = Path(out) / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


def _anchor(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text)


# --------------------------------------------------------------------- gallery


def _vehicle_neighbours(wells: pd.DataFrame, plate: str, row: int, col: int, k: int) -> list[str]:
    vehicle = wells[
        (wells["plate"] == plate)
        & (wells["role"] == "negative_control")
        & (wells["artifact"] == "")
    ]
    if vehicle.empty:
        return []
    distance = (vehicle["row"] - row) ** 2 + (vehicle["col"] - col) ** 2
    return list(vehicle.assign(d=distance).sort_values("d")["well"].head(k))


def write_gallery(
    result: Any, out: Path, max_hits: int = 48, vehicles: int = 2, size: int = 200
) -> Path | None:
    """``gallery/``: for each hit (ranked, at most ``max_hits``) and each of its replicate wells,
    the well's image next to the ``vehicles`` nearest vehicle wells of the same plate, with the
    **same** contrast for every image in the row (linked contrast, Phase 14) — so a dimmer or
    emptier well really is dimmer or emptier. Images are the runs' saved corrected images of the
    representative field (closest to the well's median readout); runs saved without images get
    a note instead. None when there are no hits."""
    from PIL import Image

    from hcs_studio.compare.render import auto_contrast, compose, default_channels
    from hcs_studio.compare.sources import load_panel, plane
    from hcs_studio.compare.spec import PanelSpec

    if result.hits.empty:
        return None
    folder = Path(out) / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    wells: pd.DataFrame = result.wells
    runs = dict(zip(result.plates["plate"], result.plates["run"], strict=True))
    run_paths = {Path(p).name: Path(p) for p in getattr(result, "run_folders", [])}
    readout = result.config.readouts[0].column
    sections, missing = [], 0
    for hit in result.hits.head(max_hits).to_dict("records"):
        compound = str(hit["compound"])
        rows_html = []
        for entry in str(hit["wells"]).split(","):
            plate, well = entry.split(":")
            run = run_paths.get(str(runs.get(plate)))
            info = wells[(wells["plate"] == plate) & (wells["well"] == well)].iloc[0]
            members = [
                well,
                *_vehicle_neighbours(wells, plate, int(info["row"]), int(info["col"]), vehicles),
            ]
            panels = []
            try:
                if run is None:
                    raise ValueError("run folder unknown")
                for name in members:
                    image = load_panel(
                        PanelSpec(run=str(run), well=name), representative_readout=readout
                    )
                    panels.append((name, plane(image.image, "max"), image.image.metadata))
            except (OSError, ValueError, KeyError, IndexError) as exc:
                missing += 1
                rows_html.append(
                    f'<p class="muted">{html.escape(plate)} {html.escape(well)}: no image ({html.escape(str(exc))})</p>'
                )
                continue
            display = auto_contrast([p for _, p, _ in panels], default_channels(panels[0][2]))
            cells = []
            for k, (name, planes, _) in enumerate(panels):
                rgb = (compose(planes, display) * 255).astype(np.uint8)
                picture = Image.fromarray(rgb)
                picture.thumbnail((size, size))
                file = f"{_anchor(plate)}_{name}.png"
                picture.save(folder / file)
                label = "hit" if k == 0 else "vehicle"
                cells.append(
                    f'<figure><img src="{file}" alt="{html.escape(name)}"><figcaption>{html.escape(name)} · {label}</figcaption></figure>'
                )
            rows_html.append(
                f'<div class="strip"><b>{html.escape(plate)}</b>{"".join(cells)}</div>'
            )
        score = hit.get(f"{result.config.primary}_score")
        sections.append(
            f'<section id="{_anchor(compound)}"><h2>#{hit["rank"]} {html.escape(compound)} '
            f"<small>score {score:.3g}</small></h2>{''.join(rows_html)}</section>"
            if isinstance(score, float)
            else f'<section id="{_anchor(compound)}"><h2>#{hit["rank"]} {html.escape(compound)}</h2>{"".join(rows_html)}</section>'
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Hit gallery</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}.strip{{display:flex;gap:8px;align-items:center;margin:4px 0}}
figure{{margin:0;font-size:11px;text-align:center}}img{{display:block;background:#000;max-width:{size}px}}h2{{margin:14px 0 4px;font-size:16px}}.muted{{color:#666}}</style></head><body>
<h1>Hit gallery</h1><p class="muted">Each hit well beside its nearest vehicle wells on the same plate, all images of a row with the same contrast. <a href="../report.html">Screen report</a>
{f" · {missing} well(s) without images" if missing else ""}</p>
{"".join(sections)}</body></html>"""
    target = folder / "index.html"
    target.write_text(page, encoding="utf-8")
    return target
