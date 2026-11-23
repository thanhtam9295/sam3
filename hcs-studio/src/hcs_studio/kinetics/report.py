"""``kinetics/report.html``: the views of a time-lapse plate (Phase 22c).

A self-contained page (inline CSS and JavaScript, data as embedded JSON, no network), like
``index.html``:

- **plate map with sparklines**: each well's readout over time, on one shared log scale, coloured
  by role / compound (darker = higher concentration); wells with kinetic-QC flags have a red
  border and list them on hover; clicking a well adds it to the overlay;
- **condition × time heat map**: one row per condition (vehicle, then each compound by
  concentration), one column per time point; the colour is the median over replicate wells of
  log2 (readout / vehicle median at the same time point) — or the log2 fold change since the
  first time point — blue below, red above;
- **curve overlays**: the median curve of every concentration of one compound with the vehicle
  in black (and wells clicked on the plate map);
- **QC**: the status of each kinetic check, and the focus / edge-effect timelines.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def report_data(result: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Everything the page draws, JSON-ready."""
    from hcs_studio.model.plate import parse_well_name

    matrix: pd.DataFrame = result.matrix
    times = [float(t) for t in matrix.columns]
    conditions = (
        result.conditions.set_index("well") if not result.conditions.empty else pd.DataFrame()
    )
    flags = result.qc.well_flags if result.qc is not None else {}
    fits = result.fits.set_index("well") if not result.fits.empty else pd.DataFrame()
    wells = {}
    for well, row in matrix.iterrows():
        name = str(well)
        r, c = parse_well_name(name)
        annotation = conditions.loc[name] if name in conditions.index else None
        fit = fits.loc[name] if name in fits.index else None
        wells[name] = {
            "row": r,
            "col": c,
            "values": [_number(v) for v in row.to_numpy()],
            "role": str(annotation["role"]) if annotation is not None else "",
            "compound": str(annotation["compound"]) if annotation is not None else "",
            "concentration": _number(annotation["concentration"])
            if annotation is not None
            else None,
            "cell_line": str(annotation["cell_line"]) if annotation is not None else "",
            "flags": flags.get(name, []),
            "doubling_h": _number(fit["doubling_h"]) if fit is not None else None,
            "model": str(fit["model"]) if fit is not None else "",
        }
    rows = int(info.get("rows") or max((w["row"] for w in wells.values()), default=0) + 1)
    cols = int(info.get("cols") or max((w["col"] for w in wells.values()), default=0) + 1)
    qc = result.qc
    timeline = qc.timeline.to_dict("list") if qc is not None and not qc.timeline.empty else {}
    return {
        "readout": result.readout,
        "time_label": "hours since treatment"
        if result.time_column == "hours_since_treatment"
        else "hours",
        "times": times,
        "rows": rows,
        "cols": cols,
        "wells": wells,
        "qc": {"status": qc.status, "checks": qc.checks} if qc is not None else None,
        "timeline": {k: [_number(v) for v in values] for k, values in timeline.items()},
        "notes": result.notes,
        "info": {
            k: info.get(k)
            for k in ("treated_at", "treatment_source", "acquired_at", "time_source", "runs")
            if info.get(k)
        },
    }


def _json(data: dict[str, Any]) -> str:
    text = json.dumps(data, allow_nan=False, default=str)
    return text.replace("</", "<\\/")


STYLE = """
body{font-family:system-ui,sans-serif;margin:16px;color:#222}h1{margin:0 0 4px}h2{margin:18px 0 6px}
.muted{color:#666;font-size:13px}.status{display:inline-block;padding:1px 6px;border-radius:3px;color:#fff;font-size:12px}
.pass{background:#2e7d32}.warn{background:#ef6c00}.fail{background:#c62828}.na{background:#9e9e9e}
#plate{display:grid;gap:2px;margin:6px 0}#plate .w{border:1px solid #ccc;border-radius:3px;position:relative;cursor:pointer;background:#fafafa}
#plate .w.flagged{border:2px solid #c62828}#plate .w.on{outline:2px solid #1565c0}#plate .w svg{display:block;width:100%;height:100%}
#plate .lbl{position:absolute;top:0;left:2px;font-size:8px;color:#777}#plate .hdr{font-size:10px;color:#666;text-align:center;align-self:center}
table.heat{border-collapse:collapse;font-size:11px}table.heat td,table.heat th{padding:2px 4px;text-align:right;border:1px solid #fff}
table.heat th{background:#eee;font-weight:normal}.legend span{display:inline-block;width:22px;height:10px}
ul.checks{padding-left:18px}ul.checks li{margin:3px 0}svg text{font-size:10px;fill:#444}
"""

SCRIPT = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const wells = Object.entries(D.wells);
const compounds = [...new Set(wells.map(([, w]) => w.compound).filter((c) => c && c !== 'combination'))].sort();
const doses = {};
for (const c of compounds) doses[c] = [...new Set(wells.filter(([, w]) => w.compound === c).map(([, w]) => w.concentration))].sort((a, b) => a - b);
const palette = ['#1565c0', '#2e7d32', '#6a1b9a', '#ef6c00', '#00838f', '#ad1457', '#5d4037', '#827717'];
function conc(m) {
  if (m === null || m === undefined) return '';
  const units = [[1e-3, 'mM'], [1e-6, 'µM'], [1e-9, 'nM'], [1e-12, 'pM']];
  for (const [f, u] of units) if (m >= f) return `${+(m / f).toPrecision(3)} ${u}`;
  return `${m.toPrecision(3)} M`;
}
function colour(w) {
  if (w.role === 'negative_control') return '#000';
  if (w.role === 'positive_control') return '#c62828';
  const i = compounds.indexOf(w.compound);
  if (i < 0) return '#888';
  const d = doses[w.compound], k = d.length > 1 ? d.indexOf(w.concentration) / (d.length - 1) : 1;
  return shade(palette[i % palette.length], 0.35 + 0.65 * k);
}
function shade(hex, a) {
  const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  const mix = (v) => Math.round(255 - (255 - v) * a);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}
const positive = wells.flatMap(([, w]) => w.values).filter((v) => v !== null && v > 0);
const logScale = positive.length > 0 && wells.every(([, w]) => w.values.every((v) => v === null || v > 0));
const tr = (v) => (logScale ? Math.log10(v) : v);
const all = wells.flatMap(([, w]) => w.values).filter((v) => v !== null && (!logScale || v > 0)).map(tr);
const lo = Math.min(...all), hi = Math.max(...all);
const t0 = Math.min(...D.times), t1 = Math.max(...D.times);
function path(values, x0, x1, y0, y1) {
  const pts = [];
  values.forEach((v, i) => {
    if (v === null || (logScale && v <= 0)) return;
    const x = x0 + (D.times[i] - t0) / ((t1 - t0) || 1) * (x1 - x0);
    const y = y1 - (tr(v) - lo) / ((hi - lo) || 1) * (y1 - y0);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  return pts.join(' ');
}
const selected = new Set();
function plate() {
  const el = document.getElementById('plate');
  el.style.gridTemplateColumns = `24px repeat(${D.cols}, minmax(36px, 1fr))`;
  let out = '<div></div>';
  for (let c = 0; c < D.cols; c++) out += `<div class="hdr">${c + 1}</div>`;
  const byPos = {};
  for (const [name, w] of wells) byPos[`${w.row},${w.col}`] = [name, w];
  for (let r = 0; r < D.rows; r++) {
    out += `<div class="hdr">${String.fromCharCode(65 + (r % 26))}</div>`;
    for (let c = 0; c < D.cols; c++) {
      const hit = byPos[`${r},${c}`];
      if (!hit) { out += '<div class="w" style="visibility:hidden;height:30px"></div>'; continue; }
      const [name, w] = hit;
      const tip = [name, w.role, w.compound && `${w.compound} ${conc(w.concentration)}`, w.doubling_h && `doubling ${w.doubling_h.toFixed(1)} h`, ...w.flags].filter(Boolean).join('\n');
      out += `<div class="w${w.flags.length ? ' flagged' : ''}" id="w-${name}" data-well="${name}" title="${tip.replace(/"/g, '&quot;')}" style="height:30px">` +
        `<span class="lbl">${name}</span><svg viewBox="0 0 60 30" preserveAspectRatio="none"><polyline fill="none" stroke="${colour(w)}" stroke-width="1.4" points="${path(w.values, 2, 58, 4, 28)}"/></svg></div>`;
    }
  }
  el.innerHTML = out;
  el.querySelectorAll('.w[data-well]').forEach((cell) => cell.addEventListener('click', () => {
    const name = cell.dataset.well;
    if (selected.has(name)) selected.delete(name); else selected.add(name);
    cell.classList.toggle('on');
    overlay();
  }));
}
function median(xs) {
  const v = xs.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function conditionRows() {
  const rows = [];
  const vehicle = wells.filter(([, w]) => w.role === 'negative_control');
  if (vehicle.length) rows.push({ label: 'vehicle', members: vehicle, colour: '#000' });
  for (const c of compounds) for (const d of doses[c]) {
    const members = wells.filter(([, w]) => w.compound === c && w.concentration === d);
    rows.push({ label: `${c} ${conc(d)}`, members, colour: colour(members[0][1]) });
  }
  const others = wells.filter(([, w]) => w.role !== 'negative_control' && !compounds.includes(w.compound));
  if (!rows.length) for (const [name, w] of others) rows.push({ label: name, members: [[name, w]], colour: '#888' });
  return rows;
}
function heat() {
  const mode = document.getElementById('heatmode').value;
  const vehicle = wells.filter(([, w]) => w.role === 'negative_control');
  const ref = D.times.map((_, i) => median(vehicle.map(([, w]) => w.values[i])));
  let out = '<tr><th></th>' + D.times.map((t) => `<th>${+t.toFixed(1)}</th>`).join('') + '</tr>';
  for (const row of conditionRows()) {
    out += `<tr><th style="text-align:left;border-left:6px solid ${row.colour}">${row.label} <span class="muted">(${row.members.length})</span></th>`;
    D.times.forEach((_, i) => {
      const values = row.members.map(([, w]) => {
        const v = w.values[i], first = w.values.find((x) => x !== null);
        if (v === null || v <= 0) return null;
        if (mode === 'vehicle') return ref[i] ? Math.log2(v / ref[i]) : null;
        return first ? Math.log2(v / first) : null;
      });
      const m = median(values);
      out += m === null ? '<td></td>' : `<td style="background:${diverging(m)}" title="log2 ${m.toFixed(2)}">${m.toFixed(2)}</td>`;
    });
    out += '</tr>';
  }
  document.getElementById('heat').innerHTML = out;
}
function diverging(v) {
  const k = Math.max(-1, Math.min(1, v / 2));
  return k < 0 ? `rgb(${Math.round(255 + 200 * k)},${Math.round(255 + 120 * k)},255)` : `rgb(255,${Math.round(255 - 150 * k)},${Math.round(255 - 200 * k)})`;
}
function overlay() {
  const compound = document.getElementById('compound').value;
  const W = 820, H = 300, x0 = 50, x1 = 640, y0 = 10, y1 = H - 30;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="none" stroke="#ccc"/>`;
  svg += `<text x="${(x0 + x1) / 2}" y="${H - 6}" text-anchor="middle">${D.time_label}</text>`;
  svg += `<text x="${x0}" y="${y1 + 14}">${+t0.toFixed(1)}</text><text x="${x1}" y="${y1 + 14}" text-anchor="end">${+t1.toFixed(1)}</text>`;
  svg += `<text x="4" y="${y0 + 8}">${logScale ? (10 ** hi).toPrecision(3) : hi.toPrecision(3)}</text><text x="4" y="${y1}">${logScale ? (10 ** lo).toPrecision(3) : lo.toPrecision(3)}</text>`;
  const series = [];
  const vehicle = wells.filter(([, w]) => w.role === 'negative_control');
  if (vehicle.length) series.push(['vehicle', '#000', D.times.map((_, i) => median(vehicle.map(([, w]) => w.values[i])))]);
  if (compound) for (const d of doses[compound]) {
    const members = wells.filter(([, w]) => w.compound === compound && w.concentration === d);
    series.push([`${compound} ${conc(d)}`, colour(members[0][1]), D.times.map((_, i) => median(members.map(([, w]) => w.values[i])))]);
  }
  for (const name of selected) series.push([name, '#1565c0', D.wells[name].values, true]);
  series.forEach(([label, col, values, dashed], k) => {
    svg += `<polyline fill="none" stroke="${col}" stroke-width="1.6" ${dashed ? 'stroke-dasharray="4 3"' : ''} points="${path(values, x0, x1, y0, y1)}"><title>${label}</title></polyline>`;
    svg += `<line x1="${x1 + 10}" x2="${x1 + 26}" y1="${y0 + 8 + 12 * k}" y2="${y0 + 8 + 12 * k}" stroke="${col}" stroke-width="2" ${dashed ? 'stroke-dasharray="4 3"' : ''}/>`;
    svg += `<text x="${x1 + 30}" y="${y0 + 12 + 12 * k}">${label}</text>`;
  });
  document.getElementById('overlay').innerHTML = svg + '</svg>';
}
function timeline() {
  const el = document.getElementById('timeline');
  const keys = Object.keys(D.timeline).filter((k) => k !== 'time_h');
  if (!keys.length) { el.innerHTML = '<p class="muted">No focus or edge timeline.</p>'; return; }
  const ts = D.timeline.time_h, W = 640, H = 160, x0 = 50, x1 = W - 10, y0 = 10, y1 = H - 25;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="none" stroke="#ccc"/>`;
  const cols = { relative_focus: '#1565c0', edge_log2: '#c62828' };
  keys.forEach((k, j) => {
    const v = D.timeline[k], ok = v.filter((x) => x !== null);
    const pad = Math.max((Math.max(...ok) - Math.min(...ok)) * 0.15, 0.1);
    const a = Math.min(...ok) - pad, b = Math.max(...ok) + pad;
    const pts = v.map((y, i) => y === null ? '' : `${(x0 + (ts[i] - ts[0]) / ((ts[ts.length - 1] - ts[0]) || 1) * (x1 - x0)).toFixed(1)},${(y1 - (y - a) / ((b - a) || 1) * (y1 - y0)).toFixed(1)}`).join(' ');
    svg += `<polyline fill="none" stroke="${cols[k] || '#444'}" stroke-width="1.6" points="${pts}"/><text x="${x0 + 6}" y="${y0 + 12 + 12 * j}" style="fill:${cols[k] || '#444'}">${k === 'edge_log2' ? 'edge effect (log2 vs interior)' : 'focus relative to each well'}: ${Math.min(...ok).toFixed(2)} to ${Math.max(...ok).toFixed(2)}</text>`;
  });
  el.innerHTML = svg + `<text x="${(x0 + x1) / 2}" y="${H - 6}" text-anchor="middle">${D.time_label}</text></svg>`;
}
const select = document.getElementById('compound');
select.innerHTML = '<option value="">(vehicle only)</option>' + compounds.map((c) => `<option>${c}</option>`).join('');
if (compounds.length) select.value = compounds[0];
select.onchange = overlay;
document.getElementById('heatmode').onchange = heat;
plate(); heat(); overlay(); timeline();
"""


def write_report(result: Any, out: Path, info: dict[str, Any]) -> Path:
    data = report_data(result, info)
    qc = data["qc"]
    checks = ""
    if qc is not None:
        items = []
        for check in qc["checks"]:
            css = {"pass": "pass", "warn": "warn", "fail": "fail"}.get(check["status"], "na")
            items.append(f'<li><span class="status {css}">{html.escape(check["status"])}</span> '
                         f'<b>{html.escape(check["check"])}</b>: {html.escape(check["message"])}</li>')  # fmt: skip
        checks = f'<ul class="checks" id="checks">{"".join(items)}</ul>'
    facts = " · ".join(
        f"{html.escape(k.replace('_', ' '))}: {html.escape(str(v))}"
        for k, v in data["info"].items()
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in data["notes"])
    flagged = sum(1 for w in data["wells"].values() if w["flags"])
    status = qc["status"] if qc else "na"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kinetics</title><style>{STYLE}</style></head><body>
<h1>Kinetics</h1>
<p class="muted">Readout <b>{html.escape(data["readout"])}</b> over {len(data["times"])} time points ({html.escape(data["time_label"])}). {facts}</p>
<h2>QC <span class="status {status if status in ("pass", "warn", "fail") else "na"}" id="qc-status">{html.escape(status)}</span></h2>
{checks}<div id="timeline"></div>
<h2>Plate</h2><p class="muted">Each well's readout over time on one shared scale (logarithmic when every value is positive); red border: kinetic-QC flags ({flagged} wells, hover for details); click wells to add them to the overlay.</p>
<div id="plate"></div>
<h2>Curves</h2><p>Compound <select id="compound"></select></p><div id="overlay"></div>
<h2>Conditions over time</h2>
<p>Colour <select id="heatmode"><option value="vehicle">log2 vs vehicle at the same time</option><option value="first">log2 fold change since the first time point</option></select>
<span class="legend"><span style="background:{_css(-2)}"></span> −2 <span style="background:#fff;border:1px solid #ddd"></span> 0 <span style="background:{_css(2)}"></span> +2</span></p>
<table class="heat" id="heat"></table>
{f"<h2>Notes</h2><ul>{notes}</ul>" if notes else ""}
<script type="application/json" id="data">{_json(data)}</script>
<script>{SCRIPT}</script></body></html>"""
    target = out / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


def _css(value: float) -> str:
    k = max(-1.0, min(1.0, value / 2))
    if k < 0:
        return f"rgb({round(255 + 200 * k)},{round(255 + 120 * k)},255)"
    return f"rgb(255,{round(255 - 150 * k)},{round(255 - 200 * k)})"


def heatmap_figure(result: Any, path: Path) -> bool:
    """Static condition × time heat map (log2 vs the same-time vehicle median) for documents."""
    import matplotlib.pyplot as plt

    matrix: pd.DataFrame = result.matrix
    info = result.conditions
    if matrix.empty or info.empty or not (info["role"] == "negative_control").any():
        path.unlink(missing_ok=True)
        return False
    vehicle = [w for w in info.loc[info["role"] == "negative_control", "well"] if w in matrix.index]
    if not vehicle:
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = pd.DataFrame(
            np.log2(matrix.to_numpy(float) / matrix.loc[vehicle].median().to_numpy(float)),
            index=matrix.index,
            columns=matrix.columns,
        )
    rows, labels = [], []
    samples = info[(info["compound"] != "") & (info["compound"] != "combination")]
    for (compound, conc), group in samples.groupby(["compound", "concentration"]):
        wells = [w for w in group["well"] if w in relative.index]
        if wells:
            rows.append(relative.loc[wells].median().to_numpy(float))
            labels.append(f"{compound} {conc:.3g}")
    if not rows:
        return False
    fig, axis = plt.subplots(figsize=(7, 0.25 * len(rows) + 1.5))
    image = axis.imshow(np.vstack(rows), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                        interpolation="nearest")  # fmt: skip
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    times = [f"{t:g}" for t in matrix.columns]
    step = max(len(times) // 12, 1)
    axis.set_xticks(range(0, len(times), step), times[::step], fontsize=7)
    axis.set_xlabel(
        "hours since treatment" if result.time_column == "hours_since_treatment" else "hours"
    )
    fig.colorbar(image, ax=axis, label="log2 vs vehicle")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
