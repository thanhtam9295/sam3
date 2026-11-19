"""Standalone HTML report of a run: ``index.html`` in the run folder.

One self-contained file (styles, scripts and data embedded; previews linked by relative
paths), so the run folder can be zipped and opened anywhere, offline.
"""

from __future__ import annotations

import contextlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from hcs_studio import __version__
from hcs_studio.output.results import RunResults
from hcs_studio.readouts import label as readout_label

#: Readouts listed in every well's statistics table (when present), after the selected one.
_KEY_MATCHES = ("_count", "_per_mm2", "fields", "coloc_")


def _round(value: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.5g}")


def report_data(results: RunResults) -> dict[str, Any]:
    readouts = results.readout_columns()
    values = {key: {w: _round(v) for w, v in results.values(key).items()} for key in readouts}
    wells = {}
    for well in results.wells:
        fields = [
            {
                "field": f.index,
                "enhanced": f.enhanced,
                "corrected": f.corrected,
                "raw": f.raw,
                "overlays": f.overlays,
                "objects": f.objects,
            }
            for f in results.well_fields(well)
        ]
        mosaic = f"wells/{well}/previews/mosaic.png"
        wells[well] = {
            "fields": fields,
            "flags": results.well_flags(well),
            "mosaic": mosaic if (results.folder / mosaic).exists() else None,
        }
    layout = {
        w: {
            "drug": " + ".join(t.compound for t in _drugs(a)),
            "log_molar": _log_molar(a),
            "compound": a.compound,
            "concentration": a.concentration,
            "treatment": a.treatment,
            "control_type": a.control_type,
            "label": a.label(),
            "cell_line": a.cell_line,
            "role": a.role,
            "replicate_group": a.replicate_group,
        }
        for w, a in results.layout.items()
    }
    key_readouts = [
        r
        for r in readouts
        if any(m in r for m in _KEY_MATCHES) and not r.endswith(("_std", "_mean"))
    ][:14]
    pharmacology, normalized = pharmacology_data(results.folder)
    for key, per_well in normalized.items():
        values[key] = per_well
    readouts = [*readouts, *normalized]
    return {
        "plate": {"name": results.plate_name, "rows": results.rows, "cols": results.cols},
        "pharmacology": pharmacology,
        "links": report_links(results.folder),
        "readouts": readouts,
        "readout_labels": {r: readout_label(r) for r in readouts},
        "key_readouts": key_readouts,
        "values": values,
        "wells": wells,
        "layout": layout,
        "embedding": results.profile_embedding(max_cells=3000),
    }


def _drugs(a: Any) -> list[Any]:
    if a.role in ("negative_control", "positive_control", "untreated"):
        return []
    return [t for t in a.active_treatments if t.kind != "percent"]


def _log_molar(a: Any) -> float | None:
    """log10 of the summed molar concentration of a well's drugs (None without drugs)."""
    molar = [t.concentration for t in _drugs(a) if t.kind == "molar"]
    return _round(math.log10(sum(molar))) if molar else None


def pharmacology_data(folder: Path) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """The run's pharmacology results (points and matrix cells of this run only) and its
    normalized readouts by well, when ``pharmacology/`` exists."""
    import pandas as pd

    pharm = folder / "pharmacology"
    data_file = pharm / "report_data.json"
    normalized: dict[str, dict[str, Any]] = {}
    table = pharm / "normalized_well_data.csv"
    if table.exists():
        frame = pd.read_csv(table)
        frame = frame[frame["run"] == folder.name] if "run" in frame else frame
        for column in frame.columns:
            if "__" in column and not column.endswith(("__excluded", "__exclusion_reason")):
                readout, kind = column.split("__", 1)
                name = f"pharmacology · {readout} · {kind.replace('_', ' ')}"
                normalized[name] = {
                    str(w): _round(v)
                    for w, v in zip(frame["well"], frame[column], strict=True)
                    if isinstance(v, (int, float))
                }
    if not data_file.exists():
        return None, normalized
    data = json.loads(data_file.read_text(encoding="utf-8"))
    mine = folder.name
    for block in data.values():
        for curve in block.get("curves", []):
            curve["points"] = [p for p in curve["points"] if p.get("run") in (mine, "")]
        for combo in block.get("combinations", []):
            combo["wells"] = {
                cell: [w[2] for w in wells if w[0] in (mine, "")]
                for cell, wells in combo["wells"].items()
            }
    return data, normalized


def report_links(folder: Path) -> list[dict[str, str]]:
    """Reports and figures of the run, as relative links."""
    links = []
    for path, title in (
        ("pharmacology/pharmacology_report.html", "Pharmacology report (HTML)"),
        ("pharmacology/pharmacology_report.pdf", "Pharmacology report (PDF)"),
        ("experiment_report/experiment_report.html", "Experiment report (HTML)"),
        ("experiment_report/experiment_report.pdf", "Experiment report (PDF)"),
    ):
        if (folder / path).exists():
            links.append({"title": title, "href": path})
    for sub, kind in (("figures", "Figure"), ("experiment_report/montages", "Montage")):
        if (folder / sub).exists():
            for f in sorted((folder / sub).iterdir()):
                if f.suffix.lower() in (".png", ".svg", ".pdf", ".tif", ".tiff"):
                    links.append({"title": f"{kind}: {f.stem}", "href": f"{sub}/{f.name}"})
    return links


def analysis_lines(folder: Path) -> list[tuple[str, str]]:
    """(title, description) of each table step applied to the run, for the report."""
    try:
        record = json.loads((folder / "tables" / "analysis_summary.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    lines = []
    for step in record.get("steps", []):
        params, state = step.get("params", {}), step.get("state", {})
        parts = [f"on {params.get('object_type', '')}"]
        if step["module"] == "gate_objects":
            parts += [
                f"{r['feature']} in [{_text(r.get('min'))}, {_text(r.get('max'))}]"
                for r in params.get("rules", [])
            ]
        elif step["module"] == "classify_objects":
            parts.append(f"classes {', '.join(state.get('classes', []))}")
            parts.append(f"{state.get('algorithm', '')}, {state.get('features', '?')} features")
            accuracy = state.get("cv_accuracy")
            if accuracy is not None:
                parts.append(f"{state.get('cv_folds')}-fold CV accuracy {accuracy:.1%}")
            parts.append(f"model SHA-256 {str(state.get('model_sha256', ''))[:12]}…")
        elif step["module"] == "phenotypic_clusters":
            sizes = state.get("cluster_sizes_in_sample", {})
            parts.append(
                f"{params.get('method')} → {len([k for k in sizes if int(k) > 0])} clusters"
            )
        elif step["module"] == "spatial_clusters":
            parts.append(f"{params.get('scope')} scope, colonies by {params.get('method')}")
            parts.append(f"Ripley radii {params.get('radii_um')} µm")
        lines.append((f"{step['step']}. {step['title']}", "; ".join(p for p in parts if p)))
    return lines


def _text(value: Any) -> str:
    return "–" if value is None else f"{value:.4g}"


def write_report(folder: Path) -> Path:
    """Write (or rewrite) ``index.html`` for a run folder; returns its path."""
    from jinja2 import Environment, PackageLoader, select_autoescape

    results = RunResults(folder)
    data = report_data(results)
    manifest = results.manifest
    pipeline = manifest.get("pipeline", {}).get("name", "pipeline")
    started = manifest.get("started", "")
    with contextlib.suppress(ValueError):
        started = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    vendor = manifest["plate"].get("vendor", "")
    environment = Environment(
        loader=PackageLoader("hcs_studio.output", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    from markupsafe import Markup

    # Embedded JSON must not end the <script> element early; it is then safe to insert as is.
    data_json = Markup(json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = environment.get_template("report.html.j2").render(
        title=f"{results.plate_name} results",
        plate_name=results.plate_name,
        subtitle=f"{vendor} · pipeline {pipeline} · {manifest.get('status', '')} · "
        f"started {started} · {len(results.wells)} wells",
        footer=f"HCS Studio {__version__} · run folder {folder.name} · "
        "colours: viridis for values; blue–white–red for robust Z, symmetric around 0 "
        "(at least ±3, at most ±6)",
        data_json=data_json,
        analysis=analysis_lines(folder),
    )
    target = folder / "index.html"
    part = target.with_name("index.html.part")
    part.write_text(html, encoding="utf-8")
    part.replace(target)
    return target
