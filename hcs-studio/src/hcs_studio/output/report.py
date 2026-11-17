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
        wells[well] = {"fields": fields, "flags": results.well_flags(well)}
    layout = {
        w: {
            "compound": a.compound,
            "concentration": a.concentration,
            "treatment": a.treatment,
            "control_type": a.control_type,
        }
        for w, a in results.layout.items()
    }
    key_readouts = [
        r
        for r in readouts
        if any(m in r for m in _KEY_MATCHES) and not r.endswith(("_std", "_mean"))
    ][:14]
    return {
        "plate": {"name": results.plate_name, "rows": results.rows, "cols": results.cols},
        "readouts": readouts,
        "key_readouts": key_readouts,
        "values": values,
        "wells": wells,
        "layout": layout,
    }


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
    )
    target = folder / "index.html"
    part = target.with_name("index.html.part")
    part.write_text(html, encoding="utf-8")
    part.replace(target)
    return target
