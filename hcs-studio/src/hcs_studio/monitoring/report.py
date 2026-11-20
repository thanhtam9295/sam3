"""Files of an instrument bridging report: HTML (self-contained, figure embedded), CSV tables
and a JSON summary."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hcs_studio import __version__
from hcs_studio.monitoring.bridging import Bridging


def _figure(bridging: Bridging) -> str:
    """Potency ratios A/B with 95 % CIs against the MSR band, as an embedded PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ratios = bridging.ratios
    fig = Figure(figsize=(7, max(2.5, 0.3 * len(ratios) + 1)), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if not ratios.empty:
        y = range(len(ratios))
        low = (ratios["ratio"] - ratios["ratio_ci_low"]).fillna(0)
        high = (ratios["ratio_ci_high"] - ratios["ratio"]).fillna(0)
        colours = ["#1b7f3b" if ok else "#b00020" for ok in ratios["within_msr"]]
        ax.errorbar(ratios["ratio"], list(y), xerr=[low, high], fmt="none", ecolor="#555")
        ax.scatter(ratios["ratio"], list(y), c=colours, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(
            [
                f"{c} ({line})"
                for c, line in zip(ratios["compound"], ratios["cell_line"], strict=True)
            ],
            fontsize=7,
        )
    ax.axvspan(1 / bridging.msr, bridging.msr, color="#1b7f3b", alpha=0.08)
    ax.axvline(1, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(
        f"EC50 ratio {bridging.instrument_a} / {bridging.instrument_b} (95 % CI); band = 1/MSR … MSR"
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>(none)</p>"

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.4g}"
        return html.escape(str(value))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def summary(bridging: Bridging) -> dict[str, Any]:
    retest = bridging.retest
    return {
        "instrument_a": bridging.instrument_a,
        "instrument_b": bridging.instrument_b,
        "readout": bridging.readout,
        "msr": bridging.msr,
        "plates": len(bridging.agreement),
        "potency_pairs": len(bridging.ratios),
        "equivalent": bridging.equivalent,
        "mean_ratio": retest.mr if retest else None,
        "limits_of_agreement": [retest.lsa_low, retest.lsa_high] if retest else None,
        "retest_msr": retest.msr if retest else None,
        "notes": bridging.notes,
        "software": f"hcs-studio {__version__}",
        "written": datetime.now(UTC).isoformat(),
    }


def write_bridging_report(bridging: Bridging, folder: Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = summary(bridging)
    files = [folder / "bridging.json", folder / "agreement.csv", folder / "potency_ratios.csv"]
    files[0].write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    bridging.agreement.to_csv(files[1], index=False)
    bridging.ratios.to_csv(files[2], index=False)
    verdict = (
        "<p class=ok>Equivalent: every EC50 ratio and the limits of agreement lie within the MSR.</p>"
        if bridging.equivalent
        else "<p class=bad>Not shown to be equivalent (see the ratios outside the MSR band).</p>"
    )
    retest = bridging.retest
    stats = (
        f"<p>Across {retest.n} compound/cell-line pairs: mean ratio {retest.mr:.3g}, limits of "
        f"agreement {retest.lsa_low:.3g} – {retest.lsa_high:.3g}, MSR {retest.msr:.3g}.</p>"
        if retest
        else ""
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in bridging.notes)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Instrument bridging</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}
table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:right}}
th{{background:#f3f3f3}}.ok{{color:#1b7f3b}}.bad{{color:#b00020}}</style></head><body>
<h1>Instrument bridging: {html.escape(bridging.instrument_a)} vs {html.escape(bridging.instrument_b)}</h1>
<p>Readout {html.escape(bridging.readout)} · MSR used {bridging.msr:g} · {data["written"]} · {data["software"]}</p>
{verdict}{stats}<ul>{notes}</ul>
<h2>Potency ratios</h2><img alt="EC50 ratios" src="data:image/png;base64,{_figure(bridging)}">
{_table(bridging.ratios)}
<h2>Per-well agreement of the plates imaged on both</h2>
<p>Bland–Altman bias and 95 % limits in the readout's unit (A − B).</p>
{_table(bridging.agreement)}
<p>MSR method: Eastwood et al. 2006 (J Biomol Screen 11:253); not yet checked against the NCATS
Assay Guidance Manual text.</p>
</body></html>"""
    report = folder / "bridging_report.html"
    report.write_text(page, encoding="utf-8")
    return [report, *files]
