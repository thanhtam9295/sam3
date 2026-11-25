"""Regression suite (Phase 15): a synthetic 384-well plate with images, known
dose–response and a known-additive and a known-synergistic combination, run end to end —
segmentation, measurement, QC, pharmacology, a figure per compound, the experiment report
and the clickable index.html.

Truth: ``pharmacology/synthetic.py``. Baseline (results of this version, to catch
unintended changes): ``tests/data/regression/baseline.json``; regenerate with
``HCS_UPDATE_BASELINE=1 pytest tests/test_regression.py``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from hcs_studio.io.base import open_plate
from hcs_studio.model.layout import load_plate_layout
from hcs_studio.model.plate import well_name
from hcs_studio.pharmacology.synthetic import TRUTH, layout, nuclei_count, write_image_plate
from hcs_studio.pipeline.executor import pipeline_hash, run_pipeline
from hcs_studio.pipeline.schema import Pipeline

BASELINE = Path(__file__).parent / "data" / "regression" / "baseline.json"
PIPELINE = """
name: regression
steps:
  - {module: segment_classical, params: {channel: 1}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
  - {module: pharmacology, params: {readouts: [nuclei_count]}}
  - {module: compound_figures, params: {dpi: 100}}
  - {module: experiment_report, params: {montages: true}}
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("regression")
    plate_path, layout_path = write_image_plate(root)
    result = run_pipeline(
        open_plate(plate_path),
        Pipeline.from_text(PIPELINE),
        output_root=root / "out",
        layout=load_plate_layout(layout_path),
        layout_path=layout_path,
        workers=2,
        checksums=False,
    )
    assert not result.failed_wells, result.failed_wells
    return result.folder


def test_counts_match_the_truth_in_every_well(run: Path) -> None:
    table = pd.read_csv(run / "tables" / "per_well_summary.csv")
    counts = dict(zip(table["well"], table["nuclei_count"].astype(int), strict=True))
    plate = layout()
    wrong = []
    for key, a in plate.wells.items():
        drugs = [(t.compound, t.concentration) for t in a.active_treatments if t.kind != "percent"]
        expected = nuclei_count(a.cell_line, a.role, drugs if a.role == "sample" else [])
        got = counts[well_name(*key)]
        if got != expected:
            wrong.append((well_name(*key), got, expected))
    assert not wrong, wrong[:10]
    assert len(counts) == len(plate.wells)


def test_run_steps_and_outputs(run: Path) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    steps = {s["module"]: s for s in manifest["run_steps"]}
    assert [s["status"] for s in manifest["run_steps"]] == ["completed"] * 3, steps
    assert manifest["status"] == "completed"
    # run steps are not part of what the images are: the pipeline hash ignores them
    images_only = Pipeline.from_text(PIPELINE.split("  - {module: pharmacology")[0])
    plate = open_plate(Path(manifest["plate"]["path"]))
    assert pipeline_hash(plate, Pipeline.from_text(PIPELINE), {}) == pipeline_hash(
        plate, images_only, {}
    )
    for name in (
        "pharmacology/pharmacology_report.html",
        "pharmacology/pharmacology_report.pdf",
        "pharmacology/report_data.json",
        "experiment_report/experiment_report.html",
        "experiment_report/experiment_report.pdf",
        "figures/Drug-D.png",
        "experiment_report/montages/Drug-A.png",
    ):
        assert (run / name).exists(), name
    report = (run / "experiment_report" / "experiment_report.html").read_text(encoding="utf-8")
    for text in (
        "Plate layout",
        "Quality control",
        "Dose–response",
        "Combinations",
        "Representative images",
        "Field QC flags",
        "out of focus",
        "Methods",
        "Segmentation: threshold + watershed",
        "percent inhibition",
        "HCS Studio",
    ):
        assert text in report, text
    assert (run / "experiment_report" / "experiment_report.pdf").read_bytes()[:4] == b"%PDF"


def _results(run: Path) -> dict[str, float]:
    fits = pd.read_csv(run / "pharmacology" / "dose_response_fits.csv")
    scores = pd.read_csv(run / "pharmacology" / "combination_scores.csv")
    out: dict[str, float] = {}
    for line, compound, ec50, hill in zip(
        fits["cell_line"], fits["compound"], fits["ec50"], fits["hill"], strict=True
    ):
        out[f"ec50 {line} {compound}"] = float(ec50)
        out[f"hill {line} {compound}"] = float(hill)
    for row in scores.itertuples():
        for model in ("bliss", "hsa", "loewe", "zip"):
            out[f"{model} {row.drug_a} x {row.drug_b}"] = float(getattr(row, f"{model}_mean"))
    out["z_prime"] = float(pd.read_csv(run / "pharmacology" / "plate_qc.csv")["z_prime"].iloc[0])
    return out


def test_dose_response_and_synergy_against_the_truth(run: Path) -> None:
    values = _results(run)
    for (line, drug), (ic50, hill) in TRUTH.items():
        assert values[f"ec50 {line} {drug}"] == pytest.approx(ic50, rel=0.10), (line, drug)
        assert values[f"hill {line} {drug}"] == pytest.approx(hill, abs=0.25), (line, drug)
    scores = pd.read_csv(run / "pharmacology" / "combination_scores.csv").set_index("drug_b")
    synergistic, additive = scores.loc["Drug B"], scores.loc["Drug C"]
    for model in ("bliss", "loewe", "zip"):
        assert synergistic[f"{model}_mean"] > 10, model
        assert synergistic[f"{model}_label"] == "likely synergistic", model
        assert synergistic[f"{model}_p"] < 0.05, model
    for model in ("bliss", "zip"):  # built Bliss-additive (ZIP agrees)
        assert abs(additive[f"{model}_mean"]) < 3, model
        assert additive[f"{model}_label"] == "likely additive", model
    assert values["z_prime"] > 0.5
    comparisons = pd.read_csv(run / "pharmacology" / "ic50_comparisons.csv")
    d = comparisons[comparisons["compound"] == "Drug D"].iloc[0]
    assert abs(d["delta_log_ec50"]) == pytest.approx(1.0, abs=0.1) and d["p"] < 1e-6


def test_matches_the_baseline(run: Path) -> None:
    values = _results(run)
    if os.environ.get("HCS_UPDATE_BASELINE") or not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(values, indent=1, sort_keys=True), encoding="utf-8")
        if not os.environ.get("HCS_UPDATE_BASELINE"):
            pytest.skip("baseline written for the first time")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert set(values) == set(baseline)
    for key, value in baseline.items():
        if math.isfinite(value):
            assert values[key] == pytest.approx(value, rel=0.02, abs=0.05), key


BROWSER = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
  await page.route('http*://**', (route) => route.abort());
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(process.argv[2]);
  const out = {};
  out.pharmVisible = await page.$eval('#pharm', (e) => !e.hidden);
  out.items = await page.$$eval('#pharm-item option', (els) => els.map((o) => o.textContent));
  const curve = out.items.findIndex((t) => t === 'HeLa · Drug D');
  await page.selectOption('#pharm-item', { index: curve });
  out.points = await page.$$eval('#curve .pt', (els) => els.length);
  const well = await page.$eval('#curve .pt', (e) => e.dataset.well);
  await page.$eval('#curve .pt',
    (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  out.pointWell = well;
  out.selectedAfterPoint = await page.textContent('#well h2');
  const combo = out.items.findIndex((t) => t === 'MCF7 · Drug A × Drug B');
  await page.selectOption('#pharm-item', { index: combo });
  out.cells = await page.$$eval('#matrix-synergy td', (els) => els.length);
  out.curveShownWithMatrix = await page.$eval('#curve',
    (e) => getComputedStyle(e).display !== 'none');
  out.note = await page.textContent('#pharm-note');
  out.cellTitle = await page.$eval('#matrix-synergy td[data-cell="3,3"]', (e) => e.title);
  await page.click('#matrix-synergy td[data-cell="3,3"]');
  out.selectedAfterCell = await page.textContent('#well h2');
  out.picked = await page.$$eval('.well.picked', (els) => els.map((e) => e.dataset.well));
  await page.selectOption('#colour', 'drug');
  out.legend = await page.$$eval('#plate-legend span', (els) => els.map((e) => e.textContent));
  out.readouts = await page.$$eval('#readout option', (els) => els.map((o) => o.value));
  out.links = await page.$$eval('#links a', (els) => els.map((a) => a.getAttribute('href')));
  out.errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})();
"""


def _node_playwright() -> bool:
    if shutil.which("node") is None:
        return False
    return (
        subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _node_playwright(), reason="Node.js with Playwright is not available")
def test_index_html_pharmacology_offline(run: Path, tmp_path: Path) -> None:
    """index.html: plate coloured by annotation, normalized readouts, curves and heat maps
    whose points and cells open their wells, links to reports and montages — offline."""
    script = tmp_path / "check.js"
    script.write_text(BROWSER, encoding="utf-8")
    result = subprocess.run(
        ["node", str(script), (run / "index.html").as_uri()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [] and out["pharmVisible"]
    assert "HeLa · Drug D" in out["items"] and "MCF7 · Drug A × Drug B" in out["items"]
    assert out["points"] == 20  # 10 concentrations in duplicate
    assert out["selectedAfterPoint"] == f"Well {out['pointWell']}"
    assert out["cells"] == 36 and not out["curveShownWithMatrix"]
    assert "p < 0.001" in out["note"]
    wells = out["cellTitle"].split("wells ")[1].split(", ")
    assert len(wells) == 3 and sorted(out["picked"]) == sorted(wells)
    assert out["selectedAfterCell"] == f"Well {sorted(wells)[0]}"
    assert {"Drug A", "Drug B", "Drug A + Drug B", "negative control", "positive control"} <= set(
        out["legend"]
    )
    assert any(r.startswith("pharmacology · nuclei_count") for r in out["readouts"])
    assert "experiment_report/experiment_report.html" in out["links"]
    assert any(h.startswith("experiment_report/montages/") for h in out["links"])
