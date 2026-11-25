"""Results model and the standalone HTML report (rendered in a real browser when available)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcs_studio.app import main
from hcs_studio.io.base import open_plate
from hcs_studio.model.layout import read_plate_layout
from hcs_studio.output.results import RunResults, robust_z, z_prime
from hcs_studio.pipeline.executor import run_pipeline
from hcs_studio.pipeline.schema import Pipeline
from tests.builders import make_imagexpress
from tests.test_segmentation import CENTRES, nuclei_image

WELLS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
COUNTS = {"A01": 1, "A02": 2, "A03": 5, "B01": 4, "B02": 3, "B03": 3}  # nuclei per field
LAYOUT = """well,compound,concentration,treatment,control_type
A01,DMSO,,vehicle,negative
A02,DMSO,,vehicle,negative
A03,Staurosporine,1,24h,positive
B01,Staurosporine,1,24h,positive
B02,Drug X,0.5,24h,
B03,Drug X,5,24h,
"""
PIPELINE = """
name: report
steps:
  - {module: segment_classical, params: {channel: 1}}
  - {module: expand_cells, params: {distance_um: 2}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
"""


def _image(row: int, col: int, site: int, wave: int) -> Any:
    from hcs_studio.model.plate import well_name

    centres = CENTRES[: COUNTS[well_name(row, col)]]
    data = nuclei_image(centres, seed=site).data[wave, 0]
    if (row, col) == (0, 2):
        data = data * 2  # brighter nuclei in A03 (for % positive)
    return np.clip(data, 0, None).astype(np.uint16)


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("results")
    root = make_imagexpress(tmp / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)
    layout = tmp / "layout.csv"
    layout.write_text(LAYOUT, encoding="utf-8")
    result = run_pipeline(
        open_plate(root),
        Pipeline.from_text(PIPELINE),
        output_root=tmp / "out",
        layout=read_plate_layout(layout),
        layout_path=layout,
        checksums=False,
    )
    return result.folder


def test_robust_z_and_z_prime() -> None:
    values = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 100.0}
    z = robust_z(values)
    assert (
        z["c"] == 0 and z["d"] == pytest.approx(1 / 1.4826) and z["e"] > 60
    )  # outlier stays visible
    assert robust_z(values, ["a", "b"])["a"] == pytest.approx(-0.5 / (1.4826 * 0.5))
    assert np.isnan(robust_z({"a": 1.0, "b": 1.0})["a"])  # MAD 0
    positive, negative = [10.0, 12.0, 11.0, 9.0], [1.0, 2.0, 1.5, 0.5]
    expected = 1 - 3 * (np.std(positive, ddof=1) + np.std(negative, ddof=1)) / (10.5 - 1.25)
    assert z_prime(positive, negative) == pytest.approx(expected)
    assert z_prime([10], [1, 2]) is None and z_prime([1, 1], [1, 1]) is None


def test_run_results(run_folder: Path) -> None:
    results = RunResults(run_folder)
    assert (results.rows, results.cols, results.plate_name) == (8, 12, "Plate1")
    assert results.wells == ["A01", "A02", "A03", "B01", "B02", "B03"]
    readouts = results.readout_columns()
    assert readouts[0].endswith("_count") and "nuclei_per_mm2" in readouts
    counts = results.values("nuclei_count")
    assert counts == {w: 2.0 * n for w, n in COUNTS.items()}
    assert results.control_wells("negative") == ["A01", "A02"]
    zp = results.z_prime(counts)
    assert zp == pytest.approx(1 - 3 * (np.std([2, 4], ddof=1) + np.std([10, 8], ddof=1)) / 6)
    negative_z = results.normalise(counts, "robust_z_negative")
    assert negative_z["A01"] == pytest.approx(-1 / 1.4826) and negative_z["A03"] == pytest.approx(
        (10 - 3) / 1.4826
    )
    with pytest.raises(KeyError):
        results.values("nope")

    features = results.object_features()
    assert "dapi_mean" in features["nuclei"]
    bright = results.percent_positive("nuclei", "dapi_mean", 1500)
    assert bright["A03"] == 100 and bright["A01"] == 0

    fields = results.well_fields("B02")
    assert [f.index for f in fields] == [1, 2] and fields[0].objects == {"nuclei": 3, "cells": 3}
    for path in (
        fields[0].enhanced,
        fields[0].raw,
        fields[0].corrected,
        *fields[0].overlays.values(),
    ):
        assert path is not None and (run_folder / path).exists()
    assert results.well_flags("B02") == []
    with pytest.raises(ValueError, match="not an HCS Studio run folder"):
        RunResults(run_folder.parent / "missing")


def _report_data(folder: Path) -> dict[str, Any]:
    html = (folder / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    return dict(json.loads(match.group(1)))


def test_report_is_self_contained(run_folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    html = (run_folder / "index.html").read_text(encoding="utf-8")
    assert (
        "<link" not in html and 'src="http' not in html and "https://" not in html
    )  # nothing external
    data = _report_data(run_folder)
    assert data["plate"] == {"name": "Plate1", "rows": 8, "cols": 12}
    assert (
        data["values"]["nuclei_count"]["A03"] == 10
        and data["layout"]["A01"]["control_type"] == "negative"
    )
    images = [
        p
        for w in data["wells"].values()
        for f in w["fields"]
        for p in [f["enhanced"], f["raw"], f["corrected"], *f["overlays"].values()]
    ]
    assert len(images) == 6 * 2 * 5 and all(
        (run_folder / p).exists() and not Path(p).is_absolute() for p in images
    )
    (run_folder / "index.html").unlink()
    assert main(["report", str(run_folder)]) == 0 and (run_folder / "index.html").exists()
    assert "Report:" in capsys.readouterr().out
    assert main(["report", str(run_folder / "missing")]) == 2


BROWSER_SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.route('http*://**', (route) => route.abort());  // offline: no network at all
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(process.argv[2]);
  const out = {};
  out.wells = await page.$$eval('.well.data', (els) => els.length);
  out.colourA01 = await page.$eval('#w-A01', (e) => e.style.background);
  out.colourA03 = await page.$eval('#w-A03', (e) => e.style.background);
  out.zprime = await page.textContent('#zprime');
  await page.click('#w-B02');
  await page.waitForSelector('.thumb img');
  await page.waitForFunction(() =>
    [...document.querySelectorAll('.thumb img, #field-image, .overlay')].every((i) => i.complete));
  out.thumbs = await page.$$eval('.thumb img', (els) => els.map((i) => i.naturalWidth));
  out.overlays = await page.$$eval('.stack img.overlay',
    (els) => els.map((i) => [i.dataset.layer, i.naturalWidth]));
  out.mainSrc = await page.$eval('#field-image', (i) => [i.getAttribute('src'), i.naturalWidth]);
  out.stats = await page.$$eval('table.stats tr',
    (rows) => rows.slice(0, 2).map((r) => r.textContent));
  await page.click('.stage button:has-text("Raw")');
  out.rawSrc = await page.$eval('#field-image', (i) => i.getAttribute('src'));
  await page.uncheck('.viewer label:has-text("nuclei") input');
  out.overlaysAfterToggle = await page.$$eval('.stack img.overlay',
    (els) => els.map((i) => i.dataset.layer));
  await page.click('.thumb:nth-child(2)');
  out.secondField = await page.$eval('#field-image', (i) => i.getAttribute('src'));
  await page.selectOption('#norm', 'negative');
  out.titleA03 = await page.getAttribute('#w-A03', 'title');
  await page.fill('#filter', 'per_mm2');
  out.readout = await page.$eval('#readout', (s) => s.value);
  await page.focus('#w-B02');
  await page.keyboard.press('ArrowRight');
  out.afterArrow = await page.evaluate(() => location.hash);
  out.errors = errors;
  await page.goto(process.argv[2] + '#A03');
  out.deepLink = await page.$eval('#well h2', (h) => h.textContent);
  console.log(JSON.stringify(out));
  await browser.close();
})();
"""


def _node_playwright() -> bool:
    if shutil.which("node") is None:
        return False
    check = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True)
    return check.returncode == 0


@pytest.mark.skipif(not _node_playwright(), reason="Node.js with Playwright is not available")
def test_report_works_offline_in_a_browser(run_folder: Path, tmp_path: Path) -> None:
    """Acceptance: clicking a well in index.html opens its enhanced previews offline."""
    zipped = shutil.make_archive(str(tmp_path / "shared"), "zip", run_folder)
    unpacked = tmp_path / "unzipped"
    shutil.unpack_archive(zipped, unpacked)  # as a colleague would receive it
    script = tmp_path / "check.js"
    script.write_text(BROWSER_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        ["node", str(script), (unpacked / "index.html").as_uri()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [] and out["wells"] == 6
    assert out["colourA01"] != out["colourA03"]  # coloured by the count readout
    assert out["zprime"].startswith("Z′ = ")
    assert len(out["thumbs"]) == 2 and all(w > 0 for w in out["thumbs"])  # previews loaded
    assert out["mainSrc"][0] == "wells/B02/previews/f001.png" and out["mainSrc"][1] > 0
    assert sorted(layer for layer, _ in out["overlays"]) == ["cells", "nuclei"] and all(
        w > 0 for _, w in out["overlays"]
    )
    assert out["stats"][0] == "cells_count6"
    assert out["rawSrc"] == "wells/B02/previews/f001_raw.png"
    assert out["overlaysAfterToggle"] == ["cells"]
    assert out["secondField"] == "wells/B02/previews/f002_raw.png"
    expected_z = RunResults(run_folder).normalise(
        RunResults(run_folder).values("nuclei_count"), "robust_z_negative"
    )["A03"]
    shown = float(re.search(r"Z ([-\d.]+)", out["titleA03"]).group(1))  # type: ignore[union-attr]
    assert shown == pytest.approx(expected_z, abs=0.01)  # browser and Python agree
    assert out["readout"] == "cells_per_mm2" and out["afterArrow"] == "#B03"
    assert out["deepLink"] == "Well A03"
