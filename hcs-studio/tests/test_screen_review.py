"""Phase 23b: the screen report, the hit gallery (linked contrast) and the Screen tab."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.screening.screen import ScreenConfig, run_screen

CONFIG = {
    "readouts": [
        {"column": "reporter_mean", "alias": "rep"},
        {"column": "nuclei_count", "alias": "count", "activity": "percent_of_control"},
    ],
    "hit_rule": "rep_score <= -3 and count_pct > 60",
}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


@pytest.fixture(scope="module")
def screened(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    from hcs_studio.synthetic.screening import write_screen

    root = tmp_path_factory.mktemp("screen")
    folders, _ = write_screen(root / "runs", libraries=2)
    result = run_screen(folders, ScreenConfig.model_validate(CONFIG), root / "screen")
    return root / "screen", result


def _payload(page: Path) -> dict[str, Any]:
    text = page.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text.split('id="data">')[1].split("</script>")[0])
    return data


def test_report_data(screened: tuple[Path, Any]) -> None:
    out, result = screened
    data = _payload(out / "report.html")
    assert data["summary"]["hits"] == len(result.hits) and data["summary"]["plates"] == 4
    assert data["thresholds"] == {"rep_score": [["<=", -3.0]], "count_pct": [[">", 60.0]]}
    assert data["default"] == "rep_score" and "rep_score" in data["columns"]
    assert len(data["wells"]["role"]) == 4 * 384  # samples and both controls
    pairs = data["replicates"]["values"]["rep_score"]
    assert len(pairs) == len(data["replicates"]["names"]) > 500  # compounds with 2 clean replicates
    hit_names = {
        n for n, h in zip(data["replicates"]["names"], data["replicates"]["hit"], strict=True) if h
    }
    assert hit_names == set(result.hits["compound"])
    assert data["hits"][0]["rank"] == 1 and "override_reason" in data["hit_columns"]
    assert "http" not in (out / "report.html").read_text().replace("http-equiv", "")


BROWSER = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1300, height: 900 } });
  await page.route('http*://**', (route) => route.abort());
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(process.argv[2]);
  const out = {};
  out.maps = await page.$$eval('#maps canvas', (e) => e.map((c) => +c.dataset.wells));
  out.order = +(await page.$eval('#order', (c) => c.dataset.points));
  out.hist = +(await page.$eval('#hist', (c) => c.dataset.wells));
  out.reps = +(await page.$eval('#reps', (c) => c.dataset.points));
  await page.selectOption('#score', 'count_pct');
  out.histPct = +(await page.$eval('#hist', (c) => c.dataset.wells));
  const box = await page.$eval('#maps canvas',
    (c) => { const r = c.getBoundingClientRect(); return [r.x, r.y]; });
  await page.mouse.move(box[0] + 30, box[1] + 30);
  out.hover = await page.$eval('#hover', (h) => h.textContent);
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
def test_report_works_offline_in_a_browser(screened: tuple[Path, Any], tmp_path: Path) -> None:
    out, result = screened
    script = tmp_path / "check.js"
    script.write_text(BROWSER)
    done = subprocess.run(
        ["node", str(script), (out / "report.html").as_uri()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["errors"] == []
    assert seen["maps"] == [384] * 4
    samples = int((result.wells["role"] == "sample").sum())
    assert seen["order"] == seen["hist"] == samples
    assert seen["reps"] > 500 and seen["histPct"] == samples
    assert seen["hover"].startswith(("SCR01-R1", "SCR01-R2", "SCR02-R1", "SCR02-R2"))


# ----------------------------------------------------------------- gallery


def _count(counts: dict[tuple[int, int], int], row: int, col: int, field: int) -> int:
    return counts[(row, col)]


def _image_screen(root: Path) -> list[Path]:
    """Two replicate 96-well plates with images: vehicle in column 1, 22 samples; compound in
    row C column 4 ('HIT') has 8 nuclei per field instead of 36."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    wells = [(r, c) for r in range(4) for c in range(6)]
    layout = PlateLayout.for_plate_type(96)
    layout = set_role(layout, [(r, 0) for r in range(4)], "negative_control")
    for r, c in wells:
        if c > 0:
            name = "HIT" if (r, c) == (2, 3) else f"C{r}{c}"
            layout = apply_fields(layout, [(r, c)], treatments=(Treatment(name, 1e-5, "µM"),))
    steps = [
        {"module": "segment_classical", "params": {"channel": 1, "method": "otsu"}},
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
    ]
    folders = []
    for rep in range(2):
        rng = np.random.default_rng(rep)
        counts = {k: 8 if k == (2, 3) else int(rng.integers(32, 40)) for k in wells}
        plate = SyntheticPlate(
            wells=wells,
            features=[Nuclei(count=partial(_count, counts)), ShotNoise()],
            channels=("DAPI",),
            size=160,
            plate_size=(8, 12),
            name=f"IMG-R{rep + 1}",
            seed=rep,
        )
        path, _ = plate.write(root / f"raw{rep}")
        layout_rep = replace(layout, barcode=f"IMG-R{rep + 1}")
        folders.append(
            run_pipeline(
                open_plate(path),
                Pipeline.model_validate({"name": "s", "steps": steps}),
                output_root=root / f"runs{rep}",
                layout=layout_rep,
            ).folder
        )
    return folders


def test_hit_gallery_has_linked_contrast(home: Path) -> None:
    from PIL import Image

    folders = _image_screen(home)
    config = ScreenConfig.model_validate(
        {
            "readouts": [
                {"column": "nuclei_count", "alias": "count", "activity": "percent_of_control"}
            ],
            "score": "z",
            "hit_rule": "count_z <= -3",
            "artifact_filters": [],
        }
    )
    result = run_screen(folders, config, home / "screen", gallery=10)
    assert list(result.hits["compound"]) == ["HIT"]
    page = (home / "screen" / "gallery" / "index.html").read_text()
    assert 'id="HIT"' in page and page.count('<div class="strip">') == 2  # both replicates
    for plate in ("IMG-R1", "IMG-R2"):
        hit = np.asarray(Image.open(home / "screen" / "gallery" / f"{plate}_C04.png")).mean()
        vehicles = [
            np.asarray(Image.open(home / "screen" / "gallery" / f"{plate}_{w}.png")).mean()
            for w in ("C01", "B01")
        ]  # the nearest vehicle wells
        assert hit < 0.5 * min(vehicles)  # same contrast: 8 nuclei look emptier than 36
    report = (home / "screen" / "report.html").read_text()
    assert 'href="gallery/index.html#HIT"' in report


def test_gallery_without_images_says_so(screened: tuple[Path, Any], tmp_path: Path) -> None:
    from hcs_studio.screening.report import write_gallery

    _, result = screened
    page = write_gallery(result, tmp_path, max_hits=2)
    assert page is not None and "no image" in page.read_text()


def test_screen_report_command(screened: tuple[Path, Any], home: Path) -> None:
    from hcs_studio.app import main

    out, _ = screened
    copy = home / "copy"
    shutil.copytree(out, copy)
    (copy / "report.html").unlink()
    assert main(["screen", "report", str(copy)]) == 0
    assert (copy / "report.html").exists()


# --------------------------------------------------------------------- GUI


def test_screen_tab_reviews_hits(screened: tuple[Path, Any], home: Path, qtbot: Any) -> None:
    from hcs_studio.gui.screen_panel import ScreenPanel

    out, _ = screened
    copy = home / "screen"
    shutil.copytree(out, copy)
    panel = ScreenPanel()
    qtbot.addWidget(panel)
    panel.open(copy)
    qtbot.waitUntil(lambda: panel.hit_table.rowCount() > 0, timeout=20000)
    hits = pd.read_csv(copy / "hits.csv")
    assert panel.hit_table.rowCount() == len(hits)
    assert panel.plate_combo.count() == 4 and panel.score_combo.currentText() == "rep_score"
    assert panel.map._fills
    panel.hit_table.selectRow(0)
    assert panel.compound.text() == hits["compound"].iloc[0]
    panel.review("exclude")  # no reason: refused
    assert "reason" in panel.status.text()
    panel.reason.setText("autofluorescent in the source plate")
    panel.review("exclude")
    qtbot.waitUntil(lambda: panel.hit_table.rowCount() == len(hits) - 1, timeout=30000)
    assert json.loads((copy / "overrides.json").read_text())[0]["action"] == "exclude"
