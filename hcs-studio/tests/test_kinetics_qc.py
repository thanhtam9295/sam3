"""Phase 22c: kinetic QC (time points, focus drift, edge evaporation, control growth) and the
kinetics report page."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcs_studio.kinetics.analysis import KineticsParams, analyse, write_kinetics
from hcs_studio.kinetics.qc import (
    FAIL,
    NA,
    PASS,
    WARN,
    KineticQc,
    KineticQcOptions,
    check_controls,
    control_history,
)
from hcs_studio.synthetic.kinetics import response_course

TIMES = [0.0, 6, 12, 18, 24, 30, 36, 42, 48]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _plate() -> tuple[dict[str, tuple[str, float]], object]:
    from tests.test_kinetics_analysis import _plate as plate

    return plate()


def _course(times: list[float] = TIMES) -> tuple[pd.DataFrame, object]:
    wells, layout = _plate()
    course = response_course(wells, times, n0=3000, doubling_h=24, emax=1.5, ec50=1e-7)
    course["focus_score"] = 1.0 + np.random.default_rng(3).normal(0, 0.02, len(course))
    return course, layout


def _statuses(result: object) -> dict[str, str]:
    return {c["check"]: c["status"] for c in result.qc.checks}  # type: ignore[attr-defined]


def test_clean_plate_passes_every_check() -> None:
    course, layout = _course()
    result = analyse(course, layout, KineticsParams(dose_response=False), (8, 12))
    assert _statuses(result) == {
        "time points": PASS,
        "focus drift": PASS,
        "edge evaporation": PASS,
        "control growth": PASS,
    }
    assert result.qc.status == PASS and not result.qc.well_flags
    assert "history: 0 earlier run(s)" in result.qc.checks[-1]["message"]


def test_planted_problems_are_found() -> None:
    """A skipped acquisition, a well missing a time point, an out-of-focus time point, focus
    drifting at the end, edge wells (row A) growing 30 % less by 48 h, a vehicle well that
    stops growing: each check reports it and flags the wells concerned."""
    times = [t for t in TIMES if t not in (24.0, 30.0)]  # 18 h → 36 h: a gap
    course, layout = _course(times)
    course.loc[(course.well == "C05") & (course.time_h == 12), "nuclei_objects_measured"] = np.nan
    course.loc[(course.well == "B04") & (course.time_h == 6), "focus_score"] = 0.3
    late = course["time_h"] >= 36
    course.loc[late, "focus_score"] *= 0.6  # the whole plate drifts out of focus
    edge = course["well"].str.startswith("A")
    course.loc[edge, "nuclei_objects_measured"] *= 0.7 ** (course.loc[edge, "time_h"] / 48)
    course.loc[course.well == "C01", "nuclei_objects_measured"] = 3000.0  # vehicle not growing
    result = analyse(course, layout, KineticsParams(dose_response=False), (8, 12))
    statuses = _statuses(result)
    assert statuses == {"time points": WARN, "focus drift": WARN, "edge evaporation": WARN,
                        "control growth": WARN}  # fmt: skip
    checks = {c["check"]: c for c in result.qc.checks}
    assert checks["time points"]["gaps"] == [(18.0, 36.0)]
    assert checks["edge evaporation"]["final_log2"] == pytest.approx(math.log2(0.7), abs=0.03)
    assert checks["focus drift"]["late"] < 0.8 * checks["focus drift"]["early"]
    flags = result.qc.well_flags
    assert "missing time point(s) 12 h" in flags["C05"]
    assert "out of focus at 6 h" in flags["B04"]
    assert any(f.startswith("edge well below") for f in flags["A05"])
    assert "vehicle well not growing" in flags["C01"]
    assert not any(f.startswith("edge well") for f in flags.get("B05", []))  # interior
    assert result.per_well()["C05"]["kn_qc_flags"] == "missing time point(s) 12 h"


def test_edges_not_assessed_without_a_condition_on_both() -> None:
    course, layout = _course()
    inside = course[~course["well"].str.match(r"^A|.*(01|12)$")]
    result = analyse(inside, layout, KineticsParams(dose_response=False), (8, 12))
    assert _statuses(result)["edge evaporation"] == NA
    assert _statuses(result)["control growth"] == NA  # the vehicle wells were all on the edge


def test_control_growth_against_history(home: Path) -> None:
    """Earlier runs in the registry: vehicle doubling times 20, 21, 20.5, 19.5 h for HeLa. A run
    at 20.4 h is within; one at 30 h warns; this run's own rows are not history."""
    from hcs_studio.db.catalog import Plate, PlateWell, Run, WellResult
    from hcs_studio.db.engine import session_scope

    with session_scope() as session:
        plate = Plate(key="barcode:K1", barcode="K1", rows=8, cols=12)
        session.add(plate)
        session.flush()
        for well, role in (("A01", "negative_control"), ("A02", "sample")):
            session.add(PlateWell(plate_id=plate.id, well=well, cell_line="HeLa", role=role))
        for i, doubling in enumerate((20.0, 21.0, 20.5, 19.5, 31.0)):
            run = Run(folder=f"ABSOLUTE:/runs/r{i}", plate_id=plate.id)
            session.add(run)
            session.flush()
            session.add_all([
                WellResult(run_id=run.id, well="A01", readout_id="kn_doubling_h", value=doubling),
                WellResult(run_id=run.id, well="A02", readout_id="kn_doubling_h", value=99.0),
            ])  # fmt: skip
    history = control_history(["HeLa", "U2OS"], {"ABSOLUTE:/runs/r4"})
    assert sorted(history["HeLa"]) == [19.5, 20.0, 20.5, 21.0] and "U2OS" not in history
    info = pd.DataFrame({"well": ["V1", "V2"], "role": ["negative_control"] * 2,
                         "cell_line": ["HeLa"] * 2, "compound": "",
                         "concentration": math.nan})  # fmt: skip
    for doubling, expected in ((20.4, PASS), (30.0, WARN)):
        qc = KineticQc()
        fits = pd.DataFrame({"well": ["V1", "V2"], "doubling_h": [doubling, doubling]})
        check_controls(qc, fits, info, KineticQcOptions(), history)
        assert qc.checks[0]["status"] == expected, qc.checks[0]
    qc = KineticQc()
    fits = pd.DataFrame({"well": ["V1", "V2"], "doubling_h": [math.nan, math.nan]})
    check_controls(qc, fits, info, KineticQcOptions(), history)
    assert qc.checks[0]["status"] == FAIL


def test_report_page_and_link(tmp_path: Path) -> None:
    from hcs_studio.output.report import report_links

    course, layout = _course()
    result = analyse(course, layout, KineticsParams(), (8, 12))
    files = write_kinetics(result, tmp_path / "kinetics", {"rows": 8, "cols": 12})
    assert {"report.html", "heatmap.png", "qc_wells.csv", "qc_timeline.csv"} <= set(files)
    page = (tmp_path / "kinetics" / "report.html").read_text(encoding="utf-8")
    data = json.loads(page.split('id="data">')[1].split("</script>")[0])
    assert data["rows"] == 8 and len(data["times"]) == len(TIMES)
    assert data["wells"]["A02"]["compound"] == "Drug"
    assert "http" not in page.replace("http-equiv", "")  # self-contained
    summary = json.loads((tmp_path / "kinetics" / "kinetics.json").read_text(encoding="utf-8"))
    assert summary["qc"]["status"] == PASS
    assert {"title": "Kinetics report", "href": "kinetics/report.html"} in report_links(tmp_path)


BROWSER = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.route('http*://**', (route) => route.abort());
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(process.argv[2]);
  const out = {};
  out.wells = await page.$$eval('#plate .w[data-well]', (e) => e.length);
  out.flagged = await page.$$eval('#plate .w.flagged', (e) => e.map((x) => x.dataset.well));
  out.sparklines = await page.$$eval('#plate polyline',
    (e) => e.filter((p) => p.getAttribute('points').split(' ').length > 3).length);
  out.heatRows = await page.$$eval('#heat tr', (e) => e.length);
  out.curves = await page.$$eval('#overlay polyline', (e) => e.length);
  await page.click('#w-B05');
  out.curvesAfterClick = await page.$$eval('#overlay polyline', (e) => e.length);
  await page.selectOption('#heatmode', 'first');
  out.firstCell = await page.$eval('#heat tr:nth-child(2) td', (td) => td.textContent);
  out.status = await page.textContent('#qc-status');
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
def test_report_page_works_offline_in_a_browser(tmp_path: Path) -> None:
    course, layout = _course()
    course.loc[(course.well == "B05") & (course.time_h == 12), "focus_score"] = 0.2
    result = analyse(course, layout, KineticsParams(), (8, 12))
    write_kinetics(result, tmp_path / "kinetics", {"rows": 8, "cols": 12})
    script = tmp_path / "check.js"
    script.write_text(BROWSER, encoding="utf-8")
    done = subprocess.run(["node", str(script), (tmp_path / "kinetics" / "report.html").as_uri()],
                          capture_output=True, text=True, timeout=120)  # fmt: skip
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["errors"] == []
    assert out["wells"] == 33 and out["sparklines"] == 33
    assert out["flagged"] == ["B05"]
    assert out["heatRows"] == 1 + 1 + 9  # header, vehicle, 9 concentrations
    assert out["curves"] == 10 and out["curvesAfterClick"] == 11  # vehicle + 9; + B05
    assert out["firstCell"] == "0.00" and out["status"] == "warn"
