"""The acceptance-test toolkit (PROMPT.md §10) on synthetic data with known outcomes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from hcs_studio import acceptance
from hcs_studio.app import main
from hcs_studio.io.base import open_plate
from hcs_studio.pipeline.executor import run_pipeline
from hcs_studio.pipeline.schema import Pipeline
from tests.builders import make_imagexpress
from tests.test_batch import PIPELINE, _image

WELLS = [(r, c) for r in range(2) for c in range(3)]


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("acceptance")
    plate = make_imagexpress(tmp / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)
    return run_pipeline(
        open_plate(plate), Pipeline.from_text(PIPELINE), output_root=tmp / "out", checksums=False
    ).folder


def test_counts_against_manual_counts(run_folder: Path, tmp_path: Path) -> None:
    fields = pd.concat(pd.read_csv(p) for p in run_folder.glob("wells/*/fields.csv"))
    manual = fields[["well", "field", "n_nuclei"]].rename(columns={"n_nuclei": "count"})
    manual = manual.reset_index(drop=True)
    counts = manual["count"].to_numpy(copy=True)
    counts[0] = round(float(counts[0]) * 1.5)  # a miscounted field
    manual["count"] = counts
    path = tmp_path / "manual.csv"
    manual.to_csv(path, index=False)
    report = acceptance.check_counts(run_folder, path)
    assert [i.passed for i in report.items].count(False) == 1
    assert not report.items[0].passed and all(i.passed for i in report.items[1:])
    assert len(report.items) == 12 and not any("10 annotated" in n for n in report.notes)
    out = tmp_path / "report.json"
    assert (
        main(
            [
                "acceptance",
                "--report",
                str(out),
                "counts",
                "--run",
                str(run_folder),
                "--manual",
                str(path),
            ]
        )
        == 1
    )
    assert json.loads(out.read_text())["passed"] is False


COLOC2_LOG = """Coloc_2 version 3.0.6
Images used (Channel 1, Channel 2): channel_a.tif, channel_b.tif
Mask: mask.tif
Pearson's R value (no threshold), {pearson}
Pearson's R value (below threshold), 0.12
Pearson's R value (above threshold), 0.81
Li's ICQ value, {icq}
Spearman's rank correlation value, {spearman}
Manders' M1 (Above zero intensity of Ch2), {m1}
Manders' M2 (Above zero intensity of Ch1), {m2}
Manders' tM1 (Above autothreshold of Ch2), {tm1}
Manders' tM2 (Above autothreshold of Ch1), {tm2}
Costes P-Value, 1.0
"""


def test_coloc_against_coloc2_log(tmp_path: Path) -> None:
    from hcs_studio.analysis.coloc_export import export_for_fiji

    rng = np.random.default_rng(0)
    a = ndimage.gaussian_filter(rng.uniform(0, 100, (64, 64)), 2)
    b = a + rng.normal(0, 1, a.shape)
    folder = tmp_path / "field1"
    result = export_for_fiji(folder, a, b, np.ones(a.shape, bool), randomizations=5)
    assert "coloc2_log.txt" in (folder / "coloc2_compare.ijm").read_text()
    ours = result.row()
    close = {k: ours[k] + 0.01 for k in ("pearson", "m1", "m2", "tm1", "tm2")}
    (folder / "coloc2_log.txt").write_text(
        COLOC2_LOG.format(icq=ours["icq"], spearman=ours["spearman"], **close)
    )
    assert acceptance.parse_coloc2_log((folder / "coloc2_log.txt").read_text())["pearson"] == (
        pytest.approx(ours["pearson"] + 0.01)
    )
    report = acceptance.check_coloc([folder])
    assert report.passed and len(report.items) == 5
    far = dict(close, tm1=ours["tm1"] + 0.05)
    (folder / "coloc2_log.txt").write_text(
        COLOC2_LOG.format(icq=ours["icq"], spearman=ours["spearman"], **far)
    )
    report = acceptance.check_coloc([folder])
    assert [i.name for i in report.items if not i.passed] == ["field1 tm1"]
    (folder / "coloc2_log.txt").unlink()
    assert not acceptance.check_coloc([folder]).passed


def test_run_folder_check(run_folder: Path, tmp_path: Path) -> None:
    assert acceptance.check_run(run_folder).passed
    broken = tmp_path / "broken"
    shutil.copytree(run_folder, broken)
    (broken / "wells" / "A02" / "previews" / "f001.png").unlink()
    html = broken / "index.html"
    html.write_text(
        html.read_text().replace(
            "</head>", '<script src="https://cdn.example/x.js"></script></head>'
        )
    )
    failed = {i.name for i in acceptance.check_run(broken).items if not i.passed}
    assert failed == {"report is self-contained", "report previews"}


def test_kill_and_resume(tmp_path: Path) -> None:
    plate = make_imagexpress(tmp_path / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)
    slow = PIPELINE.replace("steps:\n", "steps:\n  - {module: denoise, params: {method: nlm}}\n")
    pipeline = tmp_path / "slow.yaml"
    pipeline.write_text(slow)
    report = acceptance.check_kill_resume(
        pipeline, plate, tmp_path / "out", kill_after_wells=1, workers="2"
    )
    assert report.passed, report.text()
    names = [i.name for i in report.items]
    assert names[:2] == ["killed part-way", "resume exit code"]
    assert "resumed after interruption" in names


def test_beads_fwhm_narrows_after_deconvolution(tmp_path: Path) -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.output.writers import write_ome_tiff
    from hcs_studio.processing import psf as psf_tools

    pixel, z_step = 0.1, 0.25
    spec = psf_tools.PsfSpec(
        na=1.2, emission_nm=520, pixel_size_um=pixel, immersion_ri=1.33, z_step_um=z_step
    )
    kernel = psf_tools.theoretical_psf(spec, three_d=True)
    rng = np.random.default_rng(0)
    beads = np.zeros((32, 160, 160), np.float32)
    for z, y, x in [(12, 30, 30), (16, 30, 110), (14, 80, 70), (18, 130, 30), (13, 130, 125)]:
        beads[z, y, x] = 300_000  # bright, isolated: more than a crop apart
    from scipy.signal import fftconvolve

    stack = fftconvolve(beads, kernel, mode="same") + 50 + rng.normal(0, 1, beads.shape)
    metadata = ImageMetadata(
        channels=(ChannelInfo("beads"),), pixel_size_um=pixel, z_step_um=z_step
    )
    path = tmp_path / "beads.ome.tif"
    write_ome_tiff(path, np.clip(stack, 0, None)[None, None].astype(np.float32), metadata)
    psf_file = tmp_path / "psf.ome.tif"
    psf_tools.save_psf(psf_file, kernel, pixel, z_step)
    report = acceptance.check_beads(path, psf_file=psf_file, iterations=30)
    assert report.passed, report.text()
    lateral = next(i for i in report.items if i.name == "FWHM x").values
    assert lateral["after_um"] < 0.9 * lateral["before_um"]
