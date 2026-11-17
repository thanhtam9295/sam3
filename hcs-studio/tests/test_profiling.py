"""Known-answer tests for Phase 9: feature selection, control normalisation, PCA/UMAP,
distance to control, dose-response and the output files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcs_studio.analysis.profiling import (
    ProfileOptions,
    drop_correlated,
    fit_dose_response,
    four_pl,
    load_profile,
    normalise,
    run_profiling,
    select_features,
)

CONCENTRATIONS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]  # µM; EC50 of Drug X = 1 µM


def _layout() -> dict[str, tuple[str, float | None, str]]:
    """well → (compound, concentration, control type) on a 4 × 6 plate."""
    layout: dict[str, tuple[str, float | None, str]] = {}
    for well in ["A01", "B01", "C01", "D01"]:
        layout[well] = ("DMSO", None, "negative")
    for well in ["A02", "B02", "C02", "D02"]:
        layout[well] = ("Staurosporine", 1.0, "positive")
    for k, c in enumerate(CONCENTRATIONS):
        layout[f"{'ABCD'[k % 4]}0{3 + k // 4}"] = ("Drug X", c, "")
    for k, c in enumerate(CONCENTRATIONS):
        layout[f"{'ABCD'[k % 4]}0{5 + k // 4}"] = ("Drug Y", c, "")
    return layout


def _effect(compound: str, concentration: float | None) -> float:
    """Shift of f_size, in units of its SD."""
    if compound == "Staurosporine":
        return 4.0
    if compound == "Drug X" and concentration:
        return 4.0 / (1 + 1.0 / concentration)  # 4PL, bottom 0, top 4, EC50 1, Hill 1
    return 0.0


def make_profile_run(
    root: Path, name: str, seed: int, offset: float = 0.0, cells: int = 150
) -> Path:
    """A minimal run folder: per-well objects, layout and manifest (no images)."""
    rng = np.random.default_rng(seed)
    folder = root / f"{name}__profile__run"
    layout = _layout()
    lines = ["well,compound,concentration,treatment,control_type"]
    summary_rows = []
    for well, (compound, concentration, control) in layout.items():
        shift = _effect(compound, concentration)
        size = rng.normal(10 + 2 * shift, 2, cells)  # SD 2 → shift in SD units
        objects = pd.DataFrame(
            {
                "object_type": "nuclei",
                "object_id": np.arange(1, cells + 1),
                "well": well,
                "row": "ABCD".index(well[0]) + 1,
                "col": int(well[1:]),
                "field": rng.integers(1, 4, cells),
                "timepoint": 0,
                "f_size": size + offset,
                "f_size_twin": 2 * size + rng.normal(0, 0.01, cells) + offset,
                "f_intensity": rng.normal(100 + 10 * shift, 10, cells) + 20 * offset,
                "f_noise_a": rng.normal(0, 1, cells),
                "f_noise_b": rng.normal(5, 2, cells),
                "f_constant": np.ones(cells),
                "f_missing": np.where(rng.uniform(size=cells) < 0.5, np.nan, 1.0),
                "centroid_y_um": rng.uniform(0, 500, cells),
                "centroid_x_um": rng.uniform(0, 500, cells),
            }
        )
        d = folder / "wells" / well
        d.mkdir(parents=True)
        objects.to_parquet(d / "objects.parquet", index=False)
        (d / "well_summary.json").write_text(json.dumps({"well": well, "fields": []}))
        lines.append(f"{well},{compound},{concentration or ''},,{control}")
        summary_rows.append({"well": well, "nuclei_count": cells, "readout_shift": shift})
    (folder / "plate_layout.csv").write_text("\n".join(lines) + "\n")
    (folder / "tables").mkdir()
    pd.DataFrame(summary_rows).to_csv(folder / "tables" / "per_well_summary.csv", index=False)
    manifest = {
        "plate": {"name": name, "rows": 4, "cols": 6, "path": str(root), "metadata": {}},
        "pipeline": {"name": "profile", "steps": []},
        "status": "completed",
    }
    (folder / "run_manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture(scope="module")
def plates(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("profiling")
    return make_profile_run(root, "P1", seed=1), make_profile_run(root, "P2", seed=2, offset=5)


# ------------------------------------------------------------------- units


def test_feature_selection_and_correlation_filter() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=300),
            "flat": np.r_[np.ones(290), np.arange(10)],
            "holes": np.where(rng.uniform(size=300) < 0.3, np.nan, 1.0) * rng.normal(size=300),
            "b": rng.normal(size=300),
        }
    )
    frame["a_twin"] = frame.a * 3 + rng.normal(0, 0.01, 300)
    options = ProfileOptions()
    kept, dropped = select_features(frame, list(frame.columns), options)
    assert kept == ["a", "b", "a_twin"]
    assert "near-constant" in dropped["flat"] and "missing" in dropped["holes"]
    matrix = frame[kept].to_numpy()
    survivors, correlated = drop_correlated(matrix, kept, 0.95)
    assert len(survivors) == 2 and "b" in survivors
    assert len(correlated) == 1 and "|r| = 1.000" in next(iter(correlated.values()))


def test_robust_z_relative_to_controls() -> None:
    frame = pd.DataFrame(
        {
            "plate": ["p"] * 6 + ["q"] * 6,
            "x": [1.0, 2, 3, 10, 20, 30, 101, 102, 103, 110, 120, 130],
        }
    )
    controls = pd.Series([True, True, True, False, False, False] * 2)
    out = normalise(frame, ["x"], controls)
    scale = 1.4826  # MAD of 1, 2, 3 is 1
    assert out.x.tolist()[:6] == pytest.approx(
        [-1 / scale, 0, 1 / scale, 8 / scale, 18 / scale, 28 / scale]
    )
    assert out.x.tolist()[6:] == pytest.approx(out.x.tolist()[:6])  # plate offset removed


def test_four_parameter_logistic_fit() -> None:
    x = np.repeat([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100], 3)
    rng = np.random.default_rng(0)
    y = four_pl(x, 5.0, 85.0, math.log10(2.5), 1.3) + rng.normal(0, 1.5, len(x))
    fit = fit_dose_response(x, y)
    assert fit["fit"] and fit["ec50"] == pytest.approx(2.5, rel=0.1)
    assert fit["hill"] == pytest.approx(1.3, rel=0.15)
    assert fit["bottom"] == pytest.approx(5, abs=3) and fit["top"] == pytest.approx(85, abs=3)
    assert fit["r2"] > 0.99
    down = fit_dose_response(x, 100 - y)
    assert down["fit"] and down["top"] < down["bottom"]
    assert down["ec50"] == pytest.approx(2.5, rel=0.1)
    assert not fit_dose_response(np.array([1, 2, 3.0]), np.array([1, 2, 3.0]))["fit"]


# ------------------------------------------------------------------ profiling


def test_profiling_two_plates(plates: tuple[Path, Path], tmp_path: Path) -> None:
    options = ProfileOptions(
        max_cells=4000,
        pca_components=4,
        n_neighbors=10,
        dose_readouts=[
            "profile_distance_to_control",
            "z_f_size",
            "z_f_size_twin",
            "readout_shift",
        ],
    )
    result = run_profiling(list(plates), options, output=tmp_path / "advanced")
    assert set(result.dropped) >= {"f_constant", "f_missing"}
    assert "near-constant" in result.dropped["f_constant"]
    # f_size and its twin are perfectly correlated: exactly one remains.
    assert ("f_size" in result.features) != ("f_size_twin" in result.features)
    size = "f_size" if "f_size" in result.features else "f_size_twin"
    assert {"f_intensity", "f_noise_a", "f_noise_b"} <= set(result.features)

    # The effect lies along f_size and f_intensity, so they dominate PC1.
    pc1 = result.loadings["PC1"].abs().sort_values(ascending=False)
    assert set(pc1.index[:2]) <= {"f_size", "f_size_twin", "f_intensity"}
    assert result.variance["explained_variance_ratio"].iloc[0] > 0.4

    negative = result.wells[result.wells.control_type == "negative"]
    assert negative.profile_distance_ratio.median() == pytest.approx(1)  # by definition
    wells = result.wells.set_index(["plate", "well"])
    assert len(wells) == 44 and wells.n_objects.eq(150).all()  # 22 wells per plate
    for plate in ("P1", "P2"):
        p = result.wells[result.wells.plate == plate].set_index("well")
        distance = p.profile_distance_to_control  # in units of object-to-object SD
        controls = p[p.control_type == "negative"]
        positives = distance[p.control_type == "positive"]
        drug_y = distance[p.compound == "Drug Y"]
        # Staurosporine shifts two independent features by 4 SD: √(4² + 4²) = 5.66.
        assert positives.median() == pytest.approx(5.66, rel=0.1)
        assert distance[controls.index].max() < 0.5 and drug_y.max() < 0.5
        drug_x = p[p.compound == "Drug X"].sort_values("concentration")
        assert (
            drug_x.profile_distance_to_control.iloc[-1]
            > 5 * drug_x.profile_distance_to_control.iloc[0]
        )
    # Plate 2 is offset in every raw feature; after control normalisation the plates agree.
    z1 = wells.loc["P1"].z_f_intensity
    z2 = wells.loc["P2"].z_f_intensity
    assert np.corrcoef(z1, z2.loc[z1.index])[0, 1] > 0.95
    assert abs(z1.median() - z2.median()) < 0.5

    fits = {(r["plate"], r["compound"], r["readout"]): r for r in result.dose.to_dict("records")}
    for plate in ("P1", "P2"):
        exact = fits[(plate, "Drug X", "readout_shift")]
        assert exact["fit"] and exact["ec50"] == pytest.approx(1.0, rel=1e-3)
        measured = fits[(plate, "Drug X", f"z_{size}")]
        assert measured["fit"] and measured["ec50"] == pytest.approx(1.0, rel=0.35)
        assert measured["r2"] > 0.95

    cells = result.cells
    assert {"PC1", "PC2", "UMAP1", "UMAP2", "compound", "control_type"} <= set(cells.columns)
    assert len(cells) <= 4000 and cells.groupby(["plate", "well"]).size().nunique() == 1

    folder = tmp_path / "advanced"
    for name in (
        "pca_coordinates.parquet",
        "umap_coordinates.parquet",
        "pca_loadings.csv",
        "pca_variance.csv",
        "well_profiles.parquet",
        "dose_response.csv",
        "profiling.json",
        "figures/pca_scree.svg",
        "figures/umap_cells.png",
        "figures/pca_wells.svg",
        "figures/dose_Drug_X_z_f_size.svg"
        if "f_size" in result.features
        else "figures/pca_scree.png",
    ):
        assert (folder / name).exists(), name
    record = json.loads((folder / "profiling.json").read_text())
    assert record["options"]["seed"] == 0 and record["features_used"] == result.features
    again = load_profile(folder)
    assert again is not None and again.features == result.features
    assert np.allclose(again.cells["UMAP1"], cells["UMAP1"])


def test_profiling_is_reproducible(plates: tuple[Path, Path], tmp_path: Path) -> None:
    options = ProfileOptions(max_cells=600, level="well", n_neighbors=8, figures=False)
    a = run_profiling([plates[0]], options, output=tmp_path / "a")
    b = run_profiling([plates[0]], options, output=tmp_path / "b")
    assert len(a.cells) == 22  # one point per well at well level
    assert np.allclose(a.cells["UMAP1"], b.cells["UMAP1"])
    assert np.allclose(a.cells["PC1"], b.cells["PC1"])


def test_profiling_without_controls(plates: tuple[Path, Path], tmp_path: Path) -> None:
    layout = plates[0] / "plate_layout.csv"
    original = layout.read_text()
    try:
        layout.write_text(original.replace("negative", ""))
        result = run_profiling(
            [plates[0]], ProfileOptions(umap=False, figures=False, max_cells=500), tmp_path
        )
    finally:
        layout.write_text(original)
    assert any("no negative-control wells" in n for n in result.notes)
    assert result.wells.profile_distance_to_control.isna().all()


# ------------------------------------------------- CLI, results plate, report

EMBED_SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
  await page.route('http*://**', (route) => route.abort());
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(process.argv[2]);
  const out = {};
  out.visible = await page.isVisible('#profiles');
  out.spaces = await page.$$eval('#space option', (os) => os.map((o) => o.value));
  out.legend = await page.$$eval('#legend span', (s) => s.map((x) => x.textContent));
  // Click the point of well A02 (a positive control): the well opens.
  const [x, y] = await page.evaluate(() => window.hcsEmbedding.position('A02'));
  await page.mouse.click(x, y);
  out.opened = await page.$eval('#well h2', (h) => h.textContent);
  out.selectedClass = await page.$eval('#w-A02', (e) => e.classList.contains('sel'));
  // Drag a rectangle around the positive controls: they are marked on the plate.
  const positions = await page.evaluate(
    () => ['A02', 'B02', 'C02', 'D02'].map((w) => window.hcsEmbedding.position(w)));
  const xs = positions.map((p) => p[0]), ys = positions.map((p) => p[1]);
  await page.mouse.move(Math.min(...xs) - 6, Math.min(...ys) - 6);
  await page.mouse.down();
  await page.mouse.move(Math.max(...xs) + 6, Math.max(...ys) + 6, { steps: 5 });
  await page.mouse.up();
  out.picked = await page.evaluate(() => window.hcsEmbedding.picked());
  out.pickedOnPlate = await page.$$eval('.well.picked',
    (els) => els.map((e) => e.dataset.well).sort());
  // Plate → plot: selecting a well redraws its point highlighted (no error).
  await page.click('#w-A01');
  await page.selectOption('#space', 'UMAP');
  await page.selectOption('#level', 'cells');
  out.readouts = await page.$$eval('#readout option', (os) => os.map((o) => o.value));
  out.errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})();
"""


def test_cli_profile_results_and_report(plates: tuple[Path, Path], tmp_path: Path) -> None:
    import shutil
    import subprocess

    from hcs_studio.app import main
    from hcs_studio.output.results import RunResults

    run = tmp_path / "run"
    shutil.copytree(plates[0], run)
    code = main(
        [
            "profile",
            str(run),
            "--max-cells",
            "1500",
            "--neighbors",
            "8",
            "--dose-readouts",
            "profile_distance_to_control,readout_shift",
        ]
    )
    assert code == 0
    assert (run / "advanced" / "umap_coordinates.parquet").exists()
    results = RunResults(run)
    distance = results.values("profile_distance_to_control")
    assert distance["A02"] > 10 * distance["A05"]  # positive control vs inactive compound
    assert "profile_distance_to_control" in results.readout_columns()[:6]
    embedding = results.profile_embedding(max_cells=500)
    assert embedding is not None and embedding["spaces"] == ["PC", "UMAP"]
    assert len(embedding["wells"]) == 22 and len(embedding["cells"]) == 500
    html = (run / "index.html").read_text(encoding="utf-8")
    assert 'id="profiles"' in html and '"embedding":{' in html

    if (
        shutil.which("node") is None
        or subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("Node.js with Playwright is not available")
    script = tmp_path / "embed.js"
    script.write_text(EMBED_SCRIPT)
    result = subprocess.run(
        ["node", str(script), (run / "index.html").as_uri()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [] and out["visible"]
    assert out["spaces"] == ["PC", "UMAP"]
    assert out["legend"][:2] == ["negative control", "positive control"]
    assert out["opened"] == "Well A02" and out["selectedClass"]
    assert {"A02", "B02", "C02", "D02"} <= set(out["picked"])
    assert out["pickedOnPlate"] == out["picked"]
    assert "profile_distance_to_control" in out["readouts"]
