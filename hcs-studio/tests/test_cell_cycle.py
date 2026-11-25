"""Phase 20a: the assay framework and the cell-cycle module on data with known answers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.assays.cell_cycle import CellCycle, CellCycleParams, fit_dna, g1_peak
from hcs_studio.synthetic.cell_cycle import (
    ARRESTED,
    PHASES,
    VEHICLE,
    sample_nuclei,
    write_cycle_run,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_dean_jett_fox_fit_recovers_known_phase_fractions() -> None:
    """Done-when (20a): a known G1/G2 ratio is recovered within 2 percentage points."""
    rng = np.random.default_rng(3)
    for truth in (VEHICLE, ARRESTED):
        nuclei = sample_nuclei(truth, 5000, rng)
        fit = fit_dna(nuclei["dna_n"].to_numpy())
        for phase in PHASES:
            assert fit.fractions[phase] == pytest.approx(truth[phase], abs=2.0), (
                phase,
                fit.fractions,
            )
        assert fit.cv_pct == pytest.approx(4.0, abs=0.5) and fit.g2_g1_ratio == pytest.approx(
            2.0, abs=0.03
        )
        assert not fit.warnings


def test_g1_peak_is_found_also_on_a_g2_arrested_population() -> None:
    rng = np.random.default_rng(0)
    raw = sample_nuclei({"g1": 25.0, "s": 10.0, "g2m": 65.0}, 4000, rng)["dna_n"] * 30_000
    assert g1_peak(raw.to_numpy()) == pytest.approx(
        60_000, rel=0.03
    )  # G1 = 2N, not the taller 4N peak
    with pytest.raises(Exception, match="too few nuclei"):
        g1_peak(np.ones(5))


def run_cycle(folder: Path, **params: Any) -> Any:
    from hcs_studio.analysis.table_steps import run_table_steps

    module = CellCycle(CellCycleParams(dna_channel="DAPI", **params))
    return run_table_steps(folder, [module])


def test_cell_cycle_step_calibrates_on_vehicle_wells_and_gates_edu_and_ph3(tmp_path: Path) -> None:
    truth = write_cycle_run(
        tmp_path / "run",
        {"A01": ("negative_control", VEHICLE), "A02": ("negative_control", VEHICLE),
         "B01": ("sample", ARRESTED), "B02": ("positive_control", ARRESTED)},
        staining={"B01": 1.05},  # 5 % brighter staining: G1 shifts to 2.1N but is still G1
    )  # fmt: skip
    result = run_cycle(tmp_path / "run", edu_channel="EdU", ph3_channel="pH3")
    state = result.steps[0]["state"]
    assert state["calibrated_on"] == "vehicle wells" and state["vehicle_nuclei"] == 8000
    assert state["g1_peak_raw"] == pytest.approx(50_000, rel=0.01)  # 2N × 25 000
    for well, values in result.per_well.items():
        expected = truth[well]
        for phase in PHASES:
            assert values[f"cc_{phase}_pct"] == pytest.approx(expected["fractions"][phase], abs=2.0)
            assert values[f"cc_edu_{phase}_pct"] == pytest.approx(
                expected["fractions"][phase], abs=1.0
            )
        assert values["cc_edu_pct"] == pytest.approx(expected["edu_pct"], abs=0.5)
        assert values["cc_mitotic_index_pct"] == pytest.approx(expected["mitotic_pct"], abs=0.5)
    assert result.per_well["A01"]["cc_g2m_arrest_index"] == pytest.approx(1.0, abs=0.05)
    assert result.per_well["B02"]["cc_g2m_arrest_index"] == pytest.approx(54 / 16, rel=0.05)
    assert result.per_well["B01"]["cc_g1_peak_n"] == pytest.approx(2.1, abs=0.03)
    out = tmp_path / "run" / "assays" / "cell_cycle"
    assert {p.name for p in out.iterdir()} == {
        "calibration.json",
        "cell_cycle_wells.csv",
        "report.html",
    }
    calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["edu_threshold"] > 60 and calibration["params"]["edu_channel"] == "EdU"
    assert "data:image/png;base64," in (out / "report.html").read_text(encoding="utf-8")
    phases = pd.read_parquet(tmp_path / "run" / "wells" / "A01" / "objects_annotations.parquet")
    assert set(phases["cc_phase"].dropna()) == set(PHASES) and "cc_dna_n" in phases


def test_warnings_for_max_projections_and_missing_layouts(tmp_path: Path) -> None:
    write_cycle_run(
        tmp_path / "run", {"A01": ("negative_control", VEHICLE), "A02": ("sample", ARRESTED)}
    )
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pipeline"]["steps"] = [{"module": "projection", "params": {"method": "max"}}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state = run_cycle(tmp_path / "run").steps[0]["state"]
    assert any("only a sum projection" in w for w in state["warnings"])
    (tmp_path / "run" / "plate_layout.json").unlink()
    state = run_cycle(tmp_path / "run").steps[0]["state"]
    assert state["calibrated_on"].startswith("all wells")
    with pytest.raises(Exception, match="not measured"):
        run_cycle(tmp_path / "run", edu_channel="Missing")


def test_preset_pipeline_measures_dna_content_from_images(home: Path) -> None:
    """Images → segmentation → integrated DNA → cell cycle: within 2 points of the truth."""
    from hcs_studio.assays.cell_cycle import preset
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import Background, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.cell_cycle import CycleNuclei

    vehicle, arrested = {"g1": 58.0, "s": 25.0, "g2m": 17.0}, {"g1": 22.0, "s": 15.0, "g2m": 63.0}
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (1, 0)],
        features=[
            Background(level=100.0),
            CycleNuclei(fractions=lambda r, c: vehicle if r == 0 else arrested, grid=12),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI", "EdU"),
        size=384,
        n_fields=8,
        name="CC",
    )
    path, truth = plate.write(home / "raw")
    layout = set_role(PlateLayout.for_plate_type(96), [(0, 0), (0, 1)], "negative_control")
    pipeline = preset({"dna": "DAPI", "edu": "EdU"}, channels=["DAPI", "EdU"],
                      nucleus_diameter_um=12 * 0.65)  # fmt: skip
    result = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False)  # fmt: skip
    summary = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    for well, data in truth["wells"].items():
        counts: dict[str, int] = {}
        for field in data["fields"].values():
            for phase, k in field["cycle"]["phases"].items():
                counts[phase] = counts.get(phase, 0) + k
        n = sum(counts.values())
        row = summary.loc[well].to_dict()
        assert row["cc_nuclei"] == n
        for phase in ("g1", "s", "g2m"):
            expected = 100 * counts[phase] / n
            assert row[f"cc_{phase}_pct"] == pytest.approx(expected, abs=2.0), (well, phase)
            assert row[f"cc_edu_{phase}_pct"] == pytest.approx(expected, abs=1.0), (well, phase)
        assert row["cc_g1_cv_pct"] == pytest.approx(4.0, abs=0.6)
    page = (result.folder / "index.html").read_text(encoding="utf-8")
    assert "assays/cell_cycle/report.html" in page


def test_command_line_preset_and_run(tmp_path: Path, home: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.services import recent_events

    out = tmp_path / "cc.yaml"
    command = ["assay", "preset", "cell_cycle", "--map", "dna=Hoechst", "--map", "edu=EdU"]
    command += ["--channels", "EdU,Hoechst", "--z-planes", "5", "--output", str(out)]
    assert main(command) == 0
    steps = Pipeline.load(out).steps
    assert [s.module for s in steps][:2] == ["projection", "background"]
    assert steps[0].params["method"] == "sum" and steps[2].params["channel"] == 2
    assert steps[-1].params["dna_channel"] == "Hoechst"
    run = tmp_path / "run"
    write_cycle_run(
        run, {"A01": ("negative_control", VEHICLE), "A02": ("sample", ARRESTED)}, n=1500
    )
    (run / "wells" / "A01" / "well_summary.json").write_text(
        json.dumps({"well": "A01"}), encoding="utf-8"
    )
    (run / "wells" / "A02" / "well_summary.json").write_text(
        json.dumps({"well": "A02"}), encoding="utf-8"
    )
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["pipeline"] = {"name": "synthetic", "steps": []}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        main(["assay", "run", str(run), "cell_cycle", "--map", "dna=DAPI", "--reason", "test"]) == 0
    )
    assert (run / "assays" / "cell_cycle" / "report.html").exists()
    summary = pd.read_csv(run / "tables" / "per_well_summary.csv").set_index("well")
    assert float(summary["cc_g2m_pct"].to_dict()["A02"]) > 45
    assert recent_events(1, "run.analyze")[0]["reason"] == "test"
    assert main(["assay", "run", str(run), "no_such_assay"]) == 2


def test_cell_cycle_readouts_are_registered() -> None:
    from hcs_studio.readouts import resolve

    for column in ("cc_g1_pct", "cc_edu_s_pct", "cc_edu_pct", "cc_g2m_arrest_index",
                   "cc_mitotic_index_pct", "cc_g1_cv_pct", "cc_nuclei"):  # fmt: skip
        assert resolve(column, "well") is not None, column
    assert resolve("cc_dna_n", "object") is not None


pytest.importorskip("PySide6")


@pytest.mark.gui
def test_assay_wizard_previews_and_applies(qtbot: Any, tmp_path: Path, home: Path) -> None:
    from hcs_studio.gui.assay_wizard import AssayWizard

    run = tmp_path / "run"
    write_cycle_run(
        run, {"A01": ("negative_control", VEHICLE), "A02": ("sample", ARRESTED)}, n=1500
    )
    for well in ("A01", "A02"):
        (run / "wells" / well / "well_summary.json").write_text(
            json.dumps({"well": well}), encoding="utf-8"
        )
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["plate"]["metadata"]["channels"] = [{"name": "DAPI"}, {"name": "EdU"}]
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    wizard = AssayWizard(run)
    qtbot.addWidget(wizard)
    assert wizard.assay.currentText() == "apoptosis"  # alphabetical
    assert set(wizard.role_combos) == {"dna", "apoptosis", "dead"}
    wizard.assay.setCurrentText("cell_cycle")
    assert wizard.role_combos["dna"].currentText() == "DAPI"
    assert wizard.role_combos["edu"].currentText() == "EdU"
    assert "A01" in wizard.controls.text()
    wizard.preview()
    qtbot.waitUntil(lambda: wizard.calibration is not None, timeout=30000)
    assert "vehicle wells" in wizard.summary.text()
    added: list[Any] = []
    wizard.add_to_pipeline.connect(lambda name, params: added.append((name, params)))
    wizard._add()
    assert added == [
        ("cell_cycle", {"dna_channel": "DAPI", "edu_channel": "EdU", "ph3_channel": ""})
    ]
    wizard.reason.setText("first cell cycle")
    wizard.apply()
    qtbot.waitUntil(lambda: "Applied" in wizard.status.text(), timeout=60000)
    assert (run / "assays" / "cell_cycle" / "report.html").exists()
