"""Phase 18b: image QC inside runs and the job queue, object-level QC, compound artifacts, and
exclusions propagated to tables, pharmacology, reports and the registry."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.model.plate import well_name
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.qc.rules import default_rules

STEPS = """
name: qc-run
steps:
  - {module: segment_classical, params: {channel: 1, min_distance_um: 2}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
"""


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def qc_pipeline(diameter: float, **plate: Any) -> Pipeline:
    rules = default_rules(nucleus_diameter_um=diameter)
    rules = rules.model_copy(update={"plate": rules.plate.model_copy(update=plate)})
    return Pipeline.model_validate({**Pipeline.from_text(STEPS).to_dict(), "qc": rules.to_dict()})


def test_qc_runs_before_segmentation_and_excludes_fields(home: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.output.report import write_report
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic.artifacts import qc_plate

    path, truth = qc_plate(home / "raw", per_kind=3, clean=9, kinds=("blur", "bubble", "empty"))
    plate = open_plate(path)
    pipeline = qc_pipeline(truth["nucleus_diameter_um"], max_failed_well_fraction=1.0)
    events: list[dict[str, Any]] = []
    result = run_pipeline(plate, pipeline, output_root=home / "runs", events=events.append)
    kinds = [e["type"] for e in events]
    assert kinds.index("qc_finished") < kinds.index("well_started")  # before segmentation
    folder = result.folder
    assert {p.name for p in (folder / "qc").iterdir()} >= {"field_qc.parquet", "well_qc.csv"}
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["image_qc"]["excluded_fields"] == 6  # blur + bubble
    assert "qc" in manifest["pipeline"] and manifest["status"] == "completed"

    flags = pd.read_csv(folder / "tables" / "qc_flags.csv")
    labels = {k: v for k, v in truth["labels"].items()}
    flags["truth"] = [labels[f"{w}/{f}"] for w, f in zip(flags.well, flags.field, strict=True)]
    excluded = flags[flags.image_qc_excluded]
    assert set(excluded.truth) == {"blur", "bubble"} and excluded.excluded.all()
    assert (flags.loc[flags.truth == "empty", "image_qc_flags"] == "empty").all()

    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert {"qc_failed", "qc_flags", "qc_reasons"} <= set(summary.columns)
    objects = pd.read_parquet(folder / "tables" / "per_object_all.parquet")
    for well, row in summary.iterrows():
        kept = flags[(flags.well == well) & ~flags.excluded].field
        measured = objects[(objects.well == well) & objects.field.isin(kept)]
        expected = int((measured.object_type == "nuclei").sum())
        assert row.get("nuclei_objects_measured", 0) == expected  # excluded fields left out
    page = write_report(folder).read_text()
    assert "Image QC" in page and "Excluded fields" in page and "blur (" in page


def test_a_plate_that_fails_qc_can_stop_the_run(home: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic.artifacts import qc_plate

    path, truth = qc_plate(home / "raw", per_kind=6, clean=0, kinds=("blur",), fields_per_well=2)
    pipeline = qc_pipeline(truth["nucleus_diameter_um"], stop_run=True)
    assert pipeline.qc is not None
    # every field is blurred: a fixed limit (not the plate median) catches the whole plate
    rules = [
        r.model_copy(update={"rel_below": None, "below": 0.06}) if r.flag == "blur" else r
        for r in pipeline.qc.field
    ]
    pipeline = pipeline.model_copy(update={"qc": pipeline.qc.model_copy(update={"field": rules})})
    events: list[dict[str, Any]] = []
    plate = open_plate(path)
    result = run_pipeline(plate, pipeline, output_root=home / "runs", events=events.append)
    assert "well_started" not in [e["type"] for e in events]
    manifest = json.loads((result.folder / "run_manifest.json").read_text())
    assert manifest["status"] == "qc_failed" and not manifest["image_qc"]["passed"]
    assert events[-2]["status"] == "qc_failed"


def test_qc_of_a_384_well_plate_runs_in_the_queue_before_segmentation(home: Path) -> None:
    """Done-when (18): QC on a 384-well plate runs from raw data in the queue and finishes
    before segmentation starts."""
    from hcs_studio.pipeline.jobs import Job
    from hcs_studio.pipeline.queue import JobSpec, LocalJobQueue
    from hcs_studio.synthetic import Background, Nuclei, ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.artifacts import Artifacts

    wells = [(r, c) for r in range(16) for c in range(24)]
    blurred = {(3, 5), (10, 17)}
    plate = SyntheticPlate(
        wells=wells,
        features=[
            Background(level=100.0),
            Nuclei(count=lambda r, c, f: 12 + (r + c) % 8, grid=5),
            Artifacts(kind=lambda r, c, f: "blur" if (r, c) in blurred else None),
            ShotNoise(sd=5.0),
        ],
        channels=("DAPI",),
        size=80,
        name="QC384",
    )
    path, _ = plate.write(home / "raw")
    from hcs_studio.io.base import open_plate

    source = open_plate(path).source
    job = Job(
        plate_source=source,
        pipeline=qc_pipeline(4.55, max_failed_well_fraction=1.0).to_dict(),
        output_root=str(home / "runs"),
        workers=2,
        save_images=False,
        checksums=False,
        label="QC384",
    )
    queue = LocalJobQueue(max_running=1)
    job_id = queue.submit(JobSpec.for_job(job, priority="batch"))
    started = time.monotonic()
    while queue.status(job_id).status not in ("completed", "completed_with_failures", "failed"):
        queue.poll()
        assert time.monotonic() - started < 400, queue.status(job_id).message
        time.sleep(0.2)
    state = queue.status(job_id)
    assert state.status == "completed", state.message
    kinds = [e["type"] for e in queue.events(job_id)]
    assert "qc_started" in kinds and kinds.index("qc_finished") < kinds.index("well_started")
    assert state.folder
    folder = Path(state.folder)
    qc = pd.read_parquet(folder / "qc" / "field_qc.parquet")
    assert len(qc) == 384
    assert set(qc.loc[qc.excluded, "well"]) == {well_name(*w) for w in blurred}
    manifest = json.loads((folder / "run_manifest.json").read_text())
    wall = manifest["performance"]["wall_s"]
    assert wall["image_qc"] == manifest["image_qc"]["seconds"] < wall["total"]


def _write_qc(folder: Path, failed: dict[str, str], flags: dict[str, str] | None = None) -> None:
    qc = folder / "qc"
    qc.mkdir(exist_ok=True)
    wells = sorted(set(failed) | set(flags or {}))
    pd.DataFrame(
        {
            "well": wells,
            "failed": [w in failed for w in wells],
            "flags": [(flags or {}).get(w, "") for w in wells],
            "reasons": [failed.get(w, "") for w in wells],
        }
    ).to_csv(qc / "well_qc.csv", index=False)
    plate = {"passed": True, "reasons": [], "fields": 0, "excluded_fields": 0,
             "rules": {"name": "test", "version": 1, "sha256": "x"}}  # fmt: skip
    (qc / "plate_qc.json").write_text(json.dumps(plate))


def test_wells_failing_image_qc_leave_ic50_fits_and_are_listed(tmp_path: Path) -> None:
    """Done-when (18): excluded wells are removed from IC50 fits and listed in the report."""
    from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology
    from hcs_studio.pharmacology.synthetic import write_run

    run = write_run(tmp_path / "run")
    table = run / "tables" / "per_well_summary.csv"
    summary = pd.read_csv(table, index_col=0)
    bad = ["O02", "O03", "P02"]  # Drug D in MCF7 at 10 and 3.3 µM: readouts corrupted
    summary.loc[bad, "nuclei_count"] = 1000.0
    summary.to_csv(table)
    options = PharmOptions(readouts=["nuclei_count"], outlier_mad=None)

    def drug_d(result: Any) -> Any:
        curves = result.readouts["nuclei_count"].curves
        return next(c for c in curves if c.compound == "Drug D" and c.cell_line == "MCF7")

    corrupted = drug_d(run_pharmacology([run], options, write=False, refresh_reports=False))
    reference = drug_d(
        run_pharmacology(
            [run],
            PharmOptions(
                readouts=["nuclei_count"],
                outlier_mad=None,
                excluded_wells=[("SynthPlate", w) for w in bad],
            ),
            write=False,
            refresh_reports=False,
        )
    )
    blur = "blur (c1_laplacian_variance = 0.01)"
    _write_qc(run, dict.fromkeys(bad, blur), {"O04": "precipitation"})
    result = run_pharmacology([run], options)
    curve = drug_d(result)
    points = curve.points.set_index("well")
    assert points.loc[bad, "excluded"].all()  # drawn hollow, left out of the fit
    assert curve.fit.ec50 == pytest.approx(reference.fit.ec50, rel=1e-6)
    assert abs(np.log10(corrupted.fit.ec50 / curve.fit.ec50)) > 0.1
    assert any("precipitation" in f for f in curve.fit.flags)
    listed = pd.read_csv(run / "pharmacology" / "excluded_wells.csv")
    reasons = listed.set_index("well").loc[bad, "exclusion_reason"]
    assert reasons.str.startswith("image QC: blur").all()
    page = (run / "pharmacology" / "pharmacology_report.html").read_text()
    assert all(w in page for w in bad) and "image QC: blur" in page


def test_failed_controls_do_not_set_control_levels(tmp_path: Path) -> None:
    from hcs_studio.pharmacology.normalize import Normalization, load_wells, normalize
    from hcs_studio.pharmacology.synthetic import write_run

    run = write_run(tmp_path / "run")
    table = run / "tables" / "per_well_summary.csv"
    summary = pd.read_csv(table, index_col=0)
    summary.loc["A01", "nuclei_count"] = 5000.0  # a vehicle well with debris counted as cells
    summary.to_csv(table)
    before = normalize(load_wells([run]), "nuclei_count", Normalization())
    _write_qc(run, {"A01": "debris"})
    data = normalize(load_wells([run]), "nuclei_count", Normalization())
    assert any("vehicle well(s) failed image QC" in n for n in data.notes)
    old = before.controls[before.controls.cell_line == "MCF7"].iloc[0]
    new = data.controls[data.controls.cell_line == "MCF7"].iloc[0]
    assert new.vehicle_n == old.vehicle_n - 1 and new.vehicle_median <= old.vehicle_median


def test_object_qc_flags_merged_cells_and_fragments() -> None:
    from hcs_studio.analysis.table_steps import WellContext
    from hcs_studio.qc.objects import ObjectQc, ObjectQcParams

    rng = np.random.default_rng(0)
    n = 400
    objects = pd.DataFrame(
        {
            "well": "A01",
            "field": 1,
            "timepoint": 0,
            "object_type": "nuclei",
            "object_id": np.arange(n),
            "area_um2": rng.lognormal(np.log(80), 0.12, n),
            "solidity": rng.normal(0.95, 0.01, n).clip(0, 1),
            "DAPI_mean_intensity": rng.lognormal(np.log(1500), 0.1, n),
        }
    )
    objects.loc[0, ["area_um2", "solidity"]] = [400.0, 0.7]  # two cells merged
    objects.loc[1, "area_um2"] = 8.0  # fragment
    objects.loc[2, "DAPI_mean_intensity"] = 30000.0  # debris-bright
    module = ObjectQc(ObjectQcParams())
    state = module.fit_table(objects)
    assert set(state["features"]) == {"area_um2", "solidity", "DAPI_mean_intensity"}
    out = module.transform(objects, state, WellContext("A01", Path(".")))
    flagged = set(np.flatnonzero(out["qc_outlier"].fillna(False).to_numpy(bool)))
    assert {0, 1, 2} <= flagged and len(flagged) <= 5
    reasons = out["qc_reason"].tolist()
    assert "area_um2 high" in reasons[0] and "solidity low" in reasons[0]
    assert reasons[1] == "area_um2 low"
    summary, _ = module.summarize(
        pd.concat([objects, out], axis=1), state, WellContext("A01", Path("."))
    )
    assert summary["qc_nuclei_outlier_count"] == len(flagged)

    from hcs_studio.output.tables import object_statistics

    rows, _ = object_statistics(pd.concat([objects, out], axis=1), set())
    assert rows["A01"]["nuclei_objects_measured"] == n - len(flagged)
    assert rows["A01"]["nuclei_area_um2_mean"] < 85  # the merged object no longer pulls it up


def test_compound_autofluorescence_and_precipitation_follow_the_dose() -> None:
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import dilution_series, set_role
    from hcs_studio.qc.decide import apply_rules

    layout = PlateLayout.for_plate_type(96)
    layout = set_role(layout, [(r, 0) for r in range(8)], "negative_control", cell_line="MCF7")
    for row, drug in ((0, "Glow"), (2, "Crash"), (4, "Clean")):
        layout = dilution_series(layout, (row, 1), drug, 30.0, "µM", factor=3, points=8,
                                 replicates=2, cell_line="MCF7")  # fmt: skip
    rng = np.random.default_rng(1)
    rows = []
    for (r, c), annotation in layout.wells.items():
        treatments = annotation.active_treatments
        conc = float(treatments[0].concentration) if treatments else 0.0
        drug = treatments[0].compound if treatments else ""
        background = 100 + rng.normal(0, 1)
        particles = 0.001 + abs(rng.normal(0, 0.0002))
        if drug == "Glow":
            background += 400 * conc / (conc + 3e-6)  # rises with the dose in channel 2
        if drug == "Crash" and conc >= 3e-6:
            particles += 0.02 * conc / 30e-6  # precipitate at the top concentrations
        rows.append({"well": well_name(r, c), "row": r, "col": c, "field": 1,
                     "c1_background": 100 + rng.normal(0, 1), "c2_background": background,
                     "particle_fraction": particles, "object_count": 50.0})  # fmt: skip
    rules = default_rules(brightfield_channel=3)
    result = apply_rules(pd.DataFrame(rows), rules, layout=layout)
    wells = result.wells.set_index("well")
    glow = {well_name(r, c) for (r, c), a in layout.wells.items()
            if a.active_treatments and a.active_treatments[0].compound == "Glow"}  # fmt: skip
    auto = set(wells.index[wells.autofluorescence != ""])
    assert auto and auto <= glow and (wells.loc[sorted(auto), "autofluorescence"] == "c2").all()
    top = {well_name(0, c) for c in (1, 2, 3)} | {well_name(1, c) for c in (1, 2, 3)}
    assert top <= auto  # the three highest doses in both replicates
    crash = set(wells.index[wells.precipitation])
    assert crash and all(w[0] in "CD" for w in crash)
    assert not wells.loc[[well_name(4, c) for c in range(1, 9)], "flags"].astype(bool).any()
    assert not wells.failed.any()  # flagged, not failed (CompoundChecks.exclude is off)
    assert result.plate["compound_flags"]["precipitation"] == len(crash)


def test_registry_keeps_the_run_qc_and_skips_failed_wells(home: Path) -> None:
    from hcs_studio.db import session_scope
    from hcs_studio.db.catalog import Run, WellResult
    from hcs_studio.pharmacology.synthetic import write_run
    from hcs_studio.services import context
    from hcs_studio.services.registry import scan_folders

    run = write_run(home / "runs" / "R1__run")
    summary = pd.read_csv(run / "tables" / "per_well_summary.csv")
    summary["qc_failed"] = summary["well"].isin(["A01", "B05"])
    summary.to_csv(run / "tables" / "per_well_summary.csv", index=False)
    _write_qc(run, {"A01": "too few good fields", "B05": "debris"})
    scan_folders(context(), [home / "runs"])
    with session_scope() as s:
        record = s.query(Run).one()
        assert record.image_qc["passed"] is True and record.image_qc["rules"]["name"] == "test"
        flags = {
            r.well: r.qc_flag for r in s.query(WellResult).filter_by(readout_id="nuclei_count")
        }
        assert flags["A01"] == "qc_failed" and flags["A02"] == ""
        assert not s.query(WellResult).filter_by(readout_id="qc_failed").count()
