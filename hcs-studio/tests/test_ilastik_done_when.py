"""Phase 16 done-when (PROMPT_ILASTIK.md §9): a 384-well projected plate runs through ilastik
Pixel Classification → objects → object analyzer unattended, resumes after a forced kill, and
produces the run-folder structure; the manifest records the project's hash, ilastik's version
and the runner. CI uses 24 wells of a 384-well plate; ``HCS_SLOW=1`` runs all 384."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from tests import test_ilastik_steps
from tests.test_ilastik_steps import calls

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)


def _well_of(call: dict[str, Any]) -> str:
    spec = next(a for a in call["argv"] if a.endswith(".h5/data"))
    return Path(spec).parent.parent.name  # <run>/ilastik/inputs/<well>/fNNN_t0.h5/data


@pytest.mark.skipif(sys.platform.startswith("win"), reason="process groups")
def test_384_well_projected_plate_unattended_with_kill_and_resume(
    ilastik: dict[str, Any], tmp_path: Path
) -> None:
    from hcs_studio.app import main
    from hcs_studio.segmentation.ilastik_backend.presets import pixel_preset
    from hcs_studio.segmentation.ilastik_backend.project import read_project
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    rows = 16 if os.environ.get("HCS_SLOW") else 2
    wells = [(r, c) for r in range(rows) for c in range(24)]
    plate = SyntheticPlate(
        wells=wells,
        channels=("DAPI",),
        size=64,
        planes=3,
        plate_size=(16, 24),
        name="ILA384",
        features=[Nuclei(count=lambda r, c, f: 3 + c % 4, intensity=2000), ShotNoise(sd=3)],
    )
    raw, truth = plate.write(tmp_path / "raw")
    project = ilastik["project"]
    pipeline = pixel_preset(project, read_project(project), channels=[1], projection=True)
    modules = [s.module for s in pipeline.steps]
    assert modules[:3] == ["projection", "ilastik_pixel", "probabilities_to_objects"]
    assert "measure_objects" in modules
    pipeline_file = tmp_path / "ilastik.yaml"
    pipeline_file.write_text(yaml.safe_dump(pipeline.to_dict(), sort_keys=False))
    out = tmp_path / "runs"
    command = [
        sys.executable, "-m", "hcs_studio", "run", "--pipeline", str(pipeline_file),
        "--input", str(raw), "--output", str(out), "--workers", "2", "--no-checksums",
        "--no-images",
    ]  # fmt: skip
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    deadline = time.monotonic() + 300
    folder = None
    while time.monotonic() < deadline and process.poll() is None:
        found = list(out.glob("*/wells/*/well_summary.json"))
        if len(found) >= 3:
            folder = found[0].parents[2]
            break
        time.sleep(0.1)
    os.killpg(process.pid, signal.SIGKILL)  # the whole process group, workers and ilastik
    process.wait(timeout=30)
    assert folder is not None, "no wells finished before the deadline"
    done_before = {p.parent.name for p in folder.glob("wells/*/well_summary.json")}
    assert 3 <= len(done_before) < len(wells), "the run should have been interrupted part-way"
    assert json.loads((folder / "run_manifest.json").read_text())["status"] == "running"

    assert main(["run", "--resume", str(folder), "--workers", "2", "--no-checksums"]) == 0
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["history"]) == 2
    per_well = Counter(_well_of(c) for c in calls(ilastik["log"]))
    assert set(per_well) == {f"{'ABCDEFGHIJKLMNOP'[r]}{c + 1:02d}" for r, c in wells}
    assert all(per_well[w] == 1 for w in done_before)  # finished wells were not redone

    # run-folder structure (PROMPT_ILASTIK.md §8) and provenance
    used = json.loads((folder / "ilastik" / "projects_used.json").read_text())
    assert used[0]["runner"] == "headless" and used[0]["ilastik_version"] == "1.4.2"
    assert len(list((folder / "ilastik" / "logs").glob("*.log"))) == len(wells)
    assert not any((folder / "ilastik" / "inputs").rglob("*.h5"))
    assert len(list(folder.glob("wells/*/ilastik/*_Probabilities.h5"))) == len(wells)
    assert len(list(folder.glob("wells/*/ilastik/class_summary.json"))) == len(wells)
    fitted = next(s for s in manifest["fitted_state"] if s["module"] == "ilastik_pixel")
    assert fitted["sha256"] == used[0]["sha256"] and fitted["runner"] == "headless"
    assert fitted["ilastik_version"] == "1.4.2"
    # the object analyzer's counts equal the synthetic truth in every well
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    for name, expected in truth["wells"].items():
        assert summary.loc[name, "nuclei_objects_measured"] == expected["counts"]["nuclei"], name
