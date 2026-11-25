"""Acceptance tests on real data (PROMPT.md §10): one function per criterion, each
returning a report with PASS/FAIL per item, and written as JSON by ``hcs-studio
acceptance``. They read run folders and exported files; they never change raw data.

* ``counts``: object counts per field against manual counts (±5 %).
* ``coloc``: Pearson / Manders against Fiji Coloc 2 on the exported inputs (±0.02).
* ``run``: a run folder is complete and has the layout of PROMPT.md §7; the report works
  offline (no external resources; every linked preview exists).
* ``kill_resume``: an unattended run is killed part-way, resumed, and completes.
* ``beads``: deconvolution narrows the measured FWHM of beads.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Item:
    name: str
    passed: bool
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    check: str
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(i.passed for i in self.items)

    def add(self, name: str, passed: bool, detail: str, **values: Any) -> None:
        self.items.append(Item(name, bool(passed), detail, values))

    def text(self) -> str:
        lines = [f"Acceptance check: {self.check}"]
        lines += [f"  {'PASS' if i.passed else 'FAIL'}  {i.name}: {i.detail}" for i in self.items]
        lines += [f"  note: {n}" for n in self.notes]
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        payload = {"check": self.check, "passed": self.passed, **asdict(self)}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------- counts


def check_counts(
    run: Path, manual_csv: Path, object_type: str = "nuclei", tolerance: float = 0.05
) -> Report:
    """Compare the objects counted per field with manual counts.

    ``manual_csv`` columns: ``well``, ``field`` (as in the run, starting at 1) and
    ``count``; optional ``timepoint`` (default 0). Each annotated field passes when
    |automatic − manual| ≤ ``tolerance`` × manual.
    """
    report = Report("counts")
    manual = pd.read_csv(manual_csv)
    missing = {"well", "field", "count"} - set(manual.columns)
    if missing:
        raise ValueError(f"{manual_csv.name} needs the columns well, field, count")
    if "timepoint" not in manual:
        manual["timepoint"] = 0
    column = f"n_{object_type}"
    parts = [pd.read_csv(p) for p in sorted(Path(run).glob("wells/*/fields.csv"))]
    if not parts:
        raise ValueError(f"{run} has no fields.csv tables (add the QC step to the pipeline)")
    fields = pd.concat(parts, ignore_index=True)
    if column not in fields:
        raise ValueError(f"The run counted no '{object_type}' objects (no {column} column)")
    automatic = fields.set_index(["well", "field", "timepoint"])[column]
    errors = []
    for record in manual.to_dict("records"):
        key = (str(record["well"]), int(record["field"]), int(record["timepoint"]))
        expected = float(record["count"])
        found = automatic.get(key)
        name = f"{key[0]} field {key[1]}" + (f" t{key[2]}" if key[2] else "")
        if found is None or (isinstance(found, float) and math.isnan(found)):
            report.add(name, False, "field not in the run", manual=expected)
            continue
        error = (float(found) - expected) / expected if expected else math.inf
        errors.append(error)
        report.add(
            name,
            abs(error) <= tolerance,
            f"{int(found)} found, {int(expected)} counted by hand ({100 * error:+.1f} %)",
            automatic=int(found),
            manual=expected,
            relative_error=error,
        )
    if errors:
        report.notes.append(
            f"mean error {100 * float(np.mean(errors)):+.1f} %, mean |error| "
            f"{100 * float(np.mean(np.abs(errors))):.1f} % over {len(errors)} fields "
            f"(tolerance ±{100 * tolerance:.0f} %)"
        )
    if len(manual) < 10:
        report.notes.append(f"§10 asks for 10 annotated fields; {len(manual)} given")
    return report


# ------------------------------------------------------------------------ coloc

#: Coloc 2 log labels → HCS Studio result keys. Coloc 2 writes "<label>, <value>" lines.
COLOC2_LABELS = {
    "pearson's r value (no threshold)": "pearson",
    "manders' m1 (above zero intensity of ch2)": "m1",
    "manders' m2 (above zero intensity of ch1)": "m2",
    "manders' tm1 (above autothreshold of ch2)": "tm1",
    "manders' tm2 (above autothreshold of ch1)": "tm2",
    "li's icq value": "icq",
    "spearman's rank correlation value": "spearman",
    "costes p-value": "costes_p",
}
#: Compared against the ±tolerance criterion; the others are reported for information.
COLOC_CRITERION = ("pearson", "m1", "m2", "tm1", "tm2")


def parse_coloc2_log(text: str) -> dict[str, float]:
    """Values from Fiji's Log after Coloc 2 (lines ``label, value``)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(.+?)\s*[,=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line)
        if not match:
            continue
        label = match.group(1).strip().lower().replace("’", "'")
        key = COLOC2_LABELS.get(label)
        if key and key not in values:
            values[key] = float(match.group(2))
    return values


def check_coloc(folders: list[Path], tolerance: float = 0.02) -> Report:
    """Each folder was written by **Export for Fiji…** and holds ``hcs_studio_results.json``
    and ``coloc2_log.txt`` (saved by the exported macro after running Coloc 2)."""
    report = Report("coloc")
    for folder in folders:
        folder = Path(folder)
        ours = json.loads((folder / "hcs_studio_results.json").read_text(encoding="utf-8"))
        log = folder / "coloc2_log.txt"
        if not log.exists():
            report.add(folder.name, False, "coloc2_log.txt missing: run coloc2_compare.ijm in Fiji")
            continue
        fiji = parse_coloc2_log(log.read_text(encoding="utf-8", errors="replace"))
        if not fiji:
            report.add(folder.name, False, "no Coloc 2 values found in coloc2_log.txt")
            continue
        results = ours["results"]
        for key in COLOC_CRITERION:
            if key not in fiji or results.get(key) is None:
                report.notes.append(f"{folder.name}: {key} not available from both programs")
                continue
            difference = float(results[key]) - fiji[key]
            report.add(
                f"{folder.name} {key}",
                abs(difference) <= tolerance,
                f"HCS Studio {float(results[key]):.4f}, Coloc 2 {fiji[key]:.4f} "
                f"(difference {difference:+.4f})",
                hcs_studio=float(results[key]),
                coloc2=fiji[key],
            )
        for key in ("icq", "spearman"):
            if key in fiji and results.get(key) is not None:
                report.notes.append(
                    f"{folder.name}: {key} HCS Studio {float(results[key]):.4f}, Coloc 2 "
                    f"{fiji[key]:.4f} (not part of the criterion)"
                )
    return report


# -------------------------------------------------------------------------- run

#: PROMPT.md §7: files every run folder must have.
RUN_FILES = ("run_manifest.json", "pipeline.yaml", "index.html", "logs/run.log")
TABLE_FILES = ("tables/per_well_summary.csv",)
WELL_FILES = ("well_summary.json",)


def check_run(run: Path, expect_resumed: bool = False) -> Report:
    """The run finished, every well is complete and the folder has the §7 layout; the
    report has no external resources and every image it links exists."""
    from hcs_studio.output.results import RunResults
    from hcs_studio.pipeline.executor import well_is_complete

    run = Path(run)
    report = Report("run")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    report.add("status", manifest.get("status") == "completed", f"run {manifest.get('status')}")
    missing = [f for f in (*RUN_FILES, *TABLE_FILES) if not (run / f).exists()]
    report.add(
        "run folder layout", not missing, "all present" if not missing else f"missing {missing}"
    )
    wells = list(manifest.get("wells", []))
    incomplete = [w for w in wells if not well_is_complete(run, w, manifest["pipeline_hash"])]
    report.add(
        "wells complete",
        not incomplete and bool(wells),
        f"{len(wells) - len(incomplete)} of {len(wells)}"
        + (f"; incomplete: {incomplete[:10]}" if incomplete else ""),
    )
    failed = manifest.get("failed_wells", {})
    report.add(
        "no failed wells",
        not failed,
        f"{len(failed)} failed" + (f": {list(failed)[:10]}" if failed else ""),
    )
    per_well_missing = []
    for well in wells:
        directory = run / "wells" / well
        for name in WELL_FILES:
            if not (directory / name).exists():
                per_well_missing.append(f"{well}/{name}")
        if not list((directory / "previews").glob("f*.png")):
            per_well_missing.append(f"{well}/previews")
    report.add(
        "well folders",
        not per_well_missing,
        "previews and summaries present"
        if not per_well_missing
        else f"missing {per_well_missing[:10]}",
    )
    leftovers = [str(p.relative_to(run)) for p in run.rglob("*.part")]
    report.add("no partial files", not leftovers, "none" if not leftovers else f"{leftovers[:5]}")
    if (run / "tables" / "per_object_all.parquet").exists() or any(
        run.glob("wells/*/objects.parquet")
    ):
        report.add(
            "object table",
            (run / "tables" / "per_object_all.parquet").exists(),
            "tables/per_object_all.parquet",
        )
    if (run / "index.html").exists():
        html = (run / "index.html").read_text(encoding="utf-8")
        external = re.findall(r'(?:src|href)\s*=\s*"(https?:[^"]+)"', html)
        external += re.findall(r"@import|<link[^>]+stylesheet", html)
        report.add(
            "report is self-contained",
            not external,
            "no external resources" if not external else f"{external[:3]}",
        )
        results = RunResults(run)
        linked = [
            path
            for well in results.wells
            for f in results.well_fields(well)
            for path in (f.enhanced, f.corrected, f.raw, *f.overlays.values())
            if path
        ]
        broken = [p for p in linked if not (run / p).exists()]
        report.add(
            "report previews",
            bool(linked) and not broken,
            f"{len(linked)} linked images exist" if not broken else f"missing {broken[:5]}",
        )
        report.notes.append(
            "Open index.html from a copy of the folder (or a zip) with the network off and "
            "click a few wells: that is the interactive half of this criterion."
        )
    history = manifest.get("history", [])
    if expect_resumed:
        report.add(
            "resumed after interruption",
            len(history) >= 2 and history[0].get("status") is None,
            f"{len(history)} attempts recorded; first attempt "
            f"{history[0].get('status') or 'never finished (killed)'}"
            if history
            else "no history",
        )
    return report


# ------------------------------------------------------------------ kill/resume


def check_kill_resume(
    pipeline: Path,
    plate: Path,
    output: Path,
    *,
    kill_after_wells: int = 10,
    workers: str = "auto",
    layout: Path | None = None,
    timeout_s: float = 24 * 3600,
) -> Report:
    """Start an unattended CLI run, kill it (with its workers) once ``kill_after_wells``
    wells are complete, resume it, and check the finished run folder."""
    command = (
        [sys.executable, "-m", "hcs_studio"]
        if not getattr(sys, "frozen", False)
        else [sys.executable]
    )
    base = [*command, "run", "--pipeline", str(pipeline), "--input", str(plate)]
    base += ["--output", str(output), "--workers", workers, "--no-checksums"]
    if layout is not None:
        base += ["--layout", str(layout)]
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.iterdir())
    options: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined,unused-ignore]
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options
    )
    deadline = time.monotonic() + timeout_s
    folder = None
    while time.monotonic() < deadline and process.poll() is None:
        new = [p for p in output.iterdir() if p not in before and p.is_dir()]
        if new:
            folder = new[0]
            if len(list(folder.glob("wells/*/well_summary.json"))) >= kill_after_wells:
                break
        time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(
            "the run finished (or failed) before it could be killed; use a larger plate or a "
            "smaller --kill-after"
        )
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=120)
    if folder is None:
        raise RuntimeError("no run folder appeared before the deadline")
    killed_at = len(list(folder.glob("wells/*/well_summary.json")))
    resumed = subprocess.run(
        [*command, "run", "--resume", str(folder), "--workers", workers, "--no-checksums"]
    )
    report = check_run(folder, expect_resumed=True)
    report.check = "kill_resume"
    report.items.insert(
        0,
        Item(
            "killed part-way",
            killed_at > 0,
            f"killed after {killed_at} complete wells",
            {"killed_after": killed_at},
        ),
    )
    report.items.insert(
        1, Item("resume exit code", resumed.returncode == 0, f"exit code {resumed.returncode}")
    )
    report.notes.append(f"run folder: {folder}")
    return report


# ------------------------------------------------------------------------ beads


def check_beads(
    path: Path,
    channel: int = 1,
    *,
    psf_file: Path | None = None,
    iterations: int = 40,
    position: int = 0,
) -> Report:
    """Deconvolve a bead Z stack and compare the averaged bead's FWHM before and after.

    The PSF is theoretical (from the metadata, with the user's overrides) unless
    ``psf_file`` (saved by Tools → PSF tool, e.g. from a separate bead stack) is given.
    """
    from hcs_studio.io.base import open_plate
    from hcs_studio.io.sidecar import apply_overrides
    from hcs_studio.processing import psf as psf_tools
    from hcs_studio.processing.deconvolution import deconvolve

    report = Report("beads")
    plate = apply_overrides(open_plate(Path(path)))
    metadata = plate.metadata
    if metadata.pixel_size_um is None:
        raise ValueError("The pixel size is unknown: set it in the Metadata panel first")
    fields = [f for key in sorted(plate.wells) for f in plate.wells[key].fields]
    stack = np.asarray(fields[position].load()[0, channel - 1], dtype=np.float32)  # (Z, Y, X)
    voxel = (metadata.z_step_um, metadata.pixel_size_um, metadata.pixel_size_um)
    if psf_file is not None:
        kernel, psf_xy, psf_z = psf_tools.load_psf(Path(psf_file))
        if psf_xy is not None:  # resample to the image's voxel size
            kernel = psf_tools.resample_psf(kernel, (psf_z, psf_xy, psf_xy), voxel)
        source = f"measured PSF {Path(psf_file).name}"
    else:
        spec, assumptions = psf_tools.spec_from_metadata(metadata, channel - 1)
        kernel = psf_tools.theoretical_psf(spec, three_d=stack.shape[0] > 1)
        source = "theoretical PSF from the metadata"
        report.notes += [f"PSF assumption: {a}" for a in assumptions]
    # Isolated beads, found on the raw stack; FWHM measured on each bead at the same voxel
    # before and after deconvolution, without resampling (sub-pixel re-centring would blur
    # a sharpened bead by about a pixel and hide the gain).
    nz = stack.shape[0]
    reach = int(np.ceil(3.0 / metadata.z_step_um)) if metadata.z_step_um else nz
    size_z = min(2 * reach + 1, nz if nz % 2 else nz - 1)  # ±3 µm: beads at any depth
    found = psf_tools.extract_psf_from_beads(stack, size_z=max(1, size_z))
    result = deconvolve(
        stack,
        kernel,
        iterations=iterations,
        z_to_xy_ratio=(metadata.pixel_size_um / metadata.z_step_um) if metadata.z_step_um else 1.0,
    )
    widths: dict[str, list[tuple[float, float]]] = {"x": [], "y": [], "z": []}
    for z, y, x in found.positions_zyx:
        for axis in widths:
            if axis == "z" and (stack.shape[0] < 3 or metadata.z_step_um is None):
                continue
            before = _bead_fwhm(stack, (z, y, x), axis, voxel)
            after = _bead_fwhm(result.image, (z, y, x), axis, voxel)
            if before is not None and after is not None:
                widths[axis].append((before, after))
    report.notes.append(
        f"{source}; {len(found.positions_zyx)} isolated beads ({found.n_rejected} rejected); "
        f"{result.iterations} iterations on {result.backend}"
    )
    for axis, pairs in widths.items():
        if not pairs:
            if axis != "z" or stack.shape[0] > 2:
                report.add(f"FWHM {axis}", False, "not measurable on any bead")
            continue
        b = float(np.median([p[0] for p in pairs]))
        a = float(np.median([p[1] for p in pairs]))
        report.add(
            f"FWHM {axis}",
            a < b,
            f"median of {len(pairs)} beads {1000 * b:.0f} nm → {1000 * a:.0f} nm "
            f"({100 * (a / b - 1):+.0f} %)",
            before_um=b,
            after_um=a,
            beads=len(pairs),
        )
    return report


def _bead_fwhm(
    volume: np.ndarray, centre: tuple[int, int, int], axis: str, voxel: tuple[Any, float, float]
) -> float | None:
    """FWHM (µm) along one axis through the brightest voxel near ``centre``."""
    from hcs_studio.processing.psf import profile_fwhm

    z, y, x = centre
    window = volume[max(0, z - 1) : z + 2, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
    dz, dy, dx = (int(v) for v in np.unravel_index(int(np.argmax(window)), window.shape))
    z, y, x = max(0, z - 1) + dz, max(0, y - 2) + dy, max(0, x - 2) + dx
    half = 12
    if axis == "x":
        profile, spacing = volume[z, y, max(0, x - half) : x + half + 1], voxel[2]
    elif axis == "y":
        profile, spacing = volume[z, max(0, y - half) : y + half + 1, x], voxel[1]
    else:
        profile, spacing = volume[max(0, z - half) : z + half + 1, y, x], voxel[0]
    return profile_fwhm(profile, float(spacing))
