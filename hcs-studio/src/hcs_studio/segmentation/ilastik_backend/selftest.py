"""Test an ilastik installation (PROMPT_ILASTIK.md §4 "Test installation"): run a trained Pixel
Classification / Autocontext project on a small synthetic image through every runner that is
available here (headless, Fiji, Python API) and compare each with the headless result.

HCS Studio cannot ship a trained project (one can only be made in ilastik), so the check uses a
project the user chooses; without one it only checks that ilastik starts and reports its version.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TOLERANCE = 1e-4  # probabilities: float32 output of two routes through the same classifier


@dataclass
class RunnerCheck:
    runner: str
    ok: bool
    seconds: float = 0.0
    max_difference: float | None = None  # vs headless
    message: str = ""


@dataclass
class InstallationCheck:
    ilastik: str = ""
    version: str | None = None
    project: str = ""
    checks: list[RunnerCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def text(self) -> str:
        lines = [f"ilastik {self.version or '?'}: {self.ilastik or 'not found'}"]
        if self.project:
            lines.append(f"Project: {Path(self.project).name}")
        for c in self.checks:
            status = "PASS" if c.ok else "FAIL"
            diff = (
                f", max difference vs headless {c.max_difference:.2g}"
                if c.max_difference is not None
                else ""
            )
            lines.append(f"{status} {c.runner} ({c.seconds:.1f} s{diff}) {c.message}".rstrip())
        lines += self.notes
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def _image(info: Any, size: int = 64) -> tuple[np.ndarray, str]:
    """A synthetic (T=1, C, Z, Y, X) image with the project's dimensionality and channels."""
    from hcs_studio.segmentation.ilastik_backend.axes import ilastik_axes

    channels = int(info.channels or 1)
    planes = 8 if info.axes and "z" in info.axes else 1
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    blobs = np.zeros((size, size))
    for _ in range(6):
        cy, cx = rng.integers(8, size - 8, 2)
        blobs += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0)
    plane = 100 + 900 * np.clip(blobs, 0, 1) + rng.normal(0, 10, (size, size))
    data = np.broadcast_to(plane, (1, channels, planes, size, size)).astype(np.float32).copy()
    return data, ilastik_axes(data.shape)


def check_installation(
    project: str | Path | None = None,
    runners: list[str] | None = None,
    settings: Any = None,
) -> InstallationCheck:
    """``settings``: the ilastik settings to test (default: the saved ones)."""
    from hcs_studio.segmentation.ilastik_backend.axes import from_ilastik, to_ilastik
    from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, read_h5, write_h5
    from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob
    from hcs_studio.segmentation.ilastik_backend.project import read_project
    from hcs_studio.segmentation.ilastik_backend.runners import (
        FijiRunner,
        HeadlessRunner,
        PythonApiRunner,
        api_available,
        is_python,
    )
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings, detect

    settings = settings or IlastikSettings.load()
    found = detect(settings)
    out = InstallationCheck(ilastik=found.ilastik, version=found.ilastik_version)
    if not found.ilastik_ok:
        out.notes.append(found.version_check.message if found.ilastik else "ilastik was not found")
        return out
    if project is None:
        out.notes.append(
            "ilastik starts. Choose a trained Pixel Classification project to test running it "
            "through every runner."
        )
        return out
    info = read_project(Path(project))
    out.project = str(project)
    if info.workflow not in ("pixel", "autocontext"):
        out.notes.append("Choose a Pixel Classification or Autocontext project for this check")
        return out
    available: dict[str, Any] = {"headless": HeadlessRunner(found.ilastik)}
    if found.fiji_ok and not is_python(found.ilastik):
        available["fiji"] = FijiRunner(
            found.fiji_launcher, found.ilastik, configure=settings.fiji_configure
        )
    elif found.fiji:
        out.notes.append("Fiji runner not tested (needs Fiji with ilastik4ij and the program)")
    if is_python(found.ilastik):
        if api_available(found.ilastik):
            available["python_api"] = PythonApiRunner(found.ilastik)
        else:
            out.notes.append("Python API runner not tested: no ilastik.experimental.api")
    if runners:
        available = {k: v for k, v in available.items() if k in runners}
    data, axes = _image(info)
    source = "Probabilities" if info.workflow == "pixel" else "Probabilities Stage 2"
    results: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="hcs-ilastik-test-") as tmp:
        spec = input_spec(write_h5(Path(tmp) / "test.h5", to_ilastik(data, axes), axes))
        for name, runner in available.items():
            job = IlastikJob(
                project=str(project),
                workflow=info.workflow,
                raw_data=[spec],
                export_source=source,
                output_folder=str(Path(tmp) / name),
                output_suffix="Probabilities",
                threads=settings.threads,
                ram_mb=settings.ram_mb,
                timeout_s=1800,
            )
            started = time.monotonic()
            try:
                result = runner.run(job)
                array, out_axes = read_h5(Path(result.outputs[0]))
                results[name] = np.asarray(from_ilastik(array, out_axes), dtype=np.float64)
                check = RunnerCheck(name, True, round(time.monotonic() - started, 1))
            except (IlastikError, OSError, ValueError) as exc:
                check = RunnerCheck(name, False, round(time.monotonic() - started, 1),
                                    message=str(exc).splitlines()[0])  # fmt: skip
            out.checks.append(check)
    reference = results.get("headless")
    for check in out.checks:
        if check.runner == "headless" or reference is None or check.runner not in results:
            continue
        other = results[check.runner]
        if other.shape != reference.shape:
            check.ok, check.message = False, f"shape {other.shape} ≠ headless {reference.shape}"
            continue
        check.max_difference = float(np.max(np.abs(other - reference)))
        if check.max_difference > TOLERANCE:
            check.ok = False
            check.message = f"differs from headless by more than {TOLERANCE:g}"
    return out
