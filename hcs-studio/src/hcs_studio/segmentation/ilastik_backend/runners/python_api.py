"""PythonApiRunner (PROMPT_ILASTIK.md §3.2): Pixel Classification / Autocontext probabilities
through ilastik's Python API, in ilastik's **own** conda environment (a subprocess running
``api_worker.py``); ilastik is never imported into HCS Studio.

Needs the ilastik "program" setting to be the Python of an ilastik environment. One worker
call per well loads the project once and predicts every field. Jobs the API does not cover
(Simple Segmentation, other workflows, prediction masks) run headless; when the installed
ilastik has no ``ilastik.experimental.api``, the step falls back to the headless runner.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob, IlastikResult
from hcs_studio.segmentation.ilastik_backend.project import file_sha256
from hcs_studio.segmentation.ilastik_backend.settings import ilastik_command

LOG_TAIL_LINES = 40
WORKER = Path(__file__).resolve().parent.parent / "api_worker.py"
#: what the API computes: workflow → export source
SOURCES = {"pixel": "Probabilities", "autocontext": "Probabilities Stage 2"}
NO_API = 3
_AVAILABLE: dict[str, bool] = {}


def is_python(executable: str | Path) -> bool:
    return len(ilastik_command(executable)) > 1


def api_available(python: str | Path, timeout: float = 300) -> bool:
    """Whether this Python imports ``ilastik.experimental.api`` (cached per interpreter)."""
    key = str(python)
    if key not in _AVAILABLE:
        try:
            done = subprocess.run(
                [key, "-c", "import ilastik.experimental.api"],
                capture_output=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            _AVAILABLE[key] = done.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _AVAILABLE[key] = False
    return _AVAILABLE[key]


class PythonApiRunner:
    name = "python_api"

    def __init__(self, python: str | Path) -> None:
        self.python = str(python)

    def unsupported(self, job: IlastikJob) -> str | None:
        if not is_python(self.python):
            return "the Python API runner needs the Python of an ilastik environment"
        if SOURCES.get(job.workflow) != job.export_source:
            return f"the Python API gives probabilities only (not {job.export_source!r})"
        if job.roles:
            return "the Python API takes no second input (e.g. prediction masks)"
        if job.output_format != "hdf5" or job.export_plugin:
            return "the Python API runner writes HDF5 probabilities only"
        return None

    def environment(self, job: IlastikJob) -> dict[str, str]:
        env = dict(os.environ)
        if job.threads != -1:
            env["LAZYFLOW_THREADS"] = str(job.threads)
        if job.ram_mb:
            env["LAZYFLOW_TOTAL_RAM_MB"] = str(job.ram_mb)
        return env

    def run(self, job: IlastikJob, log_path: Path | None = None) -> IlastikResult:
        reason = self.unsupported(job)
        if reason:
            raise IlastikError(f"The Python API runner cannot run this step: {reason}")
        folder = Path(job.output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        outputs = job.expected_outputs()
        for output in outputs:
            Path(output).unlink(missing_ok=True)
        spec = {
            "project": job.project,
            "workflow": job.workflow,
            "inputs": [[raw, out] for raw, out in zip(job.raw_data, outputs, strict=True)],
        }
        job_file = folder / f"api_{job.output_suffix}.json"
        job_file.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        command = [self.python, str(WORKER), str(job_file)]
        before = file_sha256(Path(job.project))
        started = time.monotonic()
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=self.environment(job),
                timeout=job.timeout_s,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise IlastikError(f"ilastik's Python could not be started ({self.python})") from exc
        except subprocess.TimeoutExpired as exc:
            raise IlastikError(f"ilastik did not finish within {job.timeout_s:g} s") from exc
        seconds = time.monotonic() - started
        output = (done.stdout or "") + (done.stderr or "")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n" + output + "\n")
        tail = "\n".join(output.strip().splitlines()[-LOG_TAIL_LINES:])
        if done.returncode == NO_API:
            raise IlastikError(f"The installed ilastik has no usable Python API\n{tail}")
        if done.returncode != 0:
            raise IlastikError(f"ilastik's Python API failed (exit {done.returncode})\n{tail}")
        after = file_sha256(Path(job.project))
        if after != before:
            raise IlastikError(f"The project file {job.project} changed during the run")
        missing = [p for p in outputs if not Path(p).exists()]
        if missing:
            raise IlastikError(f"No output for {len(missing)} input(s)\n{tail}")
        return IlastikResult(
            outputs=outputs,
            command=command,
            runner=self.name,
            returncode=done.returncode,
            seconds=round(seconds, 2),
            log_tail=tail,
            project_sha256_before=before,
            project_sha256_after=after,
        )
