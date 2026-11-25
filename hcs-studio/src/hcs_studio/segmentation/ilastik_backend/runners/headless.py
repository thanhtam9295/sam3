"""HeadlessRunner: ilastik's own command line in a subprocess (PROMPT_ILASTIK.md §3.1).

``<ilastik> --headless --project=<ilp> --export_source=<source> --output_format=hdf5
--output_filename_format=<folder>/{nickname}_<suffix>.h5 --raw_data <in1> <in2> …``

All the fields of a well go into one call (``--raw_data`` takes several inputs), which saves
ilastik's start-up time (tens of seconds) per field. ilastik is never imported: its pinned
dependencies stay in its own installation. Threads and RAM limits go through
``LAZYFLOW_THREADS`` / ``LAZYFLOW_TOTAL_RAM_MB``.

Headless ilastik opens projects read-only; the project's SHA-256 is compared before and after
every call, and a changed project is an error.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob, IlastikResult
from hcs_studio.segmentation.ilastik_backend.project import file_sha256
from hcs_studio.segmentation.ilastik_backend.settings import ilastik_command

LOG_TAIL_LINES = 40


class HeadlessRunner:
    name = "headless"

    def __init__(self, executable: str | Path) -> None:
        self.executable = str(executable)

    def command(self, job: IlastikJob) -> list[str]:
        pattern = str(
            Path(job.output_folder) / f"{{nickname}}_{job.output_suffix}.{job.output_extension}"
        )
        command = [
            *ilastik_command(self.executable),
            "--headless",
            f"--project={job.project}",
            f"--export_source={job.export_source}",
            f"--output_format={job.output_format}",
            f"--output_filename_format={pattern}",
            *([f"--export_plugin={job.export_plugin}"] if job.export_plugin else []),
            "--raw_data",
            *job.raw_data,
        ]
        for role, specs in sorted(job.roles.items()):
            if len(specs) != len(job.raw_data):
                raise ValueError(f"--{role} needs one input per raw image")
            command += [f"--{role}", *specs]
        return command

    def environment(self, job: IlastikJob) -> dict[str, str]:
        env = dict(os.environ)
        if job.threads != -1:
            env["LAZYFLOW_THREADS"] = str(job.threads)
        if job.ram_mb:
            env["LAZYFLOW_TOTAL_RAM_MB"] = str(job.ram_mb)
        return env

    def run(self, job: IlastikJob, log_path: Path | None = None) -> IlastikResult:
        Path(job.output_folder).mkdir(parents=True, exist_ok=True)
        command = self.command(job)
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
            )
        except FileNotFoundError as exc:
            raise IlastikError(
                f"ilastik could not be started ({self.executable}): set its path in Tools → "
                "ilastik settings"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise IlastikError(f"ilastik did not finish within {job.timeout_s:g} s") from exc
        seconds = time.monotonic() - started
        output = (done.stdout or "") + (done.stderr or "")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n" + output + "\n")
        tail = "\n".join(output.strip().splitlines()[-LOG_TAIL_LINES:])
        if done.returncode != 0:
            raise IlastikError(_explain(done.returncode, tail))
        after = file_sha256(Path(job.project))
        if after != before:
            raise IlastikError(
                f"ilastik changed the project file {job.project} during a headless run; "
                "results are not traceable to the recorded classifier"
            )
        outputs = [_find_output(Path(p)) for p in job.expected_outputs()]
        missing = [
            p for p, found in zip(job.expected_outputs(), outputs, strict=True) if found is None
        ]
        if missing:
            raise IlastikError(
                f"ilastik finished but wrote no {job.export_source!r} output for "
                f"{len(missing)} input(s), e.g. {Path(missing[0]).name}\n{tail}"
            )
        return IlastikResult(
            outputs=[str(p) for p in outputs if p is not None],
            command=command,
            runner=self.name,
            returncode=done.returncode,
            seconds=round(seconds, 2),
            log_tail=tail,
            project_sha256_before=before,
            project_sha256_after=after,
        )


def _find_output(expected: Path) -> Path | None:
    """The export file for one input. ilastik's ``{nickname}`` of an HDF5 input may carry the
    internal dataset name as well (unconfirmed): accept ``<stem>*_<suffix>.h5``."""
    if expected.exists():
        return expected
    stem, _, suffix = expected.stem.rpartition("_")
    matches = sorted(expected.parent.glob(f"{stem}*_{suffix}{expected.suffix}"))
    return matches[0] if len(matches) == 1 else None


def _explain(code: int, tail: str) -> str:
    hints = {
        1: "ilastik reported an error",
        2: "ilastik did not accept the command line (an option this version does not know?)",
    }
    text = hints.get(code, f"ilastik stopped with exit code {code}")
    if "MemoryError" in tail or "Cannot allocate memory" in tail:
        text = (
            "ilastik ran out of memory: lower the RAM limit in the ilastik settings or the "
            "field size"
        )
    return f"{text}\n{tail}" if tail else text
