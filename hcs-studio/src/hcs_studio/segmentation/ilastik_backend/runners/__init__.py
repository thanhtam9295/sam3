"""ilastik runners: each takes an ``IlastikJob`` and returns an ``IlastikResult``.

Steps call ``runner_state()`` in ``fit`` (which ilastik, which runner, threads/RAM → the
manifest) and ``run_job(state, job, log)`` per call. With the Fiji runner selected, a job the
ilastik4ij plugin cannot do runs headless instead; the result's ``note`` and the log say so.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob, IlastikResult
from hcs_studio.segmentation.ilastik_backend.runners.fiji import FijiRunner
from hcs_studio.segmentation.ilastik_backend.runners.headless import HeadlessRunner

__all__ = ["FijiRunner", "HeadlessRunner", "make_runner", "run_job", "runner_state"]


def runner_state() -> dict[str, Any]:
    """The installation and runner a run uses (from the ilastik settings); raises
    ``IlastikError`` with a message for users when it cannot run."""
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings, detect

    settings = IlastikSettings.load()
    override = os.environ.get("HCS_ILASTIK_RUNNER", "")
    if override:  # one run with another runner (CLI ``--runner``), settings unchanged
        if override not in ("headless", "fiji"):
            raise IlastikError(f"HCS_ILASTIK_RUNNER must be headless or fiji, not {override!r}")
        settings = replace(settings, runner=override)  # type: ignore[arg-type]
    found = detect(settings)
    if not found.ilastik_ok:
        raise IlastikError(
            "ilastik is not available: "
            + (
                found.version_check.message
                if found.ilastik
                else "set its path in Tools → ilastik settings"
            )
        )
    state: dict[str, Any] = {
        "ilastik": found.ilastik,
        "ilastik_version": found.ilastik_version,
        "runner": settings.runner,
        "threads": settings.threads,
        "ram_mb": settings.ram_mb,
    }
    if settings.runner == "fiji":
        if not found.fiji_ok:
            raise IlastikError(
                "The Fiji runner is selected, but "
                + (
                    "Fiji was not found"
                    if not found.fiji_launcher
                    else "Fiji has no ilastik4ij plugin (Help → Update → Manage update sites → "
                    "ilastik)"
                )
                + ": set Fiji's folder in Tools → ilastik settings or choose the headless runner"
            )
        problem = FijiRunner.check_ilastik(found.ilastik)
        if problem:
            raise IlastikError(problem)
        state.update(
            fiji=found.fiji_launcher,
            ilastik4ij=Path(found.ilastik4ij).name,
            fiji_configure=settings.fiji_configure,
        )
    return state


def make_runner(state: dict[str, Any], job: IlastikJob) -> tuple[Any, str]:
    """The runner for one job, and a note when the chosen runner cannot run it."""
    if state.get("runner") == "fiji":
        fiji = FijiRunner(
            state["fiji"], state["ilastik"], configure=bool(state.get("fiji_configure", True))
        )
        reason = fiji.unsupported(job)
        if reason is None:
            return fiji, ""
        return HeadlessRunner(state["ilastik"]), f"Fiji runner not used: {reason}; ran headless"
    return HeadlessRunner(state["ilastik"]), ""


def run_job(state: dict[str, Any], job: IlastikJob, log_path: Path | None = None) -> IlastikResult:
    runner, note = make_runner(state, job)
    if note and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# {note}\n")
    result: IlastikResult = runner.run(job, log_path=log_path)
    return result.model_copy(update={"note": note}) if note else result
