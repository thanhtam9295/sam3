"""The job queue behind an interface (PROMPT_PLATFORM.md §2.5).

``JobQueue`` is what the GUI, the command line and (Phase 19) automatic runs talk to:
``submit / status / events / cancel / retry / resume / jobs``. ``LocalJobQueue`` runs jobs on this
computer, each in its own process (``pipeline.jobs.JobProcess``), one at a time by default,
highest priority first (interactive > auto > batch; a running job is never pre-empted).
It is headless and driven by :meth:`LocalJobQueue.poll` (the GUI calls it on a timer, the
command line in a loop). Job states are stored in the registry (``job`` table) so the list
survives restarts; jobs that were running when the app stopped are marked ``interrupted``.
Phase 27 adds a remote implementation of the same protocol.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from hcs_studio.model.plate import parse_well_name
from hcs_studio.pipeline.jobs import Job, JobProcess
from hcs_studio.services import Actor, os_actor

Priority = Literal["interactive", "auto", "batch"]
PRIORITY_RANK: dict[str, int] = {"interactive": 0, "auto": 1, "batch": 2}
#: queued, running, paused, completed, completed_with_failures, cancelled, error, interrupted
ACTIVE = ("running", "paused")
FINAL = ("completed", "completed_with_failures", "cancelled", "error", "interrupted")
MAX_EVENTS = 2000
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resources:
    cpu_workers: int | None = 1
    gpu: bool = False
    ram_gb: float | None = None


@dataclass(frozen=True)
class JobSpec:
    job: Job
    kind: str = "pipeline"
    priority: Priority = "interactive"
    owner: Actor = field(default_factory=os_actor)
    resources: Resources = field(default_factory=Resources)

    @property
    def label(self) -> str:
        return self.job.label

    @classmethod
    def for_job(cls, job: Job, **kwargs: Any) -> JobSpec:
        from hcs_studio.pipeline.schema import Pipeline

        gpu = False
        try:
            steps = Pipeline.model_validate(job.pipeline).image_steps()
            gpu = any(step.create().uses_gpu() for step in steps)
        except Exception:  # unknown module, bad params: the job itself reports it
            gpu = False
        resources = Resources(cpu_workers=job.workers, gpu=gpu, ram_gb=job.max_memory_gb)
        return cls(job=job, resources=resources, **kwargs)


@dataclass
class JobState:
    spec: JobSpec
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "queued"
    folder: str | None = None
    done: int = 0
    total: int = 0
    eta_s: float | None = None
    wells: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    message: str = ""
    created: datetime = field(default_factory=lambda: datetime.now(UTC))
    started: datetime | None = None
    finished: datetime | None = None
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def job(self) -> Job:
        return self.spec.job


class JobQueue(Protocol):
    def submit(self, spec: JobSpec) -> str: ...
    def status(self, job_id: str) -> JobState: ...
    def events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]: ...
    def cancel(self, job_id: str) -> None: ...
    def retry(self, job_id: str, failed_only: bool = True) -> str | None: ...
    def resume(self, job_id: str) -> str | None: ...
    def jobs(self) -> list[JobState]: ...


ProcessFactory = Callable[[Job], JobProcess]


class LocalJobQueue:
    def __init__(
        self,
        *,
        process_factory: ProcessFactory = JobProcess,
        max_running: int = 1,
        persist: bool = True,
    ) -> None:
        self.states: list[JobState] = []
        self.running: dict[str, JobProcess] = {}
        self.process_factory = process_factory
        self.max_running = max(1, max_running)
        self.persist = persist

    # ----------------------------------------------------------- protocol

    def submit(self, spec: JobSpec) -> str:
        state = JobState(spec=spec)
        self.states.append(state)
        self._save(state)
        self.start_next()
        return state.id

    def status(self, job_id: str) -> JobState:
        for state in self.states:
            if state.id == job_id:
                return state
        raise KeyError(f"No job {job_id}")

    def events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]:
        return list(self.status(job_id).events[since:])

    def jobs(self) -> list[JobState]:
        return list(self.states)

    def cancel(self, job_id: str) -> None:
        state = self.status(job_id)
        if state.status == "queued":
            state.status, state.message = "cancelled", "Cancelled before it started."
            state.finished = datetime.now(UTC)
            self._save(state)
            return
        process = self.running.get(job_id)
        if process is not None:
            process.pause.clear()
            process.cancel.set()
            state.message = "Cancelling: finishing the wells in progress …"

    def retry(self, job_id: str, failed_only: bool = True) -> str | None:
        """A new job for the failed wells (or all wells) of a finished job, in its folder."""
        state = self.status(job_id)
        if not state.folder or (failed_only and not state.failed):
            return None
        wells = [parse_well_name(w) for w in state.failed] if failed_only else state.job.wells
        job = replace(
            state.job,
            wells=wells,
            resume=state.folder,
            label=f"{state.job.label} (retry {len(state.failed)} wells)",
        )
        return self.submit(replace(state.spec, job=job))

    def resume(self, job_id: str) -> str | None:
        state = self.status(job_id)
        if not state.folder:
            return None
        job = replace(state.job, resume=state.folder, label=f"{state.job.label} (resumed)")
        return self.submit(replace(state.spec, job=job))

    # ------------------------------------------------------------ control

    def toggle_pause(self, job_id: str) -> None:
        state, process = self.status(job_id), self.running.get(job_id)
        if process is None:
            return
        if process.pause.is_set():
            process.pause.clear()
            state.status = "running"
        else:
            process.pause.set()
            state.status = "paused"
        self._save(state)

    def is_paused(self, job_id: str) -> bool:
        process = self.running.get(job_id)
        return bool(process is not None and process.pause.is_set())

    def start_next(self) -> list[JobState]:
        started = []
        while len(self.running) < self.max_running:
            if any(self.status(i).spec.resources.gpu for i in self.running):
                break  # a GPU job runs alone
            queued = [
                s
                for s in self.states
                if s.status == "queued" and not (s.spec.resources.gpu and self.running)
            ]
            if not queued:
                break
            state = min(queued, key=lambda s: (PRIORITY_RANK[s.spec.priority], s.created))
            process = self.process_factory(state.job)
            self.running[state.id] = process
            state.status, state.started = "running", datetime.now(UTC)
            process.start()
            self._save(state)
            started.append(state)
        return started

    def poll(self) -> list[tuple[JobState, dict[str, Any]]]:
        """Collect events of running jobs, finish ended ones, start queued ones."""
        out: list[tuple[JobState, dict[str, Any]]] = []
        for job_id, process in list(self.running.items()):
            state = self.status(job_id)
            for event in process.poll():
                self._handle(state, event)
                out.append((state, event))
            if job_id in self.running and state.status in ACTIVE and not process.alive():
                for event in process.poll():  # anything sent just before exiting
                    self._handle(state, event)
                    out.append((state, event))
                if state.status in ACTIVE:
                    state.status = "interrupted"
                    state.message = "The job stopped unexpectedly; resume it from its run folder."
                    self._finish(state)
                    out.append((state, {"type": "finished", "status": "interrupted"}))
        self.start_next()
        return out

    def shutdown(self, timeout: float = 10) -> None:
        """Stop running jobs (they can be resumed later)."""
        for job_id, process in list(self.running.items()):
            process.cancel.set()
            process.process.join(timeout=timeout)
            if process.alive():
                process.kill()
            state = self.status(job_id)
            if state.status in ACTIVE:
                state.status, state.message = "interrupted", "Stopped when the app closed."
                self._finish(state)

    # ----------------------------------------------------------- internals

    def _handle(self, state: JobState, event: dict[str, Any]) -> None:
        if len(state.events) < MAX_EVENTS:
            state.events.append(event)
        kind = event.get("type")
        if kind == "run_started":
            state.folder = event["folder"]
            state.total = event["total"]
            for well in event["wells"]:
                state.wells[well] = {"status": "queued", "fields": 0, "error": ""}
            for well in event["skipped"]:
                state.wells[well] = {"status": "done (earlier)", "fields": 0, "error": ""}
            self._save(state)
        elif kind == "well_started":
            state.wells.setdefault(event["well"], {"fields": 0, "error": ""})["status"] = "running"
        elif kind == "field_done":
            state.done, state.total, state.eta_s = event["done"], event["total"], event["eta_s"]
            entry = state.wells.setdefault(event["well"], {"status": "running", "error": ""})
            entry["fields"] = entry.get("fields", 0) + 1
        elif kind == "well_finished":
            entry = state.wells.setdefault(event["well"], {"fields": 0})
            entry["status"], entry["error"] = event["status"], event.get("error") or ""
        elif kind in ("status", "qc_finished"):
            state.message = event["text"]
        elif kind == "finished":
            state.status = event["status"]
            state.failed = dict(event.get("failed") or {})
            state.folder = event.get("folder", state.folder)
            if event.get("error"):
                state.message = event["error"]
            self._finish(state)

    def _finish(self, state: JobState) -> None:
        process = self.running.pop(state.id, None)
        if process is not None:
            process.process.join(timeout=5)
        state.finished = datetime.now(UTC)
        self._save(state)

    def _save(self, state: JobState) -> None:
        if not self.persist:
            return
        try:
            from hcs_studio.db import session_scope
            from hcs_studio.db.models import JobRecord

            with session_scope() as session:
                record = session.get(JobRecord, state.id) or JobRecord(id=state.id)
                record.kind = state.spec.kind
                record.label = state.spec.label
                record.owner = state.spec.owner.id
                record.priority = state.spec.priority
                record.status = state.status
                record.folder = state.folder or ""
                record.resources = asdict(state.spec.resources)
                record.payload = {
                    "plate": state.job.plate_source,
                    "wells": len(state.job.wells or []),
                }
                record.message = state.message
                record.created = state.created
                record.started = state.started
                record.finished = state.finished
                session.merge(record)
        except Exception:  # the registry is a record; never stop a job for it
            log.exception("Could not store job %s", state.id)


def recover_interrupted() -> int:
    """At start-up: jobs left running or paused by a previous session become interrupted."""
    from sqlalchemy import update

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import JobRecord

    with session_scope() as session:
        result = session.execute(
            update(JobRecord)
            .where(JobRecord.status.in_(("running", "paused")))
            .values(status="interrupted", message="The app stopped while the job was running.")
        )
        return int(getattr(result, "rowcount", 0) or 0)
