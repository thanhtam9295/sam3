"""Job queue (PROMPT_PLATFORM.md §2.5): priorities, GPU exclusivity, cancel, retry, resume,
interrupted jobs and persistence."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from hcs_studio.pipeline.jobs import Job
from hcs_studio.pipeline.queue import JobSpec, LocalJobQueue, Resources, recover_interrupted
from hcs_studio.services import Actor


class FakeProcess:
    """Stands in for JobProcess: the test decides which events it sends and when it ends."""

    instances: ClassVar[list[FakeProcess]] = []

    def __init__(self, job: Job) -> None:
        self.job = job
        self.pause, self.cancel = threading.Event(), threading.Event()
        self.queued: list[dict[str, Any]] = []
        self.started = self.ended = False
        self.process = self  # JobProcess exposes .process.join
        FakeProcess.instances.append(self)

    def start(self) -> None:
        self.started = True
        self.queued.append(
            {
                "type": "run_started",
                "folder": self.job.resume or f"/runs/{self.job.label}",
                "total": 2,
                "wells": ["A01", "A02"],
                "skipped": [],
            }
        )

    def finish(self, status: str = "completed", failed: dict[str, str] | None = None) -> None:
        self.queued.append({"type": "finished", "status": status, "failed": failed or {}})
        self.ended = True

    def crash(self) -> None:
        self.ended = True

    def alive(self) -> bool:
        return self.started and not self.ended

    def poll(self) -> list[dict[str, Any]]:
        out, self.queued = self.queued, []
        return out

    def join(self, timeout: float | None = None) -> None:
        pass

    def kill(self) -> None:
        self.ended = True


@pytest.fixture
def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalJobQueue]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    FakeProcess.instances.clear()
    yield LocalJobQueue(process_factory=FakeProcess, max_running=2)  # type: ignore[arg-type]
    dispose_engines()


def spec(label: str, priority: str = "interactive", gpu: bool = False) -> JobSpec:
    job = Job(plate_source={"path": f"/raw/{label}"}, pipeline={"steps": []}, label=label)
    return JobSpec(
        job=job,
        priority=priority,  # type: ignore[arg-type]
        owner=Actor("tester"),
        resources=Resources(gpu=gpu),
    )


def test_priority_order_and_gpu_jobs_run_alone(queue: LocalJobQueue) -> None:
    queue.max_running = 1
    first = queue.submit(spec("batch-1", "batch"))  # starts at once: nothing else queued
    queue.submit(spec("batch-2", "batch"))
    queue.submit(spec("auto-1", "auto"))
    queue.submit(spec("interactive-1"))
    order = [first]
    for _ in range(3):
        FakeProcess.instances[-1].finish()
        queue.poll()
        order.append(next(iter(queue.running)))
    labels = [queue.status(i).spec.label for i in order]
    assert labels == ["batch-1", "interactive-1", "auto-1", "batch-2"]  # never pre-empted

    queue.max_running = 2
    FakeProcess.instances[-1].finish()
    queue.poll()
    gpu = queue.submit(spec("gpu", gpu=True))
    cpu = queue.submit(spec("cpu"))
    assert list(queue.running) == [gpu]  # the CPU job waits while the GPU job runs
    FakeProcess.instances[-1].finish()
    queue.poll()
    assert list(queue.running) == [cpu]


def test_progress_cancel_retry_resume_and_interrupted(queue: LocalJobQueue) -> None:
    job_id = queue.submit(spec("plate"))
    process = FakeProcess.instances[-1]
    process.queued += [
        {"type": "well_started", "well": "A01"},
        {"type": "field_done", "well": "A01", "done": 1, "total": 2, "eta_s": 3.0},
        {"type": "well_finished", "well": "A01", "status": "done"},
    ]
    queue.poll()
    state = queue.status(job_id)
    assert state.folder == "/runs/plate" and state.done == 1 and state.wells["A01"]["fields"] == 1
    assert len(queue.events(job_id)) == 4 and len(queue.events(job_id, since=3)) == 1
    queue.toggle_pause(job_id)
    assert state.status == "paused" and queue.is_paused(job_id)
    queue.toggle_pause(job_id)
    process.finish("completed_with_failures", {"A02": "unreadable image"})
    queue.poll()
    assert state.status == "completed_with_failures" and not queue.running
    retry = queue.retry(job_id)
    assert retry is not None
    retried = queue.status(retry)
    assert retried.job.wells == [(0, 1)] and retried.job.resume == "/runs/plate"
    FakeProcess.instances[-1].crash()  # the process dies without a "finished" event
    queue.poll()
    assert retried.status == "interrupted" and "resume" in retried.message
    resumed = queue.resume(retry)
    assert resumed is not None and queue.status(resumed).job.resume == "/runs/plate"

    queue.max_running = 1
    waiting = queue.submit(spec("waiting"))
    queue.cancel(waiting)
    assert queue.status(waiting).status == "cancelled"
    queue.cancel(resumed)
    assert FakeProcess.instances[-1].cancel.is_set()  # running jobs finish their wells first


def test_jobs_are_stored_and_interrupted_ones_recovered(queue: LocalJobQueue) -> None:
    from hcs_studio.db import session_scope
    from hcs_studio.db.models import JobRecord

    done = queue.submit(spec("done"))
    FakeProcess.instances[-1].finish()
    queue.poll()
    running = queue.submit(spec("left running"))
    with session_scope() as s:
        rows = {r.id: r for r in s.query(JobRecord)}
        assert rows[done].status == "completed" and rows[done].owner == "tester"
        assert rows[done].folder == "/runs/done" and rows[running].status == "running"
    assert recover_interrupted() == 1  # e.g. at the next start-up
    with session_scope() as s:
        assert s.get(JobRecord, running).status == "interrupted"  # type: ignore[union-attr]


def test_job_services_record_submit_cancel_retry(queue: LocalJobQueue) -> None:
    from hcs_studio.services import context, recent_events
    from hcs_studio.services.operations import cancel_job, retry_job, submit_job

    job = Job(plate_source={"path": "/raw/p"}, pipeline={"steps": []}, label="p")
    job_id = submit_job(context(), queue, job)
    FakeProcess.instances[-1].finish("completed_with_failures", {"A01": "x"})
    queue.poll()
    new = retry_job(context(), queue, job_id)
    cancel_job(context("wrong plate"), queue, str(new))
    types = [e["type"] for e in recent_events(3)]
    assert types == ["run.cancel", "run.retry", "run.submit"]
    assert recent_events(1)[0]["reason"] == "wrong plate"
