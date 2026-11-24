"""Workers of the server queue (Phase 27).

A worker claims a job from :class:`~hcs_studio.server.queue.DbJobQueue`, runs it in its own
process (``pipeline.jobs.JobProcess`` — a crash takes down the job, not the worker), forwards
its progress events to the database, keeps its heartbeat fresh, and passes a cancel request on
to the job. One worker per GPU (``CUDA_VISIBLE_DEVICES`` set to that GPU) plus CPU workers.

``run_pool`` supervises the workers of one machine: it starts them, restarts any that died
(a killed worker is back within seconds) and reaps jobs whose worker stopped beating, which
then resume from their run folder on the next free worker.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hcs_studio.pipeline.jobs import Job, JobProcess
from hcs_studio.server.queue import DbJobQueue

log = logging.getLogger(__name__)


@dataclass
class Worker:
    gpu: int | None = None
    worker_id: str = ""
    poll_s: float = 1.0
    heartbeat_s: float = 5.0
    stale_s: float = 30.0
    process_factory: Callable[[Job], Any] = JobProcess
    queue: DbJobQueue = field(default_factory=DbJobQueue)

    def __post_init__(self) -> None:
        if not self.worker_id:
            kind = f"gpu{self.gpu}" if self.gpu is not None else "cpu"
            self.worker_id = f"{socket.gethostname()}-{kind}-{uuid.uuid4().hex[:6]}"
        self.queue.stale_s = self.stale_s
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def run_once(self) -> str | None:
        """Claim and run one job; returns its final status (None when the queue is empty)."""
        claimed = self.queue.claim(self.worker_id, gpu=self.gpu is not None)
        if claimed is None:
            return None
        job_id, job = claimed
        log.info("Worker %s runs job %s (%s)", self.worker_id, job_id, job.label)
        process = self.process_factory(job)
        process.start()
        finished = False
        last_beat = 0.0
        while True:
            for event in process.poll():
                self.queue.record(job_id, event)
                finished = finished or event.get("type") == "finished"
            if time.monotonic() - last_beat >= self.heartbeat_s:
                last_beat = time.monotonic()
                if self.queue.heartbeat(self.worker_id, job_id):
                    process.pause.clear()
                    process.cancel.set()
            if not process.alive():
                for event in process.poll():
                    self.queue.record(job_id, event)
                    finished = finished or event.get("type") == "finished"
                break
            time.sleep(min(self.poll_s, 0.2))
        status = self.queue.release(self.worker_id, job_id, crashed=not finished)
        if finished:
            register_run(self.queue, job_id)
        return status

    def run_forever(self, stop: Any = None) -> None:
        self.queue.register_worker(self.worker_id, self.gpu, os.getpid())
        last_reap = 0.0
        while stop is None or not stop.is_set():
            self.queue.heartbeat(self.worker_id)
            if time.monotonic() - last_reap > self.stale_s / 2:
                last_reap = time.monotonic()
                self.queue.reap()
            if self.run_once() is None:
                time.sleep(self.poll_s)


def _worker_main(gpu: int | None, worker_id: str, poll_s: float, stale_s: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    Worker(gpu=gpu, worker_id=worker_id, poll_s=poll_s, stale_s=stale_s).run_forever()


def run_pool(
    gpus: list[int],
    cpu_workers: int = 1,
    *,
    poll_s: float = 1.0,
    stale_s: float = 30.0,
    check_s: float = 2.0,
    stop: Any = None,
    on_restart: Callable[[str], None] | None = None,
) -> None:
    """Run and supervise the workers of this machine until ``stop`` is set."""
    context = multiprocessing.get_context("spawn")
    slots: dict[str, tuple[int | None, Any]] = {}
    host = socket.gethostname()
    plan = [(f"{host}-gpu{g}", g) for g in gpus] + [
        (f"{host}-cpu{i}", None) for i in range(cpu_workers)
    ]

    def start(name: str, gpu: int | None) -> None:
        process = context.Process(target=_worker_main, args=(gpu, name, poll_s, stale_s), name=name)
        process.start()
        slots[name] = (gpu, process)

    for name, gpu in plan:
        start(name, gpu)
    queue = DbJobQueue(stale_s)
    try:
        while stop is None or not stop.is_set():
            time.sleep(check_s)
            for name, (gpu, process) in list(slots.items()):
                if not process.is_alive():
                    log.warning("Worker %s stopped (exit %s); restarting", name, process.exitcode)
                    if on_restart is not None:
                        on_restart(name)
                    start(name, gpu)
            queue.reap()
    finally:
        for _, process in slots.values():
            process.terminate()
        for _, process in slots.values():
            process.join(timeout=10)


def register_run(queue: DbJobQueue, job_id: str) -> None:
    """Index a finished job's run folder in the registry, so clients find it (best effort)."""
    try:
        folder = queue.status(job_id).folder
        if not folder:
            return
        from pathlib import Path

        from hcs_studio.db import session_scope
        from hcs_studio.db.scanner import scan

        with session_scope() as session:
            scan(session, [Path(folder)])
    except Exception:  # the run is fine; indexing can be repeated with `db scan`
        log.exception("Could not index the run of job %s", job_id)
