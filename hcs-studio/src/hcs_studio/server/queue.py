"""The server job queue (Phase 27): jobs in the registry database, claimed by workers.

``DbJobQueue`` implements the same ``JobQueue`` protocol as the local queue, but the jobs live
in the ``queue_job`` table of the registry (PostgreSQL in server mode), so any client can
submit and watch them and any worker on any machine can run them.

- **Claiming.** A worker takes the next queued job with one atomic conditional update; on
  PostgreSQL the candidate is chosen ``FOR UPDATE SKIP LOCKED`` so workers never wait on each
  other. Order: priority (interactive, auto, batch), then **fair share** (the owner with the
  fewest jobs running goes first), then age. GPU jobs only go to GPU workers; GPU workers take
  CPU jobs when no GPU job waits.
- **Liveness.** Workers write a heartbeat every few seconds for themselves and their job. A
  job whose heartbeat is older than ``stale_s`` lost its worker (killed, crashed, machine off):
  ``reap`` queues it again with ``resume`` set to its run folder, so finished wells are kept
  and the next worker carries on — up to ``max_attempts`` times, then it is ``interrupted``.
- **Paths.** Payload paths (plate, layout, output, resume) are stored as data-root paths
  (``ROOT:relative``) and resolved on the worker, so a laptop and a Linux server agree.
"""

from __future__ import annotations

import socket
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hcs_studio.model.plate import ChannelInfo, ImageMetadata, parse_well_name, well_name
from hcs_studio.pipeline.jobs import Job
from hcs_studio.pipeline.queue import PRIORITY_RANK, JobSpec, JobState, Resources
from hcs_studio.services import Actor

ACTIVE = ("running",)
FINAL = ("completed", "completed_with_failures", "cancelled", "error", "interrupted", "failed")
PATH_KEYS = ("layout_path", "output_root", "resume")


def _rooted(path: str | None) -> str | None:
    if not path:
        return path
    from hcs_studio.output.manifest import rooted_text

    return rooted_text(Path(path))


def _resolved(text: str | None) -> str | None:
    if not text:
        return text
    if ":" in text.split("/", 1)[0] and not Path(text).exists():
        from hcs_studio.paths import resolve

        try:
            return str(resolve(text))
        except (KeyError, ValueError):
            return text
    return text


def job_to_payload(job: Job) -> dict[str, Any]:
    """A JSON-safe job; paths as data-root paths."""
    source = dict(job.plate_source)
    if source.get("path"):
        source["path"] = _rooted(str(source["path"]))
    payload: dict[str, Any] = {
        "plate_source": source,
        "pipeline": job.pipeline,
        "metadata": _metadata_dict(job.metadata),
        "wells": [list(w) for w in job.wells] if job.wells is not None else None,
        "fields": {well_name(*k): list(v) for k, v in job.fields.items()} if job.fields else None,
        "workers": job.workers,
        "checksums": job.checksums,
        "save_images": job.save_images,
        "max_memory_gb": job.max_memory_gb,
        "image_dtype": job.image_dtype,
        "label": job.label,
        "extra": job.extra,
    }
    for key in PATH_KEYS:
        payload[key] = _rooted(getattr(job, key))
    return payload


def payload_to_job(payload: dict[str, Any]) -> Job:
    """The job of a payload, paths resolved for this computer."""
    source = dict(payload.get("plate_source") or {})
    if source.get("path"):
        source["path"] = _resolved(str(source["path"]))
    fields = payload.get("fields")
    return Job(
        plate_source=source,
        pipeline=dict(payload.get("pipeline") or {}),
        metadata=_metadata(payload.get("metadata")),
        wells=[(int(w[0]), int(w[1])) for w in payload["wells"]]
        if payload.get("wells") is not None
        else None,
        fields={parse_well_name(k): list(v) for k, v in fields.items()} if fields else None,
        layout_path=_resolved(payload.get("layout_path")),
        output_root=_resolved(payload.get("output_root")),
        workers=payload.get("workers", 1),
        checksums=payload.get("checksums", True),
        save_images=payload.get("save_images", True),
        max_memory_gb=payload.get("max_memory_gb"),
        image_dtype=payload.get("image_dtype", "float32"),
        resume=_resolved(payload.get("resume")),
        label=payload.get("label", ""),
        extra=dict(payload.get("extra") or {}),
    )


def _metadata_dict(metadata: ImageMetadata | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    data = asdict(metadata)
    data["channels"] = [asdict(c) for c in metadata.channels]
    return data


def _metadata(data: dict[str, Any] | None) -> ImageMetadata | None:
    if not data:
        return None
    data = dict(data)
    channels = tuple(ChannelInfo(**c) for c in data.pop("channels", []))
    return ImageMetadata(channels=channels, **data)


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class DbJobQueue:
    """The job queue in the registry database (server mode)."""

    def __init__(self, stale_s: float = 30.0) -> None:
        self.stale_s = stale_s

    # ------------------------------------------------------------------ protocol

    def submit(self, spec: JobSpec) -> str:
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob

        job_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(
                QueueJob(
                    id=job_id,
                    kind=spec.kind,
                    label=spec.label,
                    owner=spec.owner.id,
                    owner_role=spec.owner.role,
                    priority=spec.priority,
                    priority_rank=PRIORITY_RANK[spec.priority],
                    gpu=spec.resources.gpu,
                    payload=job_to_payload(spec.job),
                )
            )
        return job_id

    def _state(self, row: Any) -> JobState:
        from hcs_studio.paths import resolve

        spec = JobSpec(
            job=payload_to_job(row.payload),
            kind=row.kind,
            priority=row.priority,
            owner=Actor(row.owner, row.owner, "account", row.owner_role),
            resources=Resources(gpu=row.gpu),
        )
        folder = None
        if row.folder:
            try:
                folder = str(resolve(row.folder))
            except (KeyError, ValueError):
                folder = row.folder
        return JobState(
            spec=spec,
            id=row.id,
            status=row.status,
            folder=folder,
            done=row.done,
            total=row.total,
            failed=dict(row.failed_wells or {}),
            message=row.message,
            created=_aware(row.created) or _now(),
            started=_aware(row.started),
            finished=_aware(row.finished),
        )

    def status(self, job_id: str) -> JobState:
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob

        with session_scope() as session:
            row = session.get(QueueJob, job_id)
            if row is None:
                raise KeyError(f"No job {job_id}")
            return self._state(row)

    def events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueEvent

        with session_scope() as session:
            rows = session.scalars(
                select(QueueEvent)
                .where(QueueEvent.job_id == job_id, QueueEvent.seq > since)
                .order_by(QueueEvent.seq)
            ).all()
            return [{**r.event, "seq": r.seq} for r in rows]

    def jobs(self, limit: int = 200) -> list[JobState]:
        from sqlalchemy import select

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob

        with session_scope() as session:
            rows = session.scalars(select(QueueJob).order_by(QueueJob.created.desc()).limit(limit))
            return [self._state(r) for r in rows]

    def cancel(self, job_id: str) -> None:
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob

        with session_scope() as session:
            row = session.get(QueueJob, job_id)
            if row is None:
                raise KeyError(f"No job {job_id}")
            if row.status == "queued":
                row.status, row.message, row.finished = (
                    "cancelled",
                    "Cancelled before it started.",
                    _now(),
                )
            elif row.status in ACTIVE:
                row.cancel_requested = True
                row.message = "Cancelling: finishing the wells in progress …"

    def retry(self, job_id: str, failed_only: bool = True) -> str | None:
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

    # ------------------------------------------------------------------ workers

    def register_worker(self, worker_id: str, gpu: int | None, pid: int) -> None:
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueWorker

        with session_scope() as session:
            row = session.get(QueueWorker, worker_id) or QueueWorker(id=worker_id)
            row.host, row.pid, row.gpu, row.status = socket.gethostname(), pid, gpu, "idle"
            row.started = row.heartbeat = _now()
            row.job_id, row.message = "", ""
            session.add(row)

    def claim(self, worker_id: str, gpu: bool) -> tuple[str, Job] | None:
        """Atomically take the next job for this worker (None when there is none)."""
        from sqlalchemy import func, select, update

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob, QueueWorker

        other = QueueJob.__table__.alias("other")
        fair = (
            select(func.count())
            .select_from(other)
            .where(other.c.status == "running", other.c.owner == QueueJob.owner)
            .scalar_subquery()
        )
        with session_scope() as session:
            query = select(QueueJob.id).where(QueueJob.status == "queued")
            if not gpu:
                query = query.where(QueueJob.gpu.is_(False))
            order = [QueueJob.gpu.desc()] if gpu else []
            query = query.order_by(*order, QueueJob.priority_rank, fair, QueueJob.created).limit(1)
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                query = query.with_for_update(skip_locked=True, of=QueueJob)
            job_id = session.scalar(query)
            if job_id is None:
                return None
            now = _now()
            result: Any = session.execute(
                update(QueueJob)
                .where(QueueJob.id == job_id, QueueJob.status == "queued")
                .values(
                    status="running",
                    worker=worker_id,
                    started=now,
                    heartbeat=now,
                    attempts=QueueJob.attempts + 1,
                    message="",
                )
            )
            if not result.rowcount:  # another worker was faster (SQLite)
                return None
            row = session.get(QueueJob, job_id)
            assert row is not None
            worker = session.get(QueueWorker, worker_id)
            if worker is not None:
                worker.status, worker.job_id, worker.heartbeat = "busy", job_id, now
            return job_id, payload_to_job(row.payload)

    def heartbeat(self, worker_id: str, job_id: str | None = None) -> bool:
        """Refresh the worker's (and its job's) heartbeat; True when cancellation was asked."""
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob, QueueWorker

        now = _now()
        with session_scope() as session:
            worker = session.get(QueueWorker, worker_id)
            if worker is not None:
                worker.heartbeat = now
                worker.status = "busy" if job_id else "idle"
                worker.job_id = job_id or ""
            if job_id:
                row = session.get(QueueJob, job_id)
                if row is not None and row.worker == worker_id:
                    row.heartbeat = now
                    return bool(row.cancel_requested)
        return False

    def record(self, job_id: str, event: dict[str, Any]) -> None:
        """Store a progress event and update the job from it."""
        from sqlalchemy import update

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueEvent, QueueJob

        with session_scope() as session:
            # increment first: the UPDATE locks the job row, so event numbers never collide
            session.execute(
                update(QueueJob).where(QueueJob.id == job_id).values(events=QueueJob.events + 1)
            )
            row = session.get(QueueJob, job_id)
            if row is None:
                return
            session.refresh(row)
            session.add(QueueEvent(job_id=job_id, seq=row.events, event=_jsonable(event)))
            kind = event.get("type")
            if kind == "run_started":
                row.folder = _rooted(str(event["folder"])) or ""
                row.total = int(event.get("total", 0))
            elif kind == "field_done":
                row.done, row.total = int(event["done"]), int(event["total"])
            elif kind == "status":
                row.message = str(event.get("text", ""))[:2000]
            elif kind == "finished":
                row.status = str(event.get("status", "completed"))
                row.failed_wells = dict(event.get("failed") or {})
                if event.get("folder"):
                    row.folder = _rooted(str(event["folder"])) or row.folder
                if event.get("error"):
                    row.message = str(event["error"])[:2000]
                row.finished = _now()

    def release(self, worker_id: str, job_id: str, crashed: bool, message: str = "") -> str:
        """The worker is done with a job: a crashed job goes back to the queue (resuming from
        its folder) or, after ``max_attempts``, becomes interrupted. Returns the new status."""
        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob, QueueWorker

        with session_scope() as session:
            row = session.get(QueueJob, job_id)
            worker = session.get(QueueWorker, worker_id)
            if worker is not None:
                worker.status, worker.job_id = "idle", ""
            if row is None:
                return "missing"
            if crashed and row.status in ACTIVE:
                _requeue(row, message or "The job's process stopped unexpectedly")
            return row.status

    def reap(self) -> list[str]:
        """Jobs whose worker stopped beating go back to the queue (or are interrupted)."""
        from sqlalchemy import select

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueJob, QueueWorker

        limit = _now() - timedelta(seconds=self.stale_s)
        changed = []
        with session_scope() as session:
            for row in session.scalars(select(QueueJob).where(QueueJob.status == "running")):
                beat = _aware(row.heartbeat)
                if beat is None or beat < limit:
                    _requeue(row, f"Worker {row.worker} stopped responding")
                    changed.append(row.id)
            for worker in session.scalars(
                select(QueueWorker).where(QueueWorker.status != "stopped")
            ):
                beat = _aware(worker.heartbeat)
                if beat is not None and beat < limit:
                    worker.status, worker.message = "stopped", "no heartbeat"
        return changed

    def workers(self) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from hcs_studio.db import session_scope
        from hcs_studio.db.server import QueueWorker

        with session_scope() as session:
            return [
                {
                    "id": w.id,
                    "host": w.host,
                    "pid": w.pid,
                    "gpu": w.gpu,
                    "status": w.status,
                    "job_id": w.job_id,
                    "heartbeat": _aware(w.heartbeat),
                    "message": w.message,
                }
                for w in session.scalars(select(QueueWorker).order_by(QueueWorker.id))
            ]


def _requeue(row: Any, message: str) -> None:
    if row.attempts >= row.max_attempts:
        row.status, row.finished = "interrupted", _now()
        row.message = f"{message}; gave up after {row.attempts} attempts (resume it by hand)"
        return
    payload = dict(row.payload)
    if row.folder:
        payload["resume"] = row.folder  # carry on in the same run folder
    row.payload = payload
    row.status, row.worker, row.heartbeat, row.cancel_requested = "queued", "", None, False
    row.message = f"{message}; queued again (attempt {row.attempts + 1} of {row.max_attempts})"


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    import json

    return dict(json.loads(json.dumps(value, default=str)))
