"""Phase 27a: the database job queue and its workers (SQLite always; PostgreSQL when
HCS_TEST_POSTGRES_URL points at a server where databases may be created)."""

from __future__ import annotations

import os
import signal
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hcs_studio.pipeline.jobs import Job
from hcs_studio.pipeline.queue import JobSpec, Resources
from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.server.queue import DbJobQueue, job_to_payload, payload_to_job
from hcs_studio.services import Actor

PIPELINE = Pipeline(
    steps=[
        PipelineStep(module="segment_classical", params={"channel": 1}),
        PipelineStep(module="measure_objects", params={"labels": ["nuclei"], "texture": False}),
    ]
).to_dict()

BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("HCS_TEST_POSTGRES_URL") else [])


@pytest.fixture(params=BACKENDS)
def home(request: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    name = ""
    if request.param == "postgresql":
        import psycopg

        admin = os.environ["HCS_TEST_POSTGRES_URL"]
        name = f"hcs_{uuid.uuid4().hex[:10]}"
        with psycopg.connect(
            admin.replace("postgresql+psycopg", "postgresql"), autocommit=True
        ) as c:
            c.execute(f"CREATE DATABASE {name}")
        monkeypatch.setenv("HCS_DATABASE_URL", admin.rsplit("/", 1)[0] + "/" + name)
    else:
        monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()
    if name:
        import psycopg

        with psycopg.connect(
            admin.replace("postgresql+psycopg", "postgresql"), autocommit=True
        ) as c:
            c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def _plate(tmp: Path, wells: int = 4, size: int = 96, name: str = "Q") -> Path:
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(r, c) for r in range(2) for c in range(wells // 2)],
        channels=("DAPI",),
        size=size,
        n_fields=2,
        name=name,
        features=[Nuclei(count=lambda r, c, f: 8 + c), ShotNoise(sd=5)],
    )
    path, _ = plate.write(tmp / f"raw_{name}")
    return path


def _spec(
    raw: Path,
    out: Path,
    owner: str,
    priority: str = "interactive",
    gpu: bool = False,
    label: str = "",
) -> JobSpec:
    from hcs_studio.io.base import open_plate

    job = Job(
        plate_source=open_plate(raw).source,
        pipeline=PIPELINE,
        output_root=str(out),
        label=label or owner,
        checksums=False,
        save_images=False,
    )
    return JobSpec(
        job=job,
        priority=priority,  # type: ignore[arg-type]
        owner=Actor(owner, owner, "account", "analyst"),
        resources=Resources(gpu=gpu),
    )


def test_payload_round_trip(home: Path, tmp_path: Path) -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata

    job = Job(
        plate_source={"reader": "omezarr", "path": str(tmp_path / "p.zarr")},
        pipeline=PIPELINE,
        wells=[(0, 1), (2, 3)],
        fields={(0, 1): [1, 2]},
        metadata=ImageMetadata(channels=(ChannelInfo("DAPI", 405.0),), pixel_size_um=0.65),
        output_root=str(tmp_path / "out"),
        label="x",
        extra={"a": 1},
    )
    back = payload_to_job(job_to_payload(job))
    assert back.wells == [(0, 1), (2, 3)] and back.fields == {(0, 1): [1, 2]}
    assert back.metadata == job.metadata and back.plate_source == job.plate_source
    assert back.output_root == job.output_root and back.extra == {"a": 1}


def test_priority_fair_share_and_gpu_routing(home: Path, tmp_path: Path) -> None:
    raw = _plate(tmp_path)
    q = DbJobQueue()
    out = tmp_path / "out"
    a1 = q.submit(_spec(raw, out, "anna", "batch", label="a1"))
    a2 = q.submit(_spec(raw, out, "anna", "interactive", label="a2"))
    b1 = q.submit(_spec(raw, out, "bert", "interactive", label="b1"))
    g1 = q.submit(_spec(raw, out, "carl", "batch", gpu=True, label="g1"))
    q.register_worker("cpu-1", None, 1)
    q.register_worker("gpu-0", 0, 2)
    first = q.claim("cpu-1", gpu=False)
    assert first is not None and first[0] == a2  # interactive before batch, older first
    second = q.claim("cpu-1", gpu=False)
    assert second is not None and second[0] == b1  # fair share: anna already has one running
    assert q.claim("gpu-0", gpu=True)[0] == g1  # type: ignore[index]  # GPU jobs to GPU workers
    assert q.claim("cpu-1", gpu=False)[0] == a1  # type: ignore[index]
    assert q.claim("cpu-1", gpu=False) is None
    assert q.status(g1).status == "running" and q.status(g1).spec.resources.gpu


def test_events_cancel_and_crash_requeue(home: Path, tmp_path: Path) -> None:
    raw = _plate(tmp_path)
    q = DbJobQueue(stale_s=0.5)
    job_id = q.submit(_spec(raw, tmp_path / "out", "anna"))
    q.register_worker("w", None, 1)
    claimed = q.claim("w", gpu=False)
    assert claimed is not None
    run = tmp_path / "out" / "run1"
    q.record(
        job_id, {"type": "run_started", "folder": str(run), "total": 8, "wells": [], "skipped": []}
    )
    q.record(job_id, {"type": "field_done", "done": 3, "total": 8, "well": "A01", "eta_s": 1})
    assert [e["seq"] for e in q.events(job_id)] == [1, 2] and q.events(job_id, since=1)[0][
        "done"
    ] == 3
    state = q.status(job_id)
    assert (state.done, state.total, state.folder) == (3, 8, str(run))
    q.cancel(job_id)
    assert q.heartbeat("w", job_id) is True  # the worker learns about the cancellation
    # the worker dies: its job is queued again and will resume in the same folder
    time.sleep(0.7)
    assert q.reap() == [job_id]
    requeued = q.status(job_id)
    assert requeued.status == "queued" and requeued.job.resume == str(run)
    assert "attempt 2 of 3" in requeued.message
    assert q.workers()[0]["status"] == "stopped"
    for _ in range(2):  # attempts 2 and 3 crash too
        q.claim("w", gpu=False)
        q.release("w", job_id, crashed=True)
    final = q.status(job_id)
    assert final.status == "interrupted" and "gave up after 3 attempts" in final.message


def test_two_clients_jobs_complete(home: Path, tmp_path: Path) -> None:
    """Done-when (part): two clients submit at the same time; both jobs complete."""
    from hcs_studio.server.worker import Worker

    raw = _plate(tmp_path)
    ids: list[str] = []

    def client(owner: str) -> None:
        ids.append(DbJobQueue().submit(_spec(raw, tmp_path / owner, owner)))

    threads = [threading.Thread(target=client, args=(o,)) for o in ("anna", "bert")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop = threading.Event()
    workers = [Worker(worker_id=f"w{i}", poll_s=0.2) for i in range(2)]
    runners = [threading.Thread(target=w.run_forever, args=(stop,)) for w in workers]
    for r in runners:
        r.start()
    q = DbJobQueue()
    deadline = time.time() + 180
    while time.time() < deadline and any(q.status(i).status in ("queued", "running") for i in ids):
        time.sleep(0.5)
    stop.set()
    for r in runners:
        r.join(timeout=30)
    states = [q.status(i) for i in ids]
    assert [s.status for s in states] == ["completed", "completed"], [s.message for s in states]
    assert {s.spec.owner.id for s in states} == {"anna", "bert"}
    for s in states:
        assert (
            s.folder and (Path(s.folder) / "run_manifest.json").exists() and s.done == s.total == 8
        )
        assert q.events(s.id)[-1]["type"] in ("finished", "status")


def test_killed_worker_is_restarted_and_job_resumes(home: Path, tmp_path: Path) -> None:
    """Done-when (part): a worker killed mid-job is restarted by the pool and the job resumes
    in the same run folder (finished wells are kept)."""
    from hcs_studio.server.worker import run_pool

    raw = _plate(tmp_path, wells=16, size=192, name="Big")
    q = DbJobQueue(stale_s=3)
    job_id = q.submit(_spec(raw, tmp_path / "out", "anna"))
    stop = threading.Event()
    restarts: list[str] = []
    pool = threading.Thread(
        target=run_pool,
        args=([], 1),
        kwargs={
            "poll_s": 0.3,
            "stale_s": 3,
            "check_s": 0.5,
            "stop": stop,
            "on_restart": restarts.append,
        },
    )
    pool.start()
    try:
        deadline = time.time() + 120
        while time.time() < deadline and q.status(job_id).done < 6:
            time.sleep(0.2)
        state = q.status(job_id)
        assert 6 <= state.done < state.total, state
        worker = next(w for w in q.workers() if w["job_id"] == job_id)
        os.kill(worker["pid"], signal.SIGKILL)  # the worker dies; its job process becomes an orphan
        deadline = time.time() + 240
        while time.time() < deadline and q.status(job_id).status != "completed":
            time.sleep(0.5)
        final = q.status(job_id)
    finally:
        stop.set()
        pool.join(timeout=60)
    assert restarts, "the pool did not restart the killed worker"
    assert final.status == "completed", final.message
    assert final.folder == state.folder  # the same run folder, resumed
    started = [e for e in q.events(job_id) if e["type"] == "run_started"]
    assert len(started) >= 2 and started[-1]["skipped"], "the resumed run kept no finished wells"
