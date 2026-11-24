"""Phase 27b: the server API — sessions, permissions, jobs with a worker, WebSocket progress,
runs, files, image tiles, lab events, the admin view, and the remote job queue client."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.security import accounts
from hcs_studio.server.client import RemoteJobQueue, ServerClient, ServerError

PIPELINE = Pipeline(
    steps=[
        PipelineStep(module="segment_classical", params={"channel": 1}),
        PipelineStep(module="measure_objects", params={"labels": ["nuclei"], "texture": False}),
        PipelineStep(module="qc"),
    ]
).to_dict()


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    from hcs_studio.config import DataRootConfig, load_settings, save_settings
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.server.api import create_app
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    lab = tmp_path / "lab"
    (lab / "results").mkdir(parents=True)
    settings = load_settings()
    settings.data_roots = [DataRootConfig(name="LAB", linux=str(lab))]
    settings.server.results_root = "LAB:results"
    save_settings(settings)
    accounts.create_account("admin", "admin", "Ada Admin", password="admin password")
    accounts.create_account("anna", "analyst", "Anna Analyst", password="anna password")
    accounts.create_account("vera", "viewer", "Vera Viewer", password="vera password")
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1)],
        channels=("DAPI",),
        size=96,
        n_fields=2,
        name="API01",
        features=[Nuclei(count=lambda r, c, f: 6 + 3 * c), ShotNoise(sd=4)],
    )
    raw, _ = plate.write(lab / "raw")
    yield {"app": create_app(), "lab": lab, "plate": f"LAB:{raw.relative_to(lab).as_posix()}"}
    dispose_engines()


def _client(app: Any, user: str = "anna") -> ServerClient:
    client = ServerClient(http=TestClient(app))
    client.login(user, f"{user} password")
    return client


def _work_until_done(job_ids: list[str], timeout: float = 120) -> None:
    from hcs_studio.server.queue import DbJobQueue
    from hcs_studio.server.worker import Worker

    worker = Worker(worker_id="test-worker", poll_s=0.2)
    queue = DbJobQueue()
    deadline = time.time() + timeout
    while time.time() < deadline and any(
        queue.status(j).status in ("queued", "running") for j in job_ids
    ):
        worker.run_once()


def test_sessions_and_permissions(server: dict[str, Any]) -> None:
    http = TestClient(server["app"])
    assert http.get("/api/health").json()["database"] == "sqlite"
    assert http.post("/api/login", json={"username": "anna", "password": "nope"}).status_code == 401
    assert http.get("/api/me").status_code == 401
    anna = _client(server["app"])
    assert anna.get("/api/me") == {"user": "anna", "name": "Anna Analyst", "role": "analyst"}
    vera = _client(server["app"], "vera")
    with pytest.raises(ServerError) as denied:
        vera.submit(server["plate"], PIPELINE)
    assert denied.value.status == 403
    with pytest.raises(ServerError) as not_admin:
        anna.admin_status()
    assert not_admin.value.status == 403
    anna.logout()
    with pytest.raises(ServerError) as gone:
        anna.get("/api/me")
    assert gone.value.status == 401


def test_run_through_the_server(server: dict[str, Any]) -> None:
    """A run submitted through the API, processed by a worker, then browsed: registry, run
    summary, report file, image tiles, progress over the WebSocket, the audit trail."""
    from hcs_studio.security.audit import query

    app = server["app"]
    anna = _client(app)
    queue = RemoteJobQueue(anna)
    from hcs_studio.io.base import open_plate
    from hcs_studio.paths import resolve
    from hcs_studio.pipeline.jobs import Job
    from hcs_studio.pipeline.queue import JobSpec

    raw = resolve(server["plate"])
    spec = JobSpec(
        job=Job(
            plate_source=open_plate(raw).source, pipeline=PIPELINE, label="api run", checksums=False
        )
    )
    job_id = queue.submit(spec)
    assert queue.status(job_id).status == "queued"
    _work_until_done([job_id])
    state = queue.status(job_id)
    assert state.status == "completed", state.message
    assert state.folder and Path(state.folder).is_relative_to(server["lab"] / "results")
    polled = queue.poll()
    assert [e for _, e in polled if e["type"] == "finished"]
    # progress over the WebSocket (the job is done: all events, then "closed")
    token = anna.token
    with TestClient(app).websocket_connect(f"/api/jobs/{job_id}/ws?token={token}") as socket:
        messages = []
        while True:
            message = socket.receive_json()
            messages.append(message)
            if message["type"] == "closed":
                break
    assert messages[0]["type"] == "run_started" and messages[-1]["status"] == "completed"
    with (
        pytest.raises(WebSocketDisconnect),
        TestClient(app).websocket_connect(f"/api/jobs/{job_id}/ws?token=bad") as s,
    ):
        s.receive_json()
    # the worker indexed the run: plates and runs are browsable
    plates = anna.plates(text="API01")
    assert len(plates) == 1
    page = anna.get(f"/api/plates/{plates[0]['plate_id']}")
    run_id = page["runs"][0]["id"]
    run = anna.run(run_id)
    assert run["state"] == "completed" and {w["well"] for w in run["wells"]} == {"A01", "A02"}
    assert "index.html" in run["files"] and b"<html" in anna.run_file(run_id, "index.html")
    with pytest.raises(ServerError) as escape:
        anna.run_file(run_id, "../../home/registry.sqlite")
    assert escape.value.status in (403, 404)
    info = anna.image_info(run_id, "A01", 1)
    assert info["shape"][-2:] == [96, 96] and info["channels"] == ["DAPI"]
    plane = anna.plane(run_id, "A01", 1)
    from hcs_studio.io.base import open_plate as reopen

    expected = np.asarray(reopen(raw).well("A01").fields[0].load()[0, 0, 0])
    assert np.array_equal(plane, expected.astype(np.float32))
    assert anna.tile(run_id, "A01", 1, level=1).shape == (48, 48)
    # audited as anna, from the API
    submitted = [r for r in query(action="run.submit") if r["actor_id"] == "anna"]
    assert submitted and submitted[0]["client"].startswith("api:")


def test_lab_events_and_admin_view(server: dict[str, Any]) -> None:
    app = server["app"]
    admin = _client(app, "admin")
    receipt = admin.send_event({"type": "plate.imaged", "payload": {}})
    assert receipt["status"] == "rejected" and receipt["errors"]
    status = admin.admin_status()
    assert set(status) == {"jobs", "workers", "gpus", "disk", "failed"}
    assert status["disk"]["path"] == "LAB:results" and status["disk"]["free_gb"] > 0


def test_two_clients_at_once(server: dict[str, Any]) -> None:
    """Done-when (part): two clients submit at the same time and both jobs complete."""
    app = server["app"]
    accounts.create_account("bert", "analyst", password="bert password")
    clients = [_client(app, "anna"), _client(app, "bert")]
    ids: list[str] = []

    def submit(client: ServerClient) -> None:
        ids.append(
            client.submit(server["plate"], PIPELINE, label=client.user["user"], checksums=False)
        )

    threads = [threading.Thread(target=submit, args=(c,)) for c in clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _work_until_done(ids)
    jobs = {clients[0].job(i)["owner"]: clients[0].job(i) for i in ids}
    assert set(jobs) == {"anna", "bert"} and all(j["status"] == "completed" for j in jobs.values())
    assert jobs["anna"]["folder"] != jobs["bert"]["folder"]


@pytest.mark.gui
def test_jobs_dock_in_connected_mode(server: dict[str, Any], qtbot: Any) -> None:
    pytest.importorskip("PySide6")
    from hcs_studio.gui.jobs_panel import JobsPanel
    from hcs_studio.io.base import open_plate
    from hcs_studio.paths import resolve
    from hcs_studio.pipeline.jobs import Job

    panel = JobsPanel()
    qtbot.addWidget(panel)
    client = _client(server["app"])
    panel.queue = RemoteJobQueue(client)  # type: ignore[assignment]
    accounts.adopt_session("anna", client.user["role"], client.user["name"])  # as connect_server
    finished: list[Any] = []
    panel.job_finished.connect(finished.append)
    job = Job(
        plate_source=open_plate(resolve(server["plate"])).source,
        pipeline=PIPELINE,
        label="from the desktop",
        checksums=False,
    )
    panel.add_jobs([job])
    assert panel.jobs_table.rowCount() == 1 and panel.queue.running
    _work_until_done([s.id for s in panel.states])
    panel.poll()
    assert finished and finished[0].status == "completed" and not panel.queue.running
    accounts.sign_out()
