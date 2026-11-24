"""Clients of the server API (Phase 27): the desktop app in connected mode, scripts, the
scheduler. ``ServerClient`` wraps the REST calls; ``RemoteJobQueue`` implements the same
``JobQueue`` protocol as the local queue, so the Jobs dock works unchanged against a server.

Paths sent to the server are data-root paths (``ROOT:relative``): the server resolves them on
its own mounts, so a laptop's ``Z:\\screening`` and the server's ``/mnt/screening`` agree.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio.pipeline.jobs import Job
from hcs_studio.pipeline.queue import JobSpec, JobState, Resources
from hcs_studio.services import Actor


class ServerError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


class ServerClient:
    """REST calls to an HCS Studio server. ``http`` may be any httpx-compatible client (tests
    pass FastAPI's ``TestClient``)."""

    def __init__(
        self, url: str = "", http: Any = None, verify: bool = True, timeout: float = 60
    ) -> None:
        import httpx

        self.http = http or httpx.Client(base_url=url.rstrip("/"), verify=verify, timeout=timeout)
        self.token = ""
        self.user: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _check(self, response: Any) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ServerError(response.status_code, str(detail))
        return response

    def get(self, path: str, **params: Any) -> Any:
        return self._check(self.http.get(path, params=params, headers=self._headers())).json()

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._check(self.http.post(path, json=body or {}, headers=self._headers())).json()

    def login(self, username: str, password: str) -> dict[str, Any]:
        self.user = self._check(
            self.http.post("/api/login", json={"username": username, "password": password})
        ).json()
        self.token = self.user["token"]
        return self.user

    def logout(self) -> None:
        if self.token:
            self.post("/api/logout")
        self.token = ""

    def health(self) -> dict[str, Any]:
        return dict(self.get("/api/health"))

    def plates(self, **query: str) -> list[dict[str, Any]]:
        return list(self.get("/api/plates", **query))

    def run(self, run_id: int) -> dict[str, Any]:
        return dict(self.get(f"/api/runs/{run_id}"))

    def run_file(self, run_id: int, path: str) -> bytes:
        response = self.http.get(f"/api/runs/{run_id}/files/{path}", headers=self._headers())
        return bytes(self._check(response).content)

    def image_info(self, run_id: int, well: str, field: int) -> dict[str, Any]:
        return dict(self.get(f"/api/images/{run_id}/{well}/{field}/info"))

    def tile(self, run_id: int, well: str, field: int, **where: int) -> np.ndarray:
        """One raw chunk of a plane (``c``, ``z``, ``t``, ``level``, ``ty``, ``tx``)."""
        response = self._check(
            self.http.get(
                f"/api/images/{run_id}/{well}/{field}/tile", params=where, headers=self._headers()
            )
        )
        rows, cols = (int(v) for v in response.headers["X-Shape"].split(","))
        dtype = "<u2" if response.headers.get("X-Dtype") == "uint16" else "<f4"
        return np.frombuffer(response.content, dtype=dtype).reshape(rows, cols)

    def plane(
        self,
        run_id: int,
        well: str,
        field: int,
        *,
        c: int = 0,
        z: int = 0,
        t: int = 0,
        level: int = 0,
    ) -> np.ndarray:
        """A whole plane, assembled from tiles."""
        info = self.image_info(run_id, well, field)
        _, _, _, height, width = info["shape"]
        step, size = 2**level, info["tile"]
        rows, cols = -(-height // step), -(-width // step)
        out = np.zeros((rows, cols), dtype=np.float32)
        for ty in range(-(-rows // size)):
            for tx in range(-(-cols // size)):
                tile = self.tile(run_id, well, field, c=c, z=z, t=t, level=level, ty=ty, tx=tx)
                out[
                    ty * size : ty * size + tile.shape[0], tx * size : tx * size + tile.shape[1]
                ] = tile
        return out

    def submit(self, plate: str, pipeline: dict[str, Any], **options: Any) -> str:
        """Submit a run; ``plate`` (and ``layout``, ``output_root``) are data-root paths."""
        return str(self.post("/api/jobs", {"plate": plate, "pipeline": pipeline, **options})["id"])

    def job(self, job_id: str) -> dict[str, Any]:
        return dict(self.get(f"/api/jobs/{job_id}"))

    def jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self.get("/api/jobs", limit=limit))

    def events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]:
        return list(self.get(f"/api/jobs/{job_id}/events", since=since))

    def action(self, job_id: str, action: str, reason: str = "") -> dict[str, Any]:
        return dict(self.post(f"/api/jobs/{job_id}/{action}", {"reason": reason}))

    def send_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return dict(self.post("/api/events", event))

    def admin_status(self) -> dict[str, Any]:
        return dict(self.get("/api/admin/status"))


def _rooted(path: str | None) -> str | None:
    if not path:
        return None
    from hcs_studio.output.manifest import rooted_text

    return rooted_text(Path(path))


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class RemoteJobQueue:
    """The ``JobQueue`` protocol over the server API (connected desktop mode)."""

    def __init__(self, client: ServerClient) -> None:
        self.client = client
        self.known: dict[str, JobState] = {}
        self.seen: dict[str, int] = {}

    def submit(self, spec: JobSpec) -> str:
        job = spec.job
        options: dict[str, Any] = {
            "label": job.label,
            "priority": spec.priority,
            "workers": job.workers,
            "save_images": job.save_images,
            "checksums": job.checksums,
            "wells": [list(w) for w in job.wells] if job.wells else None,
            "layout": _rooted(job.layout_path),
        }
        if job.output_root:
            options["output_root"] = _rooted(job.output_root)
        plate = _rooted(str(job.plate_source.get("path", "")))
        job_id = self.client.submit(str(plate), job.pipeline, **options)
        self.known[job_id] = self.status(job_id)
        return job_id

    def _state(self, data: dict[str, Any], spec: JobSpec | None = None) -> JobState:
        if spec is None:
            job = Job(plate_source={}, pipeline={}, label=data["label"])
            spec = JobSpec(
                job=job,
                priority=data["priority"],
                owner=Actor(data["owner"], data["owner"], "account"),
                resources=Resources(gpu=bool(data.get("gpu"))),
            )
        return JobState(
            spec=spec,
            id=data["id"],
            status=data["status"],
            folder=data.get("folder"),
            done=int(data.get("done") or 0),
            total=int(data.get("total") or 0),
            failed=dict(data.get("failed") or {}),
            message=data.get("message") or "",
            created=_parse(data.get("created")) or datetime.now(UTC),
            started=_parse(data.get("started")),
            finished=_parse(data.get("finished")),
        )

    def status(self, job_id: str) -> JobState:
        known = self.known.get(job_id)
        state = self._state(self.client.job(job_id), known.spec if known else None)
        self.known[job_id] = state
        return state

    def events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]:
        return self.client.events(job_id, since)

    def jobs(self) -> list[JobState]:
        return [self._state(d) for d in self.client.jobs()]

    def cancel(self, job_id: str) -> None:
        self.client.action(job_id, "cancel")

    def retry(self, job_id: str, failed_only: bool = True) -> str | None:
        new = self.client.action(job_id, "retry").get("new_job")
        return str(new) if new else None

    def resume(self, job_id: str) -> str | None:
        new = self.client.action(job_id, "resume").get("new_job")
        return str(new) if new else None

    def poll(self) -> list[tuple[JobState, dict[str, Any]]]:
        """New events of the jobs this client watches (as the local queue's ``poll``)."""
        out = []
        for job_id, state in list(self.known.items()):
            if state.status not in ("queued", "running") and job_id in self.seen:
                continue
            events = self.events(job_id, self.seen.get(job_id, 0))
            fresh = self.status(job_id)
            for event in events:
                self.seen[job_id] = int(event["seq"])
                out.append((fresh, event))
            self.seen.setdefault(job_id, 0)
        return out

    # the attributes the Jobs dock reads from a local queue
    @property
    def states(self) -> list[JobState]:
        return list(self.known.values())

    @property
    def running(self) -> dict[str, None]:
        return {i: None for i, s in self.known.items() if s.status in ("queued", "running")}

    def toggle_pause(self, job_id: str) -> None:
        """Server jobs cannot be paused (cancel and resume instead)."""

    def is_paused(self, job_id: str) -> bool:
        return False

    def shutdown(self, timeout: float = 10) -> None:
        """Server jobs keep running when the desktop closes."""

    def watch(self, job_id: str) -> None:
        self.known[job_id] = replace(self.status(job_id))
