"""The REST + WebSocket API of server mode (Phase 27).

Run it with ``hcs-studio serve`` (uvicorn) behind an HTTPS reverse proxy. Everything that
changes state goes through the service layer, so it is permission-checked and audited exactly
as in the desktop app; the acting user is the API session's account.

Endpoints (JSON unless noted; all but ``/api/health`` and ``/api/login`` need
``Authorization: Bearer <token>``):

=================================  ===========================================================
``GET  /api/health``               version, database, queue counts
``POST /api/login``                {username, password} → {token, user, role, expires}
``POST /api/logout`` · ``GET /api/me``
``GET  /api/plates``               registry plates (``text``, ``barcode``, ``assay`` …)
``GET  /api/runs/{id}``            run summary, state, signatures, files, per-well table
``GET  /api/runs/{id}/files/{p}``  a file of the run folder (reports, previews, tables)
``GET  /api/images/{id}/{well}/{field}/info`` · ``…/tile``  raw pixel chunks for viewers
``POST /api/jobs``                 submit a run {plate, pipeline, wells, layout, priority}
``GET  /api/jobs`` · ``/api/jobs/{id}`` · ``/api/jobs/{id}/events?since=``
``POST /api/jobs/{id}/cancel|retry|resume``
``WS   /api/jobs/{id}/ws?token=``  progress events as they happen
``POST /api/events``               lab events from the scheduler (Phase 19 contract)
``GET  /api/admin/status``         queue, workers, GPU and disk use, failed jobs (admins)
=================================  ===========================================================

Image tiles are the raw values of one channel of one plane (``uint16``, or ``float32`` for
float data), little-endian, ``tile`` × ``tile`` at a power-of-two downsampling ``level``; the
shape is in the ``X-Shape`` header. Viewers set their own contrast, so nothing is lost.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the endpoint annotations, and the
# FastAPI types are imported inside ``create_app`` (the server extra is optional).
import hashlib
import json
import secrets
import shutil
import time
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from hcs_studio import __version__
from hcs_studio.services import Actor, ServiceContext

TILE = 512


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- sessions


def issue_token(username: str, client: str = "") -> tuple[str, datetime]:
    from hcs_studio.config import load_settings
    from hcs_studio.db import session_scope
    from hcs_studio.db.server import ApiToken

    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(hours=load_settings().server.session_hours)
    with session_scope() as session:
        session.add(
            ApiToken(
                username=username,
                client=client,
                expires=expires,
                token_sha256=hashlib.sha256(token.encode()).hexdigest(),
            )
        )
    return token, expires


def actor_for_token(token: str) -> Actor | None:
    """The account behind a token (None when unknown, expired, revoked or idle too long)."""
    from sqlalchemy import select

    from hcs_studio.config import load_settings
    from hcs_studio.db import session_scope
    from hcs_studio.db.server import ApiToken
    from hcs_studio.security.accounts import get_account

    digest = hashlib.sha256(token.encode()).hexdigest()
    idle = timedelta(minutes=load_settings().security.session_timeout_min)
    now = _now()
    with session_scope() as session:
        row = session.scalar(select(ApiToken).where(ApiToken.token_sha256 == digest))
        if (
            row is None
            or row.revoked
            or _aware(row.expires) < now
            or now - _aware(row.last_used) > idle
        ):
            return None
        row.last_used = now
        username = row.username
    account = get_account(username)
    if account is None or not account.active:
        return None
    return Actor(account.username, account.display_name, "account", account.role)


def revoke_token(token: str) -> None:
    from sqlalchemy import update

    from hcs_studio.db import session_scope
    from hcs_studio.db.server import ApiToken

    digest = hashlib.sha256(token.encode()).hexdigest()
    with session_scope() as session:
        session.execute(
            update(ApiToken).where(ApiToken.token_sha256 == digest).values(revoked=True)
        )


# --------------------------------------------------------------------------- helpers


def _run_folder(run_id: int) -> tuple[Any, Path]:
    from hcs_studio.db import session_scope
    from hcs_studio.db.catalog import Run
    from hcs_studio.paths import resolve

    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise KeyError(f"No run {run_id}")
        folder = Path(resolve(run.folder))
        session.expunge(run)
    return run, folder


def _safe_file(folder: Path, relative: str) -> Path:
    target = (folder / relative).resolve()
    if folder.resolve() not in target.parents and target != folder.resolve():
        raise PermissionError("Outside the run folder")
    if not target.is_file():
        raise FileNotFoundError(relative)
    return target


class PlateCache:
    """Plates opened for image tiles (a few at a time; opening reads metadata)."""

    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.plates: OrderedDict[str, Any] = OrderedDict()

    def get(self, folder: Path) -> Any:
        from hcs_studio.io.base import reopen_plate
        from hcs_studio.paths import resolve

        key = str(folder)
        if key in self.plates:
            self.plates.move_to_end(key)
            return self.plates[key]
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        source = dict(manifest["plate"]["source"])
        rooted = manifest["plate"].get("rooted_path")
        if rooted and not Path(source.get("path", "")).exists():
            source["path"] = str(resolve(rooted))
        plate = reopen_plate(source)
        self.plates[key] = plate
        while len(self.plates) > self.size:
            self.plates.popitem(last=False)
        return plate


def field_info(plate: Any, well: str, field: int) -> dict[str, Any]:
    fld = next((f for f in plate.well(well).fields if f.index == field), None)
    if fld is None:
        raise KeyError(f"{well} has no field {field}")
    data = fld.load()
    t, c, z, y, x = (int(v) for v in data.shape)
    levels = max(1, int(np.ceil(np.log2(max(y, x) / TILE))) + 1)
    return {
        "shape": [t, c, z, y, x],
        "dtype": str(data.dtype),
        "tile": TILE,
        "levels": levels,
        "channels": [ch.name for ch in plate.metadata.channels],
        "pixel_size_um": plate.metadata.pixel_size_um,
    }


def field_tile(
    plate: Any, well: str, field: int, *, c: int, z: int, t: int, level: int, ty: int, tx: int
) -> tuple[bytes, tuple[int, int], str]:
    fld = next((f for f in plate.well(well).fields if f.index == field), None)
    if fld is None:
        raise KeyError(f"{well} has no field {field}")
    step = 2**level
    span = TILE * step
    plane = fld.load()[
        t, c, z, ty * span : (ty + 1) * span : step, tx * span : (tx + 1) * span : step
    ]
    values = np.asarray(plane.compute() if hasattr(plane, "compute") else plane)
    if np.issubdtype(values.dtype, np.integer) and values.dtype.itemsize <= 2:
        out, kind = values.astype("<u2"), "uint16"
    else:
        out, kind = values.astype("<f4"), "float32"
    return out.tobytes(), (int(out.shape[0]), int(out.shape[1])), kind


def gpu_status() -> list[dict[str, Any]]:
    """nvidia-smi's view of the GPUs (empty without NVIDIA GPUs)."""
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                }
            )
    return rows


# --------------------------------------------------------------------------- the app


def create_app() -> Any:
    """The FastAPI application (needs the ``server`` extra)."""
    from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket
    from fastapi.responses import FileResponse, Response

    from hcs_studio.server.queue import DbJobQueue

    app = FastAPI(title="HCS Studio server", version=__version__)
    queue = DbJobQueue()
    plates = PlateCache()
    app.state.queue = queue

    def user(request: Request, authorization: str = Header(default="")) -> Actor:
        token = authorization.removeprefix("Bearer ").strip()
        actor = actor_for_token(token) if token else None
        if actor is None:
            raise HTTPException(401, "Sign in first (POST /api/login)")
        request.state.token = token
        return actor

    def ctx(request: Request, actor: Actor, reason: str = "") -> ServiceContext:
        client = request.client.host if request.client else "api"
        return ServiceContext(actor=actor, reason=reason, client=f"api:{client}")

    def fail(exc: Exception) -> HTTPException:
        if isinstance(exc, PermissionError):
            return HTTPException(403, str(exc))
        if isinstance(exc, KeyError | FileNotFoundError):
            return HTTPException(404, str(exc).strip("'"))
        return HTTPException(400, str(exc))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from collections import Counter

        from hcs_studio.db.engine import get_engine

        counts = Counter(s.status for s in queue.jobs(500))
        return {"version": __version__, "database": get_engine().dialect.name, "jobs": dict(counts)}

    @app.post("/api/login")
    def login(request: Request, body: dict[str, str] = Body(...)) -> dict[str, Any]:
        from hcs_studio.security.accounts import AuthError, authenticate

        try:
            info = authenticate(body.get("username", ""), body.get("password", ""))
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from None
        token, expires = issue_token(info.username, request.client.host if request.client else "")
        return {
            "token": token,
            "user": info.username,
            "name": info.display_name,
            "role": info.role,
            "expires": expires.isoformat(),
        }

    @app.post("/api/logout")
    def logout(request: Request, actor: Actor = Depends(user)) -> dict[str, str]:
        revoke_token(request.state.token)
        return {"status": "signed out"}

    @app.get("/api/me")
    def me(actor: Actor = Depends(user)) -> dict[str, str]:
        return {"user": actor.id, "name": actor.name, "role": actor.role}

    @app.get("/api/plates")
    def list_plates(
        actor: Actor = Depends(user),
        text: str = "",
        barcode: str = "",
        assay: str = "",
        experiment: str = "",
        cell_line: str = "",
        compound: str = "",
    ) -> list[dict[str, Any]]:
        from hcs_studio.db import session_scope
        from hcs_studio.db.queries import PlateQuery, search_plates

        query = PlateQuery(
            text=text,
            barcode=barcode,
            assay=assay,
            experiment=experiment,
            cell_line=cell_line,
            compound=compound,
        )
        with session_scope() as session:
            frame = search_plates(session, query)
        return list(json.loads(frame.to_json(orient="records", date_format="iso")))

    @app.get("/api/plates/{plate_id}")
    def plate(plate_id: int, actor: Actor = Depends(user)) -> dict[str, Any]:
        from hcs_studio.db import session_scope
        from hcs_studio.db.queries import plate_page

        try:
            with session_scope() as session:
                page = plate_page(session, plate_id)
        except Exception as exc:
            raise fail(exc) from None
        return dict(json.loads(json.dumps(page, default=_json)))

    @app.get("/api/runs/{run_id}")
    def run(run_id: int, actor: Actor = Depends(user)) -> dict[str, Any]:
        from hcs_studio.security.signoff import status

        try:
            row, folder = _run_folder(run_id)
        except Exception as exc:
            raise fail(exc) from None
        state = status(folder)
        table = folder / "tables" / "per_well_summary.csv"
        wells: list[dict[str, Any]] = []
        if table.exists():
            import pandas as pd

            wells = list(json.loads(pd.read_csv(table).to_json(orient="records")))
        files = sorted(str(p.relative_to(folder)) for p in folder.glob("*") if p.is_file())
        return {
            "id": row.id,
            "name": row.name,
            "folder": row.folder,
            "status": row.status,
            "pipeline": row.pipeline_name,
            "state": state.state,
            "locked": state.locked,
            "version": state.version,
            "signatures": state.signatures,
            "files": files,
            "wells": wells,
        }

    @app.get("/api/runs/{run_id}/files/{path:path}")
    def run_file(run_id: int, path: str, actor: Actor = Depends(user)) -> Any:
        try:
            _, folder = _run_folder(run_id)
            return FileResponse(_safe_file(folder, path))
        except Exception as exc:
            raise fail(exc) from None

    @app.get("/api/images/{run_id}/{well}/{field}/info")
    def image_info(
        run_id: int, well: str, field: int, actor: Actor = Depends(user)
    ) -> dict[str, Any]:
        try:
            _, folder = _run_folder(run_id)
            return field_info(plates.get(folder), well, field)
        except Exception as exc:
            raise fail(exc) from None

    @app.get("/api/images/{run_id}/{well}/{field}/tile")
    def image_tile(
        run_id: int,
        well: str,
        field: int,
        c: int = 0,
        z: int = 0,
        t: int = 0,
        level: int = 0,
        ty: int = 0,
        tx: int = 0,
        actor: Actor = Depends(user),
    ) -> Any:
        try:
            _, folder = _run_folder(run_id)
            data, shape, kind = field_tile(
                plates.get(folder), well, field, c=c, z=z, t=t, level=level, ty=ty, tx=tx
            )
        except Exception as exc:
            raise fail(exc) from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Shape": f"{shape[0]},{shape[1]}", "X-Dtype": kind},
        )

    @app.post("/api/jobs")
    def submit(
        request: Request, body: dict[str, Any] = Body(...), actor: Actor = Depends(user)
    ) -> dict[str, str]:
        from hcs_studio.services.operations import submit_job

        try:
            spec = job_spec_from_request(body, actor)
            job_id = submit_job(
                ctx(request, actor, body.get("reason", "")),
                queue,
                spec.job,
                spec.priority,
                spec=spec,
            )
        except Exception as exc:
            raise fail(exc) from None
        return {"id": job_id}

    @app.get("/api/jobs")
    def jobs(
        actor: Actor = Depends(user), limit: int = Query(100, le=1000)
    ) -> list[dict[str, Any]]:
        return [job_dict(s) for s in queue.jobs(limit)]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, actor: Actor = Depends(user)) -> dict[str, Any]:
        try:
            return job_dict(queue.status(job_id))
        except Exception as exc:
            raise fail(exc) from None

    @app.get("/api/jobs/{job_id}/events")
    def job_events(
        job_id: str, since: int = 0, actor: Actor = Depends(user)
    ) -> list[dict[str, Any]]:
        return queue.events(job_id, since)

    @app.post("/api/jobs/{job_id}/{action}")
    def job_action(
        request: Request,
        job_id: str,
        action: str,
        body: dict[str, Any] = Body(default={}),
        actor: Actor = Depends(user),
    ) -> dict[str, Any]:
        from hcs_studio.services import operations as ops

        functions = {"cancel": ops.cancel_job, "retry": ops.retry_job, "resume": ops.resume_job}
        if action not in functions:
            raise HTTPException(404, f"Unknown action {action}")
        try:
            new = functions[action](ctx(request, actor, body.get("reason", "")), queue, job_id)
        except Exception as exc:
            raise fail(exc) from None
        return {"id": job_id, "new_job": new}

    @app.websocket("/api/jobs/{job_id}/ws")
    async def job_socket(
        websocket: WebSocket, job_id: str, token: str = "", since: int = 0
    ) -> None:
        import asyncio

        if actor_for_token(token) is None:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        seq = since
        while True:
            events = await asyncio.to_thread(queue.events, job_id, seq)
            for event in events:
                seq = event["seq"]
                await websocket.send_json(event)
            state = await asyncio.to_thread(queue.status, job_id)
            if state.status not in ("queued", "running") and not events:
                await websocket.send_json({"type": "closed", "status": state.status})
                await websocket.close()
                return
            await asyncio.sleep(0.5)

    @app.post("/api/events")
    def lab_event(body: dict[str, Any] = Body(...), actor: Actor = Depends(user)) -> dict[str, Any]:
        from hcs_studio.automation.intake import receive

        return receive(body, "http").to_dict()

    @app.get("/api/admin/status")
    def admin_status(actor: Actor = Depends(user)) -> dict[str, Any]:
        from collections import Counter

        from hcs_studio.config import load_settings
        from hcs_studio.paths import resolve

        if actor.role != "admin":
            raise HTTPException(403, "The admin page is for admins")
        recent = queue.jobs(500)
        disk: dict[str, Any] = {}
        root = load_settings().server.results_root
        if root:
            try:
                usage = shutil.disk_usage(resolve(root) if ":" in root.split("/", 1)[0] else root)
                disk = {"path": root, "total_gb": usage.total / 1e9, "free_gb": usage.free / 1e9}
            except (OSError, KeyError, ValueError) as exc:
                disk = {"path": root, "error": str(exc)}
        return {
            "jobs": dict(Counter(s.status for s in recent)),
            "workers": queue.workers(),
            "gpus": gpu_status(),
            "disk": disk,
            "failed": [
                job_dict(s)
                for s in recent
                if s.status in ("error", "interrupted", "completed_with_failures")
            ][:50],
        }

    app.state.started = time.time()
    return app


def job_spec_from_request(body: dict[str, Any], actor: Actor) -> Any:
    """A job spec from an API request; the plate is opened on the server (paths are data-root
    paths or server paths)."""
    from hcs_studio.config import load_settings
    from hcs_studio.io.base import open_plate
    from hcs_studio.paths import resolve
    from hcs_studio.pipeline.jobs import Job
    from hcs_studio.pipeline.queue import JobSpec, Resources
    from hcs_studio.pipeline.schema import Pipeline

    def local(path: str | None) -> str | None:
        if not path:
            return None
        head = path.split("/", 1)[0]
        return str(resolve(path)) if ":" in head and not Path(path).exists() else path

    plate_path = local(body.get("plate"))
    if not plate_path:
        raise ValueError("Give the plate (a data-root path such as SCREENING:2026/plate01)")
    plate = open_plate(Path(plate_path))
    pipeline = Pipeline.model_validate(body["pipeline"])
    output = local(body.get("output_root")) or local(load_settings().server.results_root)
    if not output:
        raise ValueError("The server has no results folder (settings → server.results_root)")
    wells = [tuple(w) for w in body["wells"]] if body.get("wells") else None
    job = Job(
        plate_source=plate.source,
        pipeline=pipeline.to_dict(),
        wells=wells,
        layout_path=local(body.get("layout")),
        output_root=output,
        workers=int(body.get("workers", 1)),
        save_images=bool(body.get("save_images", True)),
        checksums=bool(body.get("checksums", True)),
        label=str(body.get("label") or f"{plate.name} · {pipeline.name}"),
    )
    gpu = False
    try:
        gpu = any(step.create().uses_gpu() for step in pipeline.image_steps())
    except Exception:
        gpu = False
    return JobSpec(
        job=job,
        priority=body.get("priority", "interactive"),
        owner=actor,
        resources=Resources(gpu=gpu),
    )


def job_dict(state: Any) -> dict[str, Any]:
    return {
        "id": state.id,
        "status": state.status,
        "label": state.job.label,
        "owner": state.spec.owner.id,
        "priority": state.spec.priority,
        "done": state.done,
        "total": state.total,
        "folder": state.folder,
        "message": state.message,
        "failed": state.failed,
        "gpu": state.spec.resources.gpu,
        "created": state.created.isoformat() if state.created else None,
        "started": state.started.isoformat() if state.started else None,
        "finished": state.finished.isoformat() if state.finished else None,
    }


def _json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
