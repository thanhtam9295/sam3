"""Phase 27 done-when: a 384-well plate is processed on the server while the submitting client
is gone; afterwards its results are visible in the web portal and the desktop client. The
server runs for real (uvicorn on a local port, a worker pool in its own processes); the
deployment files are checked for consistency (Docker itself is not available here)."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

DEPLOY = Path(__file__).parents[1] / "deploy"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    import uvicorn

    from hcs_studio.config import DataRootConfig, load_settings, save_settings
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.security import accounts
    from hcs_studio.server.api import create_app
    from hcs_studio.server.worker import run_pool

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    lab = tmp_path / "lab"
    (lab / "results").mkdir(parents=True)
    settings = load_settings()
    settings.data_roots = [DataRootConfig(name="LAB", linux=str(lab))]
    settings.server.results_root = "LAB:results"
    save_settings(settings)
    accounts.create_account("anna", "analyst", "Anna Analyst", password="anna password")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    api = threading.Thread(target=server.run, daemon=True)
    api.start()
    stop = threading.Event()
    pool = threading.Thread(
        target=run_pool,
        args=([], 1),
        kwargs={"poll_s": 0.3, "stale_s": 30, "check_s": 1, "stop": stop},
    )
    pool.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    yield {"url": f"http://127.0.0.1:{port}", "lab": lab}
    stop.set()
    pool.join(timeout=60)
    server.should_exit = True
    api.join(timeout=30)
    dispose_engines()


def test_384_well_plate_while_the_client_is_away(live_server: dict[str, Any]) -> None:
    from hcs_studio.output.results import RunResults
    from hcs_studio.pharmacology.synthetic import write_image_plate
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep
    from hcs_studio.server.client import RemoteJobQueue, ServerClient
    from hcs_studio.server.queue import DbJobQueue

    lab = live_server["lab"]
    plate_path, layout_path = write_image_plate(lab / "raw")
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="segment_classical", params={"channel": 1}),
            PipelineStep(module="measure_objects", params={"texture": False, "granularity": False}),
            PipelineStep(module="qc"),
        ]
    ).to_dict()
    laptop = ServerClient(live_server["url"])
    laptop.login("anna", "anna password")
    job_id = laptop.submit(
        f"LAB:{plate_path.relative_to(lab).as_posix()}",
        pipeline,
        layout=f"LAB:{layout_path.relative_to(lab).as_posix()}",
        workers=2,
        checksums=False,
        save_images=True,
        label="384 wells",
    )
    laptop.http.close()  # the laptop is closed: nobody watches the job
    del laptop
    queue = DbJobQueue()
    deadline = time.time() + 600
    while time.time() < deadline and queue.status(job_id).status in ("queued", "running"):
        time.sleep(2)
    state = queue.status(job_id)
    assert state.status == "completed", state.message
    from hcs_studio.io.base import open_plate

    imaged = open_plate(plate_path)
    assert state.total == state.done == imaged.n_fields > 300  # the 384-well plate's imaged wells
    # the next morning, the desktop client signs in again
    desktop = ServerClient(live_server["url"])
    desktop.login("anna", "anna password")
    remote = RemoteJobQueue(desktop)
    assert remote.status(job_id).status == "completed"
    results = RunResults(Path(remote.status(job_id).folder or ""))
    assert len(results.wells) == len(imaged.wells)
    plate = desktop.plates(text="")[0]
    run_id = desktop.get(f"/api/plates/{plate['plate_id']}")["runs"][0]["id"]
    run = desktop.run(run_id)
    assert len(run["wells"]) == len(imaged.wells) and "nuclei_count" in run["wells"][0]
    # … and the web portal shows the same run (page, report, an image tile)
    import httpx

    with httpx.Client(base_url=live_server["url"]) as browser:
        assert "HCS Studio" in browser.get("/").text
        assert browser.get(f"/api/runs/{run_id}").status_code == 401  # not signed in
        browser.post("/api/login", json={"username": "anna", "password": "anna password"})
        assert browser.cookies.get("hcs_token")  # the portal session (HttpOnly cookie)
        assert browser.get(f"/api/runs/{run_id}").json()["state"] == "completed"
        report = browser.get(f"/api/runs/{run_id}/files/index.html")
        assert report.status_code == 200 and "nuclei_count" in report.text
        tile = browser.get(f"/api/images/{run_id}/A01/1/tile")
        assert tile.status_code == 200 and tile.headers["X-Shape"] == "96,96"


def test_deployment_files_are_consistent() -> None:
    import yaml

    compose = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())
    services = compose["services"]
    assert set(services) == {"db", "api", "worker-gpu", "worker-cpu", "proxy"}
    assert services["db"]["image"] == "postgres:16"
    assert services["worker-gpu"]["command"][:2] == ["hcs-studio", "worker-pool"]
    assert (
        services["worker-gpu"]["deploy"]["resources"]["reservations"]["devices"][0]["driver"]
        == "nvidia"
    )
    assert services["proxy"]["ports"] == ["${HTTPS_PORT:-443}:443"]  # the only published port
    for name in ("api", "worker-gpu", "worker-cpu"):
        assert "ports" not in services[name]
    dockerfile = (DEPLOY / "Dockerfile").read_text()
    assert "[server,segmentation]" in dockerfile and "USER hcs" in dockerfile
    nginx = (DEPLOY / "nginx.conf").read_text()
    assert "proxy_pass http://api:8000" in nginx and "Upgrade $http_upgrade" in nginx
    env = (DEPLOY / ".env.example").read_text()
    assert "POSTGRES_PASSWORD" in env and os.sep not in "x"
    import json

    from hcs_studio.config import AppSettings

    AppSettings.model_validate(
        json.loads((DEPLOY / "settings" / "settings.example.json").read_text())
    )


PORTAL_SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const [url, runId] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(url);
  await page.fill('#user', 'anna');
  await page.fill('#pass', 'anna password');
  await page.click('#login button');
  await page.waitForSelector('#list table');
  const plates = await page.$$eval('#list tr.link', (rows) => rows.length);
  await page.click('#list tr.link');
  await page.waitForSelector('tr.link');
  await page.click('tr.link');                       // the run
  await page.waitForSelector('text=State:');
  const state = await page.textContent('.card');
  await page.click('tr.link');                       // a well → the viewer
  await page.waitForFunction(() => {
    const c = document.querySelector('#canvas');
    return c && c.width === 96;
  });
  const lit = await page.evaluate(() => {
    const c = document.querySelector('#canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 60) n++;
    return n;
  });
  await page.click('nav a[data-view=jobs]');
  await page.waitForSelector('text=Progress');
  console.log(JSON.stringify({plates, state, lit, errors}));
  await browser.close();
})();
"""


def _node_playwright() -> bool:
    import subprocess

    try:
        return (
            subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode
            == 0
        )
    except OSError:
        return False


@pytest.mark.skipif(not _node_playwright(), reason="Node.js with Playwright is not available")
def test_portal_in_a_browser(live_server: dict[str, Any], tmp_path: Path) -> None:
    """The portal: sign in, plates, the run (state), the image viewer draws, jobs."""
    import json
    import subprocess

    from hcs_studio.server.client import ServerClient
    from hcs_studio.server.queue import DbJobQueue
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    lab = live_server["lab"]
    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1)],
        channels=("DAPI", "GFP"),
        size=96,
        name="WEB01",
        features=[Nuclei(count=lambda r, c, f: 10), ShotNoise(sd=4)],
    )
    raw, _ = plate.write(lab / "raw")
    client = ServerClient(live_server["url"])
    client.login("anna", "anna password")
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    pipeline = Pipeline(
        steps=[
            PipelineStep(module="segment_classical", params={"channel": 1}),
            PipelineStep(module="measure_objects", params={"texture": False}),
        ]
    ).to_dict()
    job_id = client.submit(f"LAB:{raw.relative_to(lab).as_posix()}", pipeline, checksums=False)
    queue = DbJobQueue()
    deadline = time.time() + 120
    while time.time() < deadline and queue.status(job_id).status in ("queued", "running"):
        time.sleep(1)
    assert queue.status(job_id).status == "completed"
    script = tmp_path / "portal.js"
    script.write_text(PORTAL_SCRIPT)
    done = subprocess.run(
        ["node", str(script), live_server["url"], "1"], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["errors"] == [], result["errors"]
    assert result["plates"] == 1 and "completed" in result["state"]
    assert result["lit"] > 200  # nuclei drawn on the canvas
