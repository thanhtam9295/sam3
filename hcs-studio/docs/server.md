# Server mode

Heavy processing runs on a shared GPU workstation or server; people browse, submit runs, watch
progress and review results from any PC — in the desktop app or in a web browser.

## Two modes, one codebase

| | Local mode (default) | Server mode |
|---|---|---|
| Registry | SQLite file in the settings folder | PostgreSQL |
| Job queue | in the app (`LocalJobQueue`), jobs run on this PC | in the database (`queue_job`), workers on the server |
| Who acts | the computer's user (or an app account) | the account signed in to the server |
| Clients | the desktop app | the desktop app (connected) and the web portal |

## Architecture

```
lab PCs ── HTTPS ──► reverse proxy (nginx, institutional certificate)
                         │
                         ▼
                    API (FastAPI: REST + WebSocket)  ◄── scheduler (lab events)
                         │
                         ▼
                    PostgreSQL: registry, audit trail, job queue
                         ▲
              ┌──────────┴──────────┐
        GPU worker(s)          CPU worker(s)     (one per GPU; supervised, restarted)
              │                      │
   raw data share (read-only)   results share (runs)
```

- **The queue** is a table: workers take the next job with `SELECT … FOR UPDATE SKIP LOCKED`,
  by priority (interactive, auto, batch), then fair share (the user with the fewest jobs
  running first), then age. GPU jobs go only to GPU workers.
- **Workers** run each job in its own process and write a heartbeat every few seconds. A job
  whose worker stops beating (killed, crashed, server rebooted) is queued again and **resumes
  in its run folder** (finished wells are kept), up to three attempts. `hcs-studio
  worker-pool` starts one worker per GPU plus CPU workers and restarts any that die.
- **Paths** travel as data-root paths (`SCREENING:2026/plate01`), so a laptop's
  `Z:\screening` and the server's `/mnt/raw` mean the same plate. Raw data is mounted
  read-only on the server; runs go to `server.results_root` on the results share.
- **Accounts** are HCS Studio's own local accounts (Phase 26) with argon2id passwords; API
  sessions are random tokens of which only the SHA-256 is stored, ending after
  `server.session_hours` (12) or 30 minutes idle. Every change goes through the audited
  service layer as the signed-in account.

## Clients

**Desktop app (connected).** Set `server.url` in the settings (`https://hcs.example.org`). At
start the app asks for the server account; the Jobs dock then shows and controls the server's
jobs, runs are submitted there, and results open from the results share (or in the browser
when the share is not mounted on that PC). Closing the laptop does not stop a run.

**Web portal** (`https://<server>/`): sign in; search plates; open a run (state, signatures,
per-well table, report, pharmacology tables); look at the images of a well (channels,
contrast, pan and zoom — pixel tiles are streamed, nothing is converted); follow the jobs.
Admins see the admin page: jobs by status, workers and heartbeats, GPU use, free disk on the
results share, failed jobs.

**Scripts and the scheduler** use the same API (`hcs_studio.server.client.ServerClient`):
`login`, `submit`, `job`, `events`, `send_event` (lab events, Phase 19 contract). Give the
scheduler its own account (role analyst).

**Lab assistant** (if enabled on the server): the desktop's Assistant tab sends questions to the
server, which holds the Claude API key and calls the API (`/api/assistant/…`, see
[assistant](assistant.md)).

## Running it

### Docker Compose on a Linux GPU server (recommended)

Needs Docker, the NVIDIA Container Toolkit, the raw-data share mounted read-only and the
results share mounted read-write on the host.

```bash
cd hcs-studio/deploy
cp .env.example .env                    # database password, share paths, GPUs
cp settings/settings.example.json settings/settings.json   # data roots, results root
mkdir certs && cp /path/to/server.crt /path/to/server.key certs/
docker compose up -d --build
docker compose exec api hcs-studio accounts add admin --role admin --name "…" --password --reason "first admin"
```

Services: `db` (PostgreSQL 16, data in a volume), `api`, `worker-gpu` (all GPUs; set `GPUS` to
`0,1` …), `worker-cpu`, `proxy` (nginx; the only published port, 443). The containers run as an
unprivileged user and restart automatically.

### Windows service

Not yet tested. Install PostgreSQL for Windows, the HCS Studio installer, and register two
services with a service wrapper such as NSSM, both with `HCS_DATABASE_URL` set (the password
from the environment or the Windows credential store):

```bat
nssm install HCSStudioAPI "C:\Program Files\HCS Studio\hcs-studio.exe" serve --host 127.0.0.1 --port 8000
nssm install HCSStudioWorkers "C:\Program Files\HCS Studio\hcs-studio.exe" worker-pool --gpus 0 --cpu 2
```

Put IIS or nginx with the institutional certificate in front of port 8000.

### Commands

```bash
hcs-studio serve --host 0.0.0.0 --port 8000      # the API (behind the proxy)
hcs-studio worker-pool --gpus 0,1 --cpu 2        # supervised workers of this machine
hcs-studio worker [--gpu 0]                       # one worker
hcs-studio queue list | workers | cancel <id> | reap
```

## Upgrades and rollback

The database schema is versioned with Alembic migrations shipped in the package; the API and
the workers migrate the database to the newest version when they start.

1. Back up first (below).
2. `docker compose pull` / rebuild with the new version; `docker compose up -d`. The first
   container to start migrates the database.
3. Check `GET /api/health` and the admin page.

**Rollback:** stop the containers, restore the database backup taken in step 1 (the simplest
and safest way back), and start the previous image (`HCS_VERSION=<previous>`). Alembic can also
step down one revision (`alembic downgrade -1` with the package's migration folder), but a
downgrade drops columns and tables added since — restoring the backup is preferred.

## Backups

- **Database:** `docker compose exec db pg_dump -U hcs -Fc hcs > hcs-$(date +%F).dump` daily
  (restore with `pg_restore`). It holds the registry, the job queue and the audit trail.
- **Audit chain head:** keep `security.audit_head_copies` pointing at a folder on another
  machine; back it up with the database.
- **Run folders** live on the results share; back them up with the share. Raw data is never
  changed by HCS Studio.
- **Settings:** `deploy/settings/` and `deploy/.env` (the latter contains the database
  password: store it like any secret).

See [it_security.md](it_security.md) for the checklist to review with IT before deployment.
