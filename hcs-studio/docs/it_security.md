# IT security checklist (server mode)

For review with IT before HCS Studio runs as a server. Items marked ☐ are decisions or checks
for the installation.

## Network and ports

| From | To | Port | Purpose |
|---|---|---|---|
| lab PCs, scheduler | reverse proxy | 443/TCP (HTTPS) | desktop app, web portal, API, WebSocket progress |
| reverse proxy | API container | 8000/TCP (internal Docker network only) | |
| API, workers | PostgreSQL | 5432/TCP (internal Docker network only) | registry, audit trail, job queue |
| server | raw-data share | SMB/NFS | read-only mount of the imaging data |
| server | results share | SMB/NFS | run folders (read-write) |

- ☐ Only port 443 is published; the server is reachable from the internal network only.
- ☐ TLS 1.2+ with the institutional certificate (`deploy/certs/`); HSTS is set by nginx.
- No outbound internet access is needed at run time. Cellpose/SAM weights come from
  huggingface.co once (pre-load them into the image or the settings volume if the server has
  no internet access).

## Accounts and access

- Local HCS Studio accounts (no SSO in this version); roles viewer < analyst < reviewer <
  admin (permission matrix: `hcs-studio accounts matrix`).
- Passwords: argon2id hashes; 5 wrong attempts lock an account for 15 minutes.
- API sessions: random 256-bit tokens; only their SHA-256 is stored; expiry 12 h, idle
  timeout 30 min; sign-out revokes. The portal keeps its token in an HttpOnly, SameSite=Strict,
  Secure cookie.
- ☐ Who are the admins? (They manage accounts and settings.)
- ☐ The database account `hcs` is used only by HCS Studio; for an append-only audit table,
  apply `security.audit.postgres_append_only_sql("<app role>")` with a separate owner role.

## Data flows

- Raw images are read from the raw-data share (mounted read-only) and never changed.
- Results (tables, images, reports) are written to the results share, one folder per run.
- The database holds metadata, per-well results, the job queue and the audit trail — no
  images.
- The web portal streams image pixels on request; nothing is cached outside the server.

## Secrets

- The database password comes from the environment (`deploy/.env`, not committed) or a secret
  store; settings files never contain secrets (HCS Studio refuses plain secrets in them).
- ☐ Restrict read access to `deploy/.env` and `deploy/certs/` to the service account.

## Audit trail and integrity

- Every change is recorded with user, role, time (UTC), reason, client and before/after,
  hash-chained (SHA-256). `hcs-studio audit verify` detects edited, deleted or rewritten
  records; chain-head checkpoints are kept outside the database (`audit_head_copies`).
- ☐ Point `audit_head_copies` at a folder on another machine.

## Backups and recovery

- ☐ Daily `pg_dump` of the database; retention per lab policy; test a restore.
- ☐ Results share and checkpoint folder covered by the institutional backup.
- Upgrades migrate the schema automatically; roll back by restoring the pre-upgrade backup
  (docs/server.md).

## Updates

- The image is built from the repository (`deploy/Dockerfile`, Python 3.12 slim base, pinned
  dependency ranges). ☐ Rebuild monthly for OS security updates.

## Compliance

These controls support regulated work but do not make an installation compliant by
themselves; see the note in [security.md](security.md).
