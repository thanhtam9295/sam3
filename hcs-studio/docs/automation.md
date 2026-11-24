# Automatic runs and the scheduler

When a plate finishes imaging, HCS Studio can pick the right layout and pipeline, run QC and the
analysis, write the report and email people — with no clicks — and talk back to the lab
scheduler (plate handler, washer, imager).

```
plate.imaged event (scheduler)  ─┐
                                 ├→ acquisition registered (once) → raw data complete?
finished export in a watched     ┘     → routing rule → image QC → pipeline → pharmacology /
folder (ImageXpress, Harmony, .nd2)      hit calling / kinetics → registry → report
                                          → analysis.completed + email
                                   QC failed → reimage.request + analysis.failed + email
```

## Triggers

**Scheduler events (preferred).** The scheduler sends a `plate.imaged` event (format: the
shared `lab_events` JSON Schemas, `sam3/lab_events/src/lab_events/schemas/v1/`) in either of two
ways — both are on when configured:

- **HTTP:** `POST http://<pc>:<http_port>/events` (local automation) or
  `POST https://<server>/api/events` (server mode, with an account token). Replies: 202 accepted,
  200 duplicate, 400 rejected with the schema errors.
- **Drop folder:** the scheduler writes `<inbox>/<anything>.json`. Files are read once they are
  a second old (not half-written), then moved to `processed/` or `rejected/` (with a
  `.error.txt` listing the reasons).

The event carries the barcode, instrument, protocol, acquisition start/end, the raw-data location
(preferably a data root + path, e.g. `{"root": "SCREENING", "path": "IX/2026-11-24/P0042"}`) and,
for time courses, the time-point index. A `plate.dispensed` event records the treatment time
(used by kinetics).

**Watched folders (fallback).** Each watched folder is checked every `poll_seconds`. A plate
export inside it is started only when **nothing changed for `quiet_seconds`** and the format's
**completeness check** passes, so a plate that is still being copied is never started:

| Export | Complete when |
|---|---|
| ImageXpress (MetaXpress export: `.HTD` + TIFFs) | the number of image TIFFs the `.HTD` implies (selected wells × sites × wavelengths × time points × Z planes, plus the projection MetaXpress saves with a Z series) is present and every TIFF's pixel data is fully written |
| Operetta / Harmony export (`Images/Index.idx.xml`) | the index XML parses and every image it lists exists and is readable |
| Nikon NIS-Elements `.nd2` (e.g. the Ji) | every file opens and its last frame can be read |
| OME-Zarr and other formats | the plate opens |

A new imager needs a reader (`io/`) and a checker (`automation/completeness.py`); nothing else.
Watching works by polling, which behaves the same on local disks and network shares.

**Each acquisition runs once.** The key is barcode + acquisition ID (or start time) + time
point. The same raw data reported again within `duplicate_window_hours` (24 h) is also a
duplicate — e.g. when both the scheduler and a watcher report the plate. Duplicates are logged
and ignored; a re-imaged plate (new acquisition ID, new export folder) runs again.

## Routing rules

A YAML file (`automation.routing_file`) chooses the pipeline. Rules are tried in order; a rule
matches when **every** criterion it gives matches; `default: true` matches everything (last).

```yaml
rules:
  - name: viability screen
    barcode_prefix: VIA              # or barcode_regex: "^VIA-\\d{4}$"
    instrument: ImageXpress          # case-insensitive
    protocol: "20x*"                 # shell-style pattern
    assay: viability                 # the registry's assay of the barcode
    pipeline: pipelines/viability.yaml   # paths are relative to this file
    qc_rules: qc/ix_rules.yaml       # image QC before the analysis (set plate.stop_run: true)
    pharmacology: {readouts: [nuclei_count], normalization: percent_inhibition}
    screen: screens/primary.yaml     # hit calling (ScreenConfig)
    kinetics: false                  # analyse the plate's time points together
    notify: [owner@lab.org]          # the people who own these plates
  - name: everything else
    default: true
    pipeline: pipelines/basic.yaml
```

`hcs-studio auto check` lists problems (missing files, invalid pipelines, rules that can never
match, missing folders).

- **Layouts** are attached by barcode (the latest saved version, Phase 11). Without one, the
  image analysis still runs, pharmacology and hit calling wait, and the owners get an email
  asking them to annotate the plate (then `hcs-studio annotate <run> --layout …`).
- **Time courses:** an acquisition with a time-point index is its own run; with
  `kinetics: true` the plate's finished time points are analysed together after each one.
- **QC:** with `qc_rules` and `plate.stop_run: true`, a plate that fails QC stops before the
  analysis; the scheduler gets a `reimage.request` (failed wells and fields, the reasons) when
  `reimage_on_qc_fail` is on, and an `analysis.failed` (stage `qc`).

## Priorities

Automatic runs go into the same job queue as interactive ones with the lower priority `auto`:
anything a user starts runs first (running jobs are never interrupted). At night
(`night_hours`, e.g. `"20-7"`) automatic runs may use `night_workers` processes — the whole
machine.

## Notifications

Email (SMTP) is the lab's channel; Teams/Slack webhooks and desktop messages exist too. A
message names the barcode, assay, routing rule, QC verdict, key numbers (mean cell counts, Z′)
and the report. Every channel lists the kinds it wants (`on`: `completed`, `qc_failed`,
`failed`, `needs_layout`, `unrouted`); failures also go to the rule's owners and the `admins`.

```json
"channels": [{"name": "lab mail", "kind": "email", "to": ["screening@lab.org"],
              "smtp_host": "smtp.lab.org", "smtp_port": 587, "smtp_tls": true,
              "username": "hcs-studio", "password": "secret:smtp/password"}],
"admins": ["it-admin@lab.org"]
```

The SMTP password (and any webhook URL) is a keychain reference: store it with the OS keychain
(service `hcs-studio`, name `smtp/password`) or, on a server, the environment variable
`HCS_SECRET_SMTP_PASSWORD`. `hcs-studio auto test-notify` sends a test message to every channel.

## Events sent to the scheduler

Every outgoing event is validated against the schema before it is sent, written to the
`outbox` folder (`<time>_<type>_<id>.json`, in sending order) and/or POSTed to
`scheduler_url`, and carries the incoming event's ID as `correlation_id`:

| Event | When |
|---|---|
| `analysis.started` | the job is queued (run ID = job ID, pipeline) |
| `analysis.completed` | the report is written: QC verdict (pass/warn/fail), report path (data-root form), key numbers |
| `analysis.failed` | QC failure (stage `qc`), missing or incomplete data (`registration`), job errors (`analysis`) |
| `reimage.request` | QC failure, when `reimage_on_qc_fail` is on |
| `cherrypick.request` | after cherry-picking hits (Phase 23) |

## Running it

```
hcs-studio auto serve            # events + watchers + runs, until Ctrl-C (a lab PC or the server)
hcs-studio auto status           # the automatic runs
hcs-studio auto events [--status rejected] [--direction in]   # the event log
hcs-studio auto retry <key> --reason "data copied again"
hcs-studio auto check | test-notify | send <event.json>
```

In server mode (PostgreSQL registry), `auto serve` submits to the server's job queue, so the GPU
workers run the jobs; run it once, next to the API (scheduler events can also go to the API's
`/api/events`).

**Automation tab** in the app: the automatic runs (open their results, start one again with a
reason) and the **event log** — every event received or sent, with its status (accepted,
duplicate, rejected with the reasons, ignored, sent, send failed) and the full message.

Settings (`settings.json` → `automation`): `enabled`, `inbox`, `outbox`, `http_port`/`http_host`
(local HTTP intake; 127.0.0.1 by default), `scheduler_url`, `watch` (`path`, `instrument`,
`protocol`, `quiet_seconds`), `routing_file`, `output_root` (a data-root path is fine), `workers`,
`night_hours`, `night_workers`, `reimage_on_qc_fail`, `data_wait_minutes` (give up on data that
stays incomplete), `poll_seconds`, `duplicate_window_hours`, `channels`, `admins`.

## Validation status

Tested on synthetic exports (`tests/test_automation*.py`): completeness of ImageXpress, Harmony
and (simulated) `.nd2` exports, including half-written TIFFs; a slow copy is not started early;
duplicates (resent event, same raw data from the watcher) do not run twice; a dropped
ImageXpress export runs QC → analysis → report → email end to end with real job processes and a
local SMTP server; a QC failure sends a schema-valid `reimage.request`. **Not yet checked on
real exports** from MetaXpress, Harmony and NIS-Elements on your shares, or with the real
scheduler — please run one plate of each through `auto serve` and send the event log if
anything is rejected or waits forever.
