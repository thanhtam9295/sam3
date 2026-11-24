# Lab assistant (Claude)

Ask questions about your results in plain language — "Which combinations had a ZIP score above
10 in both cell lines?", "Why is the Z′ of plate P0042 low?", "Draft the methods paragraph for
this run" — and get answers grounded in the results registry. The assistant uses the Claude API
(Anthropic) through the official Python SDK.

The assistant is **off** until an admin turns it on. Before turning it on, read
[Data policy and retention](#data-policy-and-retention).

## What the assistant can and cannot do

It can **read** a curated copy of the registry and **suggest**. It cannot change anything:
pipelines, layouts, exclusions, QC decisions and sign-offs stay with people.

| Tool | What it does |
|---|---|
| `get_schema` | the curated views and their columns |
| `run_query` | one read-only SQL `SELECT` on the curated views (row and time limits) |
| `get_plate_layout` | a plate's wells: roles, cell lines, compounds, concentrations |
| `get_run_manifest` | how a run was made (software, pipeline steps and parameters, models, channels) |
| `get_dose_response` | dose-response fits (EC50, CI, Hill, AUC…), optionally the data points |
| `get_combination_scores` | Bliss / HSA / Loewe / ZIP synergy scores |
| `get_qc_summary` | plate QC (Z′, CVs), wells by QC flag, image QC |
| `list_readouts` | readouts with names, units and direction |
| `make_chart` | draws a chart **in HCS Studio**; the chart's data never leaves the computer |
| `highlight_wells`, `open_comparison` | change what you see (Results plate, Compare tab) |
| `propose_pipeline_change` | a parameter diff (old → new, with reasons) for you to apply — or not |
| `draft_methods_text` | a methods draft from the run manifest (structured output), for you to edit |
| `get_image_thumbnail` | **off by default**; see [Images](#images) |

### Show work and grounding

Under every answer, **Show work** lists each tool call with its input and SQL, in order. The
assistant is instructed that every number must come from a tool result and to say so when data
is missing instead of estimating. HCS Studio also checks the answer: numbers that appear in no
tool result (allowing for rounding and unit conversion, e.g. M → µM, fraction → %) are flagged
in orange — check them before using them. An answer without any tool call is flagged too.

## Using it

**Assistant tab.** Type a question and press **Ask**. Options per question:

- **Stronger model** — this question goes to the stronger model (slower, more expensive).
- **Send compound IDs only** — compound names are replaced by codes (`CPD-0007`) before anything
  is sent; the answer you see has the names back (the mapping never leaves the computer).
- **Include free-text notes** — experiment purpose/notes and plate notes are left out unless
  ticked.
- **Allow image thumbnails** — offers the image tool for this question (only if an admin allows
  images); each image is confirmed separately.
- **Data to be sent…** shows the exact first request (system prompt, schema documentation,
  tools, your question) without sending anything. Later requests add the tool results.

Changing the privacy options starts a new conversation (it needs a new copy of the data).
**New conversation** forgets the context.

**Command line:**

```
hcs-studio assistant ask "Which compound is most potent in MCF7?" [--strong] [--ids-only] [--notes] [--json]
hcs-studio assistant preview "…"          # what would be sent
hcs-studio assistant status               # settings and your usage this month
```

## Turning it on (admins)

1. Get an API key in the Claude Console (platform.claude.com). API usage is billed separately
   from Claude app subscriptions.
2. **Tools → Lab assistant settings…**: tick *Enable*, paste the key (stored in the operating
   system's keychain under `hcs-studio` / `anthropic/api_key`, never in `settings.json`), set
   the budget. Or on the command line:

   ```
   hcs-studio assistant set-key              # asks for the key
   hcs-studio assistant configure --enable --budget 2000000 [--ids-only on] [--images on]
   ```

Both need the **admin** role; every change is in the audit trail (`settings.assistant`,
`settings.assistant_key` — the key itself is never logged). Install the optional extra on
computers that call the API: `pip install "hcs-studio[assistant]"` (the Windows installer and the
server image include it).

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | off | nothing is sent while off |
| `model` / `effort` | `claude-opus-5-5` / `medium` | routine questions |
| `strong_model` / `strong_effort` | `claude-fable-5-2` / `high` | "Stronger model" questions |
| `monthly_token_budget` | 2 000 000 | weighted tokens per user and month (0 = no limit); `user_budgets` per user |
| `compound_ids_only` | off | default of the per-question option |
| `allow_images` | off | whether users may enable thumbnails at all |
| `max_rows`, `query_timeout_s` | 200, 5 s | limits of each query |
| `max_turns`, `max_tokens` | 12, 16 000 | per question |
| `refusal_fallback` | on | a request the model declines is retried on a fallback model (server-side fallback, beta) |
| `strict_tools` | on | schema-valid tool inputs (turn off in organisations with customer-managed keys) |

**Models.** The routine default is Claude Opus 5.5 at medium effort: it handles the SQL and the
statistics in these questions reliably, at a moderate price. *Stronger model* uses Claude Fable
5.2 at high effort for hard questions. Claude Sonnet 5.5 and Claude Haiku 5.5 are cheaper
alternatives for the routine model; measure on the evaluation set before switching. Current
prices: platform.claude.com/docs/en/about-claude/pricing.

**Budget.** Usage is counted per user, model and calendar month (UTC) in the registry
(`assistant_usage`) as *weighted tokens* = input + output + cache writes + 10 % of cache reads
(cache reads cost about a tenth). The tab and `assistant status` show the usage. When the budget
is used up, questions are refused until next month or until an admin raises it.

**Prompt caching.** The tool definitions and the system prompt (instructions plus the curated
schema) are identical for every question and are cached; the conversation is cached as it grows.
Nothing that changes per request (time, user, IDs) is put into the system prompt.

## What leaves the computer

Only the content of the requests: the system prompt (fixed instructions and the documentation of
the curated views), your questions, and the results of tool calls — rows of the **curated
views**, scrubbed run manifests and QC summaries. Every request is logged in the app
(`PayloadLog`; `tests/test_assistant.py` checks that nothing else is sent).

The curated views are built for each conversation as a separate read-only SQLite file:

| View | Contents |
|---|---|
| `experiments` | ID, name, project, assay, dates (purpose/notes only when included) |
| `plates` | ID, barcode, name, plate type, assay, experiment, cell lines, seeded/treated times |
| `runs` | ID, plate, name, pipeline name and hash, version, status, times, instrument, image QC passed |
| `wells` | plate, well, cell line, role, up to two compounds with concentrations (M) |
| `well_results` | run, well, readout, value, objects, QC flag |
| `readouts` | readout ID, display name, unit, level, direction, description |
| `dose_response` | fits per run, readout, cell line, compound |
| `combination_scores` | synergy scores per run, readout, cell line, pair, model |
| `plate_qc` | Z′, SSMD, signal window, CVs, means, counts, flag |
| `compounds` | name (or code), MoA, target |
| `cell_lines` | name, RRID, cancer type |

**Never sent:** file paths and folder names (runs are identified by registry IDs), user and
operator names, free-text notes (unless included for the question), compound structures
(SMILES), suppliers and lots, per-object (single-cell) data, raw images, and anything outside
these views. Manifests and QC summaries are scrubbed: path-like values, keys naming people,
hosts, files or notes are removed.

**Read-only by construction.** `run_query` opens the snapshot with SQLite's read-only mode and
`PRAGMA query_only`, and an authorizer allows only reading the curated views — writes, `ATTACH`,
`PRAGMA`, other tables and extension loading are refused by the connection, not just by the
instructions. Each query has a row limit and a time limit. The real registry is never opened by
the assistant's queries.

### Images

`get_image_thumbnail` is not offered unless an admin allows images **and** you tick *Allow image
thumbnails* for the question. Even then, each image waits for your confirmation ("Send an
image?") before anything is sent; declining tells the assistant to answer without it. A
thumbnail is the run's 8-bit preview of one field, downsampled to at most 256 px, as PNG without
metadata.

## Audit trail

Every question (`assistant.question`), tool call with its input and SQL (`assistant.tool`),
image decision (`assistant.thumbnail`) and answer with its token usage (`assistant.answer`) is
recorded in the audit trail with the user — see [security](security.md). Asking needs the
viewer role (the assistant only reads).

## Server mode

With a server (see [server mode](server.md)), the desktop sends questions to the server and the
server calls the Claude API: the key lives only on the server (its keychain or
`HCS_SECRET_ANTHROPIC_API_KEY`), and the server's settings decide whether the assistant is on.

| Endpoint | |
|---|---|
| `GET /api/assistant/status` | enabled, models, your usage and budget |
| `POST /api/assistant/ask` | `{question, strong, compound_ids_only, include_notes, images, conversation_id}` → answer with steps |
| `POST /api/assistant/{id}/confirm` | `{approved}` — answer a pending image confirmation |
| `POST /api/assistant/preview` | the request a question would send |
| `DELETE /api/assistant/{id}` | end a conversation (idle ones end after 2 h) |

Conversations belong to the account that started them. The server needs outbound HTTPS to
`api.anthropic.com` (see the [IT checklist](it_security.md)).

## Data policy and retention

**Check before enabling:**

- ☐ Your institution's data policy: may screening results, compound names and (if allowed)
  images be sent to an external AI service? Turn on *compound IDs only* by default if names are
  confidential.
- ☐ Anthropic's API data-retention terms
  (platform.claude.com/docs/en/manage-claude/api-and-data-retention) and the commercial terms of
  your organisation's agreement. At the time of writing (November 2026) they state that API
  prompts and outputs are not used for training without permission and are not retained by
  default, except for *Covered Models*, which require 30-day retention. **Claude Fable 5.2 — the
  default "stronger model" — is a Covered Model**; under a zero-data-retention (ZDR) arrangement
  it is not available unless Anthropic authorises it, so with ZDR set `strong_model` to
  `claude-opus-5-5` with `strong_effort` `high`.
- ☐ Whether you need an enterprise agreement or ZDR. Content flagged by Anthropic's safety
  systems or under legal hold may be retained longer regardless of the arrangement (the page
  above gives the terms).
- Prompt caching and the Messages API are listed as ZDR-eligible; strict tool schemas and
  structured outputs cache only the JSON schema (up to 24 h), and HCS Studio's schemas contain
  no data. The server-side refusal fallback is a beta feature; if your arrangement allows only
  listed features, turn off `refusal_fallback`.

These terms change; re-read the page when you renew the agreement.

## Evaluation

An evaluation set of 30 questions with known answers on the synthetic registry (three runs of the
known-answer combination plate) checks that answers match SQL ground truth (numbers within
rounding and unit conversion, or names) and that every answer shows its work. Done-when for this
phase: ≥ 90 % pass.

It calls the live API and costs tokens (roughly 30 questions × a few requests), so it runs only
on request:

```
hcs-studio assistant eval --workdir <new folder> [--strong] [--only zip_all ec50_b]
HCS_ASSISTANT_EVAL=1 pytest tests/test_assistant_eval.py      # key from the keychain or HCS_SECRET_ANTHROPIC_API_KEY
```

The report (`eval.json`) lists each question, the expected values, the answer, the tools and SQL
used, and tokens. The ground truth and the grading are tested without the API (an "oracle" that
runs the ground-truth SQL).

## Limitations

- The curated copy is made when a conversation starts; start a new conversation after adding
  runs.
- Only per-well results, fits, synergy scores and plate QC are available — not single-cell data,
  profiles, kinetics or screens (add views to extend it).
- The grounding check is a heuristic: it catches invented numbers but not a correct number
  attached to the wrong compound. Read the steps for anything you will report.
