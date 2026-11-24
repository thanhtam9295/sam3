# Accounts, audit trail and sign-off

HCS Studio records who changed what, when and why; locks approved results; and supports review
and electronic sign-off of runs.

> **Compliance note.** These features support regulated work (for example controls in the
> style of 21 CFR Part 11), but they do not make an installation compliant by themselves.
> Compliance also needs validation of the installation (IQ/OQ/PQ), standard operating
> procedures, training and the organisation's own controls (for example of the computers and
> the database server). The defaults suit research use; **regulated mode** tightens them.

## Accounts and roles

In local mode the person acting is the person logged in to the computer. As long as no account
exists, that person is an admin (a single-user installation works as before). Once accounts
exist, the computer user name selects the account and its role; a user without an account may
only look.

| Role | May |
|---|---|
| viewer | look at runs, reports, the registry and the audit trail |
| analyst | everything that changes data: runs, layouts, QC reviews (with reasons), exclusions, pharmacology, screens, training models |
| reviewer | as analyst, plus sign runs as reviewed or approved and change model status (promotion to production) |
| admin | as reviewer, plus settings, accounts, QC and routing rules, unlocking an approved run |

```bash
hcs-studio accounts add anna --role reviewer --name "Anna Smith" [--password]
hcs-studio accounts list
hcs-studio accounts set anna --role admin | --disable | --enable | --password
hcs-studio accounts matrix          # the permission matrix
```

Passwords are optional (required in regulated mode) and stored as argon2id hashes. Five wrong
passwords lock an account for 15 minutes; a signed-in session in the app ends after 30 minutes
without activity (all three in `settings.json` → `security`).

## The audit trail

Every change made through the application — layouts, runs, QC decisions, exclusions, reports,
model changes, settings, accounts, signatures — is an audit record with the UTC time, the user
and role, the action, the object (type and id), the state before and after, the reason
(required for exclusions, overrides, deletions, re-analysis of locked runs and model
promotion), the computer and the app version. Refused actions (missing role) are recorded too.

Records are append-only and **hash-chained**: each stores the SHA-256 of its own content and of
the record before it, so changing, deleting or inserting a record breaks the chain. The last
record's hash is also written to `<settings>/audit/chain_head.jsonl` every 50 records (and to
any extra folders in `security.audit_head_copies`, e.g. a network share), so even a rewritten
chain disagrees with the copies kept elsewhere.

```bash
hcs-studio audit list [--user anna] [--since 2026-11-01] [--action run] [--object PLATE01]
hcs-studio audit export --output audit.csv|audit.pdf [same filters]
hcs-studio audit verify            # recomputes the chain; exit code 1 on any break
hcs-studio audit checkpoint        # write the chain head now
```

In server mode (PostgreSQL, Phase 27) the application's database role may insert audit records
but not change or delete them (`security.audit.postgres_append_only_sql`).

## Run states, signatures and locking

A run is **draft** while it runs, **completed** when it finishes, **reviewed** when a reviewer
signs it as reviewed, and **approved** when a reviewer signs it as approved. Each signature shows
its meaning, the printed name, the role and the time (UTC), and fixes the SHA-256 of the run
manifest at that moment.

```bash
hcs-studio run-state <run>                         # state, version, signatures
hcs-studio sign <run> --meaning reviewed [--comment "…"]
hcs-studio sign <run> --meaning approved
hcs-studio unlock <run> --reason "…"              # admin
hcs-studio analyze <locked run> --pipeline steps.yaml --new-version --reason "…"
```

In the app: **Tools → Sign the run in Results…** shows the state and the signatures so far, the
meaning of the signature, a comment and the password field (required when the settings ask for
re-authentication).

An approved run is **locked**: its files become read-only and every change through the app is
refused. Re-analysis creates a **new version** (`<run>_v2`, linked to the original in its
manifest and in the registry; large unchanged files are hard-linked) and changes that instead.
The run report (`index.html`) shows the state and the signatures, and a **DRAFT** mark until the
run is approved; it also lists the run's audit records.

## Settings (`settings.json` → `security`)

| Setting | Research default | Regulated mode |
|---|---|---|
| `regulated_mode` | false | true |
| `require_password` (sign in to the app) | false | true |
| `reauth_for_signoff` (password again to sign) | false | true |
| approver must differ from the reviewer | no | yes |
| `session_timeout_min` | 30 | 30 |
| `lockout_attempts` / `lockout_minutes` | 5 / 15 | 5 / 15 |
| `audit_head_every`, `audit_head_copies` | 50, none | 50, add a folder on another machine |

`hcs-studio db backup --output <folder>` copies the registry (consistent online copy) and the
audit chain-head files; schedule it with your backup system. Audit records are never deleted by
the application.
