"""Command-line interface for accounts and the audit trail (Phase 26)."""

from __future__ import annotations

import argparse
import getpass
from datetime import datetime
from pathlib import Path
from typing import Any


def add_parsers(sub: Any) -> None:
    acc = sub.add_parser("accounts", help="local accounts and roles")
    acc_sub = acc.add_subparsers(dest="accounts_command", required=True)
    acc_sub.add_parser("list", help="accounts, roles, password set, locked")
    acc_sub.add_parser("matrix", help="the permission matrix")
    add = acc_sub.add_parser("add", help="add an account")
    add.add_argument("username")
    add.add_argument("--role", required=True, choices=["viewer", "analyst", "reviewer", "admin"])
    add.add_argument("--name", default="", help="printed name (signatures)")
    add.add_argument("--password", action="store_true", help="ask for a password")
    add.add_argument("--reason", required=True)
    upd = acc_sub.add_parser("set", help="change an account")
    upd.add_argument("username")
    upd.add_argument("--role", choices=["viewer", "analyst", "reviewer", "admin"])
    upd.add_argument("--name")
    upd.add_argument("--disable", action="store_true")
    upd.add_argument("--enable", action="store_true")
    upd.add_argument("--password", action="store_true", help="set a new password (asked)")
    upd.add_argument("--reason", required=True)

    aud = sub.add_parser("audit", help="the audit trail: list, export, verify")
    aud_sub = aud.add_subparsers(dest="audit_command", required=True)
    for name, text in (("list", "show audit records"), ("export", "write CSV or PDF")):
        p = aud_sub.add_parser(name, help=text)
        p.add_argument("--user")
        p.add_argument("--since", help="YYYY-MM-DD or ISO time (UTC)")
        p.add_argument("--until", help="YYYY-MM-DD or ISO time (UTC)")
        p.add_argument("--action", help="event type or prefix, e.g. run or qc.review")
        p.add_argument("--object", help="text in the object id (plate, run folder …)")
        p.add_argument("--limit", type=int)
        if name == "export":
            p.add_argument("--output", type=Path, required=True, help=".csv or .pdf")
    aud_sub.add_parser("verify", help="recompute the hash chain; exit code 1 on any break")
    check = aud_sub.add_parser("checkpoint", help="write the chain head to the checkpoint files")
    check.add_argument("--reason", default="")


def _context(args: argparse.Namespace) -> Any:
    from hcs_studio.services import context

    return context(getattr(args, "reason", "") or "")


def _new_password() -> str:
    first = getpass.getpass("New password: ")
    if getpass.getpass("Repeat: ") != first:
        raise ValueError("The passwords differ")
    return first


def run_accounts(args: argparse.Namespace) -> int:
    from hcs_studio.security import accounts
    from hcs_studio.security.roles import matrix
    from hcs_studio.services import security as services

    try:
        if args.accounts_command == "list":
            for a in accounts.list_accounts():
                locked = f" locked until {a.locked_until:%Y-%m-%d %H:%M}" if a.locked_until else ""
                print(
                    f"{a.username:<20} {a.role:<9} {'active' if a.active else 'disabled':<9} "
                    f"password {'set' if a.has_password else 'none':<5} {a.display_name}{locked}"
                )
            if not accounts.list_accounts():
                print("No accounts: the computer's user is an admin (single-user mode)")
        elif args.accounts_command == "matrix":
            for row in matrix():
                print(f"{row['action']:<24} {row['lowest role']}")
        elif args.accounts_command == "add":
            password = _new_password() if args.password else None
            info = services.add_account(
                _context(args), args.username, args.role, args.name, password
            )
            print(f"Added {info.username} ({info.role})")
        else:
            active = False if args.disable else (True if args.enable else None)
            info = services.update_account(
                _context(args),
                args.username,
                role=args.role,
                display_name=args.name,
                active=active,
                password=_new_password() if args.password else None,
            )
            print(f"{info.username}: {info.role}, {'active' if info.active else 'disabled'}")
    except (ValueError, PermissionError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _when(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def run_audit(args: argparse.Namespace) -> int:
    from hcs_studio.security import audit
    from hcs_studio.services import security as services

    if args.audit_command == "verify":
        result = audit.verify_chain()
        print(result.summary())
        for problem in result.breaks[:20]:
            print(f"  record {problem['id']}: {problem['problem']}")
        return 0 if result.ok else 1
    if args.audit_command == "checkpoint":
        entry = services.audit_checkpoint(_context(args))
        print(f"Chain head written: record {entry['id']} {entry['hash'][:16]}…")
        return 0
    records = audit.query(
        user=args.user,
        since=_when(args.since),
        until=_when(args.until),
        action=args.action,
        object_contains=args.object,
        limit=args.limit,
    )
    if args.audit_command == "list":
        for r in records:
            print(
                f"{r['id']:>6} {audit._utc_text(r['utc'])[:19]} {r['actor_id']} "
                f"({r['actor_role'] or '–'}) "
                f"{r['type']} {r['object_id']} {r['status']}"
                + (f" — {r['reason']}" if r["reason"] else "")
            )
        return 0
    output: Path = args.output
    if output.suffix.lower() == ".pdf":
        audit.export_pdf(records, output)
    else:
        audit.export_csv(records, output)
    print(f"{len(records)} records → {output}")
    return 0
