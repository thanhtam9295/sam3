"""Command line of automatic runs (Phase 19): ``hcs-studio auto …``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any


def add_parsers(sub: Any) -> None:
    auto = sub.add_parser("auto", help="automatic runs: scheduler events, watched folders")
    a_sub = auto.add_subparsers(dest="auto_command", required=True)
    serve = a_sub.add_parser("serve", help="run the automation (events, watchers, runs)")
    serve.add_argument("--once", action="store_true", help="one round, then stop")
    serve.add_argument("--rounds", type=int, default=0, help="stop after N rounds (0 = never)")
    a_sub.add_parser("status", help="automatic runs and what the watchers see")
    events = a_sub.add_parser("events", help="the lab-event log")
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--status", help="accepted | duplicate | rejected | sent | …")
    events.add_argument("--direction", choices=["in", "out"])
    events.add_argument("--json", action="store_true")
    a_sub.add_parser("check", help="check the routing rules, folders and channels")
    retry = a_sub.add_parser("retry", help="start an acquisition again")
    retry.add_argument("key", help="the acquisition key (see 'auto status')")
    retry.add_argument("--reason", required=True)
    test = a_sub.add_parser("test-notify", help="send a test notification to every channel")
    test.add_argument("--kind", default="completed")
    send = a_sub.add_parser("send", help="send a lab event from a JSON file (testing)")
    send.add_argument("file", type=Path)


def _queue() -> Any:
    """The server's database queue in server mode (PostgreSQL), else a local queue."""
    from hcs_studio.db.engine import get_engine

    if get_engine().dialect.name == "postgresql":
        from hcs_studio.server.queue import DbJobQueue

        return DbJobQueue()
    from hcs_studio.pipeline.queue import LocalJobQueue

    return LocalJobQueue()


def run(args: argparse.Namespace) -> int:
    from hcs_studio.config import load_settings

    settings = load_settings().automation
    command = args.auto_command
    if command == "serve":
        if not settings.enabled:
            print("Automation is off (settings.json → automation.enabled).")
            return 2
        return _serve(args, settings)
    if command == "status":
        from hcs_studio.automation.status import auto_runs

        for run in auto_runs(50):
            print(
                f"{run['status']:<13} {run['barcode']:<16} {run['route'] or '–':<18} "
                f"{run['key']}  {run['detail'][:60]}"
            )
        return 0
    if command == "events":
        from hcs_studio.automation.status import lab_events

        rows = lab_events(args.limit, status=args.status, direction=args.direction)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            for row in rows:
                print(
                    f"{row['time']:%Y-%m-%d %H:%M:%S} {row['direction']:<3} {row['channel']:<6} "
                    f"{row['type']:<20} {row['status']:<10} {row['barcode']:<14} "
                    f"{row['detail'][:60]}"
                )
        return 0
    if command == "check":
        return _check(settings)
    if command == "retry":
        from hcs_studio.services import context
        from hcs_studio.services.automation import retry_auto_run

        try:
            retry_auto_run(context(args.reason), args.key)
        except Exception as exc:
            print(f"Error: {exc}")
            return 2
        print(f"{args.key} will start again at the next round")
        return 0
    if command == "test-notify":
        from hcs_studio.automation.notify import Notice, Notifier

        notifier = Notifier(settings.channels, settings.admins)
        notifier.notify(
            Notice(
                args.kind,
                "TEST-0001",
                "test message from HCS Studio",
                ["If you read this, notifications work."],
            )
        )
        for row in notifier.summary():
            print(f"{row['channel']}: {row['status']} ({row['detail']})")
        return 0 if all(r["status"] == "sent" for r in notifier.summary()) else 1
    if command == "send":
        from hcs_studio.automation.intake import receive

        receipt = receive(args.file.read_bytes(), "cli")
        print(json.dumps(receipt.to_dict(), indent=2))
        return 0 if receipt.status in ("accepted", "duplicate", "ignored") else 1
    return 2


def _check(settings: Any) -> int:
    from hcs_studio.automation.routing import check_routing, load_routing

    problems = []
    if not settings.routing_file:
        problems.append("no routing file (automation.routing_file)")
    else:
        try:
            problems += check_routing(load_routing(settings.routing_file))
        except Exception as exc:
            problems.append(f"routing file not readable: {exc}")
    for folder in [w.path for w in settings.watch] + [settings.inbox]:
        if folder and not Path(folder).is_dir():
            problems.append(f"folder not found: {folder}")
    for channel in settings.channels:
        if channel.kind == "email" and not channel.smtp_host:
            problems.append(f"{channel.name}: no SMTP host")
    for problem in problems:
        print(f"- {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def _serve(args: argparse.Namespace, settings: Any) -> int:
    from hcs_studio.automation.intake import EventServer
    from hcs_studio.automation.runner import AutoRunner

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = None
    if settings.http_port:
        server = EventServer(settings.http_host, settings.http_port).start()
        print(f"Receiving lab events at {server.url}")
    queue = _queue()
    runner = AutoRunner(settings, queue)
    rounds = 1 if args.once else args.rounds
    done = 0
    try:
        while True:
            runner.step()
            done += 1
            if rounds and done >= rounds:
                break
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        print("Stopping (running jobs finish in their own processes or resume later)")
    finally:
        if server is not None:
            server.stop()
        if hasattr(queue, "shutdown") and not args.once:
            queue.shutdown()
    return 0
