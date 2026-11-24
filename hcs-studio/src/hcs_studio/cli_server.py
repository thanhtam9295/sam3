"""Command-line interface for server mode (Phase 27): workers, the queue, the API server."""

from __future__ import annotations

import argparse
from typing import Any


def add_parsers(sub: Any) -> None:
    worker = sub.add_parser("worker", help="run one queue worker (server mode)")
    worker.add_argument("--gpu", type=int, help="GPU index (default: a CPU worker)")
    worker.add_argument("--stale", type=float, default=30.0, help="seconds without heartbeat")
    pool = sub.add_parser("worker-pool", help="run and supervise this machine's workers")
    pool.add_argument("--gpus", default="", help="GPU indexes, comma-separated (one worker each)")
    pool.add_argument("--cpu", type=int, default=1, help="CPU workers")
    pool.add_argument("--stale", type=float, default=30.0)
    serve = sub.add_parser("serve", help="run the server API (behind an HTTPS reverse proxy)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    queue = sub.add_parser("queue", help="the server job queue")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_sub.add_parser("list", help="recent jobs")
    queue_sub.add_parser("workers", help="workers and their heartbeats")
    cancel = queue_sub.add_parser("cancel", help="cancel a job")
    cancel.add_argument("job_id")
    cancel.add_argument("--reason", default="")
    queue_sub.add_parser("reap", help="re-queue jobs whose worker stopped")


def run(args: argparse.Namespace) -> int:
    from hcs_studio.server.queue import DbJobQueue

    if args.command == "worker":
        import logging

        from hcs_studio.server.worker import Worker

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        Worker(gpu=args.gpu, stale_s=args.stale).run_forever()
        return 0
    if args.command == "worker-pool":
        import logging

        from hcs_studio.server.worker import run_pool

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        run_pool(gpus, args.cpu, stale_s=args.stale)
        return 0
    if args.command == "serve":
        import uvicorn

        from hcs_studio.server.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True)
        return 0
    queue = DbJobQueue()
    if args.queue_command == "list":
        for s in queue.jobs():
            print(
                f"{s.id}  {s.status:<24} {s.spec.priority:<11} {s.spec.owner.id:<12} "
                f"{s.done}/{s.total}  {s.job.label}  {s.message}"
            )
    elif args.queue_command == "workers":
        for w in queue.workers():
            gpu = f"gpu{w['gpu']}" if w["gpu"] is not None else "cpu"
            print(f"{w['id']:<32} {gpu:<5} {w['status']:<8} {w['job_id'] or '–'}  {w['heartbeat']}")
    elif args.queue_command == "cancel":
        from hcs_studio.services import context
        from hcs_studio.services.operations import cancel_job

        cancel_job(context(args.reason), queue, args.job_id)
        print(f"Cancel requested: {args.job_id}")
    else:
        print(f"Re-queued: {queue.reap() or 'none'}")
    return 0
