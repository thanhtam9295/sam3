"""Command-line interface of the lab assistant (Phase 28)."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any


def add_parsers(sub: Any) -> None:
    a = sub.add_parser("assistant", help="the Claude lab assistant (ask, settings, evaluation)")
    a_sub = a.add_subparsers(dest="assistant_command", required=True)
    for name, text in (("ask", "ask a question"), ("preview", "show what would be sent")):
        p = a_sub.add_parser(name, help=text)
        p.add_argument("question")
        p.add_argument("--strong", action="store_true", help="use the stronger model")
        p.add_argument("--ids-only", action="store_true", help="send compound codes, not names")
        p.add_argument("--notes", action="store_true", help="include free-text notes")
        if name == "ask":
            p.add_argument("--json", action="store_true", help="print the answer as JSON")
    a_sub.add_parser("status", help="settings and this month's usage")
    conf = a_sub.add_parser("configure", help="change the assistant settings (admins)")
    switch = conf.add_mutually_exclusive_group()
    switch.add_argument("--enable", action="store_true")
    switch.add_argument("--disable", action="store_true")
    conf.add_argument("--model")
    conf.add_argument("--strong-model")
    conf.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    conf.add_argument("--budget", type=int, help="monthly weighted tokens per user (0 = no limit)")
    conf.add_argument("--user-budget", nargs=2, metavar=("USER", "TOKENS"))
    conf.add_argument("--ids-only", choices=["on", "off"], help="compound codes by default")
    conf.add_argument("--images", choices=["on", "off"], help="allow image thumbnails")
    a_sub.add_parser(
        "set-key", help="store the Claude API key in the OS keychain (asked; admins)"
    ).add_argument("--remove", action="store_true", help="remove the stored key")
    ev = a_sub.add_parser("eval", help="run the evaluation set against the live API (costs tokens)")
    ev.add_argument("--workdir", type=Path, required=True, help="a new folder for the registry")
    ev.add_argument("--strong", action="store_true", help="evaluate the stronger model")
    ev.add_argument("--only", nargs="*", help="item IDs")
    ev.add_argument("--output", type=Path, help="report JSON (default: <workdir>/eval.json)")


def _question(args: argparse.Namespace, default_ids: bool) -> Any:
    from hcs_studio.assistant import Privacy, Question

    return Question(
        args.question,
        strong=args.strong,
        privacy=Privacy(compound_ids_only=args.ids_only or default_ids, include_notes=args.notes),
    )


def _print_answer(answer: Any) -> None:
    print(answer.display_text or answer.text or f"({answer.status}: {answer.message})")
    if answer.unsupported_numbers:
        print(f"\nCheck these numbers (not in any tool result): {answer.unsupported_numbers}")
    print("\nShow work:")
    for i, step in enumerate(answer.steps, 1):
        print(f"  {i}. {step.tool}: {step.summary}{' [error]' if step.is_error else ''}")
        for sql in step.sql:
            print(f"     SQL: {sql}")
    if not answer.steps:
        print("  (no tool was used)")
    usage = answer.usage
    print(
        f"\nModel {answer.model}; tokens in {usage.get('input_tokens', 0)}, out "
        f"{usage.get('output_tokens', 0)}, cache read {usage.get('cache_read_tokens', 0)}"
    )


def run(args: argparse.Namespace) -> int:
    from hcs_studio.config import load_settings
    from hcs_studio.services import context

    command = args.assistant_command
    settings = load_settings().assistant
    if command in ("ask", "preview"):
        from hcs_studio.assistant import (
            AssistantDisabled,
            AssistantUnavailable,
            BudgetExceeded,
        )
        from hcs_studio.assistant.session import LocalAssistant

        question = _question(args, settings.compound_ids_only)
        assistant = LocalAssistant(context())
        try:
            if command == "preview":
                print(json.dumps(assistant.preview(question), indent=2, ensure_ascii=False))
                return 0
            answer = assistant.ask(question)
            while answer.status == "needs_confirmation":  # images are not offered here
                answer = assistant.resume(False)
        except (AssistantDisabled, AssistantUnavailable, BudgetExceeded) as exc:
            print(f"Error: {exc}")
            return 2
        finally:
            assistant.close()
        if args.json:
            print(json.dumps(answer.to_json(), indent=2, ensure_ascii=False, default=str))
        else:
            _print_answer(answer)
        return 0 if answer.status == "answered" else 1
    if command == "status":
        from hcs_studio.assistant.session import LocalAssistant

        print(json.dumps(LocalAssistant(context()).status(), indent=2))
        return 0
    if command == "configure":
        from hcs_studio.services.assistant import configure_assistant

        changes: dict[str, Any] = {}
        if args.enable or args.disable:
            changes["enabled"] = bool(args.enable)
        for key in ("model", "strong_model", "effort"):
            if getattr(args, key):
                changes[key] = getattr(args, key)
        if args.budget is not None:
            changes["monthly_token_budget"] = args.budget
        if args.user_budget:
            changes["user_budgets"] = {
                **settings.user_budgets,
                args.user_budget[0]: int(args.user_budget[1]),
            }
        if args.ids_only:
            changes["compound_ids_only"] = args.ids_only == "on"
        if args.images:
            changes["allow_images"] = args.images == "on"
        try:
            new = configure_assistant(context(), **changes)
        except (PermissionError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
        print(json.dumps(new.model_dump(mode="json"), indent=2))
        return 0
    if command == "set-key":
        from hcs_studio.services.assistant import set_api_key

        value = None if args.remove else getpass.getpass("Claude API key: ").strip()
        if not args.remove and not value:
            print("Error: no key given")
            return 2
        try:
            set_api_key(context(), value)
        except (PermissionError, RuntimeError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
        print("Key removed" if args.remove else "Key stored in the OS keychain")
        return 0
    if command == "eval":
        return _eval(args, settings)
    return 2


def _eval(args: argparse.Namespace, settings: Any) -> int:
    """The evaluation runs in its own settings folder (``<workdir>/home``) with a synthetic
    registry; the API key is read from the keychain (or ``HCS_SECRET_ANTHROPIC_API_KEY``)
    first."""
    from hcs_studio.assistant import AssistantUnavailable, make_client
    from hcs_studio.assistant.evaluation import ITEMS, build_eval_registry, run_evaluation

    workdir = Path(args.workdir)
    if workdir.exists() and any(workdir.iterdir()):
        print(f"Error: {workdir} is not empty (the evaluation builds a new registry there)")
        return 2
    eval_settings = settings.model_copy(update={"enabled": True})
    try:
        client = make_client(eval_settings)
    except AssistantUnavailable as exc:
        print(f"Error: {exc}")
        return 2
    os.environ["HCS_STUDIO_HOME"] = str(workdir / "home")
    os.environ.pop("HCS_DATABASE_URL", None)
    from hcs_studio.db.engine import dispose_engines

    dispose_engines()
    print("Building the synthetic registry…", file=sys.stderr)
    build_eval_registry(workdir)
    items = tuple(i for i in ITEMS if not args.only or i.id in args.only)
    report = run_evaluation(client, eval_settings, workdir, items=items, strong=args.strong)
    path = report.write(args.output or workdir / "eval.json")
    print(
        f"Score {report.score:.0%} ({sum(r.passed for r in report.results)}/{len(items)}); "
        f"every answer shows its work: {report.all_show_work}. Report: {path}"
    )
    return 0 if report.score >= 0.9 and report.all_show_work else 1
