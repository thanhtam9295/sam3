"""The service layer: every state change goes through a service function (PROMPT_PLATFORM.md §2.4).

A service takes a :class:`ServiceContext` (who acts, and why) as its first argument and is
decorated with :func:`service`, which checks the reason, runs the change and emits one
:class:`Event` to the :data:`BUS` — also when the change fails. Sinks write events to the
registry (``event`` table) and the log; Phase 26 adds the tamper-evident audit trail as
another sink without touching the callers.

Core functions (``run_pipeline``, ``annotate_run``, …) stay free of actors and events; the
GUI and the command line call the services (``tests/test_services.py`` checks this).
"""

from __future__ import annotations

import functools
import getpass
import logging
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, ParamSpec, TypeVar

from hcs_studio import __version__

ActorKind = Literal["os", "account", "scheduler", "system"]
P = ParamSpec("P")
T = TypeVar("T")
log = logging.getLogger("hcs_studio.events")


class ReasonRequired(ValueError):
    """The change needs a reason (exclusions, overrides, deletions, re-analysis, promotion)."""


@dataclass(frozen=True)
class Actor:
    id: str
    name: str = ""
    kind: ActorKind = "os"
    #: viewer | analyst | reviewer | admin (Phase 26); "" = look it up from the accounts.
    role: str = ""


def os_actor() -> Actor:
    """The person logged in to this computer (local mode)."""
    try:
        user = getpass.getuser()
    except Exception:  # no login name (some services, containers)
        user = "unknown"
    return Actor(id=user, name=user, kind="os")


@dataclass
class ServiceContext:
    """Who changes something, why, from where. ``note`` collects what changed."""

    actor: Actor = field(default_factory=os_actor)
    reason: str = ""
    client: str = field(default_factory=socket.gethostname)
    app_version: str = __version__
    _note: dict[str, Any] = field(default_factory=dict, repr=False)

    def note(
        self,
        *,
        object_type: str | None = None,
        object_id: str | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        """Called by a service to describe the object it changes and its before/after state."""
        for key, value in (
            ("object_type", object_type),
            ("object_id", object_id),
            ("before", before),
            ("after", after),
        ):
            if value is not None:
                self._note[key] = value

    def with_reason(self, reason: str) -> ServiceContext:
        return replace(self, reason=reason, _note={})


def context(reason: str = "", actor: Actor | None = None) -> ServiceContext:
    """The context of an action: the acting user (the signed-in account, else the OS user)
    with their role, and the reason."""
    return ServiceContext(actor=with_role(actor or current_actor()), reason=reason.strip())


def current_actor() -> Actor:
    """The signed-in account in this process (app with a password), else the OS user."""
    from hcs_studio.security.accounts import current_session

    try:
        session = current_session()
    except Exception:  # settings unreadable: fall back to the OS user
        session = None
    if session is not None:
        session.touch()
        return Actor(session.username, session.display_name, "account", session.role)
    return os_actor()


def with_role(actor: Actor) -> Actor:
    """The actor with its role filled in from the accounts (system actors: admin)."""
    if actor.role:
        return actor
    if actor.kind in ("system", "scheduler"):
        return replace(actor, role="admin")
    from hcs_studio.security.accounts import role_of

    return replace(actor, role=role_of(actor.id))


@dataclass(frozen=True)
class Event:
    type: str
    utc: datetime
    actor: Actor
    reason: str
    object_type: str
    object_id: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    client: str
    app_version: str
    status: Literal["ok", "failed", "denied"] = "ok"
    error: str = ""


Sink = Callable[[Event], None]


class AuditError(RuntimeError):
    """An event could not be written to the audit trail."""


class EventBus:
    """Delivers events to sinks. A failing sink is logged and never breaks the change
    (Phase 26 makes the audit sink strict)."""

    def __init__(self) -> None:
        self.sinks: list[Sink] = []
        #: Sinks whose failure is an error (the audit trail): raised after all sinks ran.
        self.strict: set[Sink] = set()
        self._lock = threading.Lock()

    def add(self, sink: Sink, strict: bool = False) -> None:
        with self._lock:
            if sink not in self.sinks:
                self.sinks.append(sink)
            if strict:
                self.strict.add(sink)

    def remove(self, sink: Sink) -> None:
        with self._lock:
            if sink in self.sinks:
                self.sinks.remove(sink)

    def emit(self, event: Event) -> None:
        with self._lock:
            sinks = list(self.sinks)
            strict = set(self.strict)
        failure: Exception | None = None
        for sink in sinks:
            try:
                sink(event)
            except Exception as exc:
                log.exception("Event sink %r failed for %s", sink, event.type)
                if sink in strict:
                    failure = exc
        if failure is not None:
            raise AuditError(
                f"{event.type} was not recorded in the audit trail: {failure}"
            ) from failure


def log_sink(event: Event) -> None:
    log.info(
        "%s by %s (%s)%s%s",
        event.type,
        event.actor.id,
        event.status,
        f" on {event.object_type} {event.object_id}" if event.object_id else "",
        f": {event.reason}" if event.reason else "",
    )


def database_sink(event: Event) -> None:
    """Append the event to the audit trail (``event`` table) and chain it (Phase 26)."""
    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord
    from hcs_studio.security.audit import chain_new_rows, maybe_checkpoint

    with session_scope() as session:
        session.add(
            EventRecord(
                type=event.type,
                utc=event.utc,
                actor_id=event.actor.id,
                actor_name=event.actor.name,
                actor_kind=event.actor.kind,
                actor_role=event.actor.role,
                reason=event.reason,
                object_type=event.object_type,
                object_id=event.object_id,
                before=_jsonable(event.before),
                after=_jsonable(event.after),
                client=event.client,
                app_version=event.app_version,
                status=event.status,
                error=event.error,
            )
        )
        session.flush()
        chain_new_rows(session)
        from hcs_studio.db.security import AuditHead

        count = session.get(AuditHead, 1).count  # type: ignore[union-attr]
    maybe_checkpoint(count)


def _jsonable(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    import json

    return dict(json.loads(json.dumps(value, default=str)))


BUS = EventBus()
BUS.add(log_sink)
BUS.add(database_sink, strict=True)


def service(
    event_type: str,
    *,
    object_type: str = "",
    reason_required: bool | Callable[..., bool] = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a service function ``fn(ctx: ServiceContext, ...)``.

    ``reason_required`` may be a predicate on the call's arguments (e.g. only when wells
    are excluded). The event is emitted after the change, with status ``failed`` and the
    error when it raises.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            ctx = args[0] if args else kwargs.get("ctx")
            if not isinstance(ctx, ServiceContext):
                raise TypeError(f"{fn.__name__}: the first argument must be a ServiceContext")
            needs = (
                reason_required(*args, **kwargs) if callable(reason_required) else reason_required
            )
            if needs and not ctx.reason.strip():
                raise ReasonRequired(f"{event_type}: give a reason for this change")
            outer_note, ctx._note = ctx._note, {}  # a nested service keeps its own note
            status: Literal["ok", "failed", "denied"] = "ok"
            error = ""
            ctx.actor = with_role(ctx.actor)
            try:
                from hcs_studio.security.roles import allowed, required_role

                if not allowed(ctx.actor.role, event_type):
                    status = "denied"
                    raise PermissionError(
                        f"{event_type} needs the {required_role(event_type)} role; "
                        f"{ctx.actor.id} is {ctx.actor.role or 'unknown'}"
                    )
                return fn(*args, **kwargs)
            except PermissionError as exc:
                status = "denied" if status == "denied" else "failed"
                error = f"{type(exc).__name__}: {exc}"
                raise
            except Exception as exc:
                status, error = "failed", f"{type(exc).__name__}: {exc}"
                raise
            finally:
                note, ctx._note = ctx._note, outer_note
                BUS.emit(
                    Event(
                        type=event_type,
                        utc=datetime.now(UTC),
                        actor=ctx.actor,
                        reason=ctx.reason,
                        object_type=str(note.get("object_type", object_type)),
                        object_id=str(note.get("object_id", "")),
                        before=note.get("before"),
                        after=note.get("after"),
                        client=ctx.client,
                        app_version=ctx.app_version,
                        status=status,
                        error=error,
                    )
                )

        wrapper.__service__ = event_type  # type: ignore[attr-defined]
        return wrapper

    return decorate


def recent_events(limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
    """The newest events in the registry (for the event-log viewer and tests)."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    with session_scope() as session:
        query = select(EventRecord).order_by(EventRecord.id.desc()).limit(limit)
        if event_type:
            query = query.where(EventRecord.type == event_type)
        rows = session.scalars(query).all()
        return [
            {
                "id": r.id,
                "type": r.type,
                "utc": r.utc,
                "actor": r.actor_id,
                "reason": r.reason,
                "object_type": r.object_type,
                "object_id": r.object_id,
                "before": r.before,
                "after": r.after,
                "status": r.status,
                "error": r.error,
            }
            for r in rows
        ]
