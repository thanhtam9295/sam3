"""The lab assistant through the server (Phase 28.1, server mode): the API key and the calls
to Claude stay on the server; desktops and scripts send questions here.

Conversations live in the server process (per account, expiring after two idle hours); each
has its own curated snapshot. Images still need the user's confirmation: an answer with
status ``needs_confirmation`` waits for ``POST /api/assistant/{id}/confirm``.
"""

# no ``from __future__ import annotations``: FastAPI resolves the endpoint annotations, and
# ``Request`` is imported inside ``register`` (see server/api.py)

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hcs_studio.services import Actor, ServiceContext

if TYPE_CHECKING:
    from fastapi import FastAPI

IDLE_S = 2 * 3600
MAX_PER_USER = 5


@dataclass
class _Entry:
    conversation: Any
    owner: str
    privacy: tuple[bool, bool]
    used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class Conversations:
    """Open conversations of the server, per account."""

    def __init__(self, client_factory: Any = None) -> None:
        self.entries: dict[str, _Entry] = {}
        self.lock = threading.Lock()
        #: settings → Anthropic client (tests inject a fake)
        self.client_factory = client_factory

    def _expire(self) -> None:
        now = time.time()
        with self.lock:
            old = [k for k, e in self.entries.items() if now - e.used > IDLE_S]
            dropped = [self.entries.pop(k) for k in old]
        for entry in dropped:
            entry.conversation.close()

    def get(self, conversation_id: str, actor: Actor) -> _Entry:
        self._expire()
        entry = self.entries.get(conversation_id)
        if entry is None or entry.owner != actor.id:
            raise KeyError(f"No conversation {conversation_id}")
        entry.used = time.time()
        return entry

    def open(self, ctx: ServiceContext, privacy: Any) -> _Entry:
        from hcs_studio.assistant import make_client, open_conversation
        from hcs_studio.config import load_settings

        self._expire()
        settings = load_settings().assistant
        if not settings.enabled:
            from hcs_studio.assistant import AssistantDisabled

            raise AssistantDisabled("The lab assistant is off on this server (ask an admin).")
        factory = self.client_factory or make_client
        conversation = open_conversation(ctx, privacy, client=factory(settings), settings=settings)
        entry = _Entry(
            conversation, ctx.actor.id, (privacy.compound_ids_only, privacy.include_notes)
        )
        with self.lock:
            mine = sorted((e.used, k) for k, e in self.entries.items() if e.owner == ctx.actor.id)
            for _, key in mine[: max(0, len(mine) - MAX_PER_USER + 1)]:
                self.entries.pop(key).conversation.close()
            self.entries[conversation.id] = entry
        return entry

    def close(self, conversation_id: str, actor: Actor) -> None:
        entry = self.get(conversation_id, actor)
        with self.lock:
            self.entries.pop(conversation_id, None)
        entry.conversation.close()


def _question(body: dict[str, Any], default_ids_only: bool) -> Any:
    from hcs_studio.assistant import Privacy, Question

    text = str(body.get("question", "")).strip()
    if not text:
        raise ValueError("Ask a question")
    ids_only = body.get("compound_ids_only")
    privacy = Privacy(
        compound_ids_only=default_ids_only if ids_only is None else bool(ids_only),
        include_notes=bool(body.get("include_notes", False)),
    )
    return Question(
        text, strong=bool(body.get("strong")), privacy=privacy, images=bool(body.get("images"))
    )


def register(app: "FastAPI", user: Any, make_ctx: Any, fail: Any) -> Conversations:
    """Add the ``/api/assistant`` endpoints to ``app``."""
    from fastapi import Body, Depends, HTTPException, Request

    conversations = Conversations()
    app.state.assistant = conversations

    def settings() -> Any:
        from hcs_studio.config import load_settings

        return load_settings().assistant

    @app.get("/api/assistant/status")
    def status(actor: Actor = Depends(user)) -> dict[str, Any]:
        from hcs_studio.assistant.usage import budget_of, usage_of

        s = settings()
        return {
            "enabled": s.enabled,
            "model": s.model,
            "strong_model": s.strong_model,
            "allow_images": s.allow_images,
            "compound_ids_only": s.compound_ids_only,
            "usage": usage_of(actor.id),
            "budget": budget_of(actor.id, s),
        }

    def _answer(entry: _Entry, run: Any) -> dict[str, Any]:
        from hcs_studio.assistant import AssistantDisabled, BudgetExceeded

        with entry.lock:
            try:
                answer = run(entry.conversation)
            except AssistantDisabled as exc:
                raise HTTPException(403, str(exc)) from exc
            except BudgetExceeded as exc:
                raise HTTPException(429, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
        data: dict[str, Any] = answer.to_json()
        data["text"] = answer.display_text  # names, not codes, for the user
        return data

    @app.post("/api/assistant/ask")
    def ask(
        request: Request, body: dict[str, Any] = Body(...), actor: Actor = Depends(user)
    ) -> dict[str, Any]:
        from hcs_studio.assistant import AssistantDisabled, AssistantUnavailable

        s = settings()
        try:
            question = _question(body, s.compound_ids_only)
        except ValueError as exc:
            raise fail(exc) from exc
        key = (question.privacy.compound_ids_only, question.privacy.include_notes)
        entry = None
        if body.get("conversation_id"):
            try:
                entry = conversations.get(str(body["conversation_id"]), actor)
            except KeyError as exc:
                raise fail(exc) from exc
            if entry.privacy != key:  # other privacy settings need another snapshot
                entry = None
        if entry is None:
            try:
                entry = conversations.open(make_ctx(request, actor), question.privacy)
            except AssistantDisabled as exc:
                raise HTTPException(403, str(exc)) from exc
            except AssistantUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
        return _answer(entry, lambda c: c.ask(question))

    @app.post("/api/assistant/{conversation_id}/confirm")
    def confirm(
        conversation_id: str, body: dict[str, Any] = Body(...), actor: Actor = Depends(user)
    ) -> dict[str, Any]:
        try:
            entry = conversations.get(conversation_id, actor)
        except KeyError as exc:
            raise fail(exc) from exc
        approved = bool(body.get("approved"))
        return _answer(entry, lambda c: c.resume(approved))

    @app.post("/api/assistant/preview")
    def preview(body: dict[str, Any] = Body(...), actor: Actor = Depends(user)) -> dict[str, Any]:
        """The request a question would send (nothing is sent)."""
        from hcs_studio.assistant import Conversation
        from hcs_studio.assistant.views import Snapshot

        s = settings()
        try:
            question = _question(body, s.compound_ids_only)
        except ValueError as exc:
            raise fail(exc) from exc
        if body.get("conversation_id"):
            try:
                entry = conversations.get(str(body["conversation_id"]), actor)
            except KeyError as exc:
                raise fail(exc) from exc
            return dict(entry.conversation.preview(question))
        import tempfile
        from pathlib import Path

        from hcs_studio.assistant.views import build_snapshot

        with tempfile.TemporaryDirectory() as folder:
            snapshot: Snapshot = build_snapshot(Path(folder) / "v.sqlite", question.privacy)
            conversation = Conversation(None, snapshot, s, count_usage=False)
            try:
                return dict(conversation.preview(question))
            finally:
                conversation.close()

    @app.delete("/api/assistant/{conversation_id}")
    def close(conversation_id: str, actor: Actor = Depends(user)) -> dict[str, Any]:
        try:
            conversations.close(conversation_id, actor)
        except KeyError as exc:
            raise fail(exc) from exc
        return {"closed": conversation_id}

    return conversations
