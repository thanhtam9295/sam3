"""One front end for the GUI and the command line: :class:`LocalAssistant` (key in this
computer's keychain, Claude called from here) and :class:`RemoteAssistant` (server mode:
questions go to the server, which holds the key). Both keep one conversation going and start
a new one when the privacy options change (they need another snapshot)."""

from __future__ import annotations

import contextlib
from typing import Any

from hcs_studio.assistant.conversation import Answer, Conversation, PayloadLog, Question, Step


def answer_from_json(data: dict[str, Any]) -> Answer:
    steps = [Step(**s) for s in data.get("steps", [])]
    fields = {k: v for k, v in data.items() if k in Answer.__dataclass_fields__ and k != "steps"}
    answer = Answer(**fields)
    answer.steps = steps
    if not answer.display_text:
        answer.display_text = answer.text
    return answer


class LocalAssistant:
    def __init__(self, ctx: Any = None, client: Any = None, settings: Any = None) -> None:
        self.ctx = ctx
        self.client = client
        self.settings = settings
        self.conversation: Conversation | None = None
        self.payload_log = PayloadLog()
        self._privacy: Any = None

    def _settings(self) -> Any:
        if self.settings is not None:
            return self.settings
        from hcs_studio.config import load_settings

        return load_settings().assistant

    def _conversation(self, question: Question) -> Conversation:
        if self.conversation is None or self._privacy != question.privacy:
            from hcs_studio.assistant import open_conversation
            from hcs_studio.services import context

            self.close()
            self.conversation = open_conversation(
                self.ctx or context(),
                question.privacy,
                client=self.client,
                settings=self._settings(),
                payload_log=self.payload_log,
            )
            self._privacy = question.privacy
        return self.conversation

    def ask(self, question: Question) -> Answer:
        return self._conversation(question).ask(question)

    def resume(self, approved: bool) -> Answer:
        if self.conversation is None:
            raise RuntimeError("No conversation")
        return self.conversation.resume(approved)

    def preview(self, question: Question) -> dict[str, Any]:
        if self.conversation is not None and self._privacy == question.privacy:
            return self.conversation.preview(question)
        import tempfile
        from pathlib import Path

        from hcs_studio.assistant.views import build_snapshot

        with tempfile.TemporaryDirectory() as folder:
            snapshot = build_snapshot(Path(folder) / "v.sqlite", question.privacy)
            conversation = Conversation(None, snapshot, self._settings(), count_usage=False)
            try:
                return conversation.preview(question)
            finally:
                conversation.close()

    def status(self) -> dict[str, Any]:
        from hcs_studio.assistant.usage import budget_of, usage_of
        from hcs_studio.services import context

        s = self._settings()
        user = (self.ctx or context()).actor.id
        return {
            "enabled": s.enabled,
            "model": s.model,
            "strong_model": s.strong_model,
            "allow_images": s.allow_images,
            "compound_ids_only": s.compound_ids_only,
            "usage": usage_of(user),
            "budget": budget_of(user, s),
        }

    def reveal(self, text: str) -> str:
        if self.conversation is None:
            return text
        return self.conversation.snapshot.pseudonyms.reveal(text)

    def new_conversation(self) -> None:
        self.close()

    def close(self) -> None:
        if self.conversation is not None:
            self.conversation.close()
            self.conversation = None


class RemoteAssistant:
    """The assistant of an HCS Studio server (``ServerClient`` signed in)."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.conversation_id: str | None = None
        self._privacy: tuple[bool, bool] | None = None

    def _body(self, question: Question) -> dict[str, Any]:
        privacy = (question.privacy.compound_ids_only, question.privacy.include_notes)
        if privacy != self._privacy:
            self.conversation_id = None
        return {
            "question": question.text,
            "strong": question.strong,
            "compound_ids_only": privacy[0],
            "include_notes": privacy[1],
            "images": question.images,
            "conversation_id": self.conversation_id,
        }

    def ask(self, question: Question) -> Answer:
        body = self._body(question)
        answer = answer_from_json(self.client.post("/api/assistant/ask", body))
        self.conversation_id = answer.conversation_id
        self._privacy = (question.privacy.compound_ids_only, question.privacy.include_notes)
        return answer

    def resume(self, approved: bool) -> Answer:
        if not self.conversation_id:
            raise RuntimeError("No conversation")
        data = self.client.post(
            f"/api/assistant/{self.conversation_id}/confirm", {"approved": approved}
        )
        return answer_from_json(data)

    def preview(self, question: Question) -> dict[str, Any]:
        return dict(self.client.post("/api/assistant/preview", self._body(question)))

    def status(self) -> dict[str, Any]:
        return dict(self.client.get("/api/assistant/status"))

    def reveal(self, text: str) -> str:
        return text  # the server sends names already

    def new_conversation(self) -> None:
        self.close()

    def close(self) -> None:
        if self.conversation_id:
            with contextlib.suppress(Exception):  # the server forgets idle ones anyway
                self.client.http.delete(
                    f"/api/assistant/{self.conversation_id}", headers=self.client._headers()
                )
            self.conversation_id = None


def assistant_for_app(server: Any = None) -> LocalAssistant | RemoteAssistant:
    """The assistant for this app: the server's in connected mode (``server`` = the signed-in
    ``ServerClient``), else the local one."""
    if server is not None:
        return RemoteAssistant(server)
    return LocalAssistant()
