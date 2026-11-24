"""The Claude lab assistant (PROMPT_PLATFORM.md Phase 28).

Plain-language questions about results, answered from a curated, read-only copy of the
registry (:mod:`~hcs_studio.assistant.views`) through read-only tools
(:mod:`~hcs_studio.assistant.tools`), with every tool call shown under the answer and recorded
in the audit trail. Off until an admin enables it (``AppSettings.assistant``).

Local mode: the API key is in the OS keychain (``secret:anthropic/api_key``). Server mode: the
desktop sends questions to the server (``RemoteAssistant``), which holds the key.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from hcs_studio.assistant.conversation import (
    Answer,
    AssistantDisabled,
    AssistantUnavailable,
    Conversation,
    PayloadLog,
    Question,
    Step,
)
from hcs_studio.assistant.usage import BudgetExceeded
from hcs_studio.assistant.views import Privacy, build_snapshot

__all__ = [
    "Answer",
    "AssistantDisabled",
    "AssistantUnavailable",
    "BudgetExceeded",
    "Conversation",
    "PayloadLog",
    "Privacy",
    "Question",
    "Step",
    "make_client",
    "open_conversation",
]


def make_client(settings: Any) -> Any:
    """An ``anthropic.Anthropic`` client with the key from the keychain (local mode)."""
    try:
        import anthropic
    except ImportError as exc:
        raise AssistantUnavailable(
            "The assistant needs the 'assistant' extra: pip install 'hcs-studio[assistant]'"
        ) from exc
    from hcs_studio.secrets import get_secret

    key = get_secret(str(settings.api_key).removeprefix("secret:"))
    if not key:
        raise AssistantUnavailable(
            "No Claude API key is stored. An admin adds it in Settings → Assistant "
            "(keys come from the Claude Console, platform.claude.com)."
        )
    return anthropic.Anthropic(api_key=key, max_retries=3, timeout=600.0)


class _OwnedSnapshotConversation(Conversation):
    """A conversation that deletes its snapshot folder when closed."""

    workdir: Path | None = None

    def close(self) -> None:
        super().close()
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)


def open_conversation(
    ctx: Any,
    privacy: Privacy | None = None,
    *,
    client: Any = None,
    settings: Any = None,
    payload_log: PayloadLog | None = None,
    seed: int | None = None,
) -> Conversation:
    """A new conversation for ``ctx.actor``: a fresh curated snapshot, the audit trail as the
    audit sink, the user's budget. ``client`` defaults to :func:`make_client`."""
    from hcs_studio.config import load_settings
    from hcs_studio.services.assistant import audit_for

    settings = settings or load_settings().assistant
    if not settings.enabled:
        raise AssistantDisabled(
            "The lab assistant is off. An admin can enable it in Settings → Assistant."
        )
    privacy = privacy or Privacy(compound_ids_only=settings.compound_ids_only)
    client = client if client is not None else make_client(settings)
    workdir = Path(tempfile.mkdtemp(prefix="hcs-assistant-"))
    snapshot = build_snapshot(workdir / "views.sqlite", privacy, seed=seed)
    conversation = _OwnedSnapshotConversation(
        client,
        snapshot,
        settings,
        user=ctx.actor.id,
        audit=audit_for(ctx),
        payload_log=payload_log,
    )
    conversation.workdir = workdir
    return conversation
