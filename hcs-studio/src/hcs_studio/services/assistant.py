"""Service functions of the lab assistant (Phase 28.4): every question, tool call, image
confirmation and answer is an event in the audit trail; settings and the API key change only
through admins."""

from __future__ import annotations

from typing import Any

from hcs_studio.services import ServiceContext, service


@service("assistant.question", object_type="assistant_conversation")
def record_question(ctx: ServiceContext, details: dict[str, Any]) -> None:
    ctx.note(object_id=str(details.get("conversation", "")), after=details)


@service("assistant.tool", object_type="assistant_conversation")
def record_tool(ctx: ServiceContext, details: dict[str, Any]) -> None:
    ctx.note(object_id=str(details.get("conversation", "")), after=details)


@service("assistant.thumbnail", object_type="assistant_conversation")
def record_thumbnail(ctx: ServiceContext, details: dict[str, Any]) -> None:
    ctx.note(object_id=str(details.get("conversation", "")), after=details)


@service("assistant.answer", object_type="assistant_conversation")
def record_answer(ctx: ServiceContext, details: dict[str, Any]) -> None:
    ctx.note(object_id=str(details.get("conversation", "")), after=details)


RECORDERS = {
    "question": record_question,
    "tool": record_tool,
    "thumbnail": record_thumbnail,
    "answer": record_answer,
}


def audit_for(ctx: ServiceContext) -> Any:
    """An ``audit`` callback for :class:`~hcs_studio.assistant.conversation.Conversation`."""

    def audit(kind: str, details: dict[str, Any]) -> None:
        RECORDERS[kind](ctx, details)

    return audit


@service("settings.assistant", object_type="settings")
def configure_assistant(ctx: ServiceContext, **changes: Any) -> Any:
    """Change the assistant settings (admin): enable it, models, budgets, privacy defaults."""
    from hcs_studio.config import AssistantSettings, load_settings, save_settings

    settings = load_settings()
    before = settings.assistant.model_dump(mode="json")
    settings.assistant = AssistantSettings.model_validate({**before, **changes})
    save_settings(settings)
    ctx.note(object_id="assistant", before=before, after=settings.assistant.model_dump(mode="json"))
    return settings.assistant


@service("settings.assistant_key", object_type="settings")
def set_api_key(ctx: ServiceContext, value: str | None) -> None:
    """Store (or, with None, remove) the Claude API key in the OS keychain (admin). The value
    is never logged or written to settings."""
    from hcs_studio.config import load_settings
    from hcs_studio.secrets import delete_secret, set_secret

    name = load_settings().assistant.api_key.removeprefix("secret:")
    if value:
        set_secret(name, value)
    else:
        delete_secret(name)
    ctx.note(object_id=name, after={"stored": bool(value)})
