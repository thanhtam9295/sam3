"""Service functions for accounts and the audit trail (Phase 26)."""

from __future__ import annotations

from typing import Any

from hcs_studio.services import ServiceContext, service


@service("accounts.add", object_type="account", reason_required=True)
def add_account(
    ctx: ServiceContext,
    username: str,
    role: str,
    display_name: str = "",
    password: str | None = None,
) -> Any:
    from hcs_studio.security.accounts import create_account

    info = create_account(username, role, display_name, password)
    ctx.note(
        object_id=username,
        after={"role": info.role, "display_name": info.display_name, "password": info.has_password},
    )
    return info


@service("accounts.update", object_type="account", reason_required=True)
def update_account(ctx: ServiceContext, username: str, **changes: Any) -> Any:
    from hcs_studio.security.accounts import get_account
    from hcs_studio.security.accounts import update_account as update

    before = get_account(username)
    info = update(username, **changes)
    shown = {
        k: ("(changed)" if k == "password" else v) for k, v in changes.items() if v is not None
    }
    ctx.note(
        object_id=username,
        before={"role": before.role, "active": before.active} if before else None,
        after={"role": info.role, "active": info.active, **shown},
    )
    return info


@service("audit.checkpoint", object_type="audit")
def audit_checkpoint(ctx: ServiceContext, note: str = "") -> dict[str, Any]:
    """Write the chain head to the checkpoint files."""
    from hcs_studio.security.audit import checkpoint

    entry = checkpoint(note or ctx.reason)
    ctx.note(object_id=str(entry.get("id", "")), after=entry)
    return entry
