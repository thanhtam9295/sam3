"""Service functions for accounts and the audit trail (Phase 26)."""

from __future__ import annotations

from pathlib import Path
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


@service("run.sign", object_type="run")
def sign_run(ctx: ServiceContext, folder: Path, meaning: str, password: str | None = None) -> Any:
    """Sign a run as reviewed or approved (reviewer role). With ``password`` the signer is
    re-authenticated (required when the settings say so). ``ctx.reason`` is the comment."""
    from hcs_studio.security import signoff
    from hcs_studio.security.accounts import AuthError, authenticate, get_account
    from hcs_studio.services.operations import object_id

    reauthenticated = False
    if password is not None:
        try:
            authenticate(ctx.actor.id, password)
        except AuthError as exc:
            raise PermissionError(f"Re-authentication failed: {exc}") from None
        reauthenticated = True
    account = get_account(ctx.actor.id)
    before = signoff.status(Path(folder)).state
    result = signoff.sign(
        Path(folder),
        meaning,
        username=ctx.actor.id,
        display_name=account.display_name if account else ctx.actor.name,
        role=ctx.actor.role,
        comment=ctx.reason,
        reauthenticated=reauthenticated,
    )
    ctx.note(
        object_id=object_id(folder),
        before={"state": before},
        after={
            "state": result.state,
            "locked": result.locked,
            "meaning": meaning,
            "reauthenticated": reauthenticated,
        },
    )
    return result


@service("run.unlock", object_type="run", reason_required=True)
def unlock_run(ctx: ServiceContext, folder: Path) -> Any:
    """Admin: unlock an approved run (back to reviewed)."""
    from hcs_studio.security import signoff
    from hcs_studio.services.operations import object_id

    result = signoff.unlock(Path(folder))
    ctx.note(
        object_id=object_id(folder),
        before={"state": "approved", "locked": True},
        after={"state": result.state, "locked": False},
    )
    return result


@service("run.new_version", object_type="run", reason_required=True)
def new_run_version(ctx: ServiceContext, folder: Path) -> Path:
    """A linked new version of a run, to change instead of the locked original."""
    from hcs_studio.security import signoff
    from hcs_studio.services.operations import object_id

    target = signoff.new_version(Path(folder), ctx.reason)
    ctx.note(
        object_id=object_id(target),
        before={"run": object_id(folder)},
        after={"version": signoff.status(target).version},
    )
    return target
