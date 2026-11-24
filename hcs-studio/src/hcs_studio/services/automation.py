"""Service functions of automatic runs (Phase 19)."""

from __future__ import annotations

from hcs_studio.services import ServiceContext, service


@service("automation.retry", object_type="auto_run", reason_required=True)
def retry_auto_run(ctx: ServiceContext, key: str) -> None:
    """Start an acquisition again (after fixing the routing, the data or the layout)."""
    from sqlalchemy import select

    from hcs_studio.automation.runner import retry
    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import AutoRun

    with session_scope() as session:
        run = session.scalars(select(AutoRun).where(AutoRun.key == key)).one()
        before = {"status": run.status, "detail": run.detail}
    retry(key)
    ctx.note(object_id=key, before=before, after={"status": "pending"})
