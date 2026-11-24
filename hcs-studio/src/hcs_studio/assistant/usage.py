"""Per-user monthly token budget (Phase 28.4).

Usage is counted in *weighted tokens*: input + output + cache writes at full weight and cache
reads at 10 % (roughly what they cost), so prompt caching stretches the budget. The budget is
``AssistantSettings.monthly_token_budget`` unless ``user_budgets`` names the user (0 = no
limit).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

CACHE_READ_WEIGHT = 0.1


class BudgetExceeded(RuntimeError):
    """The user's monthly token budget is used up."""


def month(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


def weighted(usage: dict[str, int]) -> int:
    return int(
        usage.get("input_tokens", 0)
        + usage.get("output_tokens", 0)
        + usage.get("cache_creation_tokens", 0)
        + CACHE_READ_WEIGHT * usage.get("cache_read_tokens", 0)
    )


def usage_of(user: str, when: str | None = None) -> dict[str, Any]:
    """This month's usage of ``user``: totals, per model, weighted tokens."""
    from hcs_studio.db import session_scope
    from hcs_studio.db.assistant import AssistantUsage

    keys = (
        "requests",
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    )
    totals = dict.fromkeys(keys, 0)
    models: dict[str, dict[str, int]] = {}
    with session_scope() as session:
        rows = session.scalars(
            select(AssistantUsage).where(
                AssistantUsage.user == user, AssistantUsage.month == (when or month())
            )
        ).all()
        for row in rows:
            values = {k: int(getattr(row, k)) for k in keys}
            models[row.model] = values
            for k in keys:
                totals[k] += values[k]
    return {
        "user": user,
        "month": when or month(),
        **totals,
        "weighted": weighted(totals),
        "models": models,
    }


def budget_of(user: str, settings: Any) -> int:
    return int(settings.user_budgets.get(user, settings.monthly_token_budget))


def check_budget(user: str, settings: Any) -> dict[str, Any]:
    """Raise :class:`BudgetExceeded` when the user has no budget left; else the usage."""
    used = usage_of(user)
    budget = budget_of(user, settings)
    used["budget"] = budget
    if budget and used["weighted"] >= budget:
        raise BudgetExceeded(
            f"{user} used {used['weighted']:,} of {budget:,} assistant tokens this month; "
            "ask an admin to raise the budget."
        )
    return used


def record_usage(user: str, model: str, usage: dict[str, int]) -> None:
    """Add one request's token counts to this month's row."""
    from hcs_studio.db import session_scope
    from hcs_studio.db.assistant import AssistantUsage

    with session_scope() as session:
        row = session.scalars(
            select(AssistantUsage)
            .where(
                AssistantUsage.user == user,
                AssistantUsage.month == month(),
                AssistantUsage.model == model,
            )
            .with_for_update()
        ).first()
        if row is None:
            row = AssistantUsage(
                user=user,
                month=month(),
                model=model,
                requests=0,
                input_tokens=0,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
            )
            session.add(row)
        row.requests += 1
        row.input_tokens += int(usage.get("input_tokens", 0))
        row.output_tokens += int(usage.get("output_tokens", 0))
        row.cache_creation_tokens += int(usage.get("cache_creation_tokens", 0))
        row.cache_read_tokens += int(usage.get("cache_read_tokens", 0))
